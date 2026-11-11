"""Cosmos 3 Generator client: "what-if" video generation.

Backends
  * ``local``: vLLM-Omni ``POST {COSMOS_GENERATOR_URL}/v1/videos/sync`` form request
    (fields verified against the cosmos-framework generate-media skill / vLLM-Omni
    Cosmos3-Nano recipe): prompt, negative_prompt, size=WxH, num_frames, fps,
    num_inference_steps, guidance_scale, flow_shift, seed, extra_params (JSON:
    use_resolution_template, use_duration_template, guardrails, and for
    video2video condition_frame_indexes_vision / condition_video_keep), optional
    ``input_reference`` file (image -> image2video, video -> video2video).
    Response body is the MP4.
  * ``nim``: Cosmos3 NIM ``/v1/infer`` JSON schema (model_mode, resolution key,
    input_reference data URL ...); response ``{"b64_video": ...}`` (VP9-in-MP4,
    transcoded to H.264 here). Hosted URL https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano.
  * ``replay``: cache (exact key, then same prompt+conditioning, then nearest
    showcase prompt) or a labelled placeholder clip.
"""
from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from . import cache
from .config import SOURCE_CACHE, SOURCE_HEURISTIC, Settings, backend_source, get_settings, is_local_url, resolve_backend
from .http import BackendError, client, raise_for
from .prompts import compact, looks_like_json_prompt, negative_prompt, prompt_query_text, weld_scene_prompt
from .video import data_url, ensure_h264, frames_for_seconds, render_placeholder

FN_WHATIF = "what_if"

# Cosmos3 resolution tiers (NIM generation.md "Resolution Keys", 16:9 column).
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {"width": 320, "height": 192, "fps": 24, "steps": 20, "guidance": 6.0, "flow_shift": 10.0},
    "live": {"width": 832, "height": 480, "fps": 24, "steps": 35, "guidance": 6.0, "flow_shift": 10.0},
    "showcase": {"width": 1280, "height": 720, "fps": 24, "steps": 35, "guidance": 6.0, "flow_shift": 10.0},
}
_TIER_BY_HEIGHT = {192: "256", 256: "256", 480: "480", 544: "480", 640: "480", 720: "720", 832: "720", 960: "720"}


def resolve_size(resolution: Optional[str], preset: dict) -> tuple[int, int]:
    if not resolution:
        return int(preset["width"]), int(preset["height"])
    r = str(resolution).lower().replace("p", "")
    if "x" in r:
        w, h = r.split("x", 1)
        return int(w), int(h)
    return {"256": (320, 192), "480": (832, 480), "720": (1280, 720)}.get(r, (int(preset["width"]), int(preset["height"])))


def nim_resolution_key(width: int, height: int) -> str:
    tier = _TIER_BY_HEIGHT.get(min(width, height), "720" if min(width, height) >= 600 else "480")
    ratio = width / height
    for name, val in (("16_9", 16 / 9), ("4_3", 4 / 3), ("1_1", 1.0), ("3_4", 3 / 4), ("9_16", 9 / 16)):
        if abs(ratio - val) < 0.03:
            return tier if name == "16_9" else f"{tier}_{name}"
    return tier


def _mode(image_path: Optional[str], video_path: Optional[str]) -> str:
    if video_path:
        return "video2video"
    if image_path:
        return "image2video"
    return "text2video"


def build_request(prompt: Any, *, image_path: Optional[str] = None, video_path: Optional[str] = None,
                  seconds: float = 5.0, seed: int = 0, preset: Optional[str] = None,
                  resolution: Optional[str] = None, num_frames: Optional[int] = None, fps: Optional[float] = None,
                  steps: Optional[int] = None, guidance: Optional[float] = None, flow_shift: Optional[float] = None,
                  negative: Optional[str] = None, wrap: bool = True, scene: Optional[dict] = None,
                  condition: str = "last", s: Optional[Settings] = None) -> dict:
    """Resolve knobs into one backend-neutral request description (also the cache-key input)."""
    s = s or get_settings()
    p = dict(PRESETS.get(preset or s.gen_preset, PRESETS["live"]))
    width, height = resolve_size(resolution, p)
    fps_v = float(fps or p["fps"])
    frames = int(num_frames) if num_frames else frames_for_seconds(seconds, fps_v)
    if isinstance(prompt, dict):
        prompt_text = compact(prompt)
    elif isinstance(prompt, str) and (looks_like_json_prompt(prompt) or not wrap):
        prompt_text = prompt
    else:
        sc = scene or {}
        prompt_text = compact(weld_scene_prompt(str(prompt), process=sc.get("process", "FCAW"),
                                                pass_type=sc.get("pass_type", "fill"), defect=sc.get("defect"),
                                                view=sc.get("view", "torch"), seconds=frames / fps_v, fps=fps_v,
                                                width=width, height=height))
    return {
        "mode": _mode(image_path, video_path),
        "prompt": prompt_text,
        "query_text": prompt_query_text(prompt),
        "negative_prompt": negative if negative is not None else negative_prompt("video"),
        "width": width, "height": height, "num_frames": frames, "fps": fps_v,
        "steps": int(steps or p["steps"]), "guidance": float(guidance or p["guidance"]),
        "flow_shift": float(flow_shift or p["flow_shift"]), "seed": int(seed),
        "image_path": image_path, "video_path": video_path, "condition": condition,
        "guardrails": s.guardrails,
    }


def request_key(req: dict) -> tuple[str, str]:
    inputs = {k: v for k, v in req.items() if k not in ("image_path", "video_path", "query_text", "negative_prompt", "guardrails")}
    inputs["negative_sha"] = cache.text_hash(req["negative_prompt"])
    inputs["image"] = cache.FileRef(req["image_path"]) if req.get("image_path") else None
    inputs["video"] = cache.FileRef(req["video_path"]) if req.get("video_path") else None
    key = cache.make_key(FN_WHATIF, inputs)
    alias = cache.alias_key(FN_WHATIF, req["query_text"], sha_or_none(req.get("image_path")),
                            sha_or_none(req.get("video_path")))
    return key, alias


def sha_or_none(path: Optional[str]) -> Optional[str]:
    try:
        return cache.file_sha256(path) if path else None
    except OSError:
        return None


# ------------------------------------------------------------------ backends
def _extra_params(req: dict) -> dict:
    ep: dict[str, Any] = {"use_resolution_template": False, "use_duration_template": False,
                          "guardrails": bool(req.get("guardrails"))}
    if req["mode"] == "video2video":
        ep["condition_frame_indexes_vision"] = [0, 1]
        ep["condition_video_keep"] = req.get("condition", "last")
    return ep


def omni_form(req: dict, s: Settings) -> dict[str, str]:
    form = {
        "prompt": req["prompt"],
        "negative_prompt": req["negative_prompt"],
        "size": f"{req['width']}x{req['height']}",
        "num_frames": str(req["num_frames"]),
        "fps": str(int(req["fps"]) if float(req["fps"]).is_integer() else req["fps"]),
        "num_inference_steps": str(req["steps"]),
        "guidance_scale": str(req["guidance"]),
        "flow_shift": str(req["flow_shift"]),
        "seed": str(req["seed"]),
        "extra_params": json.dumps(_extra_params(req), separators=(",", ":")),
    }
    if s.generator_model:
        form["model"] = s.generator_model
    return form


def _write_video_response(r: httpx.Response, out: Path) -> Path:
    ctype = r.headers.get("content-type", "")
    body = r.content
    if "json" in ctype or body[:1] in (b"{", b"["):
        data = r.json()
        b64 = data.get("b64_video") or (data.get("data") or [{}])[0].get("b64_json") or (data.get("data") or [{}])[0].get("b64_video")
        if not b64:
            raise BackendError(f"generator returned JSON without video: {str(data)[:300]}")
        body = base64.b64decode(b64)
    if len(body) < 1000 or b"ftyp" not in body[:64]:
        raise BackendError(f"generator response is not an MP4 ({len(body)} bytes, {ctype})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    return out


def post_omni(form: dict, files: Optional[dict], s: Settings, out: Path) -> Path:
    url = f"{s.generator_api}/videos/sync"
    try:
        with client(s.generator_timeout_s, {"Accept": "video/mp4"}) as c:
            r = c.post(url, data=form, files=files or None)
    except httpx.HTTPError as e:
        raise BackendError(f"vLLM-Omni request failed: {type(e).__name__}: {e}") from e
    raise_for(r, "vLLM-Omni /v1/videos/sync")
    return _write_video_response(r, out)


def _call_local(req: dict, s: Settings, out: Path) -> Path:
    form = omni_form(req, s)
    files = None
    handles = []
    try:
        ref = req.get("video_path") or req.get("image_path")
        if ref:
            fh = open(ref, "rb")
            handles.append(fh)
            ctype = "video/mp4" if req.get("video_path") else ("image/png" if str(ref).lower().endswith(".png") else "image/jpeg")
            files = {"input_reference": (Path(ref).name, fh, ctype)}
        return post_omni(form, files, s, out)
    finally:
        for h in handles:
            h.close()


def nim_url(s: Settings) -> str:
    u = s.nim_generator_url.rstrip("/")
    if is_local_url(u) and not u.endswith("/v1/infer"):
        u = f"{u}/v1/infer"  # self-hosted Cosmos3 NIM
    return u


def nim_body(req: dict) -> dict:
    body: dict[str, Any] = {
        "model_mode": req["mode"],
        "prompt": req["prompt"],
        "negative_prompt": req["negative_prompt"],
        "resolution": nim_resolution_key(req["width"], req["height"]),
        "num_frames": int(req["num_frames"]),
        "fps": float(req["fps"]),
        "num_inference_steps": int(req["steps"]),
        "guidance_scale": float(req["guidance"]),
        "flow_shift": float(req["flow_shift"]),
        "seed": int(req["seed"]),
    }
    if req.get("video_path"):
        body["input_reference"] = data_url(req["video_path"], "video/mp4")
        body["condition_frame_indexes_vision"] = [0, 1]
        body["condition_video_keep"] = req.get("condition", "last")
    elif req.get("image_path"):
        body["input_reference"] = data_url(req["image_path"], "image/png")
    return body


def post_nim(body: dict, s: Settings, out: Path) -> Path:
    headers = {"Accept": "application/json"}
    if s.nvidia_api_key:
        headers["Authorization"] = f"Bearer {s.nvidia_api_key}"
    elif not is_local_url(s.nim_generator_url):
        raise BackendError("nim backend needs NVIDIA_API_KEY")
    url = nim_url(s)
    try:
        with client(s.generator_timeout_s, headers) as c:
            r = c.post(url, json=body)
    except httpx.HTTPError as e:
        raise BackendError(f"NIM request failed: {type(e).__name__}: {e}") from e
    if r.status_code == 404 and "page not found" in r.text.lower():
        raise BackendError("NIM generator route returned plain-text 404 (hosted cosmos3-nano endpoint not provisioned "
                           "for this account - see worldmodels/README.md)", status=404)
    raise_for(r, f"NIM {url}")
    _write_video_response(r, out)
    if s.transcode_h264:
        ensure_h264(out)
    return out


def _call_nim(req: dict, s: Settings, out: Path) -> Path:
    return post_nim(nim_body(req), s, out)


# ------------------------------------------------------------------ public API
def _replay_lookup(fn: str, key: str, alias: str, query_text: str, t0: float, allow_nearest: bool = True) -> Optional[dict]:
    if get_settings().cache_mode == "off":
        return None
    k2 = cache.resolve_alias(fn, alias)
    hit = cache.get_media(fn, k2) if k2 else None
    match = "same-prompt"
    score = None
    if not hit and allow_nearest and query_text:
        near = cache.nearest_media(fn, query_text)
        if near:
            hit = (near[0], near[1])
            match, score = "nearest-prompt", round(near[2], 3)
    if not hit:
        return None
    path, meta = hit
    return {**meta, "path": str(path), "source": SOURCE_CACHE, "origin_source": meta.get("source"),
            "origin_latency_ms": meta.get("latency_ms"), "cache_match": match, "match_score": score,
            "latency_ms": int(round((time.perf_counter() - t0) * 1000))}


def run_media(fn: str, req: dict, key: str, alias: str, call, *, backend: Optional[str], placeholder_title: str,
              placeholder_sub: str, placeholder_bg: Optional[str], seconds: float, extra_meta: Optional[dict] = None,
              allow_nearest: bool = True) -> dict:
    """Shared cache / backend / fallback driver for every video-producing call."""
    t0 = time.perf_counter()
    s = get_settings()
    ms = lambda: int(round((time.perf_counter() - t0) * 1000))  # noqa: E731
    public_req = {k: v for k, v in req.items() if k not in ("negative_prompt",)}
    if cache.reading_enabled():
        hit = cache.get_media(fn, key)
        if hit:
            path, meta = hit
            return {**meta, "path": str(path), "source": SOURCE_CACHE, "origin_source": meta.get("source"),
                    "origin_latency_ms": meta.get("latency_ms"), "cache_match": "exact", "latency_ms": ms()}
    b = backend or resolve_backend("generator", s)
    error = None
    if b in ("local", "nim"):
        try:
            with tempfile.TemporaryDirectory(prefix="cosmos-") as td:
                tmp_out = Path(td) / "out.mp4"
                call(b, tmp_out)
                meta = {"source": backend_source(b), "backend": b, "latency_ms": ms(), "fn": fn,
                        "query_text": req.get("query_text", ""), "request": public_req, **(extra_meta or {})}
                if cache.writing_enabled():
                    dst = cache.put_media(fn, key, tmp_out, meta, aliases=[alias], move=True)
                else:  # cache off / read-only: still need a durable file
                    dst = cache.cache_root() / "uncached" / f"{key}.mp4"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp_out.replace(dst)
                return {**meta, "path": str(dst), "cache_key": key}
        except (BackendError, OSError) as e:
            error = str(e)
            if not s.fallback:
                raise
    hit = _replay_lookup(fn, key, alias, req.get("query_text", ""), t0, allow_nearest=allow_nearest)
    if hit:
        if error:
            hit["error"] = error
        return hit
    ph = cache.placeholder_path(key)
    if not ph.exists():
        render_placeholder(ph, placeholder_title, placeholder_sub, seconds=min(4.0, max(2.0, seconds)),
                           background=placeholder_bg, seed=int(req.get("seed", 0)))
    return {"path": str(ph), "source": SOURCE_HEURISTIC, "placeholder": True, "backend": b, "latency_ms": ms(),
            "fn": fn, "query_text": req.get("query_text", ""), "request": public_req, "cache_key": key, "error": error,
            **(extra_meta or {})}


def what_if_meta(prompt: Any, image_path: Optional[str] = None, video_path: Optional[str] = None,
                 seconds: float = 5, seed: int = 0, *, backend: Optional[str] = None, **knobs: Any) -> dict:
    """Generate a what-if clip; returns ``{path, source, latency_ms, cache_key, request, ...}``.

    knobs: preset (smoke|live|showcase), resolution ("832x480" | "480" | "720"), num_frames, fps,
    steps, guidance, flow_shift, negative, wrap (wrap free text into the weld-cell JSON prompt),
    scene ({process, pass_type, defect, view}), condition ("last"|"first" for video2video).
    """
    s = get_settings()
    req = build_request(prompt, image_path=image_path, video_path=video_path, seconds=seconds, seed=seed, s=s, **knobs)
    key, alias = request_key(req)

    def call(b: str, out: Path) -> Path:
        return _call_local(req, s, out) if b == "local" else _call_nim(req, s, out)

    return run_media(FN_WHATIF, req, key, alias, call, backend=backend,
                     placeholder_title=f"What if: {req['query_text'][:120]}",
                     placeholder_sub=f"Cosmos3 {req['mode']} {req['width']}x{req['height']} {req['num_frames']}f seed {seed}",
                     placeholder_bg=video_path or image_path, seconds=float(seconds), extra_meta={"mode": req["mode"]})


def what_if(prompt: Any, image_path: Optional[str] = None, video_path: Optional[str] = None,
            seconds: float = 5, seed: int = 0, **knobs: Any) -> str:
    """What-if generation with the Cosmos3 Generator; returns the MP4 path.

    ``image_path`` -> image2video (e.g. a frame of the real pass), ``video_path`` ->
    video2video continuing from the clip's last frames, neither -> text2video.
    """
    return what_if_meta(prompt, image_path, video_path, seconds, seed, **knobs)["path"]


def read_sidecar(mp4_path: str) -> dict:
    """Metadata (source, latency_ms, request ...) for an MP4 returned by this package."""
    p = Path(mp4_path)
    sp = p.with_name(p.stem + ".meta.json")
    if sp.exists():
        try:
            return json.loads(sp.read_text())
        except Exception:
            return {}
    if "placeholders" in p.parts:
        return {"source": SOURCE_HEURISTIC, "placeholder": True}
    return {}
