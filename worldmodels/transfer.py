"""Cosmos 3 Transfer: sim -> real from the WeldCell simulator's control videos.

Control-to-video ("transfer") is served by the same Cosmos3-Nano Generator:

* ``omni`` (default local engine): vLLM-Omni ``POST /v1/videos/sync`` with the
  control hint inside ``extra_params`` (``{"edge": {"control_path": ...}}``,
  plus ``resolution``, ``control_guidance``, ``num_video_frames_per_chunk``,
  ``num_conditional_frames``, ``num_first_chunk_conditional_frames``,
  ``share_vision_temporal_positions``, ``max_frames``) - shape copied from the
  cosmos3-inference-transfer skill's ``build_transfer_request``. The control
  file must be readable by the server (same pod, ``--allowed-local-media-path /``);
  with ``COSMOS_GENERATOR_SHARED_FS=0`` it is uploaded as ``control_reference``
  + ``control_type`` (vLLM-Omni main only, single control).
* ``framework``: ``python -m cosmos_framework.scripts.inference -i spec.json``
  in a cosmos-framework checkout (spec schema copied from the skill's
  ``specs/edge.json``); supports per-control ``weight`` for multi-control.
* ``nim``: Cosmos3 NIM ``/v1/infer`` with ``transfer: {<kind>: {"video": data-url}}``.
* ``replay``: cache, else a placeholder that shows the dimmed control video.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from . import cache
from .config import Settings, get_settings, probe, repo_root, resolve_backend
from .generate import nim_resolution_key, post_nim, post_omni, run_media, sha_or_none
from .http import BackendError
from .prompts import compact, looks_like_json_prompt, negative_prompt, prompt_query_text, weld_scene_prompt
from .video import frames_4k1, probe as probe_video

FN_TRANSFER = "sim2real"
CONTROL_KINDS = ("edge", "depth", "blur", "seg", "wsm")

# NIM "Resolution Keys" table (width x height per tier and aspect)
_SIZES = {
    "256": {"16_9": (320, 192), "4_3": (320, 256), "1_1": (256, 256)},
    "480": {"16_9": (832, 480), "4_3": (736, 544), "1_1": (640, 640)},
    "720": {"16_9": (1280, 720), "4_3": (1104, 832), "1_1": (960, 960)},
}
TRANSFER_PRESETS = {
    "smoke": {"tier": "256", "steps": 20},
    "live": {"tier": "480", "steps": 35},
    "showcase": {"tier": "720", "steps": 50},     # skill specs: num_steps 50
}
DEFAULTS = {"guidance": 3.0, "control_guidance": 1.5, "flow_shift": 10.0, "num_video_frames_per_chunk": 121,
            "num_conditional_frames": 1, "num_first_chunk_conditional_frames": 0, "max_frames": 121}


def _aspect_name(w: int, h: int) -> tuple[str, str]:
    r = w / h
    for name, val, spec in (("16_9", 16 / 9, "16,9"), ("4_3", 4 / 3, "4,3"), ("1_1", 1.0, "1,1")):
        if abs(r - val) < 0.05:
            return name, spec
    return "16_9", "16,9"


def build_transfer_request(controls: dict[str, str], prompt: Any, *, seed: int = 0, preset: Optional[str] = None,
                           weights: Optional[dict[str, float]] = None, wrap: bool = True,
                           scene: Optional[dict] = None, s: Optional[Settings] = None, **overrides: Any) -> dict:
    s = s or get_settings()
    for k, p in controls.items():
        if k not in CONTROL_KINDS:
            raise ValueError(f"control kind must be one of {CONTROL_KINDS}, got {k!r}")
        if not Path(p).exists():
            raise FileNotFoundError(p)
    first = next(iter(controls.values()))
    info = probe_video(first)
    tp = dict(TRANSFER_PRESETS.get(preset or s.gen_preset, TRANSFER_PRESETS["live"]))
    aspect, aspect_spec = _aspect_name(info.width, info.height)
    tier = str(overrides.pop("tier", tp["tier"]))
    width, height = _SIZES[tier][aspect]
    opts = {**DEFAULTS, **{k: v for k, v in overrides.items() if k in DEFAULTS}}
    num_frames = frames_4k1(min(info.frames, int(opts["max_frames"])))
    fps = float(info.fps or 24.0)
    if isinstance(prompt, dict):
        prompt_text = compact(prompt)
    elif isinstance(prompt, str) and (looks_like_json_prompt(prompt) or not wrap):
        prompt_text = prompt
    else:
        sc = scene or {}
        prompt_text = compact(weld_scene_prompt(str(prompt), process=sc.get("process", "FCAW"),
                                                pass_type=sc.get("pass_type", "fill"), defect=sc.get("defect"),
                                                view=sc.get("view", "torch"), seconds=num_frames / fps, fps=fps,
                                                width=width, height=height, for_transfer=True))
    return {
        "controls": {k: str(Path(v).resolve()) for k, v in controls.items()},
        "weights": weights or None,
        "prompt": prompt_text,
        "query_text": prompt_query_text(prompt),
        "negative_prompt": negative_prompt("transfer"),
        "tier": tier, "aspect": aspect, "aspect_spec": aspect_spec, "width": width, "height": height,
        "num_frames": num_frames, "fps": fps, "steps": int(overrides.get("steps", tp["steps"])),
        "seed": int(seed), "guardrails": s.guardrails, **opts,
    }


def transfer_key(req: dict) -> tuple[str, str]:
    inputs = {k: v for k, v in req.items() if k not in ("controls", "negative_prompt", "query_text", "guardrails")}
    inputs["controls"] = {k: cache.FileRef(v) for k, v in req["controls"].items()}
    inputs["negative_sha"] = cache.text_hash(req["negative_prompt"])
    key = cache.make_key(FN_TRANSFER, inputs)
    alias = cache.alias_key(FN_TRANSFER, req["query_text"], {k: sha_or_none(v) for k, v in req["controls"].items()})
    return key, [alias]


# ------------------------------------------------------------------ vLLM-Omni
def omni_transfer_form(req: dict, s: Settings) -> tuple[dict, Optional[dict]]:
    extra: dict[str, Any] = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": bool(req["guardrails"]),
    }
    files = None
    if s.generator_shared_fs:
        for kind, path in req["controls"].items():
            hint: dict[str, Any] = {"control_path": path}
            if kind == "edge":
                hint["preset_edge_threshold"] = "medium"   # as in the skill's specs/edge.json
            if req.get("weights") and kind in req["weights"]:
                hint["control_weight"] = float(req["weights"][kind])  # vLLM-Omni main; cosmos3 image ignores
            extra[kind] = hint
    else:
        if len(req["controls"]) != 1:
            raise BackendError("multi-control upload needs a shared filesystem (COSMOS_GENERATOR_SHARED_FS=1)")
        kind, path = next(iter(req["controls"].items()))
        extra[kind] = {"control_weight": 1.0}
        files = {"control_reference": (Path(path).name, open(path, "rb"), "video/mp4")}
    extra.update({
        "resolution": req["tier"],
        "control_guidance": req["control_guidance"],
        "num_video_frames_per_chunk": req["num_video_frames_per_chunk"],
        "num_conditional_frames": req["num_conditional_frames"],
        "num_first_chunk_conditional_frames": req["num_first_chunk_conditional_frames"],
        "share_vision_temporal_positions": True,
        "max_frames": req["num_frames"],
    })
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
        "extra_params": json.dumps(extra, separators=(",", ":")),
    }
    if files:
        form["control_type"] = next(iter(req["controls"]))
    if s.generator_model:
        form["model"] = s.generator_model
    return form, files


def _call_omni(req: dict, s: Settings, out: Path) -> Path:
    form, files = omni_transfer_form(req, s)
    try:
        return post_omni(form, files, s, out)
    finally:
        for f in (files or {}).values():
            f[1].close()


# ------------------------------------------------------------------ cosmos-framework
def framework_spec(req: dict, name: str, prompt_path: Path) -> dict:
    """Spec in the format of cosmos3-inference-transfer/specs/{edge,depth,multi_control}.json."""
    spec: dict[str, Any] = {
        "name": name,
        "model_mode": "video2video",
        "resolution": req["tier"],
        "aspect_ratio": req["aspect_spec"],
        "num_frames": req["num_frames"],
        "max_frames": req["num_frames"],   # framework takes the length from the control video, capped by max_frames
        "fps": int(req["fps"]) if float(req["fps"]).is_integer() else req["fps"],
        "shift": req["flow_shift"],
        "num_steps": req["steps"],
        "seed": req["seed"],
        "num_video_frames_per_chunk": req["num_video_frames_per_chunk"],
        "num_conditional_frames": req["num_conditional_frames"],
        "num_first_chunk_conditional_frames": req["num_first_chunk_conditional_frames"],
        "share_vision_temporal_positions": True,
        "negative_metadata_mode": "none",
        "negative_prompt_keep_metadata": False,
        "guidance": req["guidance"],
        "control_guidance": req["control_guidance"],
        "negative_prompt_file": str(repo_root() / "configs" / "cosmos" / "negative_prompt_transfer.json"),
        "prompt_path": str(prompt_path),
        "emphasize_control_in_prompt": False,
    }
    for kind, path in req["controls"].items():
        hint: dict[str, Any] = {"control_path": path}
        if kind == "edge":
            hint["preset_edge_threshold"] = "medium"
        if len(req["controls"]) > 1:
            hint["weight"] = float((req.get("weights") or {}).get(kind, 1.0 / len(req["controls"])))
        spec[kind] = hint
    return spec


def framework_available(s: Settings) -> bool:
    return bool(s.framework_dir) and (Path(s.framework_dir) / ".venv" / "bin" / "python").exists()


def _clean_ld_path() -> str:
    bad = ("torch", "cudnn", "nccl", "tensorrt", "triton", "site-packages", "dist-packages")
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and not any(b in p for b in bad)]
    return ":".join(parts)


def _call_framework(req: dict, s: Settings, out: Path, key: str) -> Path:
    if not framework_available(s):
        raise BackendError("cosmos-framework checkout not found (set COSMOS_FRAMEWORK_DIR, run serve/install.sh)")
    work = s.cache_dir / "framework_runs" / key[:16]
    work.mkdir(parents=True, exist_ok=True)
    prompt_path = work / "prompt.json"
    prompt_path.write_text(req["prompt"] if looks_like_json_prompt(req["prompt"]) else json.dumps({"temporal_caption": req["prompt"]}))
    name = f"weld_{'_'.join(req['controls'])}_{key[:8]}"
    spec_path = work / "spec.json"
    spec_path.write_text(json.dumps(framework_spec(req, name, prompt_path), indent=2))
    fw = Path(s.framework_dir)
    cmd = [str(fw / ".venv" / "bin" / "python"), "-m", "cosmos_framework.scripts.inference",
           "--parallelism-preset=latency", "-i", str(spec_path), "-o", str(work / "out"),
           "--checkpoint-path", s.framework_checkpoint, "--seed", str(req["seed"])]
    if not req["guardrails"]:
        cmd.append("--no-guardrails")
    env = {**os.environ, "LD_LIBRARY_PATH": _clean_ld_path()}
    if os.environ.get("COSMOS_TRANSFER_GPU"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["COSMOS_TRANSFER_GPU"]
    log = work / "inference.log"
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, cwd=fw, env=env, stdout=lf, stderr=subprocess.STDOUT,
                              timeout=s.generator_timeout_s * 2)
    produced = work / "out" / name / "vision.mp4"
    if proc.returncode != 0 or not produced.exists():
        tail = log.read_text()[-1500:] if log.exists() else ""
        raise BackendError(f"cosmos_framework inference failed (rc={proc.returncode}); log {log}\n{tail}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(produced.read_bytes())
    return out


# ------------------------------------------------------------------ NIM
def nim_transfer_body(req: dict) -> dict:
    from .video import data_url

    transfer: dict[str, Any] = {kind: {"video": data_url(path, "video/mp4")} for kind, path in req["controls"].items()}
    transfer.update({"control_guidance": float(req["control_guidance"]),
                     "num_conditional_frames": int(req["num_conditional_frames"]),
                     "num_first_chunk_conditional_frames": int(req["num_first_chunk_conditional_frames"]),
                     "num_video_frames_per_chunk": int(req["num_video_frames_per_chunk"])})
    return {
        "model_mode": "video2video",
        "prompt": req["prompt"],
        "negative_prompt": req["negative_prompt"],
        "transfer": transfer,
        "resolution": nim_resolution_key(req["width"], req["height"]),
        "num_frames": int(req["num_frames"]),
        "fps": float(req["fps"]),
        "num_inference_steps": int(req["steps"]),
        "guidance_scale": float(req["guidance"]),
        "flow_shift": float(req["flow_shift"]),
        "seed": int(req["seed"]),
    }


# ------------------------------------------------------------------ public API
def _engine(s: Settings) -> str:
    if s.transfer_backend in ("omni", "framework"):
        return s.transfer_backend
    if probe(f"{s.generator_api}/models", timeout=s.probe_timeout_s):
        return "omni"
    return "framework" if framework_available(s) else "omni"


def sim2real_meta(controls: dict[str, str], prompt: Any, seed: int = 0, *, backend: Optional[str] = None,
                  **knobs: Any) -> dict:
    s = get_settings()
    req = build_transfer_request(controls, prompt, seed=seed, s=s, **knobs)
    key, aliases = transfer_key(req)
    b = backend or resolve_backend("generator", s)
    engine = None
    if b == "replay" and s.transfer_backend in ("auto", "framework") and framework_available(s) and backend is None:
        b = "local"  # generator server down but the framework checkout can run transfer offline
    if b == "local":
        engine = _engine(s)

    def call(bk: str, out: Path) -> Path:
        if bk == "nim":
            return post_nim(nim_transfer_body(req), s, out)
        if engine == "framework":
            return _call_framework(req, s, out, key)
        return _call_omni(req, s, out)

    kinds = "+".join(req["controls"])
    first_control = next(iter(req["controls"].values()))
    return run_media(FN_TRANSFER, req, key, aliases, call, backend=b,
                     placeholder_title=f"Sim2Real ({kinds}): {req['query_text'][:100]}",
                     placeholder_sub=f"Cosmos3 Transfer {req['width']}x{req['height']} {req['num_frames']}f seed {seed} - "
                                     "background shows the input control video",
                     placeholder_bg=first_control, seconds=req["num_frames"] / req["fps"],
                     extra_meta={"kind": kinds, "engine": engine}, allow_nearest=False)


def sim2real(control_video: str, kind: str, prompt: Any, seed: int = 0, **knobs: Any) -> str:
    """Photoreal weld footage from one simulator control video (``kind``: edge|depth|blur|seg)."""
    if kind == "multi":
        raise ValueError("use sim2real_multi() for multi-control transfer")
    return sim2real_meta({kind: control_video}, prompt, seed, **knobs)["path"]


def sim2real_multi(controls: dict[str, str], prompt: Any, seed: int = 0,
                   weights: Optional[dict[str, float]] = None, **knobs: Any) -> str:
    """Multi-control transfer, e.g. ``{"edge": edge.mp4, "depth": depth.mp4}``.

    The controls must be spatially and temporally aligned (the simulator renders
    them from the same camera). Weights are honoured by the framework engine and
    by vLLM-Omni main; the released vllm/vllm-omni:cosmos3 image uses equal weights.
    """
    return sim2real_meta(controls, prompt, seed, weights=weights, **knobs)["path"]


def controls_for_pass(pass_dir: str | Path) -> dict[str, str]:
    """``data/generated/<seq>/control/{edge,depth}.mp4`` -> {kind: path} (existing only)."""
    d = Path(pass_dir) / "control"
    return {k: str(d / f"{k}.mp4") for k in CONTROL_KINDS if (d / f"{k}.mp4").exists()}


__all__ = ["sim2real", "sim2real_multi", "sim2real_meta", "controls_for_pass", "build_transfer_request",
           "framework_spec", "omni_transfer_form", "nim_transfer_body", "TRANSFER_PRESETS"]
