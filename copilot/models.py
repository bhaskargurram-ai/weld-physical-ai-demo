"""Adapter to `worldmodels` (Cosmos 3 Reasoner / Generator / Transfer) with a transparent local mock.

Contract (PLAN.md):
    reason.assess_clip(video_path, context) -> {verdict, defects[{type,confidence,t_s}], root_cause,
                                                wps_findings, recommended_action, reasoning}
    generate.what_if(prompt, image_path=None, video_path=None, seconds=5, seed=0) -> mp4 path
    transfer.sim2real(control_video, kind, prompt, seed=0) -> mp4 path

If worldmodels is not importable (or raises - e.g. replay-cache miss with no GPU), the copilot uses
a heuristic mock and labels the output `source: copilot-mock` so it is never mistaken for Cosmos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import httpx

from . import config
from .data import DataLayer, _cached, _try_import

log = logging.getLogger("copilot.models")
Progress = Callable[[float, str], None]

ROOT_CAUSES = {
    "porosity": ("Shielding-gas coverage loss: gas flow fell below the WPS minimum, letting atmospheric "
                 "N2/O2 into the pool; erratic voltage confirms arc instability from contamination.",
                 "Check gas hose/regulator and nozzle for spatter build-up or draft; grind out the affected "
                 "length and re-weld; add a gas-flow interlock (< WPS min -> arc stop)."),
    "lack_of_fusion": ("Insufficient heat input: travel speed ran high while current sagged, so the fill "
                       "did not wet into the bevel sidewalls.",
                       "Excavate and re-weld; verify robot travel-speed override and contact-tip wear; "
                       "add a heat-input lower-bound alarm."),
    "undercut": ("Excess arc voltage with fast travel melted the base metal at the toes without filling it.",
                 "Blend or re-weld toes; reduce voltage to WPS range and check torch angle/stick-out."),
    "burn_through": ("Excessive heat on the root: high current with slow travel opened the keyhole.",
                     "Cut out and replace the root; verify positioner speed and root gap / land."),
    "arc_instability": ("Irregular wire feed / stick-out producing high-frequency voltage noise.",
                        "Inspect drive rolls, liner and contact tip; minor - monitor, visual re-check."),
    "crack": ("Interpass temperature exceeded the WPS maximum, raising hydrogen / hot-cracking risk.",
              "Excavate, MT/PT to confirm sound metal, re-weld respecting max interpass; add an IR interlock."),
}


def backend_modes() -> dict:
    """Concrete backend per service. Uses worldmodels.config.resolve_backend when available (it resolves
    COSMOS_BACKEND=auto by probing the local vLLM servers / NVIDIA_API_KEY), else the env var."""
    wmc = _try_import("worldmodels.config")
    out = {}
    for svc in ("reasoner", "generator"):
        mode = None
        f = getattr(wmc, "resolve_backend", None) if wmc else None
        if callable(f):
            try:
                mode = str(f(svc)).lower()
            except Exception:  # noqa: BLE001
                mode = None
        if mode is None:
            mode = config.cosmos_backend()
            if mode == "auto":
                mode = "replay"
        out[svc] = mode
    return out


def backend_mode() -> str:
    m = _cached("cosmos_modes", 10, backend_modes)
    return m["generator"] if m["generator"] == m["reasoner"] else f"{m['reasoner']}/{m['generator']}"


def mode_label(mode: str | None = None) -> str:
    mode = mode or backend_mode()
    names = {"local": "LIVE GPU", "nim": "NIM", "replay": "REPLAY"}
    return " / ".join(names.get(x, x.upper()) for x in mode.split("/"))


def cosmos_health() -> dict:
    """Reasoner/Generator status derived from the worldmodels config (backend resolution + replay cache)."""
    def probe():
        modes = _cached("cosmos_modes", 10, backend_modes)
        wm = _try_import("worldmodels")
        mode = backend_mode()
        out = {"mode": mode, "mode_label": mode_label(mode), "worldmodels": wm is not None, "modes": modes}
        cache_dir = config.data_dir() / "cache" / "worldmodels"
        wmc = _try_import("worldmodels.config")
        if wmc is not None and hasattr(wmc, "get_settings"):
            try:
                cache_dir = Path(wmc.get_settings().cache_dir)
            except Exception:  # noqa: BLE001
                pass
        n_cache = sum(1 for _ in cache_dir.glob("**/*.json")) if cache_dir.exists() else 0
        out["replay_cache_files"] = n_cache
        for svc in ("reasoner", "generator"):
            m = modes[svc]
            if m == "local":
                out[svc] = True            # resolve_backend only picks local after a successful probe
            elif m == "nim":
                out[svc] = bool(os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY"))
            else:
                out[svc] = n_cache > 0     # replay works only with a populated cache (else heuristic/placeholder)
        return out
    return _cached("cosmos", 8, probe)


def _http_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=0.6).status_code < 500
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ ffmpeg helpers
def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True, timeout=180)


def cut_clip(src: Path, t0: float, seconds: float, out: Path) -> Path:
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-ss", f"{max(t0, 0):.2f}", "-i", str(src), "-t", f"{seconds:.2f}", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out))
    return out


def grab_frame(src: Path, t: float, out: Path) -> Path:
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg("-ss", f"{max(t, 0):.2f}", "-i", str(src), "-frames:v", "1", str(out))
    return out


# ------------------------------------------------------------------ change templates
CHANGE_TEMPLATES = {
    "travel_speed_-20": {
        "label": "Travel speed −20%",
        "text": "Robot travel speed is reduced by 20 percent. Heat input rises about 25 percent: the molten pool "
                "grows larger and hotter, the bead becomes wider and flatter with coarser ripples, and the glow "
                "trail behind the arc lengthens.",
        "ffmpeg": "setpts=1.25*PTS,eq=brightness=0.05:saturation=1.35,colorbalance=rs=0.12:gs=0.04",
    },
    "gas_loss": {
        "label": "Shielding gas loss",
        "text": "Shielding gas flow collapses from 21 to 5 litres per minute. The arc becomes erratic and "
                "flickers, spatter increases, the pool surface oxidises and gas pores appear in the trailing bead.",
        "ffmpeg": "noise=alls=22:allf=t,eq=contrast=1.25:brightness=-0.03:saturation=0.8",
    },
    "porosity": {
        "label": "Porosity synthesis",
        "text": "Clusters of spherical gas pores (porosity) form in the solidifying weld bead behind the arc, "
                "with visible surface-breaking blowholes.",
        "ffmpeg": "noise=alls=14:allf=t+u,eq=contrast=1.3:saturation=0.9,unsharp=5:5:1.2",
    },
    "voltage_+20": {
        "label": "Voltage +20% (undercut)",
        "text": "Arc voltage increases by 20 percent: the arc lengthens and widens, and an undercut groove is "
                "melted into both toes of the bead without being filled.",
        "ffmpeg": "eq=brightness=0.08:contrast=1.1,colorbalance=bs=0.1",
    },
    "interpass_hot": {
        "label": "Interpass overheat (crack)",
        "text": "Interpass temperature exceeds the procedure maximum; a fine longitudinal centreline crack opens "
                "in the cooling bead.",
        "ffmpeg": "colorbalance=rs=0.2:gs=0.05,eq=saturation=1.4:contrast=1.15",
    },
}


def parse_change(change: str) -> tuple[str | None, str]:
    """Map free text to a template key (or None) and a natural-language description."""
    c = (change or "").strip()
    if c in CHANGE_TEMPLATES:
        return c, CHANGE_TEMPLATES[c]["text"]
    lc = c.lower()
    if "travel" in lc or "speed" in lc:
        m = re.search(r"([+\-−]?\s*\d+)\s*%", lc)
        pct = int(m.group(1).replace("−", "-").replace(" ", "")) if m else -20
        if pct > 0 and ("drop" in lc or "reduc" in lc or "slow" in lc or "decreas" in lc):
            pct = -pct
        if pct == -20:
            return "travel_speed_-20", CHANGE_TEMPLATES["travel_speed_-20"]["text"]
        return ("travel_speed_-20" if pct < 0 else None), (f"Robot travel speed changes by {pct} percent; heat input changes by about "
                                    f"{round((1 / (1 + pct / 100) - 1) * 100)} percent with a matching change in "
                                    "pool size, bead width and ripple spacing.")
    if "gas" in lc:
        return "gas_loss", CHANGE_TEMPLATES["gas_loss"]["text"]
    if "poros" in lc:
        return "porosity", CHANGE_TEMPLATES["porosity"]["text"]
    if "volt" in lc or "undercut" in lc:
        return "voltage_+20", CHANGE_TEMPLATES["voltage_+20"]["text"]
    if "interpass" in lc or "crack" in lc or "temp" in lc:
        return "interpass_hot", CHANGE_TEMPLATES["interpass_hot"]["text"]
    return None, c


def build_prompt(meta: dict, change_text: str | None, base_prompt: str | dict | None = None) -> str:
    """Cosmos-3 JSON-style ('upsampled') prompt."""
    p: dict = {}
    if isinstance(base_prompt, str) and base_prompt.strip().startswith("{"):
        try:
            p = json.loads(base_prompt)
        except ValueError:
            p = {}
    elif isinstance(base_prompt, dict):
        p = dict(base_prompt)
    proc = meta.get("process", "FCAW")
    p.setdefault("subject", f"robotic {proc} {meta.get('pass_type', '')} pass on {meta.get('material', 'carbon-steel')} pipe")
    p.setdefault("scene", "ASME pressure-piping fabrication shop weld cell, pipe on a positioner")
    p.setdefault("camera", "torch-mounted camera looking at the weld pool, 24 fps, fixed framing")
    p.setdefault("lighting", "intense arc light, orange molten pool, dark surroundings")
    p.setdefault("style", "photorealistic industrial footage, sharp focus, realistic spatter")
    if change_text:
        p["action"] = change_text
    return json.dumps(p)


# ------------------------------------------------------------------ the adapter
class WorldModels:
    def __init__(self, data: DataLayer):
        self.data = data
        self._lock = threading.Lock()

    @property
    def out_dir(self) -> Path:
        d = config.state_dir() / "gen"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- clip resolution
    def resolve_clip(self, sequence: str | None = None, clip_id: str | None = None,
                     t: float | None = None, seconds: float = 5.0) -> dict:
        """-> {sequence, clip_id, t0, t1, video_path, manifest_row}"""
        row = None
        if clip_id:
            row = next((r for r in self.data.manifest() if r.get("clip_id") == clip_id), None)
            if row:
                sequence = row.get("sequence_name")
        if not sequence:
            raise ValueError("need a sequence or clip_id")
        meta = self.data.get_meta(sequence)
        dur = float(meta.get("duration_s") or 30)
        if row:
            t0, t1 = float(row["t0_s"]), float(row["t1_s"])
        else:
            if t is None:
                wins = self.data.defect_windows(sequence)
                t = (wins[0]["t_start"] + wins[0]["t_end"]) / 2 if wins else dur / 2
            t0 = max(0.0, min(float(t) - seconds / 2, dur - seconds))
            t1 = t0 + seconds
            row = next((r for r in self.data.manifest() if r.get("sequence_name") == sequence
                        and float(r.get("t0_s", -1)) <= t0 + seconds / 2 <= float(r.get("t1_s", -1))), None)
        video = None
        if row:
            uri = row.get("clip_uri") or ""
            for cand in (Path(uri), config.curated_dir() / uri):
                if uri and cand.exists() and cand.is_file():
                    video = cand
                    break
        src = self.data.pass_dir(sequence) / "torch_cam.mp4"
        if video is None and src.exists() and shutil.which("ffmpeg"):
            video = cut_clip(src, t0, t1 - t0, config.state_dir() / "clips" / f"{sequence}_{t0:06.2f}.mp4")
        return {"sequence": sequence, "clip_id": row.get("clip_id") if row else None, "t0": round(t0, 2),
                "t1": round(t1, 2), "video_path": str(video) if video else None, "manifest_row": row}

    def context_for(self, sequence: str, t0: float, t1: float) -> dict:
        meta = self.data.get_meta(sequence)
        df = self.data.get_signals(sequence, t0, t1)
        stats = {}
        for c in df.columns:
            if c in ("t_s", "t_ns"):
                continue
            s = df[c]
            if s.dtype.kind in "fi":
                stats[c] = {"mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
                            "min": round(float(s.min()), 3), "max": round(float(s.max()), 3)}
        viol = [v for v in self.data.wps_violations(sequence) if v["t_end_s"] >= t0 and v["t_start_s"] <= t1]
        # Same shape as the mosaico_io export sidecar (what worldmodels.reason expects as `context`).
        return {"clip_id": None, "sequence_name": sequence, "t0_s": t0, "t1_s": t1, "meta": meta,
                "signal_stats": stats, "wps_violations": viol,
                "defects_in_window": [d for d in self.data.defect_windows(sequence)
                                      if d["t_end"] >= t0 and d["t_start"] <= t1],
                "rt_result": meta.get("rt_result"),
                "wps_limits": self.data.wps_limits(sequence)}   # copilot-normalised (interpass as [None, max])

    # ---- Reasoner
    def assess(self, sequence: str | None = None, clip_id: str | None = None, t: float | None = None) -> dict:
        clip = self.resolve_clip(sequence, clip_id, t)
        ctx = self.context_for(clip["sequence"], clip["t0"], clip["t1"])
        ctx["clip_id"] = clip["clip_id"]
        t_start = time.time()
        res, source, note = None, None, None
        reason = _try_import("worldmodels.reason")
        if reason is not None and hasattr(reason, "assess_clip") and clip["video_path"]:
            try:
                res = reason.assess_clip(clip["video_path"], ctx)
                source = (res or {}).get("source") or f"cosmos3-reasoner · {backend_mode()}"
                if res and res.get("latency_ms") is not None:
                    res["latency_s"] = res["latency_ms"] / 1000.0
            except Exception as e:  # noqa: BLE001
                note = f"worldmodels.reason failed ({type(e).__name__}: {e}); heuristic fallback"
                log.warning(note)
                res = None
        if res is None:
            res = mock_assess(ctx)
            source = "copilot-mock (no worldmodels)" if reason is None else "copilot-mock (fallback)"
        res = dict(res)
        res.setdefault("defects", [])
        res["source"] = source
        res["latency_s"] = round(res.get("latency_s") or (time.time() - t_start), 2)
        dur = clip["t1"] - clip["t0"]
        for d in res["defects"]:   # worldmodels reports t_s relative to the clip start
            ts = d.get("t_s")
            if isinstance(ts, (int, float)):
                d["t_abs"] = round(clip["t0"] + ts if ts <= dur + 0.01 else ts, 2)
        res["sequence_name"] = clip["sequence"]
        res["clip_id"] = clip["clip_id"]
        res["t0_s"], res["t1_s"] = clip["t0"], clip["t1"]
        if clip["video_path"]:
            from .data import file_url
            res["video_url"] = file_url(clip["video_path"])
        if note:
            res["note"] = note
        res["rt_result"] = ctx.get("rt_result")
        return res

    # ---- Generator / Transfer (run inside a job thread)
    def what_if(self, sequence: str | None, change: str, clip_id: str | None = None, t: float | None = None,
                seed: int = 0, prompt: str | None = None, progress: Progress | None = None) -> dict:
        progress = progress or (lambda p, m: None)
        progress(0.04, "Resolving source clip")
        clip = self.resolve_clip(sequence, clip_id, t)
        meta = self.data.get_meta(clip["sequence"])
        key, change_text = parse_change(change)
        base = (clip.get("manifest_row") or {}).get("generator_prompt")
        full_prompt = prompt or build_prompt(meta, change_text, base)
        progress(0.1, "Extracting conditioning frame")
        img = None
        src = self.data.pass_dir(clip["sequence"]) / "torch_cam.mp4"
        if src.exists() and shutil.which("ffmpeg"):
            img = grab_frame(src, clip["t0"], config.state_dir() / "frames" / f"{clip['sequence']}_{clip['t0']:06.2f}.png")
        gen = _try_import("worldmodels.generate")
        out, info, note = None, {}, None
        t_start = time.time()
        if gen is not None and hasattr(gen, "what_if"):
            # worldmodels wraps free text into its tuned weld-scene JSON prompt; pass our JSON only on override
            wm_prompt = prompt or change_text
            scene = {"process": meta.get("process", "FCAW"), "pass_type": meta.get("pass_type", "fill"),
                     "defect": key if key in ("porosity",) else None, "view": "torch"}
            try:
                out, info = self._call_wm(gen, "what_if", progress, "Cosmos 3 Generator", wm_prompt,
                                          image_path=str(img) if img else None, seconds=5, seed=seed,
                                          **({} if prompt else {"scene": scene}))
                full_prompt = (info.get("request") or {}).get("prompt") or full_prompt
            except Exception as e:  # noqa: BLE001
                note = f"worldmodels.generate failed ({type(e).__name__}: {e}); copilot mock render"
                log.warning(note)
        if out is None:
            out = self._mock_video(clip["video_path"], key, f"whatif_{clip['sequence']}_{key}_{seed}", progress)
            info = {"source": "copilot-mock (no worldmodels)" if gen is None else "copilot-mock (fallback)"}
        return self._result("what_if", clip, out, info, time.time() - t_start, full_prompt,
                            label=CHANGE_TEMPLATES.get(key or "", {}).get("label") or change, note=note,
                            change=change)

    def sim2real(self, sequence: str, kind: str = "edge", t: float | None = None, seed: int = 0,
                 prompt: str | None = None, progress: Progress | None = None) -> dict:
        progress = progress or (lambda p, m: None)
        progress(0.04, f"Loading {kind} control video")
        clip = self.resolve_clip(sequence, None, t)
        meta = self.data.get_meta(sequence)
        ctrl_src = self.data.pass_dir(sequence) / "control" / f"{kind}.mp4"
        if not ctrl_src.exists():
            raise FileNotFoundError(f"no control/{kind}.mp4 for {sequence}")
        ctrl = cut_clip(ctrl_src, clip["t0"], clip["t1"] - clip["t0"],
                        config.state_dir() / "clips" / f"{sequence}_{kind}_{clip['t0']:06.2f}.mp4")
        full_prompt = prompt or build_prompt(meta, (
            f"Photorealistic close-up of a robotic {meta.get('process')} weld on {meta.get('material')} pipe: "
            "bright arc, molten orange pool, realistic spatter, metallic bead with temper colours."))
        tr = _try_import("worldmodels.transfer")
        out, info, note = None, {}, None
        t_start = time.time()
        if tr is not None and hasattr(tr, "sim2real"):
            try:
                out, info = self._call_wm(tr, "sim2real", progress, "Cosmos 3 Transfer", full_prompt,
                                          controls_or_video=str(ctrl), kind=kind, seed=seed)
            except Exception as e:  # noqa: BLE001
                note = f"worldmodels.transfer failed ({type(e).__name__}: {e}); copilot mock render"
                log.warning(note)
        if out is None:
            out = self._mock_video(clip["video_path"], "sim2real", f"sim2real_{sequence}_{kind}_{seed}", progress)
            info = {"source": "copilot-mock (no worldmodels)" if tr is None else "copilot-mock (fallback)"}
        res = self._result("sim2real", clip, out, info, time.time() - t_start, full_prompt,
                           label=f"Sim→Real ({kind})", note=note)
        from .data import file_url
        res["control_url"] = file_url(ctrl)
        res["kind"] = kind
        return res

    # ---- helpers
    def _call_wm(self, mod, fn: str, progress: Progress, what: str, prompt, controls_or_video: str | None = None,
                 kind: str | None = None, **kw) -> tuple[Path, dict]:
        """Call worldmodels.<fn>_meta if present (returns source/latency/placeholder info), else <fn>."""
        meta_fn = getattr(mod, f"{fn}_meta", None)
        if fn == "sim2real":
            if callable(meta_fn):
                call = lambda: meta_fn({kind: controls_or_video}, prompt, **kw)  # noqa: E731
            else:
                call = lambda: mod.sim2real(controls_or_video, kind, prompt, **kw)  # noqa: E731
        else:
            call = (lambda: meta_fn(prompt, **kw)) if callable(meta_fn) else (lambda: getattr(mod, fn)(prompt, **kw))
        res = self._run_with_progress(call, progress, what)
        info = res if isinstance(res, dict) else {}
        path = Path(str(info.get("path") if info else res))
        if not path.exists():
            raise FileNotFoundError(f"{what} returned no video ({path})")
        if not info.get("source"):
            info["source"] = f"cosmos3 · {backend_mode()}"
        return path, info

    def _run_with_progress(self, fn, progress: Progress, what: str):
        expected = {"replay": 4.0, "nim": 90.0, "local": 150.0}.get(backend_mode(), 60.0)
        box: dict = {}

        def run():
            try:
                box["out"] = fn()
            except Exception as e:  # noqa: BLE001
                box["err"] = e
        th = threading.Thread(target=run, daemon=True)
        t0 = time.time()
        th.start()
        while th.is_alive():
            el = time.time() - t0
            frac = 1 - pow(2.718, -el / expected * 2.2)
            steps = int(frac * 35)
            progress(0.15 + 0.8 * frac, f"{what}: denoising step {max(steps, 1)}/35")
            th.join(0.4)
        if "err" in box:
            raise box["err"]
        return box.get("out")

    def _mock_video(self, src: str | None, key: str | None, name: str, progress: Progress) -> Path:
        out = self.out_dir / f"{_slug(name)}.mp4"
        stages = ["Encoding prompt (T5)", "Denoising step 9/35", "Denoising step 21/35", "Denoising step 35/35",
                  "Decoding latents → mp4"]
        for i, s in enumerate(stages):
            progress(0.15 + 0.16 * i, s + " · mock")
            time.sleep(float(os.environ.get("COPILOT_MOCK_DELAY", "0.5")))
        if not src or not shutil.which("ffmpeg"):
            raise FileNotFoundError("no source video to render a mock from")
        if key == "sim2real":
            vf = ("eq=contrast=1.18:saturation=1.25:gamma=0.92,colorbalance=rs=0.08:bs=-0.06,"
                  "unsharp=5:5:0.9,noise=alls=7:allf=t,vignette=PI/4.5")
        else:
            vf = CHANGE_TEMPLATES.get(key or "", {}).get("ffmpeg", "eq=saturation=1.2")
        _ffmpeg("-i", src, "-vf", vf, "-an", "-t", "6.5", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out))
        progress(0.98, "Writing mp4")
        return out

    def _result(self, kind, clip, out, info, latency, prompt, label, note=None, change=None) -> dict:
        from .data import file_url
        if info.get("latency_ms") is not None:
            latency = info["latency_ms"] / 1000.0
        res = {"kind": kind, "sequence_name": clip["sequence"], "clip_id": clip["clip_id"],
               "t0_s": clip["t0"], "t1_s": clip["t1"], "video_path": str(out), "video_url": file_url(out),
               "source_url": file_url(clip["video_path"]) if clip["video_path"] else None,
               "source": info.get("source"), "latency_s": round(latency, 2), "prompt": prompt, "label": label,
               "mode": mode_label(), "placeholder": bool(info.get("placeholder")),
               "origin_source": info.get("origin_source")}
        if change:
            res["change"] = change
        if note:
            res["note"] = note
        return res


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120] + "_" + hashlib.sha1(s.encode()).hexdigest()[:6]


# ------------------------------------------------------------------ heuristic "reasoner" (mock)
def mock_assess(ctx: dict) -> dict:
    time.sleep(float(os.environ.get("COPILOT_MOCK_DELAY", "0.5")))
    defects = ctx.get("defects_in_window") or []
    viol = ctx.get("wps_violations") or []
    stats = ctx.get("signal_stats") or {}
    wps = ctx.get("wps_limits") or {}
    t0 = float(ctx.get("t0_s") or 0)
    seq = ctx.get("sequence_name")
    steps = [f"Clip {seq} t={ctx['t0_s']:.1f}–{ctx['t1_s']:.1f}s, {ctx['meta'].get('process')} "
             f"{ctx['meta'].get('pass_type')} pass under {ctx['meta'].get('wps_id')}."]
    for v in viol[:4]:
        lim = f"{v['limit_lo']}–{v['limit_hi']}" if v.get("limit_lo") is not None else f"≤{v['limit_hi']}"
        steps.append(f"Sensor check: {v['param']} reached {v['value']} (WPS {lim}) from "
                     f"{v['t_start_s']:.1f}s to {v['t_end_s']:.1f}s.")
    if not viol:
        steps.append("Sensor check: all monitored parameters within WPS limits in this window.")
    out_defects = []
    for d in defects:
        sev = float(d.get("severity") or 0.5)
        out_defects.append({"type": d["defect"], "confidence": round(min(0.97, 0.62 + 0.35 * sev), 2),
                            "t_s": round(max(0.0, float(d["t_start"]) - t0), 2)})
        steps.append(f"Visual: weld pool / bead appearance consistent with {d['defect'].replace('_', ' ')} "
                     f"from ~{d['t_start']:.1f}s.")
    # infer from violations when no labelled defect
    if not out_defects and viol:
        params = {v["param"] for v in viol}
        guess = ("porosity" if "gas_lpm" in params else "lack_of_fusion" if "travel_speed_mmps" in params and
                 "current_a" in params else "undercut" if "voltage_v" in params else "crack" if "interpass_c" in params
                 else None)
        if guess:
            out_defects.append({"type": guess, "confidence": 0.55, "t_s": round(max(0.0, viol[0]["t_start_s"] - t0), 2)})
            steps.append(f"Signal signature suggests possible {guess.replace('_', ' ')}; confidence limited.")
    major = [d for d in out_defects if d["confidence"] >= 0.7]
    verdict = "reject" if major else ("review" if out_defects or viol else "accept")
    if out_defects:
        rc, act = ROOT_CAUSES.get(out_defects[0]["type"], ("Out-of-procedure parameters.", "Review with CWI."))
    else:
        rc, act = ("No defect indicators; parameters nominal.", "Accept - no action. Continue per ITP.")
    findings = [{"param": v["param"], "value": v["value"], "limit_lo": v.get("limit_lo"),
                 "limit_hi": v.get("limit_hi"), "t_start_s": v["t_start_s"], "t_end_s": v["t_end_s"]} for v in viol]
    if stats.get("heat_input_kj_mm") and wps.get("heat_input_kj_mm"):
        hi = stats["heat_input_kj_mm"]["mean"]
        lo_, hi_ = wps["heat_input_kj_mm"]
        steps.append(f"Mean heat input {hi:.2f} kJ/mm vs WPS {lo_}–{hi_}.")
    steps.append(f"Verdict: {verdict.upper()}.")
    return {"verdict": verdict, "defects": out_defects, "root_cause": rc, "wps_findings": findings,
            "recommended_action": act, "reasoning": "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))}
