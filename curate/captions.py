"""Captioners: Cosmos 3 Reasoner via ``worldmodels.reason.chat`` or a deterministic CWI template."""
from __future__ import annotations

import os
import urllib.request
from typing import Any, Callable

from curate.common import defect_name, defect_window, log, normalize_signal_stats
from curate.prompts import DEFECT_CAUSE, PROCESS_DESC, WELD_CAPTION_PROMPT

TEMPLATE_MODEL = "weld_template"
REASONER_MODEL = "cosmos3_reasoner"

_PARAM_LABEL = {
    "current_a": ("current", "A", 0), "voltage_v": ("voltage", "V", 1), "wfs_mpm": ("wire feed", "m/min", 1),
    "gas_lpm": ("shielding gas", "L/min", 1), "travel_speed_mmps": ("travel speed", "mm/s", 1),
    "heat_input_kj_mm": ("heat input", "kJ/mm", 2), "interpass_c": ("interpass temperature", "°C", 0),
}


def _fmt(v: Any, nd: int) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "?"


def template_caption(ctx: dict[str, Any], quality: dict[str, Any] | None = None) -> str:
    """CWI-style description built from the sidecar (and visual quality metrics).

    ``ctx`` keys: meta, signal_stats, wps_violations, defects (in window), rt_result, t0_s, t1_s.
    """
    meta = ctx.get("meta") or {}
    stats = normalize_signal_stats(ctx.get("signal_stats") or {})
    proc = str(meta.get("process") or "").upper()
    pdesc = PROCESS_DESC.get(proc)
    t0, t1 = float(ctx.get("t0_s") or 0.0), ctx.get("t1_s")
    window = f"t={t0:.1f}-{float(t1):.1f} s" if t1 is not None else f"from t={t0:.1f} s"

    who = []
    if meta.get("pass_type"):
        who.append(f"{meta['pass_type']} pass" + (f" {meta['pass_no']}" if meta.get("pass_no") else ""))
    where = []
    if meta.get("spool_id"):
        where.append(f"spool {meta['spool_id']}")
    if meta.get("joint_id"):
        where.append(f"joint {meta['joint_id']}")
    pipe = meta.get("material") or "carbon-steel"
    if meta.get("pipe_od_mm") and meta.get("wall_mm"):
        pipe += f" pipe {_fmt(meta['pipe_od_mm'], 1)} mm OD x {_fmt(meta['wall_mm'], 1)} mm wall"
    else:
        pipe += " pipe"
    s1 = (f"Torch-camera view of a robotic {proc + ' ' if proc else ''}{' '.join(who) or 'weld pass'}"
          f"{' on ' + ', '.join(where) if where else ''}: {pipe}"
          f"{', ' + str(meta['position']) + ' position' if meta.get('position') else ''}"
          f"{', ' + str(meta['wps_id']) if meta.get('wps_id') else ''}, {window}.")
    sentences = [s1]

    if pdesc:
        sentences.append(f"Process cues: {pdesc['torch']}; {pdesc['spatter']}.")

    parts = []
    for sig in ("current_a", "voltage_v", "wfs_mpm", "travel_speed_mmps", "gas_lpm", "heat_input_kj_mm",
                "interpass_c"):
        d = stats.get(sig)
        if not d or d.get("mean") is None or d.get("mean") != d.get("mean"):
            continue
        label, unit, nd = _PARAM_LABEL[sig]
        txt = f"{label} {_fmt(d['mean'], nd)} {unit}"
        if d.get("std") is not None and d["std"] == d["std"] and d["std"] > 0:
            txt += f" (±{_fmt(d['std'], nd + 1 if nd < 2 else nd)})"
        parts.append(txt)
    if parts:
        sentences.append("Recorded arc data: " + ", ".join(parts) + ".")

    viol = ctx.get("wps_violations") or []
    if viol:
        vtxt = []
        for v in viol[:4]:
            label, unit, nd = _PARAM_LABEL.get(v.get("param", ""), (v.get("param", "parameter"), "", 2))
            lim = f"{_fmt(v.get('limit_lo'), nd)}-{_fmt(v.get('limit_hi'), nd)} {unit}".strip()
            dur = ""
            if v.get("t_start_s") is not None and v.get("t_end_s") is not None:
                dur = f" for {float(v['t_end_s']) - float(v['t_start_s']):.1f} s from t={float(v['t_start_s']):.1f} s"
            vtxt.append(f"{label} reached {_fmt(v.get('value'), nd)} {unit} against a WPS range of {lim}{dur}")
        more = f" (+{len(viol) - 4} more)" if len(viol) > 4 else ""
        sentences.append("WPS deviation: " + "; ".join(vtxt) + more + ".")
    else:
        sentences.append("All monitored parameters stayed inside the WPS window.")

    defects = [d for d in (ctx.get("defects") or []) if defect_name(d) not in ("none", "")]
    if defects:
        dtxt = []
        for d in defects:
            name = defect_name(d)
            s = f"{name.replace('_', ' ')}"
            if isinstance(d, dict):
                if d.get("severity") is not None:
                    s += f" (severity {_fmt(d['severity'], 2)})"
                a, b = defect_window(d)
                if a is not None and b is not None:
                    s += f" at {a:.1f}-{b:.1f} s"
            if name in DEFECT_CAUSE:
                s += f", consistent with {DEFECT_CAUSE[name]}"
            dtxt.append(s)
        sentences.append("Discontinuity flagged in this window: " + "; ".join(dtxt) + ".")
    else:
        sentences.append("No discontinuity is flagged in this window.")

    if quality:
        vis = []
        g = quality.get("glare_frac")
        if g is not None:
            vis.append("heavy arc glare saturating much of the frame" if g > 0.2 else
                       "a bright arc with moderate glare" if g > 0.03 else
                       "a small bright arc spot" if (quality.get("arc_on_frac") or 0) > 0 else
                       "no visible arc")
        m = quality.get("motion")
        if m is not None:
            vis.append("strong pool and surface motion" if m > 4 else
                       "steady joint motion past the torch" if m > 0.2 else "a nearly static scene")
        if vis:
            sentences.append("Visually: " + " and ".join(vis) + ".")

    rt = ctx.get("rt_result")
    if rt:
        sentences.append(f"Radiography (RT) result for this joint: {rt}.")
    return " ".join(sentences)


def _reasoner_reachable(timeout: float = 2.0) -> bool:
    """Is a Reasoner backend usable? Defers to worldmodels.config.resolve_backend when present."""
    try:
        from worldmodels.config import resolve_backend  # type: ignore[import-not-found]
        return resolve_backend("reasoner") in ("local", "nim", "replay")
    except Exception:  # noqa: BLE001 - older/stub worldmodels: decide from env ourselves
        pass
    backend = os.environ.get("COSMOS_BACKEND", "auto").lower()
    if backend == "replay":
        return True  # replay answers from cache; per-clip misses fall back to the template
    if backend == "nim":
        return bool(os.environ.get("NVIDIA_API_KEY"))
    url = os.environ.get("COSMOS_REASONER_URL", "http://localhost:8000/v1").rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local endpoint
            return r.status < 500
    except Exception:  # noqa: BLE001
        return backend == "auto" and bool(os.environ.get("NVIDIA_API_KEY"))


def get_captioner(mode: str = "auto") -> tuple[str, Callable[[str, dict, dict], str] | None]:
    """Return (model_name, fn(video_path, ctx, quality) -> caption) or (template, None).

    mode: auto (reasoner if importable and reachable), reason (require it), template.
    """
    if mode == "template":
        return TEMPLATE_MODEL, None
    try:
        from worldmodels import reason  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        if mode == "reason":
            raise RuntimeError(f"worldmodels.reason not importable: {exc}") from exc
        log.info("worldmodels.reason not importable (%s); using template captions", exc)
        return TEMPLATE_MODEL, None
    if not _reasoner_reachable():
        if mode == "reason":
            raise RuntimeError("Cosmos reasoner backend not reachable (COSMOS_BACKEND/COSMOS_REASONER_URL)")
        log.info("Cosmos reasoner not reachable; using template captions")
        return TEMPLATE_MODEL, None

    def _caption(video_path: str, ctx: dict, quality: dict) -> str:
        facts = template_caption(ctx, quality)
        messages = [
            {"role": "system", "content": WELD_CAPTION_PROMPT},
            {"role": "user", "content": "Logged process data for this clip (for context, describe what you "
                                        "see in the video): " + facts},
        ]
        if hasattr(reason, "chat_meta"):
            meta = reason.chat_meta(messages, video_path)
            # replay-mode placeholders / reused assessments are not captions
            if meta.get("source") == "heuristic-fallback" or meta.get("cache_match") or meta.get("error"):
                raise ValueError(f"no reasoner caption (source={meta.get('source')})")
            out = meta.get("text")
        else:
            out = reason.chat(messages, video_path=video_path)
        if not isinstance(out, str) or not out.strip() or out.lstrip().startswith("["):
            raise ValueError("empty or placeholder caption")
        return out.strip()

    return REASONER_MODEL, _caption
