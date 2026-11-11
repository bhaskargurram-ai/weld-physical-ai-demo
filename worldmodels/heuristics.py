"""Deterministic, rule-based stand-ins used when no Cosmos backend is reachable.

Every result produced here carries ``"source": "heuristic-fallback"`` so the
copilot / dashboard can show that it is *not* a Cosmos answer.  The rules use
the same defect signatures as the simulator (collect/schema.py).
"""
from __future__ import annotations

from typing import Optional

from .prompts import SIGNAL_LABELS, describe_pass, normalize_context

# violation param (+direction) -> (likely defect, root cause text)
_PARAM_CAUSES = {
    ("gas_lpm", "low"): ("porosity", "shielding-gas flow fell below the WPS minimum, so atmospheric contamination of the pool is likely"),
    ("current_a", "low"): ("lack_of_fusion", "arc current below the WPS range lowers heat input and sidewall wetting"),
    ("current_a", "high"): ("burn_through", "arc current above the WPS range over-penetrates the root"),
    ("travel_speed_mmps", "high"): ("lack_of_fusion", "travel speed above the WPS range starves the joint of heat"),
    ("travel_speed_mmps", "low"): ("burn_through", "travel speed below the WPS range concentrates heat in the root"),
    ("voltage_v", "high"): ("undercut", "arc voltage above the WPS range widens the arc and melts the bead toes"),
    ("voltage_v", "low"): ("arc_instability", "arc voltage below the WPS range shortens the arc and causes stubbing/spatter"),
    ("wfs_mpm", "high"): ("arc_instability", "wire feed speed above the WPS range destabilises metal transfer"),
    ("wfs_mpm", "low"): ("lack_of_fusion", "wire feed speed below the WPS range reduces deposition and current"),
    ("heat_input_kj_mm", "low"): ("lack_of_fusion", "heat input below the qualified range"),
    ("heat_input_kj_mm", "high"): ("burn_through", "heat input above the qualified range (also a Sec IX supplementary essential variable concern)"),
    ("interpass_c", "high"): ("crack", "interpass temperature above the WPS maximum raises cracking risk"),
}

_DEFECT_CAUSE = {
    "porosity": "shielding-gas coverage loss or contamination (check gas flow, nozzle spatter, drafts, wire/base-metal cleanliness)",
    "lack_of_fusion": "insufficient heat input or fast travel (current low / travel speed high), poor torch angle into the sidewall",
    "undercut": "excessive voltage and/or travel speed melting the bead toes without filling them",
    "burn_through": "excessive heat on the root pass (current high / travel slow) or too large a root gap",
    "arc_instability": "unstable metal transfer: contact-tip wear, stick-out variation, wire-feed slippage",
    "crack": "interpass temperature above the WPS maximum and/or hydrogen pickup with restraint",
}

_DEFECT_ACTION = {
    "porosity": "Grind out the porous section, verify gas flow at the nozzle (flowmeter) and nozzle condition, re-weld, then RT the repair.",
    "lack_of_fusion": "Excavate to sound metal and re-weld within the WPS current/travel window; UT or RT the repair (LOF is rejectable at any size per B31.3).",
    "undercut": "Measure undercut depth with a gauge; if > lesser of 1 mm or Tw/4, blend or add a repair pass; reduce voltage/travel speed.",
    "burn_through": "Cut out and re-weld the root; check root gap and lower current / raise travel speed on the root pass.",
    "arc_instability": "Inspect contact tip, liner and drive rolls; review spatter and bead profile visually before the next pass.",
    "crack": "Reject: remove the crack completely (PT/MT to confirm), re-weld under interpass control; review preheat/interpass and consumable hydrogen.",
}

_SEVERE = {"crack", "lack_of_fusion", "burn_through"}


def _defect_list(c: dict) -> list[dict]:
    out = []
    t0 = c.get("t0_s") if isinstance(c.get("t0_s"), (int, float)) else 0.0
    for d in c.get("defects") or []:
        if isinstance(d, str):
            d = {"defect": d}
        if not isinstance(d, dict):
            continue
        dtype = str(d.get("defect") or d.get("type") or "").lower()
        if dtype in ("", "none"):
            continue
        sev = d.get("severity")
        ts = d.get("t_start", d.get("t_start_s", d.get("t_s")))
        t_rel: Optional[float] = None
        if isinstance(ts, (int, float)):
            t_rel = float(ts) - t0 if ts >= t0 else float(ts)
            t_rel = max(0.0, t_rel)
        out.append({"type": dtype, "severity": float(sev) if isinstance(sev, (int, float)) else 0.6, "t_s": t_rel})
    return out


def _violation_direction(v: dict) -> str:
    val, lo, hi = v.get("value"), v.get("limit_lo"), v.get("limit_hi")
    if isinstance(val, (int, float)):
        if isinstance(hi, (int, float)) and val > hi:
            return "high"
        if isinstance(lo, (int, float)) and val < lo:
            return "low"
    return "high" if v.get("param") == "interpass_c" else "low"


def assess(context: Optional[dict]) -> dict:
    c = normalize_context(context)
    defects = _defect_list(c)
    violations = [v for v in (c.get("wps_violations") or []) if isinstance(v, dict)]
    t0 = c.get("t0_s") if isinstance(c.get("t0_s"), (int, float)) else 0.0

    out_defects: list[dict] = []
    for d in defects:
        out_defects.append({"type": d["type"], "confidence": round(min(0.95, 0.5 + 0.45 * d["severity"]), 2),
                            "t_s": None if d["t_s"] is None else round(d["t_s"], 2)})

    findings, causes = [], []
    seen_types = {d["type"] for d in out_defects}
    for v in violations:
        p = str(v.get("param", "?"))
        direction = _violation_direction(v)
        label, unit = SIGNAL_LABELS.get(p, (p, ""))
        note = f"{label} {v.get('value')} {unit} vs WPS {v.get('limit_lo')}-{v.get('limit_hi')} {unit}".strip()
        findings.append({"param": p, "status": "violation", "note": note})
        hit = _PARAM_CAUSES.get((p, direction))
        if hit:
            causes.append(hit[1])
            if hit[0] not in seen_types:
                ts = v.get("t_start_s")
                t_rel = (ts - t0 if ts >= t0 else ts) if isinstance(ts, (int, float)) else None
                out_defects.append({"type": hit[0], "confidence": 0.35, "t_s": None if t_rel is None else round(max(0.0, t_rel), 2)})
                seen_types.add(hit[0])

    stats = c.get("signal_stats") if isinstance(c.get("signal_stats"), dict) else {}
    wps = c.get("wps") or {}
    flagged = {f["param"] for f in findings}
    for p, st in stats.items():
        if p in flagged or not isinstance(st, dict) or p not in wps:
            continue
        rng = wps[p]
        if isinstance(rng, (list, tuple)) and len(rng) == 2 and isinstance(st.get("mean"), (int, float)):
            lo, hi = rng
            span = (hi - lo) or 1.0
            m = st["mean"]
            status = "within" if lo <= m <= hi else "violation"
            if status == "within" and (m - lo < 0.05 * span or hi - m < 0.05 * span):
                status = "marginal"
            findings.append({"param": p, "status": status, "note": f"mean {m:.3g} in WPS {lo}-{hi}"})

    max_sev = max((d["severity"] for d in defects), default=0.0)
    severe = any(d["type"] in _SEVERE for d in defects)
    if defects and (severe or max_sev >= 0.5):
        verdict = "reject"
    elif defects or violations:
        verdict = "review"
    else:
        verdict = "accept"

    top = max(out_defects, key=lambda d: d["confidence"], default=None)
    if top:
        root = f"Most likely {top['type'].replace('_', ' ')}: {_DEFECT_CAUSE.get(top['type'], 'process excursion')}."
        if causes:
            root += " Logged excursion: " + causes[0] + "."
        action = _DEFECT_ACTION.get(top["type"], "Hold the joint for RT/UT and review the process data.")
    elif causes:
        root = "WPS excursion without a visible defect: " + causes[0] + "."
        action = "Hold for NDE (RT/UT) of the affected length and correct the parameter before the next pass."
    else:
        root = "All logged parameters within the WPS envelope; no defect evidence."
        action = "Accept; proceed to the next pass / NDE per the inspection test plan."
    if c.get("rt_result") == "reject" and verdict == "accept":
        verdict = "review"
        action = "RT reported a reject on this joint although this window looks clean: review the full pass."

    return {
        "verdict": verdict,
        "defects": out_defects,
        "root_cause": root,
        "wps_findings": findings,
        "recommended_action": action,
        "code_basis": "Rule-based: B31.3 Table 341.3.2 (cracks / LOF rejectable at any size) + WPS ranges (ASME Sec IX).",
        "reasoning": ("Heuristic fallback (no Cosmos Reasoner reachable and no cached answer): verdict derived from "
                      f"{len(defects)} annotated defect(s) and {len(violations)} logged WPS excursion(s) for "
                      f"{describe_pass(c)}. Not a model judgement."),
    }


def explain(context: Optional[dict]) -> str:
    c = normalize_context(context)
    lines = [f"[heuristic-fallback] Signal summary for {describe_pass(c)}:"]
    stats = c.get("signal_stats") if isinstance(c.get("signal_stats"), dict) else {}
    wps = c.get("wps") or {}
    for p, st in stats.items():
        if not isinstance(st, dict):
            continue
        label, unit = SIGNAL_LABELS.get(p, (p, ""))
        rng = wps.get(p)
        rng_txt = f" (WPS {rng[0]}-{rng[1]})" if isinstance(rng, (list, tuple)) and len(rng) == 2 else ""
        mean = st.get("mean")
        lines.append(f"- {label}: mean {mean:.3g} {unit}{rng_txt}, range {st.get('min', '?'):.3g}-{st.get('max', '?'):.3g}"
                     if isinstance(mean, (int, float)) and isinstance(st.get("min"), (int, float)) and isinstance(st.get("max"), (int, float))
                     else f"- {label}: {st}")
    for v in c.get("wps_violations") or []:
        if not isinstance(v, dict):
            continue
        hit = _PARAM_CAUSES.get((v.get("param"), _violation_direction(v)))
        if hit:
            lines.append(f"- EXCURSION {v.get('param')}={v.get('value')}: {hit[1]} -> risk of {hit[0].replace('_', ' ')}.")
    if len(lines) == 1:
        lines.append("- No sensor statistics or excursions supplied.")
    return "\n".join(lines)

