"""Prompt construction for the Cosmos 3 Reasoner and Generator.

Reasoner prompt format follows the Cosmos3 Reasoner prompt guide
(cosmos-framework .agents/skills/cosmos3-inference-reasoning/references/prompt-guide.md):
media block *before* the text, optional ``<think>`` instruction appended after
the task text, sampling presets for reasoning on/off.

Generator prompts follow the Cosmos3 "JSON-upsampled" prompt schema used by the
model-repo / cookbook assets (subjects, background_setting, lighting,
aesthetics, cinematography, style_medium, actions, temporal_caption,
resolution, duration, fps ...), sent as compact JSON text in the ``prompt`` field.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

from .config import repo_root

PROMPT_VERSION = "weldqa-v2"

DEFECT_TYPES = ("porosity", "lack_of_fusion", "undercut", "burn_through", "arc_instability", "crack")
SIGNALS = ("current_a", "voltage_v", "wfs_mpm", "gas_lpm", "travel_speed_mmps", "heat_input_kj_mm", "interpass_c",
           "current2_a", "voltage2_v", "wfs2_mpm")
SIGNAL_LABELS = {
    "current2_a": ("trail-electrode current", "A"),
    "voltage2_v": ("trail-electrode voltage", "V"),
    "wfs2_mpm": ("trail-electrode wire feed speed", "m/min"),
    "current_rms_a": ("lead-electrode current (RMS)", "A"),
    "current2_rms_a": ("trail-electrode current (RMS; the trail is AC, its mean is ~0)", "A"),
    "voltage2_rms_v": ("trail-electrode voltage (RMS)", "V"),
    "position_mm": ("torch position along the joint", "mm"),
    "temp_left_c": ("left pyrometer", "degC"),
    "temp_right_c": ("right pyrometer", "degC"),
    "temp_back_c": ("back pyrometer (pool region)", "degC"),
    "thermo2_c": ("thermocouple 2", "degC"),
    "current_a": ("arc current", "A"),
    "voltage_v": ("arc voltage", "V"),
    "wfs_mpm": ("wire feed speed", "m/min"),
    "gas_lpm": ("shielding gas flow", "L/min"),
    "travel_speed_mmps": ("travel speed", "mm/s"),
    "heat_input_kj_mm": ("heat input", "kJ/mm"),
    "interpass_c": ("interpass temperature", "degC"),
    "positioner_deg": ("positioner angle", "deg"),
}

# Sampling presets from the Cosmos3 Reasoner prompt guide.
SAMPLING_REASONING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0, "repetition_penalty": 1.0}
SAMPLING_DIRECT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "repetition_penalty": 1.0}

THINK_SUFFIX = (
    "Answer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)

CWI_SYSTEM_PROMPT = """You are an AWS Certified Welding Inspector (CWI) and welding engineer doing visual and \
process review for a heavy-fabrication shop: pressure vessels in SA-516 Gr.70 plate (ASME BPVC Section VIII, \
procedures qualified to ASME Section IX) and structural steelwork in S355 (EN 1090 / ISO 5817), welded by robotic and mechanised \
processes including two-wire tandem submerged arc welding (SAW) and tandem GMAW.

You review real weld evidence: process-camera video (torch-mounted or fixed; colour, or grayscale \
high-dynamic-range with a data overlay) or a post-weld photo of the finished bead, together with the logged \
arc signals (lead electrode, plus the trail electrode for tandem processes), travel and thermal signals, the \
procedure's parameter ranges and, when available, the NDT result (RT / UT indications with position). Apply \
these rules:
- The parameter ranges are the qualified envelope (or, where stated, the published nominal setting +-10 %). \
Sustained operation outside a range is a non-compliance even if the bead looks acceptable.
- Acceptance: cracks and lack of fusion are rejectable at any size; porosity, undercut and incomplete \
penetration are acceptable only within size / length limits (B31.3 Table 341.3.2, ASME VIII UW-51, \
ISO 5817 quality levels), so an indication of unknown size needs review against the NDT report.
- Typical process signatures: gas-flow loss or contamination (or damp flux on SAW) -> porosity; low current \
or high travel speed (low heat input) -> lack of fusion; high voltage or fast travel -> undercut; high \
current or slow travel -> burn-through; voltage noise / oscillating wire feed -> arc instability; interpass \
temperature above the maximum -> cracking risk.
- Real defects often leave NO signature in the arc signals (e.g. porosity from flux moisture or \
contamination at normal current and voltage). Never invent a signal excursion; if the signals are normal, \
say so explicitly ("no electrical signature") and rely on the NDT result and the image.
- Be calibrated. Only report defects the image, the NDT record or the signals actually support. Subsurface \
NDT indications may not be visible in a surface photo or camera view. If the evidence is ambiguous, use \
verdict "review" and say what to check (RT/UT report, PT/MT, visual with a gauge).
Always answer with the requested JSON object."""

ASSESS_SCHEMA_HINT = """{
  "verdict": "accept" | "reject" | "review",
  "defects": [{"type": one of %s, "confidence": 0.0-1.0, "t_s": seconds from the start of this clip}],
  "root_cause": "one or two sentences linking the evidence to the most likely process cause",
  "wps_findings": [{"param": signal name, "status": "within" | "violation" | "marginal", "note": "short evidence"}],
  "recommended_action": "concrete next step for the cell operator / inspector",
  "code_basis": "which ASME Sec IX / B31.3 criterion drives the verdict"
}""" % (" | ".join(f'"{d}"' for d in DEFECT_TYPES))

ASSESS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "reject", "review"]},
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(DEFECT_TYPES)},
                    "confidence": {"type": "number"},
                    "t_s": {"type": "number"},
                },
                "required": ["type", "confidence"],
            },
        },
        "root_cause": {"type": "string"},
        "wps_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"},
                    "status": {"type": "string", "enum": ["within", "violation", "marginal"]},
                    "note": {"type": "string"},
                },
                "required": ["param", "status"],
            },
        },
        "recommended_action": {"type": "string"},
        "code_basis": {"type": "string"},
    },
    "required": ["verdict", "defects", "root_cause", "wps_findings", "recommended_action"],
}


# ------------------------------------------------------------------ context normalisation
def _wps_limits(meta: dict, ctx: dict) -> dict:
    wps = ctx.get("wps") or meta.get("wps") or {}
    if isinstance(wps, dict) and wps:
        return wps
    wps_id = meta.get("wps_id") or ctx.get("wps_id")
    if wps_id:
        try:
            from dataclasses import asdict

            from collect.schema import WPS_LIBRARY

            w = WPS_LIBRARY.get(wps_id)
            if w:
                d = asdict(w)
                d["process"] = getattr(w.process, "value", w.process)
                return d
        except Exception:
            pass
    return {}


def normalize_context(context: Optional[dict]) -> dict:
    """Accept an export sidecar, a curated-manifest row, or a free dict.

    Returns a flat dict with the fields the prompts / heuristics use.
    """
    ctx = dict(context or {})
    meta = ctx.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    summary = ctx.get("sensor_summary")
    stats = ctx.get("signal_stats") or {}
    if isinstance(summary, dict):
        # curated manifest: per-signal {mean,std,min,max} dicts next to pass metadata scalars
        meta = {**{k: v for k, v in summary.items() if not isinstance(v, dict)}, **meta}
        if not stats:
            stats = {k: v for k, v in summary.items() if isinstance(v, dict)}
    elif summary and not stats:
        stats = summary
    get = lambda k, d=None: ctx.get(k, meta.get(k, d))  # noqa: E731
    defects = ctx.get("defects_in_window")
    if defects is None:
        defects = ctx.get("defects")
    if defects is None:
        defects = meta.get("defects", [])
    return {
        "clip_id": ctx.get("clip_id"),
        "sequence_name": get("sequence_name"),
        "t0_s": ctx.get("t0_s"),
        "t1_s": ctx.get("t1_s"),
        "spool_id": get("spool_id"),
        "joint_id": get("joint_id"),
        "pass_no": get("pass_no"),
        "pass_type": get("pass_type"),
        "process": get("process"),
        "position": get("position"),
        "material": get("material"),
        "pipe_od_mm": get("pipe_od_mm"),
        "wall_mm": get("wall_mm"),
        "wps_id": get("wps_id"),
        "wps": _wps_limits(meta, ctx),
        "signal_stats": stats,
        "wps_violations": ctx.get("wps_violations") or [],
        "defects": defects or [],
        "rt_result": ctx.get("rt_result", meta.get("rt_result")),
        "caption": ctx.get("caption"),
        "question": ctx.get("question"),
        "electrodes": get("electrodes"),
        "thickness_mm": get("thickness_mm"),
        "source": _source(ctx, meta),
        "ndt": _ndt(ctx, meta),
        "media_type": ctx.get("media_type"),
        "photo_note": ctx.get("photo_note"),
        "video_kind": ctx.get("video_kind") or ((meta.get("video") or {}).get("kind")
                                                if isinstance(meta.get("video"), dict) else None),
        "wps_basis": get("wps_basis"),
    }


def _source(ctx: dict, meta: dict) -> dict:
    for cand in (ctx.get("source"), meta.get("source")):
        if isinstance(cand, dict):
            return cand
    return {}


def _ndt(ctx: dict, meta: dict) -> dict:
    for cand in (ctx.get("ndt"), meta.get("ndt")):
        if isinstance(cand, dict):
            return cand
    return {}


def is_real(c: dict) -> bool:
    src = c.get("source") or {}
    return bool(src) and not src.get("synthetic", False) and src.get("id") not in (None, "weldcell_sim")


def _fmt(v: Any, nd: int = 2) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
    return str(v)


def _fmt_range(r: Any) -> str:
    if isinstance(r, (list, tuple)) and len(r) == 2:
        return f"{_fmt(r[0])}-{_fmt(r[1])}"
    return _fmt(r)


def describe_pass(c: dict) -> str:
    bits = []
    if c.get("sequence_name"):
        bits.append(f"Sequence {c['sequence_name']}")
    loc = " / ".join(str(x) for x in (c.get("spool_id"), c.get("joint_id")) if x)
    if loc:
        bits.append(f"spool/joint {loc}")
    if c.get("pass_type") or c.get("pass_no"):
        bits.append(f"pass {c.get('pass_no', '?')} ({c.get('pass_type', '?')})")
    if c.get("process"):
        n = c.get("electrodes")
        tandem = isinstance(n, (int, float)) and n > 1
        bits.append(f"process {'tandem ' if tandem else ''}{c['process']}"
                    + (f" ({int(n)} electrodes: lead + trail)" if tandem else ""))
    if c.get("position"):
        bits.append(f"position {c['position']}")
    if c.get("material"):
        geo = ""
        if c.get("pipe_od_mm"):
            geo = f", OD {_fmt(c['pipe_od_mm'])} mm x {_fmt(c.get('wall_mm', '?'))} mm wall"
        if not geo and c.get("thickness_mm"):
            geo = f", {_fmt(c['thickness_mm'])} mm thick"
        bits.append(f"material {c['material']}{geo}")
    src = c.get("source") or {}
    if src.get("id"):
        bits.append(f"data source {src['id']}" + (" (synthetic)" if src.get("synthetic") else
                                                  f" (real, {src.get('license', 'see source')})"))
    return "; ".join(bits) or "Robotic weld pass (no metadata supplied)"


def describe_ndt(c: dict) -> str:
    """NDT record of the pass (real data): methods, indications in this window with time and position."""
    ndt = c.get("ndt") or {}
    rt = c.get("rt_result")
    if not ndt and rt not in ("indications", "no_indications", "not_inspected"):
        return ""
    methods = "/".join(ndt.get("methods") or []) or "NDT"
    if rt == "not_inspected" or ndt.get("inspected") is False:
        return "NDT record: this weld was not NDT-inspected (no ground-truth defect labels)."
    inds = ndt.get("indications_in_window")
    if inds is None:
        inds = [d for d in c.get("defects") or [] if isinstance(d, dict)]
    t0 = c.get("t0_s") if isinstance(c.get("t0_s"), (int, float)) else 0.0
    lines = []
    for d in inds:
        typ = d.get("defect") or d.get("type") or "indication"
        ts, te = d.get("t_start_s", d.get("t_start")), d.get("t_end_s", d.get("t_end"))
        when = (f" at t={_fmt(ts - t0)}-{_fmt(te - t0)} s from the start of this window"
                if isinstance(ts, (int, float)) and isinstance(te, (int, float)) else "")
        where = (f", x={_fmt(d['x_start_mm'])}-{_fmt(d['x_end_mm'])} mm along the weld"
                 if isinstance(d.get("x_start_mm"), (int, float)) and isinstance(d.get("x_end_mm"), (int, float)) else "")
        lines.append(f"- {str(typ).replace('_', ' ')}{when}{where}")
    if lines:
        return f"NDT record ({methods} of the finished weld) - indications in this window:\n" + "\n".join(lines)
    if rt == "indications":
        return f"NDT record ({methods}): no indication in this window; the pass has indications elsewhere."
    return f"NDT record ({methods}): no indication in this pass."


def describe_media(c: dict, media: str = "video") -> str:
    real = is_real(c)
    if media == "image":
        txt = ("Inspect this post-weld photo of the finished bead together with the logged process data of the "
               "window it is linked to, and give your weld-quality assessment.")
        if c.get("photo_note"):
            txt += f" {c['photo_note']}"
        return txt
    if real:
        kind = str(c.get("video_kind") or "process camera").replace("_", " ")
        src = (c.get("source") or {}).get("id")
        extra = " (grayscale high-dynamic-range process camera with the BAM data overlay)" if src == "bam_tgmaw" else ""
        return (f"Inspect this {kind} clip{extra} of a robotic weld together with the logged process data and "
                "give your weld-quality assessment.")
    return ("Inspect this torch-camera clip of a robotic weld pass together with the logged process data and give "
            "your weld-quality assessment.")


def describe_wps(c: dict) -> str:
    w = c.get("wps") or {}
    if not w:
        return "WPS limits: not supplied."
    parts = []
    for k in ("current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm", "gas_lpm", "heat_input_kj_mm"):
        if k in w:
            label, unit = SIGNAL_LABELS.get(k, (k, ""))
            parts.append(f"{label} {_fmt_range(w[k])} {unit}".strip())
    if "max_interpass_c" in w:
        parts.append(f"max interpass {_fmt(w['max_interpass_c'])} degC")
    for k in ("current2_a", "voltage2_v", "wfs2_mpm"):
        if k in w:
            label, unit = SIGNAL_LABELS.get(k, (k, ""))
            parts.append(f"{label} {_fmt_range(w[k])} {unit}".strip())
    if "gas_lpm" not in w and str(c.get("process") or w.get("process") or "").upper() == "SAW":
        parts.append("no shielding gas (flux-shielded SAW)")
    kind = "qualified ranges"
    if c.get("wps_basis"):
        kind = f"ranges ({c['wps_basis']})"
    return f"WPS {w.get('wps_id', c.get('wps_id') or '')} ({w.get('process', c.get('process') or '')}) {kind}: " + "; ".join(parts) + "."


def describe_stats(c: dict) -> str:
    st = c.get("signal_stats")
    if not st:
        return "Sensor statistics: not supplied."
    if isinstance(st, str):
        return f"Sensor summary: {st}"
    lines = []
    for k, v in st.items():
        label, unit = SIGNAL_LABELS.get(k, (k, ""))
        if isinstance(v, dict):
            s = ", ".join(f"{m} {_fmt(v[m])}" for m in ("mean", "std", "min", "max") if m in v)
            lines.append(f"- {label} [{unit}]: {s}")
        else:
            lines.append(f"- {label} [{unit}]: {_fmt(v)}")
    return "Measured sensor statistics in this clip window:\n" + "\n".join(lines)


def describe_violations(c: dict) -> str:
    vs = c.get("wps_violations") or []
    if is_real(c):
        if not vs:
            return ("Excursions of the 0.5 s signal averages outside the ranges in this window: none (no "
                    "electrical signature).")
    elif not vs:
        return "WPS excursions logged by the cell controller in this window: none."
    lines = []
    t0 = c.get("t0_s") or 0.0
    for v in vs[:20]:
        if not isinstance(v, dict):
            lines.append(f"- {v}")
            continue
        label, unit = SIGNAL_LABELS.get(v.get("param", ""), (v.get("param", "?"), ""))
        ts, te = v.get("t_start_s"), v.get("t_end_s")
        when = ""
        if isinstance(ts, (int, float)):
            rel = ts - t0 if isinstance(t0, (int, float)) and ts >= t0 else ts
            when = f" from t={_fmt(rel)} s" + (f" to {_fmt(te - t0 if te >= t0 else te)} s" if isinstance(te, (int, float)) else "")
        lim = f" (WPS {_fmt(v.get('limit_lo'))}-{_fmt(v.get('limit_hi'))} {unit})" if "limit_lo" in v else ""
        lines.append(f"- {label}: {_fmt(v.get('value'))} {unit}{lim}{when}".strip())
    head = ("Excursions of the 0.5 s signal averages outside the ranges" if is_real(c) else
            "WPS excursions logged by the cell controller")
    return head + " (times relative to clip start):\n" + "\n".join(lines)


def describe_labels(c: dict) -> str:
    parts = []
    if c.get("defects"):
        parts.append("Known defect annotations: " + json.dumps(c["defects"], default=str))
    if c.get("rt_result"):
        parts.append(f"Radiography (RT) result for the joint: {c['rt_result']}")
    return "\n".join(parts)


def build_assess_text(context: Optional[dict], *, think: bool = True, include_labels: bool = False,
                      media: str = "video") -> str:
    c = normalize_context(context)
    window = ""
    if isinstance(c.get("t0_s"), (int, float)) and isinstance(c.get("t1_s"), (int, float)):
        window = (f" The clip covers t={_fmt(c['t0_s'])}-{_fmt(c['t1_s'])} s of the pass; report defect times relative "
                  "to the clip start." if media != "image" else
                  f" The linked process window is t={_fmt(c['t0_s'])}-{_fmt(c['t1_s'])} s of the pass.")
    sections = [
        describe_media(c, media) + window,
        describe_pass(c),
        describe_wps(c),
        describe_stats(c),
        describe_violations(c),
    ]
    if is_real(c):   # real data: the NDT record is the ground truth the shop has, part of the evidence
        sections.append(describe_ndt(c))
    if include_labels:
        lab = describe_labels(c)
        if lab:
            sections.append(lab)
    if c.get("caption"):
        sections.append(f"Curator caption of the clip (may be imperfect): {c['caption']}")
    if c.get("question"):
        sections.append(f"Specific question from the shop: {c['question']}")
    sections.append(
        "Return ONLY a JSON object with exactly these keys (no markdown, no extra prose):\n" + ASSESS_SCHEMA_HINT
    )
    text = "\n\n".join(s for s in sections if s)
    if think:
        text += "\n\n" + THINK_SUFFIX
    return text


def build_explain_text(context: Optional[dict]) -> str:
    c = normalize_context(context)
    return "\n\n".join([
        "Explain, for a fabrication-shop supervisor, what these weld-cell signals say about this pass: "
        "which parameters were inside the WPS envelope, which were not, what defect each excursion is likely "
        "to cause, and what to check next. If every signal is normal, say plainly that there is no electrical "
        "signature (a defect reported by NDT can still be present). Plain text, at most 8 short bullet points.",
        describe_pass(c), describe_wps(c), describe_stats(c), describe_violations(c),
        describe_ndt(c) if is_real(c) else "",
    ]).strip()


# ------------------------------------------------------------------ generator prompts
@lru_cache(maxsize=4)
def negative_prompt(kind: str = "video") -> str:
    """Upstream Cosmos3 negative prompt (compact JSON), vendored in configs/cosmos/."""
    name = "negative_prompt_transfer.json" if kind == "transfer" else "negative_prompt_video.json"
    p = repo_root() / "configs" / "cosmos" / name
    try:
        return json.dumps(json.loads(p.read_text()), ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return "blurry, distorted, low quality, jittery, deformed, cartoon, CGI render, flickering"


SHOP_BACKGROUND = (
    "An industrial pipe-spool fabrication bay: a carbon-steel pipe section clamped in a powered rotary "
    "positioner, a six-axis robotic welding arm with a gas-cooled MIG/FCAW torch, welding curtains, steel "
    "fixtures, ground clamp and cables, grinding dust on the floor; bright overhead high-bay lighting."
)


# Real-data scenes (docs/REAL_DATA.md): what the source camera / cell actually looks like.
REAL_SCENES = {
    "bam_tgmaw": {
        "background": ("A welding laboratory cell: a 30 mm thick S355 structural-steel plate with a deep V-groove "
                       "clamped on a welding table, a robot-held tandem GMAW torch with two wires, dark surroundings."),
        "subject": "A tandem GMAW layer (lead and trail wire in one shared pool) in a deep V-groove of a thick steel plate",
        "camera": {"camera_motion": "Static process camera aimed at the arc and weld pool",
                   "framing": "Close-up of the arcs, the shared weld pool and the groove",
                   "camera_angle": "Oblique view along the groove", "depth_of_field": "Medium",
                   "focus": "Sharp on the weld pool",
                   "lens_focal_length": "Industrial high-dynamic-range process camera, grayscale"},
        "style": "Realistic grayscale high-dynamic-range process-camera footage (keep the data overlay)",
        "context": "Robotic tandem GMAW of a thick S355J2+N plate in a welding research laboratory (BAM)",
    },
    "aimen_saw": {
        "background": ("A heavy-fabrication welding cell: a 52 mm SA-516 Gr.70 plate butt joint under a "
                       "submerged-arc welding head with two wires, a flux hopper and a flux recovery nozzle."),
        "subject": "A two-wire tandem submerged-arc weld bead forming under a blanket of granular flux",
        "camera": {"camera_motion": "Static", "framing": "Medium close-up of the welding head and the joint",
                   "camera_angle": "Slightly elevated side view", "depth_of_field": "Medium",
                   "focus": "Sharp on the flux blanket and the solidified slag behind the head",
                   "lens_focal_length": "35-50mm equivalent"},
        "style": "Realistic industrial footage; the arc is hidden under the flux, only an orange glow escapes",
        "context": "Mechanised two-wire tandem SAW of a pressure-vessel steel plate (AIMEN)",
    },
}


def _camera(view: str) -> dict:
    if view == "cell":
        return {"camera_motion": "Static", "framing": "Medium-wide shot of the robot, torch and rotating pipe",
                "camera_angle": "Slightly elevated three-quarter view", "depth_of_field": "Medium",
                "focus": "Sharp on the torch tip and the joint", "lens_focal_length": "35mm equivalent"}
    return {"camera_motion": "Camera rigidly mounted on the torch, moving with it; the pipe surface slides past",
            "framing": "Extreme close-up of the arc, weld pool and freshly deposited bead",
            "camera_angle": "Torch-mounted, looking down the joint at about 30 degrees",
            "depth_of_field": "Shallow", "focus": "Sharp on the molten weld pool and bead toes",
            "lens_focal_length": "Macro, welding-camera optics with an ND / auto-darkening filter"}


def weld_scene_prompt(description: str, *, process: str = "FCAW", pass_type: str = "fill",
                      defect: Optional[str] = None, view: str = "torch", seconds: float = 5.0,
                      fps: float = 24.0, width: int = 832, height: int = 480,
                      actions: Optional[list[dict]] = None, temporal_caption: Optional[str] = None,
                      for_transfer: bool = False, source: Optional[str] = None) -> dict:
    """Cosmos3 JSON-style prompt for a weld-cell scene.

    ``description`` is the free-text scenario ("travel speed drops 20 %...");
    the rest supplies the shop context the Generator needs for photoreal,
    domain-consistent footage.
    """
    dur = f"{int(round(seconds))}s"
    defect_txt = ""
    if defect and defect != "none":
        defect_txt = {
            "porosity": "Gas pockets erupt through the weld pool and freeze as small round pores and pinholes in the bead surface (porosity).",
            "lack_of_fusion": "The bead sits on top of the bevel face with a visible cold lap line; the puddle does not wet into the sidewall (lack of fusion).",
            "undercut": "A groove melts into the base metal along the bead toe and is left unfilled (undercut).",
            "burn_through": "The root melts through, leaving a hole with sagging molten metal and an oxidised edge (burn-through).",
            "arc_instability": "The arc flickers and wanders, spatter flies, and the bead ripples become irregular (arc instability).",
            "crack": "A thin dark crack line opens along the bead centreline as it cools (centreline crack).",
        }.get(defect, f"A {defect.replace('_', ' ')} defect forms in the bead.")
    subject = {
        "description": (f"The {process} {pass_type} pass on a carbon-steel pipe butt joint: bright arc, molten weld "
                        "pool and a solidifying bead with regular ripples. " + description).strip(),
        "appearance_details": ("Glowing orange-white weld pool, dark grey solidified bead with slag/oxide film, "
                               "heat-tint colours on the base metal, occasional spatter. " + defect_txt).strip(),
        "relationship": "The robot torch deposits the bead while the positioner rotates the pipe",
        "location": "Centre of frame",
        "relative_size": "Large within frame",
        "orientation": "Joint runs left to right",
        "pose": "Torch at a constant work and travel angle",
        "action": description,
        "state_changes": defect_txt or "The bead grows steadily along the joint.",
        "number_of_subjects": 1,
    }
    prompt = {
        "subjects": [subject],
        "background_setting": SHOP_BACKGROUND,
        "lighting": {"conditions": "Intense arc light dominating a dim cell", "direction": "From the arc at the torch tip",
                     "shadows": "Hard flickering shadows cast by the torch nozzle",
                     "illumination_effect": "Arc glare with bloom, glowing pool, cooler bead behind"},
        "aesthetics": {"composition": "Weld pool centred, finished bead trailing behind",
                       "color_scheme": "Orange-white arc and pool against grey-blue steel",
                       "mood_atmosphere": "Industrial, precise, documentary"},
        "cinematography": _camera(view),
        "style_medium": "Live-action video",
        "artistic_style": "Realistic industrial welding-camera footage",
        "context": "Robotic welding of ASME B31.3 process piping in a fabrication shop",
        "actions": actions or [{"time": f"0:00-0:{int(round(seconds)):02d}", "description": description}],
        "temporal_caption": temporal_caption or description,
        "resolution": {"W": int(width), "H": int(height)},
        "aspect_ratio": "16,9" if abs(width / height - 16 / 9) < 0.05 else f"{width},{height}",
        "duration": dur,
        "fps": fps,
    }
    real = REAL_SCENES.get(str(source)) if source else None
    if real:
        subject["description"] = f"{real['subject']}. {description}".strip()
        subject["relationship"] = "The robot torch deposits the bead along the fixed plate"
        prompt["background_setting"] = real["background"]
        prompt["cinematography"] = dict(real["camera"])
        prompt["artistic_style"] = real["style"]
        prompt["context"] = real["context"]
        if source == "bam_tgmaw":
            prompt["aesthetics"]["color_scheme"] = "Grayscale: saturated white arcs, bright pool, dark steel"
    if for_transfer:
        prompt["context"] += "; photoreal rendering that follows the structure of the provided control video exactly"
    return prompt


def compact(prompt: Any) -> str:
    """Prompt object -> the compact JSON text Cosmos3 expects (strings pass through)."""
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=True, separators=(",", ":"))


def looks_like_json_prompt(text: str) -> bool:
    t = text.strip()
    if not t.startswith("{"):
        return False
    try:
        return isinstance(json.loads(t), dict)
    except Exception:
        return False


def prompt_query_text(prompt: Any) -> str:
    """Human text of a (possibly JSON) prompt, for nearest-match lookups and labels."""
    if isinstance(prompt, dict):
        subj = (prompt.get("subjects") or [{}])[0]
        return str(subj.get("action") or prompt.get("temporal_caption") or "")
    if isinstance(prompt, str) and looks_like_json_prompt(prompt):
        return prompt_query_text(json.loads(prompt))
    return str(prompt)
