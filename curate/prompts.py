"""Weld-domain prompts.

1. ``WELD_CAPTION_PROMPT`` - instruction for a video-language captioner (Cosmos3 Reasoner,
   Qwen, Cosmos-Reason2) so captions read like a Certified Welding Inspector's notes.
   Used by the lite backend (``worldmodels.reason.chat``), Curator Next
   ``multimodal-caption`` (``inference.prompt.mode: custom``) and the legacy pipeline
   (``--captioning-prompt-text``). It deliberately contains no ``{``/``}`` so it can be
   passed to template engines verbatim.

2. ``build_generator_prompt(clip, variant)`` - a Cosmos 3 "upsampled" JSON prompt
   (schema of cosmos_framework/inference/prompting_templates/external_api/
   t2v_i2v_video_json_schema.json: subjects, background_setting, lighting, aesthetics,
   cinematography, actions, segments, temporal_caption, audio_description, resolution,
   aspect_ratio, duration, fps) describing the weld scene, for
     photoreal_transfer     sim->real Transfer of the procedural render (edge/depth control)
     what_if_travel_speed   counterfactual generation with a travel-speed change
     what_if_gas_loss       counterfactual generation with shielding-gas loss
     defect_synthesis       synthetic defect footage for training data
"""
from __future__ import annotations

import copy
import json
from typing import Any

from curate.common import defect_name, normalize_signal_stats

VARIANTS = ("photoreal_transfer", "what_if_travel_speed", "what_if_gas_loss", "defect_synthesis")

WELD_CAPTION_PROMPT = (
    "You are an AWS Certified Welding Inspector (CWI) reviewing footage from a weld-process camera "
    "(torch-mounted or fixed, colour or grayscale HDR) in a robotic weld cell. Describe this clip in one dense paragraph "
    "of 4 to 7 sentences, as an inspection note that could train a video model. Cover, in order: "
    "1) the camera view and what is in frame (torch nozzle, electrode or wire, arc, weld pool, joint "
    "bevel, previous passes, pipe surface); 2) the welding process you can infer (GTAW with a tungsten "
    "electrode and filler rod, FCAW or GMAW with continuously fed wire, SMAW stick, or SAW under flux) "
    "and the pass type if visible (root, hot, fill, cap); 3) arc and weld-pool behaviour: arc length and "
    "stability, pool size and shape, fluidity, wetting into the sidewalls, keyhole, how the pool tracks "
    "the joint, travel direction and apparent speed; 4) spatter, smoke, slag, fume and arc glare; "
    "5) any visible discontinuity forming or present: porosity or pinholes, lack of fusion, undercut "
    "along the toes, burn-through or excessive root penetration, crack, irregular or ropey bead, "
    "arc wander or instability; 6) lighting and image quality, including saturation from arc glare. "
    "Be specific and physical, state only what is visible, use inspection vocabulary, and do not "
    "invent part numbers or measurements that cannot be seen."
)

WELD_PHOTO_CAPTION_PROMPT = (
    "You are an AWS Certified Welding Inspector (CWI) doing a visual examination of a post-weld photo of a "
    "completed weld bead. Describe the photo in one dense paragraph of 3 to 6 sentences, as an inspection note "
    "that could train a vision model. Cover: 1) what is in frame (bead, toes, base metal, slag or flux "
    "residue, scale, markings); 2) bead geometry: width uniformity, ripple pattern, reinforcement, wetting at "
    "the toes; 3) any visible surface discontinuity: surface pores or pinholes, undercut, overlap, cracks, "
    "spatter, irregular bead; 4) lighting and image quality. State only what is visible in the photo; "
    "subsurface indications reported by RT/UT may not be visible - say so instead of inventing them. Do not "
    "invent measurements."
)

NEGATIVE_PROMPT: dict[str, Any] = {
    "subjects": [{
        "description": "Cartoonish or CGI-looking weld, plastic metal, molten pool that glows without heat "
                       "distortion, sparks frozen in mid-air, torch floating off the joint.",
        "appearance_details": "Flat textures, no oxide tint, no heat-affected zone colours, smeared "
                              "compression artifacts, melted geometry, extra nozzles or electrodes.",
        "action": "Bead that appears without the torch moving, pool that jumps between frames, sparks "
                  "that fly upward against gravity or loop back, flickering geometry.",
        "number_of_subjects": 0, "number_of_arms": 0, "number_of_legs": 0,
    }],
    "background_setting": "Clean showroom or outdoor scene, cluttered unrelated objects, text overlays, "
                          "watermarks, logos.",
    "lighting": {"conditions": "Uniform flat lighting with no arc illumination",
                 "direction": "Contradictory light sources", "shadows": "Missing or popping shadows",
                 "illumination_effect": "Arc does not light the surroundings"},
    "cinematography": {"camera_motion": "Shaky handheld, sudden zooms, cuts",
                       "focus": "Out of focus, chromatic aberration"},
    "style_medium": "Low-quality animation, video game render",
    "physical_realism": "Molten metal that behaves like water or solid plastic; slag or spatter without "
                        "gravity; bead dimensions changing randomly; no correlation between heat and colour.",
}

# ------------------------------------------------------------------ domain lookups
PROCESS_DESC: dict[str, dict[str, str]] = {
    "GTAW": {
        "name": "gas tungsten arc welding (TIG)",
        "torch": "a ceramic TIG cup with a sharpened tungsten electrode; a filler rod is dabbed into the "
                 "leading edge of the pool",
        "arc": "a narrow, stable, bluish-white arc cone with a calm, mirror-like weld pool",
        "spatter": "no spatter and almost no smoke; a thin clean argon-shielded bead with a silver to "
                   "straw oxide tint",
        "audio": "a steady, soft high-pitched hiss of the TIG arc and the faint hum of the positioner",
    },
    "FCAW": {
        "name": "flux-cored arc welding",
        "torch": "a copper MIG-style nozzle feeding a continuous tubular flux-cored wire",
        "arc": "a bright, slightly globular-to-spray arc with a fluid pool trailing a layer of molten slag",
        "spatter": "moderate spatter sparks thrown outward, a light grey fume plume rising from the arc, "
                   "and a brown glassy slag crust solidifying behind the pool",
        "audio": "an aggressive crackling, 'frying bacon' arc sound with intermittent spatter pops",
    },
    "GMAW": {
        "name": "gas metal arc welding (MIG)",
        "torch": "a copper MIG nozzle feeding solid wire",
        "arc": "a bright spray-transfer arc with a fluid, wide pool",
        "spatter": "light spatter and thin smoke, a shiny bead with silicon islands",
        "audio": "a smooth buzzing hiss of spray transfer",
    },
    "SMAW": {
        "name": "shielded metal arc welding (stick)",
        "torch": "a flux-coated stick electrode in an electrode holder, burning down as it welds",
        "arc": "a long, crackling arc with a pool covered by slag",
        "spatter": "heavy spatter and dense smoke, thick slag covering the bead",
        "audio": "loud crackling with irregular pops",
    },
    "SAW": {
        "name": "submerged arc welding",
        "torch": "a contact tip feeding wire into a mound of granular flux",
        "arc": "an arc hidden under the flux blanket with only an orange glow and occasional flashes",
        "spatter": "no visible spatter; flux granules, a fused slag cap peeling off behind the head",
        "audio": "a low rumbling hum with the hiss of flux recovery",
    },
}

# Real public datasets (docs/REAL_DATA.md): what the data actually shows, used instead of the simulator's
# visual process cues so captions / prompts never invent visuals the footage does not have.
SOURCE_DESC: dict[str, dict[str, Any]] = {
    "aimen_saw": {
        "setup": "two-wire tandem submerged arc welding (DC lead + AC trail electrode) of a 52 mm SA-516 Gr.70N "
                 "butt-weld coupon (AIMEN, PENELOPE project)",
        "unit": "bead",
        "camera": None,
        "photo": "Post-weld photo of the complete SAW bead",
        "scene": "a submerged-arc welding head with two wires feeding into a granular flux blanket on a thick "
                 "carbon-steel plate butt joint; the arc is hidden under the flux",
    },
    "bam_tgmaw": {
        "setup": "tandem GMAW (lead + trail wire; root layer single wire) in a 30 mm S355J2+N groove (BAM)",
        "unit": "layer",
        "camera": "HDR process camera footage, grayscale, with the BAM data overlay",
        "photo": None,
        "scene": "a tandem GMAW torch with two wires depositing a fill layer in a deep V-groove of a thick "
                 "structural-steel plate, seen by a grayscale high-dynamic-range process camera",
    },
}

POSITION_DESC = {
    "1G": "the pipe is rolled on a positioner under a stationary torch in the flat position, so the "
          "surface scrolls steadily through the frame",
    "2G": "the pipe axis is vertical and the torch travels horizontally around it",
    "5G": "the pipe is fixed horizontally and the torch orbits it, moving from flat through vertical to "
          "overhead",
    "6G": "the pipe is fixed at 45 degrees and the torch orbits it through all positions",
}

DEFECT_VISUAL = {
    "porosity": "small round gas pores and pinholes erupting through the pool surface, frozen into the bead "
                "as dark cavities, with an erratic sputtering arc",
    "lack_of_fusion": "the pool rides on top of the bevel sidewall without wetting into it, leaving a cold "
                      "lapped edge and a visible unfused line along the toe",
    "undercut": "a groove melted into the base metal along the weld toe that the filler does not refill, "
                "giving the bead a notched edge",
    "burn_through": "the root overheats, the pool sags and collapses into a hole with molten metal "
                    "dripping through the back of the joint",
    "arc_instability": "the arc wanders and flickers, the pool surges and shrinks, with bursts of spatter "
                       "and an irregular, ropey bead",
    "crack": "a fine longitudinal crack opening along the bead centreline as it cools, visible through the "
             "slag-free surface",
}

DEFECT_CAUSE = {
    "porosity": "loss of shielding gas or contamination",
    "lack_of_fusion": "low heat input from low current and fast travel",
    "undercut": "high voltage and fast travel",
    "burn_through": "excessive heat input on the root from high current and slow travel",
    "arc_instability": "stick-out or wire-feed variation",
    "crack": "interpass temperature above the WPS maximum",
}


def _mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


def _ctx(clip: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields the prompt needs from a manifest row or export sidecar."""
    meta = clip.get("meta") if isinstance(clip.get("meta"), dict) else {}
    stats = normalize_signal_stats(clip.get("signal_stats") or clip.get("sensor_summary") or {})
    defects = clip.get("defects") or clip.get("defects_in_window") or []
    t0 = float(clip.get("t0_s") or 0.0)
    t1 = clip.get("t1_s")
    dur = max(1.0, float(t1) - t0) if t1 is not None else 5.0
    proc = str(meta.get("process") or clip.get("process") or "FCAW").upper()
    src = clip.get("source") if isinstance(clip.get("source"), dict) else \
        (meta.get("source") if isinstance(meta.get("source"), dict) else {})
    return {
        "source_id": src.get("id"), "synthetic": bool(src.get("synthetic", not src)),
        "media_type": clip.get("media_type") or "video",
        "electrodes": int(meta.get("electrodes") or clip.get("electrodes") or 1),
        "thickness": meta.get("thickness_mm"),
        "process": proc if proc in PROCESS_DESC else "FCAW",
        "position": str(meta.get("position") or clip.get("position") or "1G"),
        "material": str(meta.get("material") or "A106 Gr.B carbon steel"),
        "od": meta.get("pipe_od_mm"),
        "wall": meta.get("wall_mm"),
        "pass_type": str(meta.get("pass_type") or "fill"),
        "pass_no": meta.get("pass_no"),
        "stats": stats,
        "defects": [d for d in defects if defect_name(d) not in ("none", "")],
        "duration": dur,
        "t0": t0,
    }


def _mean(stats: dict, sig: str) -> float | None:
    v = stats.get(sig, {}).get("mean")
    return None if v is None or v != v else float(v)


def build_generator_prompt(clip: dict[str, Any], variant: str = "photoreal_transfer",
                           *, travel_speed_factor: float = 0.8, defect: str | None = None,
                           width: int = 1280, height: int = 720, fps: int = 24) -> dict[str, Any]:
    """Cosmos 3 JSON prompt for one clip (manifest row or sidecar dict)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    c = _ctx(clip)
    p = PROCESS_DESC[c["process"]]
    dur = int(min(10, max(2, round(c["duration"]))))
    pipe = c["material"] + " pipe"
    if c["od"] and c["wall"]:
        pipe = f"{c['material']} pipe ({float(c['od']):.0f} mm OD, {float(c['wall']):.1f} mm wall)"
    real = not c["synthetic"]
    sdesc = SOURCE_DESC.get(str(c["source_id"]), {}) if real else {}
    if real:
        pipe = f"{c['material']} plate" + (f" ({float(c['thickness']):.0f} mm thick)" if c["thickness"] else "")
    pos = POSITION_DESC.get(c["position"], POSITION_DESC["1G"])
    defects = [defect_name(d) for d in c["defects"]]

    pool_desc = f"{p['arc']}"
    state_changes = "The pool advances steadily along the bevel, leaving a uniform rippled bead."
    actions: list[dict[str, str]] = []
    narrative: list[str] = []
    context = (f"Robotic {c['pass_type']} pass by {p['name']} on a pressure-piping spool in an ASME "
               "Section IX / B31.3 fabrication shop, filmed by a torch-mounted inspection camera")
    style_extra = ""

    if real:
        context = (f"Robotic {c['pass_type']} {sdesc.get('unit', 'pass')} by "
                   f"{'tandem ' if c['electrodes'] > 1 else ''}{p['name']} on {pipe}"
                   + (f": {sdesc['scene']}" if sdesc.get("scene") else ""))
    if variant == "photoreal_transfer":
        if real:
            context += "; colour photoreal re-rendering that follows the edge map of the real process-camera clip"
            style_extra = ("Must look like real shop footage, preserving the exact geometry, torch path and "
                           "timing of the control video.")
        else:
            context += "; photoreal re-rendering of a simulated weld for sim-to-real training data"
            style_extra = ("Must look like real shop footage from a filtered weld camera, preserving the exact "
                           "geometry, torch path and timing of the control video.")
        for d in defects:
            narrative.append(f"{d.replace('_', ' ').capitalize()} develops: {DEFECT_VISUAL.get(d, d)}.")
        if narrative:
            state_changes = " ".join(narrative)
    elif variant == "what_if_travel_speed":
        f = float(travel_speed_factor)
        ts = _mean(c["stats"], "travel_speed_mmps")
        ts_txt = f" from {ts:.1f} to {ts * f:.1f} mm/s" if ts else ""
        if f < 1.0:
            state_changes = (f"Travel speed drops by {round((1 - f) * 100)}%{ts_txt}: the weld pool grows "
                             "larger and hotter, the bead widens with heavier reinforcement and coarser "
                             "ripples, the heat tint spreads further into the base metal")
            if c["pass_type"] == "root":
                state_changes += ", and the thin root begins to sag toward burn-through"
            state_changes += "."
        else:
            state_changes = (f"Travel speed rises by {round((f - 1) * 100)}%{ts_txt}: the pool shrinks and "
                             "elongates into a teardrop, the bead narrows and turns ropey and convex, the "
                             "arc cuts grooves at the toes (undercut) and the filler stops wetting the "
                             "sidewalls (lack of fusion).")
        context += f"; counterfactual: same pass with travel speed x{f:.2f}"
        narrative.append(state_changes)
    elif variant == "what_if_gas_loss":
        state_changes = ("Shielding gas flow collapses mid-clip: the arc becomes erratic and sputtering, "
                         "spatter increases sharply, the pool surface boils with gas bubbles, and the "
                         "solidifying bead turns dull grey and pocked with porosity and surface pinholes.")
        context += "; counterfactual: same pass with loss of shielding gas"
        narrative.append(state_changes)
        defects = sorted(set(defects) | {"porosity"})
    elif variant == "defect_synthesis":
        d = defect or (defects[0] if defects else "porosity")
        defects = [d]
        state_changes = f"Midway through the clip {d.replace('_', ' ')} forms: {DEFECT_VISUAL.get(d, d)}."
        context += (f"; synthetic training example of {d.replace('_', ' ')} "
                    f"(typical cause: {DEFECT_CAUSE.get(d, 'process upset')})")
        narrative.append(state_changes)

    # simple 3-beat timeline across the clip
    beats = [(0, dur / 3), (dur / 3, 2 * dur / 3), (2 * dur / 3, dur)]
    beat_txt = [
        f"The torch tracks the joint with {pool_desc}; {p['spatter']}.",
        (narrative[0] if narrative else "The pool wets evenly into both bevel sidewalls as the joint "
                                         "scrolls past the torch."),
        ("The solidified bead trails behind the pool, glowing orange then dulling as it cools."
         if not narrative else "The resulting bead trails behind, showing the effect as it cools."),
    ]
    for (a, b), txt in zip(beats, beat_txt):
        actions.append({"time": f"{_mmss(a)}-{_mmss(b)}", "description": txt})

    current = _mean(c["stats"], "current_a")
    volt = _mean(c["stats"], "voltage_v")
    params = ""
    if current and volt:
        params = f" Arc running at about {current:.0f} A and {volt:.1f} V."

    prompt = {
        "subjects": [
            {
                "description": f"Molten weld pool and welding arc at the tip of {p['torch']}.{params}",
                "appearance_details": (f"{pool_desc}. Incandescent white-yellow core fading to orange at the "
                                       "pool edges; heat-distorted air above the pool."),
                "relationship": "The arc and pool are the focus; the bead behind them records the pass.",
                "location": "Center of frame, slightly below the optical axis",
                "relative_size": "Medium within frame",
                "orientation": "Torch angled about 10-15 degrees in the direction of travel",
                "pose": "Torch fixed relative to the camera, joint moving through the frame",
                "action": f"Depositing a {c['pass_type']} pass along a single-V groove",
                "state_changes": state_changes,
                "clothing": "", "expression": "", "gender": "", "age": "",
                "skin_tone_and_texture": "", "facial_features": "",
                "number_of_subjects": 1, "number_of_arms": 0, "number_of_legs": 0,
            },
            {
                "description": f"{pipe} with a {'deep groove' if real else 'beveled butt joint'} and earlier "
                               "passes visible in the groove.",
                "appearance_details": ("Mill-scale grey steel with a ground bright bevel, rainbow heat tint "
                                       "bands beside the weld, slag-free previous passes."),
                "relationship": "Workpiece being welded",
                "location": "Fills the lower two thirds of the frame",
                "relative_size": "Large within frame",
                "orientation": "Joint runs top-to-bottom through the frame",
                "pose": "", "action": "The plate is fixed while the robot moves the torch along the joint"
                if real else pos, "state_changes": "Bead length grows steadily.",
                "clothing": "", "expression": "", "gender": "", "age": "",
                "skin_tone_and_texture": "", "facial_features": "",
                "number_of_subjects": 1, "number_of_arms": 0, "number_of_legs": 0,
            },
        ],
        "background_setting": (("Inside a robotic welding laboratory cell: the thick steel plate with its groove "
                                "fills the view, clamped to a welding table, with a dark blurred cell interior "
                                "behind; drifting fume catches the arc light.") if real else
                               ("Inside a robotic weld cell of a pipe-spool fabrication shop: the curved pipe "
                                "surface fills the view, with a dark blurred cell interior, a positioner chuck "
                                "and grey welding curtains far behind; drifting fume catches the arc light.")),
        "lighting": {
            "conditions": "The welding arc is the dominant, extremely bright light source; ambient shop light is dim",
            "direction": "Point source at the arc, lighting the pipe from the center outward",
            "shadows": "Hard shadow of the nozzle cast away from the arc; falloff to darkness at the frame edges",
            "illumination_effect": ("High dynamic range filtered weld-camera look: arc core near saturation "
                                    "with bloom, molten pool detail still readable, flicker synced to the arc"),
        },
        "aesthetics": {
            "composition": "Tight industrial close-up centred on the arc with the joint as a leading line",
            "color_scheme": "White-yellow arc, orange molten metal, steel grey pipe, blue-violet heat tint",
            "mood_atmosphere": "Industrial, precise, intense, technical",
            "patterns": "Regular solidification ripples in the bead",
        },
        "cinematography": {
            "camera_motion": "Static relative to the torch (camera rides on the robot wrist)",
            "framing": "Close-up",
            "camera_angle": "High angle looking down the torch at the joint",
            "depth_of_field": "Shallow",
            "focus": "Sharp focus on the weld pool and arc",
            "lens_focal_length": "Short telephoto macro, approximately 50-75mm equivalent, with a neutral-density weld filter",
        },
        "style_medium": "Live-action video",
        "artistic_style": ("Realistic, documentary industrial footage from a weld-monitoring camera. "
                           + style_extra).strip(),
        "context": context,
        "actions": actions,
        "text_and_signage_elements": [],
        "segments": [{
            "segment_index": 0,
            "time_range": f"0:00-{_mmss(dur)}",
            "description": " ".join(a["description"] for a in actions),
            "key_changes": state_changes,
            "camera": "Locked to the torch for the whole shot",
        }],
        "transitions": [],
        "temporal_caption": " ".join(f"At {a['time'].split('-')[0]}, {a['description'][0].lower()}"
                                     f"{a['description'][1:]}" for a in actions),
        "audio_description": p["audio"] + "; no dialogue.",
        "physical_realism": ("Molten steel behaves with correct surface tension and gravity for the welding "
                             "position; spatter follows ballistic arcs and cools from white to orange to dark; "
                             "the bead width and ripple spacing are consistent with the travel speed and heat "
                             "input; the pipe and torch stay rigid."),
        "resolution": {"W": int(width), "H": int(height)},
        "aspect_ratio": "16,9",
        "duration": f"{dur}s",
        "fps": int(fps),
        "weld_metadata": {
            "variant": variant, "process": c["process"], "position": c["position"],
            "pass_type": c["pass_type"], "material": c["material"], "defects": defects,
            "source_id": c["source_id"], "electrodes": c["electrodes"],
        },
    }
    return prompt


def prompt_text(prompt: dict[str, Any], *, include_metadata: bool = False) -> str:
    """Serialise for the generator API (it takes a string)."""
    p = copy.deepcopy(prompt)
    if not include_metadata:
        p.pop("weld_metadata", None)
    return json.dumps(p, ensure_ascii=False)


def all_variants(clip: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every variant for one clip (travel speed -20 %, gas loss, and defect synthesis)."""
    return {v: build_generator_prompt(clip, v) for v in VARIANTS}
