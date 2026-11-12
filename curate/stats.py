"""Dataset card for the curated set: ``python -m curate.stats [data/curated]``.

Writes ``<out>/dataset_card.md`` (shown by the copilot) and ``<out>/dataset_card.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from curate.common import defect_name
from curate.manifest import read_manifest


def summarize(rows: list[dict[str, Any]], backend: str | None = None) -> dict[str, Any]:
    kept = [r for r in rows if r.get("kept")]
    rejected = [r for r in rows if not r.get("kept")]

    def secs(rs):
        return sum(max(0.0, float(r.get("t1_s") or 0) - float(r.get("t0_s") or 0)) for r in rs)

    def defects(rs):
        c = Counter()
        for r in rs:
            names = {defect_name(d) for d in r.get("defects") or []} - {"none", ""}
            for n in names or {"none"}:
                c[n] += 1
        return dict(c.most_common())

    viol = Counter()
    for r in kept:
        for v in r.get("wps_violations") or []:
            viol[v.get("param", "unknown")] += 1

    q = {}
    for k in ("blur", "motion", "glare_frac", "arc_on_frac"):
        vals = [r["quality"][k] for r in rows if isinstance(r.get("quality"), dict)
                and isinstance(r["quality"].get(k), (int, float))]
        if vals:
            q[k] = {"median": round(median(vals), 4), "min": round(min(vals), 4), "max": round(max(vals), 4)}

    sources: dict[str, dict[str, Any]] = {}
    for r in rows:
        src = r.get("source") if isinstance(r.get("source"), dict) else {}
        sid = str(src.get("id") or "unknown")
        e = sources.setdefault(sid, {"license": src.get("license"), "attribution": src.get("attribution"),
                                     "doi": src.get("doi"), "url": src.get("url"),
                                     "synthetic": src.get("synthetic"), "items": 0, "kept": 0,
                                     "video": 0, "image": 0})
        e["items"] += 1
        e["kept"] += int(bool(r.get("kept")))
        mt = r.get("media_type") or "video"
        e[mt] = e.get(mt, 0) + 1

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": backend,
        "clips_total": len(rows),
        "clips_kept": len(kept),
        "clips_rejected": len(rejected),
        "rejected_by_reason": dict(Counter(r.get("reject_reason") or "unknown" for r in rejected).most_common()),
        "hours_total": round(secs(rows) / 3600, 4),
        "hours_kept": round(secs(kept) / 3600, 4),
        "sequences": len({r.get("sequence_name") for r in rows}),
        "sequences_kept": len({r.get("sequence_name") for r in kept}),
        "defects_kept": defects(kept),
        "defects_all": defects(rows),
        "rt_result_kept": dict(Counter(str(r.get("rt_result")) for r in kept).most_common()),
        "wps_violations_kept_by_param": dict(viol.most_common()),
        "clips_with_wps_violation_kept": sum(1 for r in kept if r.get("wps_violations")),
        "process_kept": dict(Counter(str((r.get("sensor_summary") or {}).get("process", "unknown"))
                                     for r in kept).most_common()),
        "pass_type_kept": dict(Counter(str((r.get("sensor_summary") or {}).get("pass_type", "unknown"))
                                       for r in kept).most_common()),
        "caption_models": dict(Counter(r.get("caption_model") or "none" for r in rows).most_common()),
        "media_type_kept": dict(Counter(r.get("media_type") or "video" for r in kept).most_common()),
        "media_type_all": dict(Counter(r.get("media_type") or "video" for r in rows).most_common()),
        "sources": sources,
        "quality": q,
    }


def _table(d: dict, h1: str, h2: str, total: int | None = None) -> str:
    if not d:
        return "_none_\n"
    lines = [f"| {h1} | {h2} |" + (" share |" if total else ""), "|---|---:|" + ("---:|" if total else "")]
    for k, v in d.items():
        lines.append(f"| {k} | {v} |" + (f" {100 * v / total:.0f}% |" if total else ""))
    return "\n".join(lines) + "\n"


def _sources_section(s: dict[str, Any]) -> list[str]:
    src = s.get("sources") or {}
    if not src:
        return []
    out = ["## Sources, licenses and attribution", "",
           "| source | license | items (video / photo) | kept |", "|---|---|---:|---:|"]
    for sid, e in src.items():
        out.append(f"| `{sid}` | {e.get('license') or 'unknown'} | {e['items']} ({e.get('video', 0)} / "
                   f"{e.get('image', 0)}) | {e['kept']} |")
    out.append("")
    for sid, e in src.items():
        att = e.get("attribution") or sid
        link = f" <{e['url']}>" if e.get("url") else ""
        out.append(f"- **{sid}**: {att}{link}")
    if any(not e.get("synthetic") and "BY" in str(e.get("license") or "") for e in src.values()):
        out += ["", "CC BY 4.0 data: redistribution and derived works (clips, photos, captions, model outputs) "
                    "must keep this attribution."]
    out.append("")
    return out


def _intro(s: dict[str, Any]) -> str:
    src = s.get("sources") or {}
    real = [k for k, e in src.items() if not e.get("synthetic") and k != "unknown"]
    if not real:
        return ("Torch-camera clips from robotic pipe-spool welding passes (GTAW root, FCAW fill/cap per WPS), "
                "cut to fixed 5 s windows, H.264 transcoded, quality-filtered (blur, motion, arc glare, exposure), "
                "near-duplicate filtered, captioned in CWI style and joined with arc/travel/thermal sensor "
                "statistics, WPS deviations, defect labels and RT results.")
    return ("Real, public welding data (" + ", ".join(f"`{k}`" for k in real) + "): 5 s windows around NDT "
            "indications plus clean windows. Passes with process-camera video give H.264 clips; passes without "
            "video (SAW) give the post-weld bead photo as an image item. Each item is quality-filtered, "
            "near-duplicate filtered, captioned from what the data records (no invented visuals) and joined "
            "with the arc/travel/thermal statistics of its window, deviations from the nominal parameter window "
            "(published setting ±10 %, a demo assumption, not a qualified WPS) and the NDT result.")


def render_card(s: dict[str, Any]) -> str:
    kept_pct = 100 * s["clips_kept"] / s["clips_total"] if s["clips_total"] else 0
    out = [
        "# WeldWorld curated dataset card",
        "",
        f"Generated {s['generated_utc']} by `curate` (backend: `{s.get('backend') or 'unknown'}`).",
        "",
        _intro(s),
        "",
        *_sources_section(s),
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| clips total | {s['clips_total']} |",
        f"| clips kept | {s['clips_kept']} ({kept_pct:.0f}%) |",
        f"| clips rejected | {s['clips_rejected']} |",
        f"| hours total | {s['hours_total']:.3f} ({s['hours_total'] * 60:.1f} min) |",
        f"| hours kept | {s['hours_kept']:.3f} ({s['hours_kept'] * 60:.1f} min) |",
        f"| weld passes (sequences) kept / total | {s['sequences_kept']} / {s['sequences']} |",
        f"| kept clips with a WPS deviation | {s['clips_with_wps_violation_kept']} |",
        f"| kept items by media type | {', '.join(f'{k}: {v}' for k, v in (s.get('media_type_kept') or {}).items()) or '-'} |",
        "",
        "## Rejected clips by reason",
        "",
        _table(s["rejected_by_reason"], "reason", "clips", s["clips_rejected"] or None),
        "## Defect distribution (kept clips)",
        "",
        _table(s["defects_kept"], "defect", "clips", s["clips_kept"] or None),
        "## Process / pass type (kept clips)",
        "",
        _table(s["process_kept"], "process", "clips"),
        _table(s["pass_type_kept"], "pass type", "clips"),
        "## RT result (kept clips)",
        "",
        _table(s["rt_result_kept"], "rt_result", "clips"),
        "## WPS deviations by parameter (kept clips)",
        "",
        _table(s["wps_violations_kept_by_param"], "parameter", "violations"),
        "## Captions",
        "",
        _table(s["caption_models"], "caption model", "clips"),
        "## Visual quality (all clips)",
        "",
    ]
    if s["quality"]:
        out += ["| metric | median | min | max |", "|---|---:|---:|---:|"]
        for k, v in s["quality"].items():
            out.append(f"| {k} | {v['median']} | {v['min']} | {v['max']} |")
        out.append("")
    else:
        out.append("_not computed_\n")
    out += [
        "## Files",
        "",
        "- `manifest.jsonl`: one row per clip (caption, sensor summary, WPS violations, defects, quality, "
        "kept/reject_reason, Cosmos 3 generator prompt)",
        "- `clips/<clip_id>.mp4`: H.264 clips; `clips/<clip_id>.jpg`: bead photos (image items)",
        "- `lance/`: Curator-Next-schema Lance table (clip_id, clip_uri, start_ns, end_ns, caption_*, embedding_*)",
        "- `prompts/<clip_id>.json`: Cosmos 3 prompt variants (photoreal_transfer, what_if_travel_speed, "
        "what_if_gas_loss, defect_synthesis)",
        "",
    ]
    return "\n".join(out)


def write_card(out_dir: str | Path, backend: str | None = None) -> Path:
    out = Path(out_dir)
    rows = read_manifest(out / "manifest.jsonl")
    s = summarize(rows, backend)
    (out / "dataset_card.json").write_text(json.dumps(s, indent=2))
    path = out / "dataset_card.md"
    path.write_text(render_card(s))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="data/curated")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args(argv)
    print(write_card(a.out, a.backend))


if __name__ == "__main__":
    main()
