"""Numbers for the "How it works" tab: one dict describing every stage of the pipeline as it
stands on disk right now, plus one weld followed end to end. Everything is computed from
data/ (no model calls), and every stage says honestly where its output came from."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import config

REAL_SOURCES = ("cosmos3", "nim", "cache")   # prefixes of `source` values produced by a real Cosmos 3 call


def _is_real(source: str) -> bool:
    s = (source or "").lower()
    return s.startswith(REAL_SOURCES) and "fallback" not in s and "placeholder" not in s and "mock" not in s


def _cache_counts() -> dict[str, dict[str, int]]:
    """Per world-model function: how many cached results came from Cosmos 3 vs a fallback."""
    root = config.data_dir() / "cache" / "worldmodels"
    out: dict[str, dict[str, int]] = {}
    if not root.is_dir():
        return out
    for fn_dir in (p for p in root.iterdir() if p.is_dir() and p.name != "placeholders"):
        c = Counter()
        for f in fn_dir.glob("*.json"):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            src = str(rec.get("source") or rec.get("meta", {}).get("source") or "")
            c["real" if _is_real(src) else "fallback"] += 1
        out[fn_dir.name] = {"real": c["real"], "fallback": c["fallback"]}
    placeholders = root / "placeholders"
    if placeholders.is_dir():
        out["placeholders"] = {"real": 0, "fallback": len(list(placeholders.glob("*.mp4")))}
    return out


def _cached_verdict(clip_prefix: str) -> dict | None:
    """A cached assess_clip result for a clip of this pass, preferring a real Cosmos 3 one."""
    d = config.data_dir() / "cache" / "worldmodels" / "assess_clip"
    best = None
    if not d.is_dir():
        return None
    for f in d.glob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        blob = json.dumps(rec.get("inputs") or rec.get("context") or rec)[:20000]
        if clip_prefix not in blob:
            continue
        if best is None or (_is_real(rec.get("source", "")) and not _is_real(best.get("source", ""))):
            best = rec
    return best


def _curation_backend() -> str:
    for p in (config.curated_dir() / "dataset_card.md", config.data_dir() / "dataset_card.md"):
        if p.exists():
            m = re.search(r"backend:\s*`?([\w-]+)`?", p.read_text(encoding="utf-8"))
            if m:
                return m.group(1)
    return ""


def build_flow(data, agent, cos: dict, mos: dict) -> dict[str, Any]:
    passes = data.all_passes()
    man = data.manifest()
    export_dir = config.data_dir() / "export"
    exported = sorted(export_dir.glob("*.json")) if export_dir.is_dir() else []
    exported = [p for p in exported if p.name != "clips.jsonl"]
    defect_clips = 0
    for p in exported:
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("defects_in_window"):
                defect_clips += 1
        except (OSError, ValueError):
            pass

    wps_break = []
    for r in passes:
        try:
            if data.wps_violations(r["sequence_name"]):
                wps_break.append(r["sequence_name"])
        except Exception:  # noqa: BLE001 - one bad pass must not blank the page
            pass

    cache = _cache_counts()
    assess = cache.get("assess_clip", {"real": 0, "fallback": 0})
    gen = cache.get("what_if", {"real": 0, "fallback": 0})
    tr = cache.get("sim2real", {"real": 0, "fallback": 0})
    placeholders = cache.get("placeholders", {}).get("fallback", 0)

    dm = data.data_mode()
    real_rows = [r for r in passes if not r.get("synthetic")]
    finding = _ndt_signal_finding(data, real_rows) if dm["real"] else None

    # one weld, end to end: real data -> the real pass with the most NDT indications (photo preferred);
    # simulator -> an RT-rejected porosity pass (the original demo story)
    ex_pass = None
    if dm["real"]:
        ind = [r for r in real_rows if r.get("rt_result") == "indications"]
        ex_pass = max(ind, key=lambda r: (sum((r.get("defect_counts") or {}).values()), bool(r.get("has_photo"))),
                      default=None)
    ex_pass = ex_pass or next((r for r in passes if "porosity" in (r.get("defect_types") or []) and r.get("rt_result") == "reject"),
                              next((r for r in passes if r.get("has_defect") and r.get("rt_result") == "reject"),
                                   next((r for r in passes if r.get("has_defect")), None)))
    example = None
    if ex_pass:
        seq = ex_pass["sequence_name"]
        viol = []
        try:
            viol = data.wps_violations(seq)
        except Exception:  # noqa: BLE001
            pass
        v0 = max(viol, key=lambda v: abs((v.get("t_end_s") or 0) - (v.get("t_start_s") or 0)), default=None)
        clip = next((r for r in man if r.get("sequence_name") == seq and r.get("defects")), None)
        verdict = _cached_verdict(seq)
        d0 = (ex_pass.get("defects") or [{}])[0]
        example = {
            "sequence": seq, "process": ex_pass.get("process"), "pass_type": ex_pass.get("pass_type"),
            "wps_id": ex_pass.get("wps_id"), "material": ex_pass.get("material"),
            "defect": d0.get("defect"), "defect_t": d0.get("t_start"), "rt_result": ex_pass.get("rt_result"),
            "violation": v0,
            "clip_id": clip.get("clip_id") if clip else None,
            "caption": (clip.get("caption") or "")[:260] if clip else None,
            "caption_model": clip.get("caption_model") if clip else None,
            "verdict": verdict.get("verdict") if verdict else None,
            "verdict_source": verdict.get("source") if verdict else None,
            "verdict_real": _is_real(verdict.get("source", "")) if verdict else False,
            "real": not ex_pass.get("synthetic"), "source": ex_pass.get("source_short"),
            "counts": ex_pass.get("defect_counts") or {}, "pass_label": ex_pass.get("pass_label"),
            "coupon": ex_pass.get("spool_id"), "has_photo": bool(ex_pass.get("has_photo")),
            "has_video": bool(ex_pass.get("has_video")),
        }
        if example["real"]:
            try:
                chk = data.indication_check(seq)
            except Exception:  # noqa: BLE001
                chk = None
            if chk and chk.get("params"):
                c = chk["params"].get("current_a") or next(iter(chk["params"].values()))
                example["signal_check"] = {"signature": chk["signature"], "n_windows": chk["n_windows"],
                                           "mean_in": c["mean_in"], "mean_clean": c["mean_clean"], "unit": c["unit"],
                                           "outside": c["windows_outside_band"]}

    srcs = [x for x in data.sources_summary()["sources"] if x.get("loaded") and not x.get("synthetic")]
    return {
        "data": dm, "sources": srcs, "finding": finding,
        "real": {"passes": len(real_rows), "inspected": sum(1 for r in real_rows if r.get("ndt_inspected")),
                 "with_indications": sum(1 for r in real_rows if r.get("rt_result") == "indications"),
                 "indications": sum(sum((r.get("defect_counts") or {}).values()) for r in real_rows),
                 "not_inspected": sum(1 for r in real_rows if not r.get("ndt_inspected")),
                 "photos": sum(r.get("n_photos") or 0 for r in real_rows),
                 "videos": sum(1 for r in real_rows if r.get("has_video")),
                 "coupons": sorted({r.get("spool_id") for r in real_rows if r.get("spool_id")}),
                 "simulated": dm["simulated"], "fixture": dm.get("fixture")},
        "collect": {"passes": len(passes), "defect_passes": sum(1 for r in passes if r.get("has_defect")),
                    "rt_reject": sum(1 for r in passes if r.get("rt_result") == "reject"),
                    "spools": sorted({r.get("spool_id") for r in passes if r.get("spool_id")}),
                    "processes": sorted({r.get("process") for r in passes if r.get("process")}),
                    "videos": sum(1 for r in passes if r.get("has_video")),
                    "fixtures": config.fixtures_enabled()},
        "mosaico": {"status": mos.get("status"), "detail": mos.get("short"), "sequences": len(passes),
                    "wps_break": len(wps_break), "source": data.last_source},
        "export": {"clips": len(exported), "defect_clips": defect_clips},
        "curate": {"backend": _curation_backend(), "total": len(man), "kept": sum(1 for r in man if r.get("kept")),
                   "reasons": dict(Counter(r.get("reject_reason") for r in man if not r.get("kept"))),
                   "caption_models": dict(Counter(r.get("caption_model") or "?" for r in man))},
        "cosmos": {"mode": cos.get("mode_label"), "reasoner_live": bool(cos.get("reasoner")),
                   "generator_live": bool(cos.get("generator")),
                   "assess": assess, "what_if": gen, "sim2real": tr, "placeholders": placeholders},
        "copilot": {"llm": agent.label, "backend": agent.name},
        "example": example,
    }


def _ndt_signal_finding(data, rows: list[dict]) -> dict | None:
    """Across every real pass with NDT indications: did the arc signals change inside the indication windows?"""
    checked, with_sig, windows, best = 0, 0, 0, None
    for r in rows:
        if r.get("rt_result") != "indications":
            continue
        try:
            chk = data.indication_check(r["sequence_name"])
        except Exception:  # noqa: BLE001
            continue
        if not chk or not chk.get("params"):
            continue
        checked += 1
        windows += chk["n_windows"]
        with_sig += int(bool(chk["signature"]))
        if best is None or chk["n_windows"] > best[1]["n_windows"]:
            best = (r, chk)
    if not checked:
        return None
    r, chk = best
    c = chk["params"].get("current_a") or next(iter(chk["params"].values()))
    return {"passes_checked": checked, "passes_with_signature": with_sig, "windows": windows,
            "example": {"sequence": r["sequence_name"], "counts": r.get("defect_counts") or {},
                        "n_windows": chk["n_windows"], "param": "current_a" if "current_a" in chk["params"] else None,
                        "mean_in": c["mean_in"], "mean_clean": c["mean_clean"], "unit": c["unit"],
                        "outside": c["windows_outside_band"]}}
