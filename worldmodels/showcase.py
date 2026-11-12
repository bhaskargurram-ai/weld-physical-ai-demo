"""Showcase jobs (configs/cosmos/showcase.json) and the warm-cache runner.

``warm_cache()`` is what we run on the RunPod pod before the customer meeting:
it assesses every kept curated clip with the Reasoner and renders the showcase
what-if / sim2real clips with the Generator, all written through to
``data/cache/worldmodels`` so the copilot can replay them with no GPU.
The run report (``data/cache/worldmodels/warm_report.json``) maps showcase ids
to MP4 paths; ``get_showcase(id)`` / ``list_showcase()`` read it back.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from .config import data_dir, get_settings, repo_root

SHOWCASE_FILE = repo_root() / "configs" / "cosmos" / "showcase.json"


def report_path() -> Path:
    return get_settings().cache_dir / "warm_report.json"


def load_showcase(path: Optional[str | Path] = None) -> dict:
    return json.loads(Path(path or SHOWCASE_FILE).read_text())


def read_manifest(manifest: str | Path) -> list[dict]:
    rows = []
    for line in Path(manifest).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def resolve_clip_path(row: dict, manifest: str | Path) -> Optional[Path]:
    mdir = Path(manifest).resolve().parent
    uri = str(row.get("clip_uri") or "")
    cands: list[Path] = []
    if uri.startswith("file://"):
        cands.append(Path(uri[7:]))
    elif uri and "://" not in uri:
        p = Path(uri)
        cands += [p] if p.is_absolute() else [mdir / p, Path.cwd() / p, data_dir().parent / p]
    if row.get("clip_id"):
        for ext in ((".jpg", ".png") if row.get("media_type") == "image" else (".mp4",)):
            cands += [mdir / "clips" / f"{row['clip_id']}{ext}", data_dir() / "curated" / "clips" / f"{row['clip_id']}{ext}"]
    for c in cands:
        if c.exists():
            return c.resolve()
    return None


def context_for_row(row: dict) -> dict:
    """Prefer the full export sidecar (signal stats + WPS) when present, else the manifest row."""
    cid = row.get("clip_id")
    base = re.sub(r"_p\d+$", "", str(cid)) if cid else cid    # photo items: <window clip_id>_p<k>
    for cand in (data_dir() / "export" / f"{cid}.json", data_dir() / "export" / f"{base}.json",
                 data_dir() / "curated" / "clips" / f"{cid}.json"):
        if cid and cand.exists():
            try:
                ctx = json.loads(cand.read_text())
                if row.get("caption") and "caption" not in ctx:
                    ctx["caption"] = row["caption"]
                if row.get("media_type"):
                    ctx["media_type"] = row["media_type"]
                return ctx
            except Exception:
                pass
    ctx = dict(row)
    if "signal_stats" not in ctx and isinstance(row.get("sensor_summary"), dict):
        ctx["signal_stats"] = row["sensor_summary"]
    return ctx


def _row_defects(row: dict) -> set[str]:
    out = set()
    for d in row.get("defects") or row.get("defects_in_window") or []:
        out.add(str(d.get("defect") or d.get("type")) if isinstance(d, dict) else str(d))
    return out


def _row_source(row: dict) -> dict:
    src = row.get("source")
    return src if isinstance(src, dict) else {}


def _is_video_row(row: dict) -> bool:
    return (row.get("media_type") or "video") == "video" and not str(row.get("clip_uri") or "").lower().endswith(
        (".jpg", ".jpeg", ".png"))


def pick_clip(rows: list[dict], spec: Optional[str], manifest: str | Path) -> Optional[tuple[dict, Path]]:
    """Kept *video* row for ``spec``: ``auto`` (real process-camera clips first, e.g. BAM), ``auto:<defect>``
    (a clip with that defect first) or ``auto@<source_id>`` / ``auto:<defect>@<source_id>``."""
    if not spec:
        return None
    spec, _, want_src = spec.partition("@")
    want = spec.split(":", 1)[1] if spec.startswith("auto:") else None
    kept = [r for r in rows if r.get("kept", True) and _is_video_row(r)]
    kept.sort(key=lambda r: (bool(want_src) and _row_source(r).get("id") != want_src,
                             bool(_row_source(r).get("synthetic", not _row_source(r)))))
    ordered = ([r for r in kept if want in _row_defects(r)] if want else []) + kept
    for r in ordered:
        p = resolve_clip_path(r, manifest)
        if p:
            return r, p
    return None


def _pass_meta(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text())
    except Exception:
        return {}


def pick_pass_dir(spec: Optional[str], rows: list[dict], kinds: Optional[list[str]] = None) -> Optional[Path]:
    """Pass dir with control videos for Transfer: an explicit sequence name, ``auto`` (real passes with
    the needed ``control/<kind>.mp4`` first, e.g. BAM edge maps of the real video) or ``auto:<defect>``."""
    gen = data_dir() / "generated"
    if not gen.exists():
        return None
    kinds = kinds or ["edge"]
    dirs = sorted(d for d in gen.iterdir() if (d / "control").is_dir()
                  and all((d / "control" / f"{k}.mp4").exists() for k in kinds))
    if spec and not spec.startswith("auto"):
        d = gen / spec
        return d if d.exists() else None
    real_first = sorted(dirs, key=lambda d: bool((_pass_meta(d).get("source") or {"synthetic": True}).get("synthetic", False)
                                               if isinstance(_pass_meta(d).get("source"), dict) else True))
    want = spec.split(":", 1)[1] if spec and spec.startswith("auto:") else None
    if want:
        for d in real_first:
            meta = _pass_meta(d)
            if any((x.get("defect") if isinstance(x, dict) else x) == want for x in meta.get("defects", [])):
                return d
    return real_first[0] if real_first else None


def warm_cache(manifest: Optional[str | Path] = None, *, showcase: Optional[str | Path] = None, limit: Optional[int] = None,
               do_assess: bool = True, do_explain: bool = True, do_generate: bool = True, do_transfer: bool = True,
               preset: str = "showcase", dry_run: bool = False, log: Callable[[str], None] = print,
               photo_limit: Optional[int] = None) -> dict:
    """``limit`` caps the kept video clips assessed, ``photo_limit`` the kept photo items (default: limit)."""
    from . import generate, reason, transfer
    from .video import extract_frame

    s = get_settings()
    t_start = time.time()
    report: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "preset": preset, "assess": [],
                              "what_if": {}, "transfer": {}, "errors": []}
    rows: list[dict] = []
    if manifest and Path(manifest).exists():
        rows = read_manifest(manifest)
    elif manifest:
        report["errors"].append(f"manifest not found: {manifest}")
        log(f"! manifest not found: {manifest}")
    kept_all = [r for r in rows if r.get("kept", True)]
    vids = [r for r in kept_all if _is_video_row(r)]
    photos = [r for r in kept_all if not _is_video_row(r)]
    plim = limit if photo_limit is None else photo_limit
    if limit:
        vids = vids[:limit]
    if plim is not None and plim >= 0 and (photo_limit is not None or limit):
        photos = photos[:plim]
    kept = vids + photos

    if do_assess:
        for i, row in enumerate(kept):
            clip = resolve_clip_path(row, manifest) if manifest else None
            if not clip:
                report["errors"].append(f"clip not found for {row.get('clip_id')}")
                continue
            if dry_run:
                log(f"[dry-run] assess {row.get('clip_id')} {clip}")
                continue
            try:
                ctx = context_for_row(row)
                res = reason.assess_clip(str(clip), ctx)
                rec = {"clip_id": row.get("clip_id"), "path": str(clip), "verdict": res["verdict"],
                       "media_type": "video" if _is_video_row(row) else "image",
                       "source": res["source"], "latency_ms": res["latency_ms"], "cache_key": res.get("cache_key")}
                if do_explain:
                    ex = reason.explain_signals_meta(ctx)
                    rec["explain_source"] = ex["source"]
                report["assess"].append(rec)
                log(f"[{i + 1}/{len(kept)}] assess {row.get('clip_id')}: {res['verdict']} ({res['source']}, {res['latency_ms']} ms)")
            except Exception as e:  # keep going: a long warm run must not die on one clip
                report["errors"].append(f"assess {row.get('clip_id')}: {e}")
                log(f"! assess {row.get('clip_id')} failed: {e}")

    sc = load_showcase(showcase)
    frames_dir = s.cache_dir / "frames"
    if do_generate:
        for job in sc.get("what_if", []):
            jid = job["id"]
            image = video = None
            scene = dict(job.get("scene") or {})
            cond = job.get("condition", "none")
            if cond in ("image", "video"):
                picked = pick_clip(rows, job.get("source_clip") or "auto", manifest) if rows else None
                if not picked:
                    log(f"- {jid}: no source clip available, falling back to text2video")
                else:
                    row, clip = picked
                    if not _row_source(row).get("synthetic", True):
                        scene.setdefault("source", _row_source(row).get("id"))
                    if cond == "image":
                        which = job.get("frame", "middle" if not _row_source(row).get("synthetic", True) else "first")
                        image = str(frames_dir / f"{row.get('clip_id', clip.stem)}_{which}.png")
                        if not dry_run:
                            extract_frame(clip, image, which)
                    else:
                        video = str(clip)
            if dry_run:
                log(f"[dry-run] what_if {jid} image={image} video={video}")
                continue
            try:
                meta = generate.what_if_meta(job["prompt"], image_path=image, video_path=video,
                                             seconds=job.get("seconds", 5), seed=job.get("seed", 0), preset=preset,
                                             scene=scene or None, match_text=f"{jid} {job.get('title', '')}")
                report["what_if"][jid] = {"title": job.get("title"), "path": meta["path"], "source": meta["source"],
                                          "latency_ms": meta["latency_ms"], "image": image, "video": video,
                                          "prompt": job["prompt"], "placeholder": meta.get("placeholder", False)}
                log(f"what_if {jid}: {meta['source']} {meta['latency_ms']} ms -> {meta['path']}")
            except Exception as e:
                report["errors"].append(f"what_if {jid}: {e}")
                log(f"! what_if {jid} failed: {e}\n{traceback.format_exc(limit=2)}")

    if do_transfer:
        for job in sc.get("transfer", []):
            jid = job["id"]
            pdir = pick_pass_dir(job.get("source_pass", "auto"), rows, job.get("kinds", ["edge"]))
            controls = transfer.controls_for_pass(pdir) if pdir else {}
            controls = {k: v for k, v in controls.items() if k in job.get("kinds", ["edge"])}
            if len(controls) != len(job.get("kinds", ["edge"])):
                report["errors"].append(f"transfer {jid}: control videos missing in {pdir}")
                log(f"! transfer {jid}: control videos missing ({pdir})")
                continue
            if dry_run:
                log(f"[dry-run] transfer {jid} {controls}")
                continue
            try:
                scene = dict(job.get("scene") or {})
                psrc = _pass_meta(pdir).get("source") if pdir else None
                if isinstance(psrc, dict) and not psrc.get("synthetic", False):
                    scene.setdefault("source", psrc.get("id"))
                meta = transfer.sim2real_meta(controls, job["prompt"], job.get("seed", 0), preset=preset,
                                              scene=scene or None)
                report["transfer"][jid] = {"title": job.get("title"), "path": meta["path"], "source": meta["source"],
                                           "latency_ms": meta["latency_ms"], "controls": controls,
                                           "sequence_name": pdir.name if pdir else None,
                                           "data_source": scene.get("source"),
                                           "placeholder": meta.get("placeholder", False)}
                log(f"transfer {jid}: {meta['source']} {meta['latency_ms']} ms -> {meta['path']}")
            except Exception as e:
                report["errors"].append(f"transfer {jid}: {e}")
                log(f"! transfer {jid} failed: {e}")

    report["elapsed_s"] = round(time.time() - t_start, 1)
    if not dry_run:
        rp = report_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        if rp.exists():  # never replace a real Cosmos result from an earlier GPU run with a placeholder
            try:
                old = json.loads(rp.read_text())
                for section in ("what_if", "transfer"):
                    for jid, prev in (old.get(section) or {}).items():
                        cur = report[section].get(jid)
                        if (cur is None or cur.get("placeholder")) and not prev.get("placeholder") and Path(prev.get("path", "")).exists():
                            report[section][jid] = {**prev, "kept_from_previous_run": True}
            except Exception:
                pass
        rp.write_text(json.dumps(report, indent=2, default=str))
        report["report_path"] = str(rp)
    return report


def list_showcase() -> dict:
    """{"what_if": {id: {...path, source}}, "transfer": {...}} from the last warm-cache run."""
    rp = report_path()
    if not rp.exists():
        return {"what_if": {}, "transfer": {}}
    rep = json.loads(rp.read_text())
    return {"what_if": rep.get("what_if", {}), "transfer": rep.get("transfer", {})}


def get_showcase(job_id: str) -> Optional[dict]:
    sc = list_showcase()
    return sc["what_if"].get(job_id) or sc["transfer"].get(job_id)
