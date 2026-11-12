"""Shared finishing stage: filter, de-duplicate, write manifest.jsonl / Lance / prompts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from curate.common import (ClipInput, defects_overlapping, log, normalize_signal_stats, round_stats,
                           violations_overlapping)
from curate.prompts import all_variants, prompt_text

# Contract (docs/PLAN.md "Curation output"): exactly these keys, in this order. ``media_type`` (video | image)
# and ``source`` ({id, license, attribution, ...} of the dataset) were added for the real-data sources
# (docs/REAL_DATA.md); image rows have clip_uri = clips/<clip_id>.jpg (a post-weld bead photo).
MANIFEST_KEYS = ("clip_id", "clip_uri", "sequence_name", "t0_s", "t1_s", "caption", "caption_model",
                 "sensor_summary", "wps_violations", "defects", "rt_result", "quality", "kept",
                 "reject_reason", "generator_prompt", "media_type", "source")


# pass metadata copied into sensor_summary next to the per-signal {mean,std,min,max} dicts
PASS_KEYS = ("process", "pass_type", "pass_no", "wps_id", "position", "material", "spool_id", "joint_id", "cell_id",
             "electrodes", "thickness_mm")


@dataclass
class Thresholds:
    min_duration_s: float = 2.0
    min_blur: float = 5.0          # Laplacian variance on 320-px-wide grey frames
    min_motion: float = 0.05       # mean |diff| per source frame (0-255)
    max_glare_frac: float = 0.35   # mean fraction of saturated pixels
    max_dark_frac: float = 0.97
    min_mean_luma: float = 6.0
    min_arc_on_frac: float = 0.0   # opt-in: reject clips where the arc is visible < this share of frames
    dedup_cosine: float = 0.98     # on mean-centred embeddings, see dedup()


@dataclass
class ClipRecord:
    clip_id: str
    clip_uri: str
    source: ClipInput
    start_s: float                  # offset inside the source video
    end_s: float
    quality: dict[str, Any] = field(default_factory=dict)
    caption: str | None = None
    caption_model: str | None = None
    embedding: np.ndarray | None = None
    embedding_name: str | None = None
    kept: bool = True
    reject_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # backend-specific Lance columns

    # -------- absolute pass time (sidecar t0_s + offset)
    @property
    def t0_s(self) -> float:
        return round(self.source.t0_s + self.start_s, 3)

    @property
    def t1_s(self) -> float:
        return round(self.source.t0_s + self.end_s, 3)

    def context(self) -> dict[str, Any]:
        """Sidecar view restricted to this clip's window (used by captioners / prompts)."""
        sc = self.source.sidecar
        whole = abs(self.start_s) < 1e-3 and (self.source.t1_s is None or
                                              abs(self.source.t0_s + self.end_s - self.source.t1_s) < 0.6)
        defects = sc.get("defects_in_window") or []
        viol = sc.get("wps_violations") or []
        if not whole:
            defects = defects_overlapping(defects, self.t0_s, self.t1_s)
            viol = violations_overlapping(viol, self.t0_s, self.t1_s)
        else:
            defects = defects_overlapping(defects, float("-inf"), float("inf"))
        return {
            "clip_id": self.clip_id, "sequence_name": self.source.sequence_name,
            "t0_s": self.t0_s, "t1_s": self.t1_s, "meta": self.source.meta,
            "signal_stats": sc.get("signal_stats") or {}, "wps_violations": viol,
            "defects": defects, "rt_result": sc.get("rt_result", self.source.meta.get("rt_result")),
            "media_type": self.source.media_type, "source": self.source.source_info,
            "video_kind": sc.get("video_kind"), "photo_note": sc.get("photo_note"),
            "ndt": sc.get("ndt") if isinstance(sc.get("ndt"), dict) else self.source.meta.get("ndt"),
            "electrodes": sc.get("electrodes") or self.source.meta.get("electrodes"),
        }


def apply_filters(records: list[ClipRecord], th: Thresholds) -> None:
    """Set kept/reject_reason from quality metrics (first failing rule wins)."""
    for r in records:
        if not r.kept:
            continue
        q = r.quality
        reason = None
        if q.get("error"):
            reason = "decode_error"
        elif not r.source.is_image and (r.end_s - r.start_s) < th.min_duration_s:
            reason = "too_short"
        elif q.get("mean_luma") is not None and (q["mean_luma"] < th.min_mean_luma
                                                 or q.get("dark_frac", 0) > th.max_dark_frac):
            reason = "underexposed"
        elif q.get("glare_frac") is not None and q["glare_frac"] > th.max_glare_frac:
            reason = "overexposed_glare"
        elif q.get("blur") is not None and q["blur"] < th.min_blur:
            reason = "blurry"
        elif q.get("motion") is not None and q["motion"] < th.min_motion:
            reason = "static"
        elif th.min_arc_on_frac > 0 and q.get("arc_on_frac") is not None and q["arc_on_frac"] < th.min_arc_on_frac:
            reason = "arc_off"
        if reason:
            r.kept, r.reject_reason = False, reason


def _label_key(r: ClipRecord) -> tuple:
    ctx = r.context()
    from curate.common import defect_name
    return (tuple(sorted({defect_name(d) for d in ctx["defects"]})),
            tuple(sorted({str(v.get("param")) for v in ctx["wps_violations"]})),
            str(ctx["rt_result"]))


def dedup(records: list[ClipRecord], cosine: float) -> int:
    """Greedy, label-aware near-duplicate removal over kept clips with embeddings.

    Only clips with the same labels (defect set, violated WPS params, RT result) can be
    duplicates of each other - a visually similar clip with different labels is a
    distinct training sample. Similarity is cosine after subtracting the mean embedding of
    the candidate set (with >= 3 clips): weld-camera clips share one layout (pipe, torch,
    arc in the centre), so raw cosines are all ~0.999; centring compares what differs.
    Sharper clips are visited first so the best member of a duplicate group survives.
    Returns the number removed.
    """
    cand = [r for r in records if r.kept and r.embedding is not None]
    cand.sort(key=lambda r: (-(r.quality.get("blur") or 0.0), r.clip_id))
    if not cand:
        return 0
    mat = np.stack([c.embedding / (np.linalg.norm(c.embedding) + 1e-9) for c in cand]).astype(np.float64)
    if len(cand) >= 3:
        mat = mat - mat.mean(axis=0, keepdims=True)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    kept_vecs: dict[tuple, list[np.ndarray]] = {}
    removed = 0
    for r, v in zip(cand, mat):
        key = _label_key(r)
        group = kept_vecs.setdefault(key, [])
        if group and float(np.max(np.stack(group) @ v)) >= cosine:
            r.kept, r.reject_reason = False, "near_duplicate"
            removed += 1
        else:
            group.append(v)
    return removed


def manifest_row(r: ClipRecord) -> dict[str, Any]:
    ctx = r.context()
    stats = round_stats(normalize_signal_stats(ctx["signal_stats"]))
    row_for_prompt = {**ctx, "signal_stats": stats}
    gen = prompt_text(all_variants(row_for_prompt)["photoreal_transfer"])
    quality = {"blur": r.quality.get("blur"), "motion": r.quality.get("motion"),
               **{k: v for k, v in r.quality.items() if k not in ("blur", "motion")}}
    row = {
        "clip_id": r.clip_id,
        "clip_uri": r.clip_uri,
        "sequence_name": r.source.sequence_name,
        "t0_s": r.t0_s,
        "t1_s": r.t1_s,
        "caption": r.caption,
        "caption_model": r.caption_model,
        "sensor_summary": {**{k: ctx["meta"][k] for k in PASS_KEYS if ctx["meta"].get(k) is not None}, **stats},
        "wps_violations": ctx["wps_violations"],
        "defects": ctx["defects"],
        "rt_result": ctx["rt_result"],
        "quality": quality,
        "kept": bool(r.kept),
        "reject_reason": r.reject_reason,
        "generator_prompt": gen,
        "media_type": r.source.media_type,
        "source": {k: v for k, v in ctx["source"].items() if k in ("id", "license", "attribution", "doi", "url",
                                                                    "title", "synthetic")},
    }
    assert tuple(row) == MANIFEST_KEYS
    return row


def write_manifest(records: list[ClipRecord], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.jsonl"
    prompts_dir = out / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    with path.open("w") as f:
        for r in sorted(records, key=lambda x: (x.source.sequence_name, x.t0_s, x.clip_id)):
            row = manifest_row(r)
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            variants = all_variants(r.context())
            (prompts_dir / f"{r.clip_id}.json").write_text(json.dumps(variants, indent=1, ensure_ascii=False))
    log.info("wrote %s (%d rows)", path, len(records))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_lance(records: list[ClipRecord], uri: Path, probes: dict[str, Any] | None = None) -> Path:
    """Write a Curator-Next-shaped Lance table (one row per clip)."""
    import lance
    import pyarrow as pa

    from curate.common import stable_id

    models = sorted({r.caption_model for r in records if r.caption_model})
    emb_names = sorted({r.embedding_name for r in records if r.embedding_name})
    cols: dict[str, list] = {k: [] for k in (
        "source_uri", "span_id", "start_ns", "end_ns", "clip_id", "clip_uri", "clip_duration_ns",
        "clip_width", "clip_height", "clip_frame_rate", "sequence_name", "t0_s", "t1_s", "kept",
        "reject_reason", "quality_blur", "quality_motion", "quality_glare_frac", "quality_mean_luma")}
    for m in models:
        cols[f"caption_{m}_v1"] = []
    for e in emb_names:
        cols[f"embedding_{e}_v1"] = []
    for r in records:
        start_ns, end_ns = int(round(r.start_s * 1e9)), int(round(r.end_s * 1e9))
        src = str(r.source.video.resolve())
        pr = (probes or {}).get(r.clip_id)
        cols["source_uri"].append(src)
        cols["span_id"].append(stable_id(src, 0, start_ns, end_ns))
        cols["start_ns"].append(start_ns)
        cols["end_ns"].append(end_ns)
        cols["clip_id"].append(r.clip_id)
        cols["clip_uri"].append(r.clip_uri)
        cols["clip_duration_ns"].append(end_ns - start_ns)
        cols["clip_width"].append(getattr(pr, "width", None))
        cols["clip_height"].append(getattr(pr, "height", None))
        cols["clip_frame_rate"].append(getattr(pr, "fps", None))
        cols["sequence_name"].append(r.source.sequence_name)
        cols["t0_s"].append(r.t0_s)
        cols["t1_s"].append(r.t1_s)
        cols["kept"].append(bool(r.kept))
        cols["reject_reason"].append(r.reject_reason)
        cols["quality_blur"].append(r.quality.get("blur"))
        cols["quality_motion"].append(r.quality.get("motion"))
        cols["quality_glare_frac"].append(r.quality.get("glare_frac"))
        cols["quality_mean_luma"].append(r.quality.get("mean_luma"))
        for m in models:
            cols[f"caption_{m}_v1"].append(r.caption if r.caption_model == m else None)
        for e in emb_names:
            cols[f"embedding_{e}_v1"].append(
                r.embedding.astype(np.float32).tolist() if r.embedding_name == e and r.embedding is not None
                else None)
    schema_fields = []
    types = {"start_ns": pa.int64(), "end_ns": pa.int64(), "clip_duration_ns": pa.int64(),
             "clip_width": pa.int32(), "clip_height": pa.int32(), "clip_frame_rate": pa.float64(),
             "t0_s": pa.float64(), "t1_s": pa.float64(), "kept": pa.bool_(),
             "quality_blur": pa.float64(), "quality_motion": pa.float64(),
             "quality_glare_frac": pa.float64(), "quality_mean_luma": pa.float64()}
    for k in cols:
        if k.startswith("embedding_"):
            dim = next((len(v) for v in cols[k] if v is not None), 0)
            t = pa.list_(pa.float32(), dim) if dim else pa.list_(pa.float32())
        else:
            t = types.get(k, pa.string())
        schema_fields.append(pa.field(k, t))
    table = pa.table(cols, schema=pa.schema(schema_fields))
    uri.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(uri), mode="overwrite")
    log.info("wrote Lance table %s (%d rows, columns %s)", uri, table.num_rows, table.column_names)
    return uri
