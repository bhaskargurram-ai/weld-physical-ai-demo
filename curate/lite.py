"""``lite`` backend: CPU-only re-implementation of the Curator chain.

cut (fixed-stride / whole clip) -> transcode H.264 -> quality metrics -> filter ->
embed -> near-duplicate removal -> caption (Cosmos 3 Reasoner or CWI template) ->
Lance + manifest.jsonl + dataset card.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from curate.captions import TEMPLATE_MODEL, get_captioner, template_caption
from curate.common import CLIP_DEFAULTS, ClipInput, discover_inputs, display_path, log
from curate.manifest import ClipRecord, Thresholds, apply_filters, dedup, write_lance, write_manifest
from curate.video import EMBED_NAME, analyze, analyze_image, embed, plan_spans, prepare_image, probe, transcode


@dataclass
class LiteOptions:
    clip_s: float = CLIP_DEFAULTS["clip_s"]
    stride_s: float = CLIP_DEFAULTS["stride_s"]
    min_s: float = CLIP_DEFAULTS["min_s"]
    captioner: str = "auto"          # auto | reason | template
    embeddings: bool = True
    dedup: bool = True
    workers: int = 4
    thresholds: Thresholds = None    # type: ignore[assignment]

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = Thresholds()


@dataclass
class ImageProbe:
    """Probe-like record for photo items (Lance columns clip_width/height/frame_rate)."""
    width: int
    height: int
    fps: float | None = None
    duration_s: float = 0.0


def image_record(src: ClipInput, clips_dir: Path, embeddings: bool = True) -> tuple[ClipRecord, object]:
    """Photo-only item: copy the bead photo to clips/<clip_id>.jpg, photo quality metrics, embedding.
    The record spans the sidecar's window (t0_s..t1_s) so it joins the window's sensor data and labels."""
    span = max(0.0, (src.t1_s if src.t1_s is not None else src.t0_s) - src.t0_s)
    dst = clips_dir / f"{src.clip_id}.jpg"
    rec = ClipRecord(clip_id=src.clip_id, clip_uri=display_path(dst), source=src, start_s=0.0, end_s=span)
    try:
        w, h = prepare_image(src.video, dst)
        q, visual = analyze_image(dst)
        q.update({"width": w, "height": h, "duration_s": None, "fps": None})
        rec.quality = q
        if embeddings:
            rec.embedding = embed(visual, src.sidecar.get("signal_stats"))
            rec.embedding_name = EMBED_NAME
        return rec, ImageProbe(w, h)
    except Exception as exc:  # noqa: BLE001
        log.warning("image %s failed: %s", src.clip_id, exc)
        rec.quality = {"error": str(exc)}
        return rec, None


def _cut_one(src: ClipInput, clips_dir: Path, opt: LiteOptions) -> list[tuple[ClipRecord, object]]:
    if src.is_image:
        return [image_record(src, clips_dir, opt.embeddings)]
    try:
        pr = probe(src.video)
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot probe %s: %s", src.video, exc)
        r = ClipRecord(clip_id=src.clip_id, clip_uri=display_path(src.video), source=src, start_s=0.0,
                       end_s=0.0, quality={"error": str(exc)})
        r.kept, r.reject_reason = False, "decode_error"
        return [(r, None)]
    spans = plan_spans(pr.duration_s, opt.clip_s, opt.stride_s, opt.min_s)
    if not spans:
        r = ClipRecord(clip_id=src.clip_id, clip_uri=display_path(src.video), source=src, start_s=0.0,
                       end_s=pr.duration_s, quality={"duration_s": round(pr.duration_s, 3)})
        r.kept, r.reject_reason = False, "too_short"
        return [(r, pr)]
    out = []
    for k, (a, b) in enumerate(spans):
        cid = src.clip_id if len(spans) == 1 else f"{src.clip_id}_{k:02d}"
        dst = clips_dir / f"{cid}.mp4"
        rec = ClipRecord(clip_id=cid, clip_uri=display_path(dst), source=src, start_s=a, end_s=b)
        try:
            whole = len(spans) == 1 and a == 0.0
            transcode(src.video, dst, None if whole else a, None if whole else b - a)
            cpr = probe(dst)
            q, visual = analyze(dst)
            q.update({"duration_s": round(cpr.duration_s, 3), "width": cpr.width, "height": cpr.height,
                      "fps": round(cpr.fps, 3)})
            rec.quality = q
            if opt.embeddings:
                rec.embedding = embed(visual, src.sidecar.get("signal_stats"))
                rec.embedding_name = EMBED_NAME
            out.append((rec, cpr))
        except Exception as exc:  # noqa: BLE001
            log.warning("clip %s failed: %s", cid, exc)
            rec.quality = {"error": str(exc)}
            out.append((rec, None))
    return out


def caption_records(records: list[ClipRecord], captioner: str = "auto") -> None:
    """Reasoner caption per clip / photo when a backend resolves, else (or on failure) the template."""
    model, fn = get_captioner(captioner)
    fallbacks = 0
    for r in records:
        ctx = r.context()
        if fn is not None and not r.quality.get("error"):
            try:
                r.caption = fn(str(Path(r.clip_uri).resolve()), ctx, r.quality)
                r.caption_model = model
                continue
            except Exception as exc:  # noqa: BLE001
                fallbacks += 1
                log.debug("reasoner caption failed for %s: %s", r.clip_id, exc)
        r.caption = template_caption(ctx, r.quality)
        r.caption_model = TEMPLATE_MODEL
    if fallbacks:
        log.warning("%d items fell back to template captions", fallbacks)


def process_images(inputs: list[ClipInput], clips_dir: Path, embeddings: bool = True) -> tuple[list[ClipRecord], dict]:
    """Lite path for photo items (used by every backend: Curator only handles video)."""
    records, probes = [], {}
    for src in inputs:
        rec, pr = image_record(src, clips_dir, embeddings)
        records.append(rec)
        if pr is not None:
            probes[rec.clip_id] = pr
    return records, probes


def run_lite(input_dir: str | Path, out_dir: str | Path, opt: LiteOptions | None = None) -> list[ClipRecord]:
    opt = opt or LiteOptions()
    out = Path(out_dir)
    inputs = discover_inputs(input_dir)
    if not inputs:
        raise SystemExit(f"no *.mp4 clips or photo sidecars found in {input_dir}")
    n_img = sum(i.is_image for i in inputs)
    log.info("lite: %d input clips + %d photo items from %s", len(inputs) - n_img, n_img, input_dir)
    clips_dir = out / "clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True)

    # 1-3: cut, transcode, quality + embedding
    records, probes = [], {}
    with ThreadPoolExecutor(max_workers=max(1, opt.workers)) as ex:
        for res in ex.map(lambda s: _cut_one(s, clips_dir, opt), inputs):
            for rec, pr in res:
                records.append(rec)
                if pr is not None:
                    probes[rec.clip_id] = pr

    # 4: filter + near-dup
    apply_filters(records, opt.thresholds)
    if opt.dedup and opt.embeddings:
        n = dedup(records, opt.thresholds.dedup_cosine)
        log.info("near-duplicate removal dropped %d clips", n)

    # 5: captions (all clips incl. rejected, so the copilot can explain rejections)
    caption_records(records, opt.captioner)

    # 6: outputs
    write_lance(records, out / "lance", probes)
    write_manifest(records, out)
    return records
