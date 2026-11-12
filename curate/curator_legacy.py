"""``curator-legacy`` backend: Cosmos Curator legacy ``video-pipeline split`` (best effort).

Unlike Curator Next it accepts a local input directory, has motion/aesthetic filters,
supports custom caption prompts and Cosmos 3 / Cosmos-Reason captioners, and can emit a
Cosmos-Predict2 post-training dataset. Equivalent manual command (run in $CURATOR_HOME):

    pixi run --as-is video-pipeline split \
        --input-video-path   /abs/data/export \
        --output-clip-path   /abs/data/curated/legacy \
        --splitting-algorithm fixed-stride --fixed-stride-split-duration 5 \
        --fixed-stride-min-clip-length-s 2 \
        --transcode-encoder libopenh264 \
        --motion-filter score-only \
        --embedding-algorithm cosmos-embed1-336p \
        --captioning-algorithm cosmos3_nano \
        --captioning-prompt-text "$(cat data/curated/configs/weld_caption_prompt.txt)" \
        --generate-cosmos-predict-dataset

Outputs: <legacy>/clips/<uuid>.mp4, <legacy>/metas/v0/<uuid>.json (source_video, start_ns,
end_ns, clip_location, windows[*].<model>_caption, motion_score),
<legacy>/cosmos_predict2_video2world_dataset/{metas,videos,t5_xxl}.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from curate.common import CLIP_DEFAULTS, ClipInput, discover_inputs, display_path, log
from curate.prompts import WELD_CAPTION_PROMPT

# cosmos_curator/pipelines/video/splitting_pipeline.py ALL_CAPTION_ALGOS (subset worth using here)
LEGACY_CAPTIONERS = ("cosmos3_nano", "cosmos3_super", "cosmos_r2", "cosmos_r1", "qwen", "qwen3_vl_30b_fp8")


def build_command(input_dir: Path, legacy_out: Path, *, captioner: str = "cosmos3_nano",
                  clip_s: float = CLIP_DEFAULTS["clip_s"], min_s: float = CLIP_DEFAULTS["min_s"],
                  prompt: str = WELD_CAPTION_PROMPT, predict_dataset: bool = True) -> list[str]:
    cmd = ["pixi", "run", "--as-is", "video-pipeline", "split",
           "--input-video-path", str(input_dir.resolve()),
           "--output-clip-path", str(legacy_out.resolve()),
           "--splitting-algorithm", "fixed-stride",
           "--fixed-stride-split-duration", str(int(round(clip_s))),
           "--fixed-stride-min-clip-length-s", str(float(min_s)),
           "--transcode-encoder", "libopenh264",
           "--motion-filter", "score-only",
           "--embedding-algorithm", "cosmos-embed1-336p",
           "--captioning-algorithm", captioner,
           "--captioning-prompt-text", prompt]
    if predict_dataset:
        cmd.append("--generate-cosmos-predict-dataset")
    return cmd


def read_back(legacy_out: Path, inputs: list[ClipInput]):
    import numpy as np

    from curate.manifest import ClipRecord

    by_name = {i.video.name: i for i in inputs}
    records = []
    for p in sorted((legacy_out / "metas" / "v0").glob("*.json")):
        m = json.loads(p.read_text())
        src = by_name.get(Path(str(m.get("source_video", ""))).name)
        if src is None:
            continue
        rec = ClipRecord(clip_id=str(m.get("span_uuid") or p.stem),
                         clip_uri=display_path(m["clip_location"]) if "://" not in str(m.get("clip_location"))
                         else str(m.get("clip_location")),
                         source=src, start_s=float(m.get("start_ns", 0)) / 1e9,
                         end_s=float(m.get("end_ns", 0)) / 1e9)
        for w in m.get("windows") or []:
            for k, v in w.items():
                if k.endswith("_caption") and not k.endswith("_enhanced_caption") and v:
                    rec.caption, rec.caption_model = str(v), k[: -len("_caption")]
                    break
            if rec.caption:
                break
        if m.get("embedding"):
            rec.embedding = np.asarray(m["embedding"], dtype=np.float32)
            rec.embedding_name = str(m.get("embedding_model_name") or "cosmos_embed1_336p").replace("-", "_")
        if m.get("valid") is False:
            rec.kept, rec.reject_reason = False, "curator_invalid"
        records.append(rec)
    return records


def run_curator_legacy(input_dir: str | Path, out_dir: str | Path, *, captioner: str = "cosmos3_nano",
                       clip_s: float = CLIP_DEFAULTS["clip_s"], min_s: float = CLIP_DEFAULTS["min_s"],
                       dry_run: bool = False, thresholds=None, dedup_enabled: bool = True):
    from curate.captions import TEMPLATE_MODEL, template_caption
    from curate.manifest import Thresholds, apply_filters, dedup, write_manifest
    from curate.video import analyze

    out = Path(out_dir).resolve()
    legacy_out = out / "legacy"
    inputs = discover_inputs(input_dir)
    cmd = build_command(Path(input_dir), legacy_out, captioner=captioner, clip_s=clip_s, min_s=min_s)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "configs" / "legacy_command.json").write_text(json.dumps(cmd, indent=1))
    (out / "configs" / "weld_caption_prompt.txt").write_text(WELD_CAPTION_PROMPT + "\n")
    if dry_run:
        log.info("legacy command written to %s", out / "configs" / "legacy_command.json")
        return None
    home = os.environ.get("CURATOR_HOME")
    if not home or not Path(home).is_dir():
        raise SystemExit("CURATOR_HOME must point at a cosmos-curator checkout (pixi env installed)")
    log.info("running: %s", " ".join(cmd[:8]) + " ...")
    r = subprocess.run(cmd, cwd=home)
    if r.returncode != 0:
        raise SystemExit(f"video-pipeline split failed with exit code {r.returncode}")
    records = read_back(legacy_out, inputs)
    for rec in records:
        if Path(rec.clip_uri).exists():
            try:
                rec.quality, _ = analyze(rec.clip_uri)
            except Exception as exc:  # noqa: BLE001
                rec.quality = {"error": str(exc)}
        if not rec.caption:
            rec.caption, rec.caption_model = template_caption(rec.context(), rec.quality), TEMPLATE_MODEL
    images = [i for i in inputs if i.is_image]
    if images:   # the legacy video pipeline cannot read photos: lite image path
        from curate.lite import caption_records, process_images
        img_records, _ = process_images(images, out / "clips")
        caption_records(img_records, "auto")
        records += img_records
    th = thresholds or Thresholds()
    apply_filters(records, th)
    if dedup_enabled:
        dedup(records, th.dedup_cosine)
    write_manifest(records, out)
    return records
