"""Curation CLI.

    python -m curate.run --input data/export --out data/curated --backend {curator-next,curator-legacy,lite}

Every backend writes <out>/manifest.jsonl and <out>/dataset_card.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from curate.common import CLIP_DEFAULTS, env
from curate.manifest import Thresholds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data = env("DATA_DIR", "data")
    ap = argparse.ArgumentParser(prog="python -m curate.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=f"{data}/export", help="dir of <clip_id>.mp4 + <clip_id>.json")
    ap.add_argument("--out", default=f"{data}/curated")
    ap.add_argument("--backend", choices=("curator-next", "curator-legacy", "lite"), default="lite")
    ap.add_argument("--clip-seconds", type=float, default=CLIP_DEFAULTS["clip_s"])
    ap.add_argument("--stride-seconds", type=float, default=None, help="default = clip seconds")
    ap.add_argument("--min-seconds", type=float, default=CLIP_DEFAULTS["min_s"])
    ap.add_argument("--captioner", default=None,
                    help="lite: auto|reason|template (default auto). curator-next: qwen3_8_27b_fp8|qwen3_8_27b|"
                         "cosmos3_nano|cosmos3_super (default qwen3_8_27b_fp8; cosmos3_* uses the weld prompt "
                         "via multimodal-caption). curator-legacy: cosmos3_nano|cosmos_r2|qwen|... "
                         "(default cosmos3_nano)")
    ap.add_argument("--embed-variant", default="336p", help="curator-next Cosmos-Embed1 variant")
    ap.add_argument("--no-embed", action="store_true", help="lite: skip embeddings and dedup")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--configs-only", action="store_true",
                    help="curator-next/legacy: only write configs/command, do not run Curator")
    ap.add_argument("--skip-upload", action="store_true", help="curator-next: inputs already in S3")
    t = Thresholds()
    ap.add_argument("--min-blur", type=float, default=t.min_blur)
    ap.add_argument("--min-motion", type=float, default=t.min_motion)
    ap.add_argument("--max-glare", type=float, default=t.max_glare_frac)
    ap.add_argument("--min-arc-on", type=float, default=t.min_arc_on_frac,
                    help="reject clips whose arc is visible in fewer than this share of frames (0 = off)")
    ap.add_argument("--dedup-cosine", type=float, default=t.dedup_cosine)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    th = Thresholds(min_duration_s=a.min_seconds, min_blur=a.min_blur, min_motion=a.min_motion,
                    max_glare_frac=a.max_glare, min_arc_on_frac=a.min_arc_on,
                    dedup_cosine=a.dedup_cosine)
    stride = a.stride_seconds or a.clip_seconds
    out = Path(a.out)
    t0 = time.time()
    if a.backend == "lite":
        from curate.lite import LiteOptions, run_lite
        opt = LiteOptions(clip_s=a.clip_seconds, stride_s=stride, min_s=a.min_seconds,
                          captioner=a.captioner or "auto", embeddings=not a.no_embed, dedup=not a.no_dedup,
                          workers=a.workers, thresholds=th)
        run_lite(a.input, out, opt)
    elif a.backend == "curator-next":
        from curate.curator_next import run_curator_next
        res = run_curator_next(a.input, out, captioner=a.captioner or "qwen3_8_27b_fp8",
                               embed_variant=a.embed_variant, clip_s=a.clip_seconds, stride_s=stride,
                               min_s=a.min_seconds, configs_only=a.configs_only, skip_upload=a.skip_upload,
                               thresholds=th, dedup_enabled=not a.no_dedup)
        if res is None:
            print(f"configs written to {out / 'configs'}")
            return 0
    else:
        from curate.curator_legacy import run_curator_legacy
        res = run_curator_legacy(a.input, out, captioner=a.captioner or "cosmos3_nano", clip_s=a.clip_seconds,
                                 min_s=a.min_seconds, dry_run=a.configs_only, thresholds=th,
                                 dedup_enabled=not a.no_dedup)
        if res is None:
            print(f"command written to {out / 'configs' / 'legacy_command.json'}")
            return 0

    from curate.stats import summarize, write_card
    from curate.manifest import read_manifest
    card = write_card(out, a.backend)
    s = summarize(read_manifest(out / "manifest.jsonl"), a.backend)
    print(f"[{a.backend}] {s['clips_kept']}/{s['clips_total']} clips kept "
          f"({s['hours_kept'] * 60:.1f} min), rejected: {s['rejected_by_reason'] or '{}'}; "
          f"captions: {s['caption_models']}; {time.time() - t0:.1f}s")
    print(f"manifest: {out / 'manifest.jsonl'}\ncard: {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
