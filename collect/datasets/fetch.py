"""Download + convert real welding datasets into pass directories (docs/REAL_DATA.md).

    python -m collect.datasets.fetch --sources aimen_saw,bam_tgmaw --data-dir data/real \
        [--limit-coupons PP5,PP6] [--bam-wids 617,630] [--skip-download] [--skip-convert]

Each source is a module collect.datasets.<id> exposing
    download(raw_dir: Path, **opts) -> None
    convert(raw_dir: Path, out_dir: Path, **opts) -> list[str]
imported lazily, so a missing/broken adapter only skips that source.
Raw files go to DATA_DIR/raw/<id>/, pass dirs to DATA_DIR/generated/ (index.jsonl rewritten at the end).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

from collect.datasets.common import write_index

KNOWN = ("aimen_saw", "bam_tgmaw")


def _split(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    v = str(v).strip()
    if v == "" or v.lower() == "all":
        return None
    return [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]


def source_opts(src: str, args) -> dict:
    if src == "aimen_saw":
        return {"coupons": _split(args.limit_coupons)}
    if src == "bam_tgmaw":
        if str(args.bam_wids or "").strip().lower() == "all":
            return {"wids": "all"}            # bam_tgmaw.parse_wids: "all" = all five; None = default 617,630
        wids = _split(args.bam_wids)
        if wids is not None:
            wids = [w.upper().replace("WID", "") for w in wids]
        return {"wids": wids}
    return {}


def run(sources, data_dir: Path, args) -> dict[str, list[str]]:
    data_dir = Path(data_dir)
    out_dir = data_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[str]] = {}
    for src in sources:
        try:
            mod = importlib.import_module(f"collect.datasets.{src}")
        except ImportError as e:
            print(f"[fetch] {src}: adapter not available ({e}) — skipped", flush=True)
            continue
        raw_dir = data_dir / "raw" / src
        opts = source_opts(src, args)
        t0 = time.time()
        try:
            if not args.skip_download:
                print(f"[fetch] {src}: download -> {raw_dir} {opts}", flush=True)
                mod.download(raw_dir, **opts)
            if not args.skip_convert:
                print(f"[fetch] {src}: convert -> {out_dir}", flush=True)
                seqs = mod.convert(raw_dir, out_dir, **opts) or []
                results[src] = list(seqs)
                print(f"[fetch] {src}: {len(seqs)} pass dirs in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep the other sources going
            print(f"[fetch] {src}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if args.strict:
                raise
    if not args.skip_convert:
        idx = write_index(out_dir)
        print(f"[fetch] index -> {idx}", flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=",".join(KNOWN), help=f"comma list of {KNOWN}")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--limit-coupons", "--aimen-coupons", dest="limit_coupons",
                    default=os.environ.get("AIMEN_COUPONS"), help="AIMEN coupons, e.g. PP5,PP6 (default all PP3-PP7)")
    ap.add_argument("--bam-wids", default=os.environ.get("BAM_WIDS", "617,630"),
                    help="BAM workpieces, e.g. 617,630 or 'all'")
    ap.add_argument("--skip-download", action="store_true", help="convert from the existing raw cache only")
    ap.add_argument("--skip-convert", action="store_true", help="download only")
    ap.add_argument("--strict", action="store_true", help="fail on the first source error")
    args = ap.parse_args(argv)
    sources = _split(args.sources) or list(KNOWN)
    unknown = [s for s in sources if s not in KNOWN]
    if unknown:
        print(f"[fetch] warning: unknown source(s) {unknown} — will try to import them anyway", flush=True)
    res = run(sources, Path(args.data_dir), args)
    total = sum(len(v) for v in res.values())
    print(f"[fetch] done: {total} pass dirs " + ", ".join(f"{k}={len(v)}" for k, v in res.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
