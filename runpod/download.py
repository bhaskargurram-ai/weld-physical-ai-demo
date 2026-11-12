#!/usr/bin/env python3
"""Download WeldWorld results from the Runpod network volume to this computer — no pod needed.

Uses Runpod's S3-compatible API for network volumes (https://docs.runpod.io/storage/s3-api), so it
works while a run is going and after the pod has terminated.

One-time setup:
  1. Runpod console -> Settings / Credentials -> "S3 API keys" -> Create. Copy the access key (user_...)
     and the secret (rps_...) into .env as RUNPOD_S3_ACCESS_KEY / RUNPOD_S3_SECRET_KEY
  2. pip install boto3

Usage:
  python runpod/download.py                 # results: run summaries, dataset card, manifest, logs (small)
  python runpod/download.py demo            # + everything the local copilot needs (Cosmos results cache,
                                            #   curated clips, simulated passes) -> ./data, then run
                                            #   COPILOT_LLM=scripted uvicorn copilot.app:app --port 7860
  python runpod/download.py all             # the whole data/ folder (generated, export, curated, cache)
  python runpod/download.py ls [PREFIX]     # list what is on the volume
Options: --dest DIR (default: repo root), --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = "weld-physical-ai-demo"          # the pod checks the repo out at /workspace/<this>

# remote prefix on the volume (= path under /workspace)  ->  local path under --dest
PRESETS: dict[str, list[tuple[str, str]]] = {
    "results": [
        ("results/", "runpod_results/"),
        ("logs/", "runpod_results/logs/"),
    ],
    # real data lives in data/real on the pod (DATA_SOURCE=real, the default), the simulator's in data/
    "demo": [
        ("results/", "runpod_results/"),
        ("logs/", "runpod_results/logs/"),
        (f"{REMOTE_REPO}/data/real/cache/", "data/real/cache/"),          # Cosmos 3 verdicts + generated videos
        (f"{REMOTE_REPO}/data/real/curated/", "data/real/curated/"),      # curated clips/photos, captions, manifest
        (f"{REMOTE_REPO}/data/real/generated/", "data/real/generated/"),  # real passes: signals, NDT, photos, video
        (f"{REMOTE_REPO}/data/cache/", "data/cache/"),                    # simulator run (DATA_SOURCE=sim), if any
        (f"{REMOTE_REPO}/data/curated/", "data/curated/"),
        (f"{REMOTE_REPO}/data/generated/", "data/generated/"),
    ],
    "all": [
        ("results/", "runpod_results/"),
        ("logs/", "runpod_results/logs/"),
        (f"{REMOTE_REPO}/data/", "data/"),                                # everything except raw/ downloads
    ],
}
def _s3_dcs() -> set:
    spec = importlib.util.spec_from_file_location("launch", REPO_ROOT / "runpod" / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.S3_DATACENTERS


S3_DATACENTERS = _s3_dcs()      # single source: runpod/launch.py
SKIP_PARTS = {".venv", "__pycache__", ".git", "raw"}   # raw/ = multi-GB source downloads, re-fetchable


def _launch():
    spec = importlib.util.spec_from_file_location("launch", REPO_ROOT / "runpod" / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def endpoint(dc: str) -> str:
    return f"https://s3api-{dc.lower()}.runpod.io/"


def local_path(key: str, remote_prefix: str, local_prefix: str, dest: Path) -> Path:
    return dest / local_prefix / key[len(remote_prefix):]


def wanted(key: str) -> bool:
    return not key.endswith("/") and not (set(key.split("/")) & SKIP_PARTS)


def find_volume(cfg: dict, launch) -> tuple[str, str]:
    """(volume_id, data_center) from .runpod_state.json, else by VOLUME_NAME through the Runpod API."""
    st = launch.load_state()
    if st.get("volume_id") and st.get("data_center"):
        return st["volume_id"], st["data_center"]
    if not cfg.get("RUNPOD_API_KEY"):
        raise SystemExit("No volume recorded in .runpod_state.json and no RUNPOD_API_KEY to look it up.")
    for v in launch.Runpod(cfg["RUNPOD_API_KEY"]).list_volumes():
        if v.get("name") == cfg["VOLUME_NAME"]:
            return v["id"], v["dataCenter"]
    raise SystemExit(f"No network volume named '{cfg['VOLUME_NAME']}' in your Runpod account.")


def s3_client(cfg: dict, dc: str):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise SystemExit("boto3 is needed: pip install boto3") from None
    ak, sk = cfg.get("RUNPOD_S3_ACCESS_KEY"), cfg.get("RUNPOD_S3_SECRET_KEY")
    if not ak or not sk:
        raise SystemExit("Set RUNPOD_S3_ACCESS_KEY (user_...) and RUNPOD_S3_SECRET_KEY (rps_...) in .env — "
                         "Runpod console -> Credentials -> S3 API keys -> Create.")
    return boto3.client("s3", aws_access_key_id=ak, aws_secret_access_key=sk, region_name=dc,
                        endpoint_url=endpoint(dc),
                        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8, "mode": "adaptive"}))


def iter_objects(s3, bucket: str, prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        yield from page.get("Contents") or []


def download(s3, bucket: str, mapping: list[tuple[str, str]], dest: Path, dry_run: bool = False) -> tuple[int, int, int]:
    n_new = n_same = total = 0
    for remote, local in mapping:
        found = 0
        for obj in iter_objects(s3, bucket, remote):
            key = obj["Key"]
            if not wanted(key):
                continue
            found += 1
            out = local_path(key, remote, local, dest)
            if out.exists() and out.stat().st_size == obj["Size"]:
                n_same += 1          # resumable: already have this file
                continue
            total += obj["Size"]
            n_new += 1
            if dry_run:
                print(f"  would get {key}  ({obj['Size'] / 1e6:.1f} MB)")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            s3.download_file(bucket, key, str(tmp))
            tmp.replace(out)
            print(f"  {key}  ->  {out.relative_to(dest)}  ({obj['Size'] / 1e6:.1f} MB)", flush=True)
        if not found:
            print(f"  (nothing under {remote} yet)")
    return n_new, n_same, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="?", default="results", choices=[*PRESETS, "ls"])
    p.add_argument("prefix", nargs="?", default="", help="ls: path on the volume, e.g. weld-physical-ai-demo/data/")
    p.add_argument("--dest", default=str(REPO_ROOT))
    p.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    launch = _launch()
    cfg = launch.load_config(Path(args.env_file))
    vid, dc = find_volume(cfg, launch)
    if dc not in S3_DATACENTERS:
        raise SystemExit(f"Volume {vid} is in {dc}, which has no S3 API. Run `python runpod/launch.py --mode serve` "
                         "and download from the copilot pod instead, or move the volume (--move-volume).")
    s3 = s3_client(cfg, dc)
    print(f"volume {vid} in {dc}  ({endpoint(dc)})")

    if args.what == "ls":
        prefix = args.prefix.lstrip("/")
        seen = set()
        resp = s3.list_objects_v2(Bucket=vid, Prefix=prefix, Delimiter="/")
        for cp in resp.get("CommonPrefixes") or []:
            print(f"  {cp['Prefix']}")
            seen.add(cp["Prefix"])
        for obj in resp.get("Contents") or []:
            if obj["Key"] not in seen:
                print(f"  {obj['Key']}  ({obj['Size'] / 1e6:.1f} MB)")
        return 0

    dest = Path(args.dest)
    n_new, n_same, total = download(s3, vid, PRESETS[args.what], dest, args.dry_run)
    verb = "would download" if args.dry_run else "downloaded"
    print(f"{verb} {n_new} files ({total / 1e9:.2f} GB); {n_same} already up to date  ->  {dest}")
    if args.what in ("demo", "all") and not args.dry_run:
        print("\nRun the copilot locally on the GPU results (no GPU needed, Cosmos outputs come from data/cache):\n"
              "  COSMOS_BACKEND=replay MOSAICO_OFFLINE=1 COPILOT_LLM=scripted uvicorn copilot.app:app --port 7860\n"
              "  (PowerShell: $env:COSMOS_BACKEND='replay'; $env:MOSAICO_OFFLINE='1'; "
              "uvicorn copilot.app:app --port 7860)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
