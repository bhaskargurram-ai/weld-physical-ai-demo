#!/usr/bin/env python3
"""Start the Weld Copilot locally — works on Windows machines that block pip's .exe shims
(e.g. "An Application Control policy has blocked this file" for uvicorn.exe).

    python scripts/run_copilot.py            # replay mode: Cosmos results from data/cache, no GPU, no Mosaico
    python scripts/run_copilot.py --live     # use whatever COSMOS_BACKEND / MOSAICO_* your .env says
    python scripts/run_copilot.py --fixtures # built-in mock data (no download needed)

Open http://localhost:7860 afterwards.
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# Runpod-only settings that must not leak into the local app
_SKIP = {"RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY", "GITHUB_TOKEN", "CURATE_BACKEND"}


def load_dotenv(path: Path) -> None:
    """Load .env (same parser as runpod/launch.py); variables already set in the shell win."""
    if not path.exists():
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("launch", ROOT / "runpod" / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    for k, v in launch.parse_env_file(path).items():
        if v and k not in _SKIP and not os.environ.get(k):
            os.environ[k] = v


def data_report(data: Path) -> int:
    """Print what the copilot will show; returns the number of weld passes found."""
    gen, cur, cache = data / "generated", data / "curated", data / "cache" / "worldmodels"
    passes = [d for d in gen.iterdir() if (d / "meta.json").exists()] if gen.is_dir() else []
    videos = sum((d / "torch_cam.mp4").exists() for d in passes)
    manifest = cur / "manifest.jsonl"
    clips = sum(1 for _ in manifest.open(encoding="utf-8")) if manifest.exists() else 0
    cached = len(list(cache.rglob("*.json"))) if cache.is_dir() else 0
    print(f"data: {data}")
    print(f"  weld passes   {len(passes):>4}  ({videos} with torch-cam video)   {gen}")
    print(f"  curated clips {clips:>4}   {manifest if manifest.exists() else '(no manifest.jsonl)'}")
    print(f"  Cosmos cache  {cached:>4} entries   {cache if cache.is_dir() else '(none: verdicts fall back to rules)'}")
    if passes and not videos:
        print("  note: no videos downloaded -> the replay shows signal plots only")
    return len(passes)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=int(os.environ.get("COPILOT_PORT", 7860)))
    p.add_argument("--live", action="store_true", help="do not force replay/offline mode")
    p.add_argument("--fixtures", action="store_true", help="serve built-in mock data")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    load_dotenv(ROOT / ".env")
    if not args.live:
        os.environ.setdefault("COSMOS_BACKEND", "replay")     # Cosmos 3 answers from data/cache
        os.environ.setdefault("MOSAICO_OFFLINE", "1")         # read data/generated directly
    if args.fixtures:
        os.environ["COPILOT_FIXTURES"] = "1"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ.setdefault("COPILOT_LLM", "scripted")      # no key: built-in scripted chat

    try:
        import uvicorn
    except ImportError:
        print("Missing packages. Run:  python -m pip install -r requirements.txt")
        return 1

    if not args.fixtures:
        from copilot import config as cconf             # same choice the app makes: data/real when it has passes
        found = data_report(cconf.data_dir())
        if not found:
            print("\nNo weld passes found, so the copilot would be empty. Either:\n"
                  "  python runpod/download.py demo          # download the GPU run's data into ./data/real\n"
                  "  python scripts/run_copilot.py --fixtures  # or start with built-in mock data")
            return 1

    url = f"http://localhost:{args.port}"
    mode = "fixtures" if args.fixtures else ("live" if args.live else "replay (no GPU)")
    print(f"Weld Copilot -> {url}   mode: {mode}   chat: {os.environ.get('COPILOT_LLM') or 'claude'}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    uvicorn.run("copilot.app:app", host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
