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

    if not (ROOT / "data" / "generated").exists() and not args.fixtures:
        print("No data/generated yet. Either download the GPU run (python runpod/download.py demo)\n"
              "or start with mock data:  python scripts/run_copilot.py --fixtures")
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
