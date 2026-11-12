"""Runtime configuration for the copilot. Everything is read lazily from env so tests can override."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
FIXTURE_DIR = REPO_ROOT / "data" / "fixtures"


def fixtures_enabled() -> bool:
    return os.environ.get("COPILOT_FIXTURES", "").lower() in ("1", "true", "yes")


def data_dir() -> Path:
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"]).resolve()
    if fixtures_enabled():
        return FIXTURE_DIR
    return (REPO_ROOT / "data").resolve()


def localize(p: str | Path) -> Path:
    """Map a path written on another machine (e.g. /workspace/weld-physical-ai-demo/data/curated/x.mp4 on the
    Runpod pod, downloaded with runpod/download.py) onto this machine's data dir. Unchanged if it exists."""
    path = Path(str(p))
    try:
        if path.exists():
            return path
    except OSError:
        pass
    parts = str(p).replace("\\", "/").split("/")
    if "data" in parts:
        i = len(parts) - 1 - parts[::-1].index("data")       # last ".../data/" segment
        cand = data_dir().joinpath(*[x for x in parts[i + 1:] if x])
        if cand.exists():
            return cand
    return path


def generated_dir() -> Path:
    return data_dir() / "generated"


def curated_dir() -> Path:
    return data_dir() / "curated"


def state_dir() -> Path:
    """Copilot-owned scratch: generated videos, gallery, cut clips."""
    p = data_dir() / "copilot"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_source() -> str:
    """auto: use mosaico_io if importable (falls back per call); local: read data/generated directly."""
    v = os.environ.get("COPILOT_DATA", "").lower()
    if v in ("local", "mosaico_io", "auto"):
        return v
    return "local" if fixtures_enabled() else "auto"


def llm_backend() -> str:
    v = os.environ.get("COPILOT_LLM", "").lower()
    if v in ("claude", "cosmos", "scripted"):
        return v
    # Sensible default: Claude if a key is configured, otherwise the zero-key scripted router.
    return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "scripted"


def cosmos_backend() -> str:
    return os.environ.get("COSMOS_BACKEND", "replay").lower()


CLAUDE_DEFAULT_MODEL = "claude-opus-5-5"
