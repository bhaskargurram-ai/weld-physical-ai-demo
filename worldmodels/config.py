"""Runtime configuration for the Cosmos 3 world-model clients.

Everything is read from the environment at call time (``get_settings()``) so a
long-running copilot process picks up a changed ``COSMOS_BACKEND`` after a
restart of the GPU servers without re-importing anything.

Backends
--------
``local``   self-hosted servers on the RunPod pod: vLLM (+ Cosmos3 Reasoner) on
            ``COSMOS_REASONER_URL`` and vLLM-Omni (Cosmos3 Generator) on
            ``COSMOS_GENERATOR_URL``.
``nim``     NVIDIA NIM API: hosted build.nvidia.com endpoints by default (needs
            ``NVIDIA_API_KEY``) or a self-hosted NIM container if the URLs are
            overridden.
``replay``  no GPU: answers come from the on-disk cache
            (``data/cache/worldmodels``); misses return clearly labelled
            heuristic / placeholder results.
``auto``    (default) probe the local servers, else ``nim`` if an API key is
            set, else ``replay``.  Resolved separately for the reasoner and the
            generator, because one may be up while the other is not.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

Backend = Literal["local", "nim", "replay"]
VALID_BACKENDS = ("auto", "local", "nim", "replay")

# Hosted endpoints (verified 2026-11-11 from the NGC endpoint specs
# api.ngc.nvidia.com/v2/endpoints/qc69jvmznzxy/<name>/spec):
#   cosmos3-nano-reasoner -> servers[0].url https://integrate.api.nvidia.com/v1,
#                            model "nvidia/cosmos3-nano-reasoner", /v1/chat/completions
#   cosmos3-nano          -> POST https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano
#                            (NIM /v1/infer request schema, response {"b64_video": ...})
# NOTE: public forum reports (forums.developer.nvidia.com/t/372044, Jun-Nov 2026)
# say the hosted *generator* route returns a plain-text 404 for most accounts.
# The reasoner route is the standard integrate.api.nvidia.com gateway.
NIM_REASONER_URL = "https://integrate.api.nvidia.com/v1"
NIM_REASONER_MODEL = "nvidia/cosmos3-nano-reasoner"
NIM_GENERATOR_URL = "https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano"


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    d = Path(_env("DATA_DIR", "data"))
    if not d.is_absolute():
        # relative DATA_DIR is resolved against the CWD (like the other packages)
        d = Path.cwd() / d
    return d


def api_root(url: str) -> str:
    """Normalise a base URL to its ``/v1`` API root (vLLM / vLLM-Omni style)."""
    u = url.rstrip("/")
    return u if u.endswith("/v1") else f"{u}/v1"


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local")


@dataclass
class Settings:
    backend: str = "auto"                      # auto|local|nim|replay (both services)
    reasoner_backend: str = "auto"             # per-service override
    generator_backend: str = "auto"
    reasoner_url: str = "http://localhost:8000/v1"
    reasoner_model: str = "nvidia/Cosmos3-Nano"
    generator_url: str = "http://localhost:8010"
    generator_model: str = ""                  # optional "model" form field for vLLM-Omni
    nvidia_api_key: str = ""
    nim_reasoner_url: str = NIM_REASONER_URL
    nim_reasoner_model: str = NIM_REASONER_MODEL
    nim_generator_url: str = NIM_GENERATOR_URL
    cache_dir: Path = field(default_factory=lambda: data_dir() / "cache" / "worldmodels")
    cache_mode: str = "readwrite"              # readwrite | write | read | off
    fallback: bool = True                      # on backend error fall back to cache/heuristic
    reasoner_timeout_s: float = 600.0
    generator_timeout_s: float = 3600.0        # matches VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600
    reasoner_retries: int = 2                  # extra attempts after the first
    reasoner_video_mode: str = "auto"          # auto | file | base64
    reasoner_fps: float = 4.0                  # Cosmos3 reasoner docs sample video at 4 fps
    reasoner_max_tokens: int = 4096
    reasoner_seed: int = 0
    reasoner_think: bool = True                # ask for <think> reasoning trace
    include_labels_in_prompt: bool = False     # never leak RT/defect ground truth by default
    gen_preset: str = "live"                   # live (832x480) | showcase (1280x720)
    guardrails: bool = False                   # gated nvidia/Cosmos-1.0-Guardrail -> off for demo
    generator_shared_fs: bool = True           # generator can read our files (same pod)
    transfer_backend: str = "auto"             # auto | omni | framework
    framework_dir: str = ""                    # cosmos-framework checkout with .venv
    framework_checkpoint: str = "Cosmos3-Nano"
    anomalygen_dir: str = ""                   # cdf-anomalygen checkout with .venv
    anomalygen_checkpoint: str = ""            # .../model/iter_000014000.pt
    anomalygen_recipe: str = ""                # .../exp_texture_ft_magnetic_tile.yaml
    transcode_h264: bool = True                # NIM returns VP9-in-MP4; browsers prefer H.264
    probe_timeout_s: float = 1.5

    @property
    def reasoner_api(self) -> str:
        return api_root(self.reasoner_url)

    @property
    def generator_api(self) -> str:
        return api_root(self.generator_url)


def get_settings() -> Settings:
    backend = _env("COSMOS_BACKEND", "auto").lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"COSMOS_BACKEND must be one of {VALID_BACKENDS}, got {backend!r}")
    cache = _env("COSMOS_CACHE_DIR", "")
    return Settings(
        backend=backend,
        reasoner_backend=_env("COSMOS_REASONER_BACKEND", backend).lower(),
        generator_backend=_env("COSMOS_GENERATOR_BACKEND", backend).lower(),
        reasoner_url=_env("COSMOS_REASONER_URL", "http://localhost:8000/v1"),
        reasoner_model=_env("COSMOS_REASONER_MODEL", "nvidia/Cosmos3-Nano"),
        generator_url=_env("COSMOS_GENERATOR_URL", "http://localhost:8010"),
        generator_model=_env("COSMOS_GENERATOR_MODEL", ""),
        nvidia_api_key=_env("NVIDIA_API_KEY", _env("NGC_API_KEY", "")),
        nim_reasoner_url=_env("COSMOS_NIM_REASONER_URL", NIM_REASONER_URL),
        nim_reasoner_model=_env("COSMOS_NIM_REASONER_MODEL", NIM_REASONER_MODEL),
        nim_generator_url=_env("COSMOS_NIM_GENERATOR_URL", NIM_GENERATOR_URL),
        cache_dir=Path(cache) if cache else data_dir() / "cache" / "worldmodels",
        cache_mode=_env("COSMOS_CACHE", "readwrite").lower(),
        fallback=_env_bool("COSMOS_FALLBACK", True),
        reasoner_timeout_s=_env_float("COSMOS_REASONER_TIMEOUT", 600.0),
        generator_timeout_s=_env_float("COSMOS_GENERATOR_TIMEOUT", 3600.0),
        reasoner_retries=_env_int("COSMOS_REASONER_RETRIES", 2),
        reasoner_video_mode=_env("COSMOS_REASONER_VIDEO_MODE", "auto").lower(),
        reasoner_fps=_env_float("COSMOS_REASONER_FPS", 4.0),
        reasoner_max_tokens=_env_int("COSMOS_REASONER_MAX_TOKENS", 4096),
        reasoner_seed=_env_int("COSMOS_REASONER_SEED", 0),
        reasoner_think=_env_bool("COSMOS_REASONER_THINK", True),
        include_labels_in_prompt=_env_bool("COSMOS_PROMPT_INCLUDE_LABELS", False),
        gen_preset=_env("COSMOS_GEN_PRESET", "live").lower(),
        guardrails=_env_bool("COSMOS_GUARDRAILS", False),
        generator_shared_fs=_env_bool("COSMOS_GENERATOR_SHARED_FS", True),
        transfer_backend=_env("COSMOS_TRANSFER_BACKEND", "auto").lower(),
        framework_dir=_env("COSMOS_FRAMEWORK_DIR", ""),
        framework_checkpoint=_env("COSMOS_FRAMEWORK_CHECKPOINT", "Cosmos3-Nano"),
        anomalygen_dir=_env("COSMOS_ANOMALYGEN_DIR", ""),
        anomalygen_checkpoint=_env("COSMOS_ANOMALYGEN_CHECKPOINT", ""),
        anomalygen_recipe=_env("COSMOS_ANOMALYGEN_RECIPE", ""),
        transcode_h264=_env_bool("COSMOS_TRANSCODE_H264", True),
        probe_timeout_s=_env_float("COSMOS_PROBE_TIMEOUT", 1.5),
    )


# --------------------------------------------------------------------- probing
_PROBE_CACHE: dict[str, tuple[float, bool]] = {}
_PROBE_TTL_S = 30.0


def probe(url: str, timeout: float = 1.5, headers: Optional[dict] = None) -> bool:
    """True if ``GET <url>`` answers 2xx within ``timeout`` (memoised 30 s)."""
    now = time.monotonic()
    hit = _PROBE_CACHE.get(url)
    if hit and now - hit[0] < _PROBE_TTL_S:
        return hit[1]
    ok = False
    try:
        from .http import client

        with client(timeout, headers) as c:
            ok = c.get(url).status_code < 300
    except Exception:
        ok = False
    _PROBE_CACHE[url] = (now, ok)
    return ok


def clear_probe_cache() -> None:
    _PROBE_CACHE.clear()


def health_url(service: Literal["reasoner", "generator"], settings: Optional[Settings] = None) -> str:
    """URL that answers 2xx only once the server can actually serve.

    vLLM-Omni (generator) answers ``/v1/models`` with 200 and an empty list while the model is still
    loading, so it must be probed on ``/health`` (503 until the diffusion engine is up). The reasoner
    (plain vLLM) only opens its port once loaded, so ``/v1/models`` is fine there.
    """
    s = settings or get_settings()
    if service == "reasoner":
        return f"{s.reasoner_api}/models"
    root = s.generator_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    return f"{root}/health"


def resolve_backend(service: Literal["reasoner", "generator"], settings: Optional[Settings] = None) -> Backend:
    """Turn ``auto`` into a concrete backend for one service."""
    s = settings or get_settings()
    choice = s.reasoner_backend if service == "reasoner" else s.generator_backend
    if choice not in VALID_BACKENDS:
        raise ValueError(f"invalid backend {choice!r} for {service}")
    if choice != "auto":
        return choice  # type: ignore[return-value]
    if probe(health_url(service, s), timeout=s.probe_timeout_s):
        return "local"
    if s.nvidia_api_key:
        return "nim"
    return "replay"


SOURCE_LOCAL = "cosmos3-local"
SOURCE_NIM = "nim"
SOURCE_CACHE = "cache"
SOURCE_HEURISTIC = "heuristic-fallback"   # also used for placeholder videos ("placeholder": true)


def backend_source(backend: str) -> str:
    return {"local": SOURCE_LOCAL, "nim": SOURCE_NIM}.get(backend, SOURCE_CACHE)
