"""WeldWorld world-model clients: NVIDIA Cosmos 3 Reasoner / Generator / Transfer.

    from worldmodels import reason, generate, transfer
    reason.assess_clip("clip.mp4", sidecar)        # weld QA verdict dict
    generate.what_if("travel speed drops 20%")     # mp4 path
    transfer.sim2real("control/edge.mp4", "edge", "photoreal FCAW fill pass")

Backend: COSMOS_BACKEND=auto|local|nim|replay (see worldmodels/config.py, README.md).
"""
from . import config

__all__ = ["config", "reason", "generate", "transfer", "anomaly", "cache"]
__version__ = "0.1.0"


def __getattr__(name: str):  # lazy submodule access: worldmodels.reason etc.
    if name in ("reason", "generate", "transfer", "anomaly", "cache", "prompts", "parsing", "heuristics", "video"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
