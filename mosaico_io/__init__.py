"""WeldWorld <-> Mosaico data layer.

    from mosaico_io import connect, ingest_pass_dir, list_passes, get_signals, wps_violations, export_clips

Online (default): talks to mosaicod at MOSAICO_HOST:MOSAICO_PORT (localhost:6726).
Offline (MOSAICO_OFFLINE=1, or automatically when mosaicod is unreachable): reads
data/generated/ directly with identical return shapes.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "connect": "mosaico_io.common", "is_offline": "mosaico_io.common",
    "ingest_pass_dir": "mosaico_io.ingest",
    "list_passes": "mosaico_io.query", "get_signals": "mosaico_io.query", "get_meta": "mosaico_io.query",
    "wps_violations": "mosaico_io.query", "find_anomalies": "mosaico_io.query",
    "export_clips": "mosaico_io.export",
}
__all__ = list(_EXPORTS)


def __getattr__(name):  # lazy, so `python -m mosaico_io.<cli>` does not pre-import the module
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
