"""Shared config, connection handling and offline (local files) readers."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from collect.schema import ARC_HZ

log = logging.getLogger("mosaico_io")

# metadata keys flattened to top-level (queryable with QuerySequence.with_user_metadata)
FLAT_KEYS = ("sequence_name", "spool_id", "joint_id", "pass_no", "pass_type", "wps_id", "process",
             "position", "material", "pipe_od_mm", "wall_mm", "cell_id", "duration_s", "rt_result",
             "joint_rt_result", "has_defect", "defect_types", "max_severity", "t0_ns",
             "start_time_utc", "heat_input_mean_kj_mm", "current_mean_a", "voltage_mean_v",
             "travel_mean_mmps", "interpass_start_c", "interpass_max_c")

SIGNALS = ("current_a", "voltage_v", "wfs_mpm", "gas_lpm", "travel_speed_mmps", "positioner_deg",
           "heat_input_kj_mm", "interpass_c")


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def generated_dir() -> Path:
    return Path(os.environ.get("WELD_GENERATED_DIR", data_dir() / "generated"))


def host_port() -> tuple[str, int]:
    return os.environ.get("MOSAICO_HOST", "localhost"), int(os.environ.get("MOSAICO_PORT", "6726"))


def _env_offline() -> Optional[bool]:
    v = os.environ.get("MOSAICO_OFFLINE")
    if v is None or v == "":
        return None
    return v.strip().lower() in ("1", "true", "yes", "on")


_CLIENT = None
_ONLINE_OK: Optional[bool] = None


def connect(host: Optional[str] = None, port: Optional[int] = None, timeout: int = 5):
    """Return a (cached) ``MosaicoClient`` connected to mosaicod (MOSAICO_HOST/MOSAICO_PORT)."""
    global _CLIENT
    from mosaicolabs import MosaicoClient
    import mosaico_io.ontology  # noqa: F401  (registers WeldArc/WeldTravel/WeldThermal)

    h, p = host_port()
    host, port = host or h, port or p
    if _CLIENT is not None and (host, port) == host_port():
        return _CLIENT
    client = MosaicoClient.connect(host=host, port=port, timeout=timeout)
    if (host, port) == host_port():
        _CLIENT = client
    return client


def is_offline() -> bool:
    """True when reading local files. MOSAICO_OFFLINE=1 forces it; MOSAICO_OFFLINE=0 forces online.
    Unset: try mosaicod once and fall back to offline if it is unreachable."""
    global _ONLINE_OK
    env = _env_offline()
    if env is not None:
        return env
    if _ONLINE_OK is None:
        try:
            connect(timeout=2)
            _ONLINE_OK = True
        except Exception as e:  # pragma: no cover - depends on environment
            log.warning("mosaicod unreachable at %s:%s (%s) -> offline mode", *host_port(), e)
            _ONLINE_OK = False
    return not _ONLINE_OK


def flatten_meta(meta: dict) -> dict:
    """Sequence user-metadata: full meta.json (nested dicts kept) + flat queryable keys."""
    out = dict(meta)
    out["defects"] = [dict(d) for d in meta.get("defects", [])]
    out["n_defects"] = len(out["defects"])
    out.setdefault("has_defect", bool(out["defects"]))
    out.setdefault("defect_types", ",".join(sorted({d["defect"] for d in out["defects"]})) or "none")
    return {k: v for k, v in out.items() if v is not None}


def meta_row(meta: dict) -> dict:
    """Row returned by list_passes (same shape online/offline)."""
    row = {k: meta.get(k) for k in FLAT_KEYS}
    row["defects"] = meta.get("defects", [])
    row["wps"] = meta.get("wps", {})
    row["setpoints"] = meta.get("setpoints", {})
    return row


# ------------------------------------------------------------------ offline readers


def pass_dir(seq: str) -> Path:
    d = generated_dir() / seq
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"pass directory not found: {d}")
    return d


@lru_cache(maxsize=256)
def _load_meta_cached(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def load_meta(seq_or_dir) -> dict:
    d = Path(seq_or_dir)
    if not (d / "meta.json").exists():
        d = pass_dir(str(seq_or_dir))
    p = d / "meta.json"
    return dict(_load_meta_cached(str(p), p.stat().st_mtime))


def local_sequences() -> list[str]:
    root = generated_dir()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "meta.json").exists())


def movavg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) == 0:
        return np.asarray(x, dtype=float)
    k = np.ones(n) / n
    pad = np.pad(np.asarray(x, dtype=float), (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def movmedian(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(n, center=True, min_periods=1).median().to_numpy()


def assemble_signals(arc: pd.DataFrame, travel: pd.DataFrame, thermal: pd.DataFrame, t0_ns: int,
                     t0_s: Optional[float] = None, t1_s: Optional[float] = None) -> pd.DataFrame:
    """Merge streams onto the ARC_HZ grid: columns t_s + SIGNALS."""
    df = arc.merge(travel, on="t_ns", how="outer").sort_values("t_ns").reset_index(drop=True)
    df[list(df.columns.drop("t_ns"))] = df[list(df.columns.drop("t_ns"))].interpolate(limit_direction="both")
    if len(thermal):
        df["interpass_c"] = np.interp(df["t_ns"].to_numpy(dtype=float), thermal["t_ns"].to_numpy(dtype=float),
                                      thermal["interpass_c"].to_numpy(dtype=float))
    else:
        df["interpass_c"] = np.nan
    df.insert(0, "t_s", (df["t_ns"].to_numpy() - t0_ns) / 1e9)
    df = df.drop(columns=["t_ns"])
    if t0_s is not None:
        df = df[df["t_s"] >= t0_s - 1e-9]
    if t1_s is not None:
        df = df[df["t_s"] < t1_s - 1e-9]
    cols = ["t_s"] + [c for c in SIGNALS if c in df.columns]
    return df[cols].reset_index(drop=True)


def offline_signals(seq: str, t0_s=None, t1_s=None) -> pd.DataFrame:
    d = pass_dir(seq)
    meta = load_meta(d)
    arc = pd.read_parquet(d / "arc.parquet")
    trv = pd.read_parquet(d / "travel.parquet")
    thm = pd.read_parquet(d / "thermal.parquet")
    t0_ns = int(meta.get("t0_ns", arc["t_ns"].iloc[0]))
    return assemble_signals(arc, trv, thm, t0_ns, t0_s, t1_s)


def avg_signals(df: pd.DataFrame) -> pd.DataFrame:
    """0.5 s moving averages (the monitor values used for WPS compliance)."""
    out = df.copy()
    n = ARC_HZ // 2
    for c in ("current_a", "voltage_v", "wfs_mpm", "gas_lpm"):
        if c in out:
            out[c] = movavg(out[c].to_numpy(), n)
    if "interpass_c" in out:
        out["interpass_c"] = movmedian(out["interpass_c"].to_numpy(), ARC_HZ)
    return out


def json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
