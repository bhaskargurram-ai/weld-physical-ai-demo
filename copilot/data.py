"""Data access layer for the copilot.

Codes strictly to the PLAN.md `mosaico_io` contract (list_passes / get_signals / wps_violations) and
the curated manifest contract. If `mosaico_io` is missing or a call fails, it falls back to reading
`DATA_DIR/generated/` directly (the same thing mosaico_io's offline mode does), so the UI never breaks.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("copilot.data")

SIGNALS = ["current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm", "gas_lpm", "heat_input_kj_mm", "interpass_c"]
WPS_WINDOW_S = {"travel_speed_mmps": 1.0, "heat_input_kj_mm": 1.0, "interpass_c": 1.0}
WPS_DEADBAND = 0.01
SIGNAL_INFO = {
    "current_a": {"label": "Current", "unit": "A"},
    "voltage_v": {"label": "Voltage", "unit": "V"},
    "travel_speed_mmps": {"label": "Travel speed", "unit": "mm/s"},
    "wfs_mpm": {"label": "Wire feed", "unit": "m/min"},
    "gas_lpm": {"label": "Shielding gas", "unit": "L/min"},
    "heat_input_kj_mm": {"label": "Heat input", "unit": "kJ/mm"},
    "interpass_c": {"label": "Interpass temp", "unit": "°C"},
}


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception as e:  # noqa: BLE001 - module may be mid-build by another agent
        log.info("optional module %s unavailable: %s", name, e)
        return None


def _mosaico():
    if config.data_source() == "local":
        return None
    return _try_import("mosaico_io")


# ------------------------------------------------------------------ normalisation helpers
def _wps_from_meta(meta: dict) -> dict:
    """Return {param: [lo, hi]} incl. interpass_c, from meta['wps'] or collect.schema.WPS_LIBRARY."""
    w = meta.get("wps") or {}
    if isinstance(w, str):
        try:
            w = json.loads(w)
        except ValueError:
            w = {}
    out: dict[str, list] = {}
    for k in ("current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm", "gas_lpm", "heat_input_kj_mm"):
        v = w.get(k)
        if v is None and (f"{k}_min" in w or f"{k}_lo" in w):
            v = [w.get(f"{k}_min", w.get(f"{k}_lo")), w.get(f"{k}_max", w.get(f"{k}_hi"))]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = [float(v[0]), float(v[1])]
    mi = w.get("max_interpass_c")
    if not out or mi is None:
        try:
            from collect.schema import WPS_LIBRARY
            lib = WPS_LIBRARY.get(meta.get("wps_id", ""))
            if lib:
                for k in ("current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm", "gas_lpm", "heat_input_kj_mm"):
                    out.setdefault(k, list(getattr(lib, k)))
                mi = mi if mi is not None else lib.max_interpass_c
        except Exception:  # noqa: BLE001
            pass
    if mi is not None:
        out["interpass_c"] = [None, float(mi)]
    return out


def _normalize_row(m: dict) -> dict:
    defects = m.get("defects") or []
    if isinstance(defects, str):
        try:
            defects = json.loads(defects)
        except ValueError:
            defects = [{"defect": d.strip()} for d in defects.split(",") if d.strip()]
    defect_types = sorted({(d.get("defect") if isinstance(d, dict) else str(d)) for d in defects} - {None, "none"})
    has_defect = m.get("has_defect")
    if isinstance(has_defect, str):
        has_defect = has_defect.lower() in ("1", "true", "yes")
    if has_defect is None:
        has_defect = bool(defect_types)
    seq = m.get("sequence_name") or m.get("name")
    row = {
        "sequence_name": seq,
        "spool_id": m.get("spool_id") or (seq.split("_")[0] if seq else None),
        "joint_id": m.get("joint_id") or (seq.split("_")[1] if seq and "_" in seq else None),
        "pass_no": int(m.get("pass_no") or 0),
        "pass_type": m.get("pass_type"),
        "process": m.get("process"),
        "wps_id": m.get("wps_id"),
        "position": m.get("position"),
        "material": m.get("material"),
        "cell_id": m.get("cell_id"),
        "duration_s": float(m.get("duration_s") or 0),
        "rt_result": m.get("rt_result") or None,
        "has_defect": bool(has_defect),
        "defect_types": defect_types,
        "defects": [d for d in defects if isinstance(d, dict)],
        "started_at": m.get("started_at") or m.get("start_time_utc"),
    }
    return row


class DataLayer:
    """Thread-safe facade with light caching. One instance per app."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sig_cache: dict[str, pd.DataFrame] = {}
        self._viol_cache: dict[str, list] = {}
        self._rows_cache: tuple[float, list[dict]] | None = None
        self.last_source = "local"

    # -------------------------------------------------------------- paths
    @property
    def gen(self) -> Path:
        return config.generated_dir()

    def pass_dir(self, seq: str) -> Path:
        return self.gen / seq

    def refresh(self) -> None:
        with self._lock:
            self._sig_cache.clear()
            self._viol_cache.clear()
            self._rows_cache = None

    # -------------------------------------------------------------- passes
    def _local_rows(self) -> list[dict]:
        rows = []
        if not self.gen.exists():
            return rows
        for d in sorted(self.gen.iterdir()):
            mp = d / "meta.json"
            if mp.exists():
                try:
                    m = json.loads(mp.read_text())
                    m.setdefault("sequence_name", d.name)
                    rows.append(m)
                except ValueError:
                    continue
        return rows

    def all_passes(self) -> list[dict]:
        with self._lock:
            if self._rows_cache and time.time() - self._rows_cache[0] < 15:
                return self._rows_cache[1]
        raw = None
        mi = _mosaico()
        if mi is not None and hasattr(mi, "list_passes"):
            try:
                raw = list(mi.list_passes())
                self.last_source = "mosaico_io"
            except Exception as e:  # noqa: BLE001
                log.warning("mosaico_io.list_passes failed, falling back to local: %s", e)
        if raw is None:
            raw = self._local_rows()
            self.last_source = "local"
        rows = [_normalize_row(r) for r in raw if (r.get("sequence_name") or r.get("name"))]
        # enrich with local meta where mosaico rows are flattened (keeps defects/timing)
        for r in rows:
            mp = self.pass_dir(r["sequence_name"]) / "meta.json"
            if not r["defects"] and mp.exists() and self.last_source != "local":
                try:
                    r.update({k: v for k, v in _normalize_row(json.loads(mp.read_text())).items() if v})
                except ValueError:
                    pass
            r["has_video"] = (self.pass_dir(r["sequence_name"]) / "torch_cam.mp4").exists()
        rows.sort(key=lambda r: (r["spool_id"] or "", r["joint_id"] or "", r["pass_no"]))
        with self._lock:
            self._rows_cache = (time.time(), rows)
        return rows

    def list_passes(self, process: str | None = None, spool_id: str | None = None, joint_id: str | None = None,
                    pass_type: str | None = None, has_defect: bool | None = None, rt_result: str | None = None,
                    wps_id: str | None = None, with_violations: bool = False, **_: Any) -> list[dict]:
        rows = self.all_passes()

        def ok(r):
            if process and (r["process"] or "").upper() != process.upper():
                return False
            if spool_id and (r["spool_id"] or "").upper() != spool_id.upper():
                return False
            if joint_id and (r["joint_id"] or "").upper() != joint_id.upper():
                return False
            if pass_type and (r["pass_type"] or "").lower() != pass_type.lower():
                return False
            if has_defect is not None and r["has_defect"] != has_defect:
                return False
            if rt_result and (r["rt_result"] or "pending") != rt_result.lower():
                return False
            if wps_id and r["wps_id"] != wps_id:
                return False
            return True

        out = [dict(r) for r in rows if ok(r)]
        if with_violations:
            for r in out:
                v = self.wps_violations(r["sequence_name"])
                r["violation_count"] = len(v)
                r["violated_params"] = sorted({x["param"] for x in v})
        return out

    def get_pass(self, seq: str) -> dict | None:
        for r in self.all_passes():
            if r["sequence_name"] == seq:
                return r
        return None

    def get_meta(self, seq: str) -> dict:
        mp = self.pass_dir(seq) / "meta.json"
        if mp.exists():
            return json.loads(mp.read_text())
        r = self.get_pass(seq)
        return dict(r) if r else {}

    def wps_limits(self, seq: str) -> dict:
        return _wps_from_meta(self.get_meta(seq))

    # -------------------------------------------------------------- signals
    def _local_signals(self, seq: str) -> pd.DataFrame:
        d = self.pass_dir(seq)
        arc = pd.read_parquet(d / "arc.parquet").sort_values("t_ns")
        t0 = int(self.get_meta(seq).get("t0_ns") or arc["t_ns"].iloc[0])
        base = arc.copy()
        for name in ("travel", "thermal"):
            p = d / f"{name}.parquet"
            if p.exists():
                other = pd.read_parquet(p).sort_values("t_ns")
                base = pd.merge_asof(base, other, on="t_ns", direction="nearest")
        if "interpass_c" in base:
            base["interpass_c"] = base["interpass_c"].interpolate(limit_direction="both")
        base.insert(0, "t_s", (base["t_ns"] - t0) / 1e9)
        return base

    def get_signals(self, seq: str, t0_s: float | None = None, t1_s: float | None = None) -> pd.DataFrame:
        with self._lock:
            df = self._sig_cache.get(seq)
        if df is None:
            mi = _mosaico()
            if mi is not None and hasattr(mi, "get_signals"):
                try:
                    df = mi.get_signals(seq)
                    if "t_s" not in df.columns and "t_ns" in df.columns:
                        df.insert(0, "t_s", (df["t_ns"] - df["t_ns"].iloc[0]) / 1e9)
                except Exception as e:  # noqa: BLE001
                    log.warning("mosaico_io.get_signals(%s) failed, local fallback: %s", seq, e)
                    df = None
            if df is None:
                df = self._local_signals(seq)
            if "heat_input_kj_mm" not in df.columns and {"current_a", "voltage_v", "travel_speed_mmps"} <= set(df.columns):
                df["heat_input_kj_mm"] = 0.8 * df["voltage_v"] * df["current_a"] / (1000 * df["travel_speed_mmps"].clip(lower=0.05))
            with self._lock:
                self._sig_cache[seq] = df
        if t0_s is not None:
            df = df[df["t_s"] >= t0_s]
        if t1_s is not None:
            df = df[df["t_s"] <= t1_s]
        return df

    # -------------------------------------------------------------- WPS
    def _local_violations(self, seq: str) -> list[dict]:
        df = self.get_signals(seq)
        lim = self.wps_limits(seq)
        hz = 1.0 / max(np.median(np.diff(df["t_s"].values[:200])), 1e-3)
        # arc-on window from events (skip 1 s start transient and the arc-stop crater fill)
        ev = self.events(seq)
        starts = [e["t_s"] for e in ev if e.get("type") == "arc_start"]
        stops = [e["t_s"] for e in ev if e.get("type") == "arc_stop"]
        t_on = (min(starts) if starts else float(df["t_s"].iloc[0])) + 1.0
        t_off = (max(stops) if stops else float(df["t_s"].iloc[-1])) - 0.6
        out = []
        for param, (lo, hi) in lim.items():
            if param not in df.columns:
                continue
            # monitor value: 0.5 s moving average (1 s for travel/heat input to remove torch weave),
            # 1 % deadband on the limit, ignore the first second after arc start (start transient)
            win = WPS_WINDOW_S.get(param, 0.5)
            x = df[param].rolling(max(int(hz * win), 1), center=True, min_periods=1).mean().values
            bad = np.zeros(len(x), bool)
            if lo is not None:
                bad |= x < lo - abs(lo) * WPS_DEADBAND
            if hi is not None:
                bad |= x > hi + abs(hi) * WPS_DEADBAND
            bad &= (df["t_s"].values >= t_on) & (df["t_s"].values <= t_off)
            # close short gaps (<0.3 s), drop blips (<0.25 s)
            idx = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
            runs = list(zip(idx[::2], idx[1::2]))
            merged: list[list[int]] = []
            for a, b in runs:
                if merged and (a - merged[-1][1]) / hz < 0.3:
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            for a, b in merged:
                if (b - a) / hz < 0.4:
                    continue
                seg = x[a:b]
                worst = float(seg.max() if hi is not None and seg.max() > hi else seg.min())
                out.append({"param": param, "t_start_s": round(float(df["t_s"].iloc[a]), 2),
                            "t_end_s": round(float(df["t_s"].iloc[b - 1]), 2), "value": round(worst, 3),
                            "limit_lo": lo, "limit_hi": hi})
        out.sort(key=lambda v: v["t_start_s"])
        return out

    def wps_violations(self, seq: str) -> list[dict]:
        with self._lock:
            if seq in self._viol_cache:
                return self._viol_cache[seq]
        res = None
        mi = _mosaico()
        if mi is not None and hasattr(mi, "wps_violations"):
            try:
                res = list(mi.wps_violations(seq))
            except Exception as e:  # noqa: BLE001
                log.warning("mosaico_io.wps_violations(%s) failed, local fallback: %s", seq, e)
        if res is None:
            try:
                res = self._local_violations(seq)
            except FileNotFoundError:
                res = []
        with self._lock:
            self._viol_cache[seq] = res
        return res

    # -------------------------------------------------------------- events & media
    def events(self, seq: str) -> list[dict]:
        p = self.pass_dir(seq) / "events.jsonl"
        out = []
        if p.exists():
            raw = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            t0 = None
            arc = self.pass_dir(seq) / "arc.parquet"
            if arc.exists():
                t0 = int(self.get_meta(seq).get("t0_ns") or pd.read_parquet(arc, columns=["t_ns"])["t_ns"].min())
            if t0 is None and raw:
                t0 = min(e.get("t_ns", 0) for e in raw)
            for e in raw:
                t_s = e.get("t_s", (e.get("t_ns", 0) - (t0 or 0)) / 1e9)
                out.append({**e, "t_s": round(float(t_s), 3)})
        else:
            for d in self.get_meta(seq).get("defects", []) or []:
                out.append({"type": "defect_on", "defect": d.get("defect"), "t_s": d.get("t_start"), "severity": d.get("severity")})
                out.append({"type": "defect_off", "defect": d.get("defect"), "t_s": d.get("t_end"), "severity": d.get("severity")})
        return out

    def defect_windows(self, seq: str) -> list[dict]:
        meta = self.get_meta(seq)
        wins = [{"defect": d.get("defect"), "t_start": float(d.get("t_start", 0)), "t_end": float(d.get("t_end", 0)),
                 "severity": d.get("severity")} for d in (meta.get("defects") or []) if isinstance(d, dict)]
        if wins:
            return wins
        on: dict[str, dict] = {}
        for e in self.events(seq):
            if e.get("type") == "defect_on":
                on[e.get("defect")] = e
            elif e.get("type") == "defect_off" and e.get("defect") in on:
                s = on.pop(e["defect"])
                wins.append({"defect": e["defect"], "t_start": s["t_s"], "t_end": e["t_s"], "severity": s.get("severity")})
        return wins

    def media(self, seq: str) -> dict:
        d = self.pass_dir(seq)
        out = {}
        for key, rel in (("torch", "torch_cam.mp4"), ("edge", "control/edge.mp4"), ("depth", "control/depth.mp4")):
            if (d / rel).exists():
                out[key] = file_url(d / rel)
        return out

    # -------------------------------------------------------------- curated
    def manifest(self) -> list[dict]:
        p = config.curated_dir() / "manifest.jsonl"
        if not p.exists():
            return []
        return _read_manifest(str(p), p.stat().st_mtime)

    def dataset_card(self) -> str:
        for p in (config.curated_dir() / "dataset_card.md", config.curated_dir() / "DATASET_CARD.md",
                  config.data_dir() / "dataset_card.md"):
            if p.exists():
                return p.read_text()
        return ""

    def clip_video(self, row: dict) -> str | None:
        uri = row.get("clip_uri") or ""
        for cand in (config.localize(uri), config.curated_dir() / uri, config.curated_dir() / "clips" / f"{row.get('clip_id')}.mp4"):
            try:
                if uri and cand.exists():
                    return file_url(cand)
            except OSError:
                continue
        src = self.pass_dir(row.get("sequence_name", "")) / "torch_cam.mp4"
        if src.exists():
            return f"{file_url(src)}#t={row.get('t0_s', 0)},{row.get('t1_s', '')}"
        return None

    def dataset_stats(self, defect: str | None = None) -> dict:
        rows = self.manifest()
        if defect:
            rows = [r for r in rows if defect in _defect_list(r)]
        kept = [r for r in rows if r.get("kept")]
        by_defect: dict[str, dict] = {}
        for r in rows:
            ds = _defect_list(r) or ["none"]
            for d in ds:
                b = by_defect.setdefault(d, {"total": 0, "kept": 0})
                b["total"] += 1
                b["kept"] += int(bool(r.get("kept")))
        reasons: dict[str, int] = {}
        for r in rows:
            if not r.get("kept"):
                k = r.get("reject_reason") or "unspecified"
                reasons[k] = reasons.get(k, 0) + 1
        return {"total": len(rows), "kept": len(kept), "rejected": len(rows) - len(kept),
                "by_defect": by_defect, "reject_reasons": reasons,
                "sequences": len({r.get("sequence_name") for r in rows}),
                "rt_reject_clips": sum(1 for r in kept if r.get("rt_result") == "reject")}


def _defect_list(r: dict) -> list[str]:
    ds = r.get("defects") or []
    out = []
    for d in ds:
        name = d.get("defect") if isinstance(d, dict) else d
        if name and name != "none":
            out.append(name)
    return out


@lru_cache(maxsize=4)
def _read_manifest(path: str, mtime: float) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def file_url(p: Path | str) -> str:
    """URL under /files/ for a path inside DATA_DIR (served by the app)."""
    p = Path(p).resolve()
    try:
        rel = p.relative_to(config.data_dir())
        return "/files/" + rel.as_posix()
    except ValueError:
        return "/files/abs" + p.as_posix()


# ------------------------------------------------------------------ health
_health_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    hit = _health_cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _health_cache[key] = (now, val)
    return val


def mosaico_health() -> dict:
    if os.environ.get("MOSAICO_OFFLINE", "") in ("1", "true"):
        return {"status": "offline", "detail": "MOSAICO_OFFLINE=1 (reading data/generated)", "short": "offline mode · data/generated"}
    host = os.environ.get("MOSAICO_HOST", "localhost")
    port = int(os.environ.get("MOSAICO_PORT", "6726"))

    def probe():
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return {"status": "ok", "detail": f"mosaicod {host}:{port}", "short": f"mosaicod :{port} connected"}
        except OSError:
            return {"status": "down", "detail": f"mosaicod {host}:{port} unreachable - offline fallback",
                    "short": f"mosaicod :{port} down · offline fallback"}
    return _cached("mosaico", 5, probe)
