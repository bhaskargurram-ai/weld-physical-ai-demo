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
from .provenance import public as public_source
from .provenance import source_of

log = logging.getLogger("copilot.data")

# Lead electrode (or the only wire) first; `*2_*` = trail electrode of a tandem process (optional).
SIGNALS = ["current_a", "current2_a", "voltage_v", "voltage2_v", "travel_speed_mmps", "wfs_mpm", "wfs2_mpm",
           "gas_lpm", "heat_input_kj_mm", "interpass_c"]
TRAIL_OF = {"current_a": "current2_a", "voltage_v": "voltage2_v", "wfs_mpm": "wfs2_mpm"}
WPS_WINDOW_S = {"travel_speed_mmps": 1.0, "heat_input_kj_mm": 1.0, "interpass_c": 1.0}
WPS_DEADBAND = 0.01
WPS_KEYS = ("current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm", "gas_lpm", "heat_input_kj_mm",
            "current2_a", "voltage2_v", "wfs2_mpm")
SIGNAL_INFO = {
    "current_a": {"label": "Current", "unit": "A"},
    "current2_a": {"label": "Current (trail)", "unit": "A"},
    "voltage_v": {"label": "Voltage", "unit": "V"},
    "voltage2_v": {"label": "Voltage (trail)", "unit": "V"},
    "travel_speed_mmps": {"label": "Travel speed", "unit": "mm/s"},
    "wfs_mpm": {"label": "Wire feed", "unit": "m/min"},
    "wfs2_mpm": {"label": "Wire feed (trail)", "unit": "m/min"},
    "gas_lpm": {"label": "Shielding gas", "unit": "L/min"},
    "heat_input_kj_mm": {"label": "Heat input", "unit": "kJ/mm"},
    "interpass_c": {"label": "Interpass temp", "unit": "°C"},
}
GOOD_RT = {"accept", "no_indications"}
BAD_RT = {"reject", "indications"}


def _finite(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def clean_json(v):
    """NaN/inf -> None, numpy scalars -> python (Starlette's JSONResponse refuses NaN)."""
    if isinstance(v, dict):
        return {k: clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


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
    for k in WPS_KEYS:
        v = w.get(k)
        if v is None and (f"{k}_min" in w or f"{k}_lo" in w):
            v = [w.get(f"{k}_min", w.get(f"{k}_lo")), w.get(f"{k}_max", w.get(f"{k}_hi"))]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = [float(v[0]), float(v[1])]
    mi = w.get("max_interpass_c")
    if (not out or mi is None) and source_of(meta).get("synthetic"):
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
    src = source_of(m)
    ndt = m.get("ndt") if isinstance(m.get("ndt"), dict) else {}
    counts: dict[str, int] = {}
    for d in row["defects"]:
        if d.get("defect") and d.get("defect") != "none":
            counts[d["defect"]] = counts.get(d["defect"], 0) + 1
    row.update({
        "source_id": src.get("id"), "source_short": src.get("short"), "synthetic": bool(src.get("synthetic")),
        "license": src.get("license"), "doi": src.get("doi"),
        "pass_label": _pass_label(src, row["pass_no"]),
        "electrodes": int(m.get("electrodes") or 1),
        "ndt_inspected": bool(ndt.get("inspected")) if ndt else row["rt_result"] in ("accept", "reject", "indications", "no_indications"),
        "ndt_methods": list(ndt.get("methods") or ([] if ndt else (["RT"] if row["rt_result"] in ("accept", "reject") else []))),
        "defect_counts": counts,
        "thickness_mm": m.get("thickness_mm") or m.get("wall_mm"),
        "n_photos": len(m.get("photos") or []),
    })
    if not row["defects"] and ndt.get("indications"):
        row["defect_counts"] = {}
        for i in ndt["indications"]:
            row["defect_counts"][i.get("type")] = row["defect_counts"].get(i.get("type"), 0) + 1
    return row


def _fix_ac_channels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """AC electrodes (AIMEN's trail wire) swing positive and negative: their 10 ms means are noise around zero.
    Use the `<name>_rms_<unit>` column when the conversion provides one, else blank the channel (never plot or
    check a meaningless mean)."""
    ac: dict[str, str] = {}
    for p in ("current_a", "voltage_v", "current2_a", "voltage2_v"):
        if p not in df.columns or not df[p].notna().any():
            continue
        x = df[p].values.astype(float)
        on = np.abs(x) > (5.0 if p.startswith("current") else 2.0)
        if on.sum() < 20 or (x[on] < 0).mean() < 0.1:
            continue
        rms = p.rsplit("_", 1)[0] + "_rms_" + p.rsplit("_", 1)[1]
        df = df.copy()
        if rms in df.columns and df[rms].notna().any():
            df[p] = df[rms]
            ac[p] = "rms"
        else:
            df[p] = np.nan
            ac[p] = "no_rms"
    return df, ac


def _pass_label(src: dict, n: int) -> str:
    w = src.get("pass_word") or "P"
    return f"{w} {n}" if len(w) > 1 else f"{w}{n}"


class DataLayer:
    """Thread-safe facade with light caching. One instance per app."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sig_cache: dict[str, pd.DataFrame] = {}
        self._viol_cache: dict[str, list] = {}
        self._ac_cache: dict[str, dict] = {}
        self._rows_cache: tuple[float, list[dict]] | None = None
        self.last_source = "local"

    # -------------------------------------------------------------- paths
    @property
    def gen(self) -> Path:
        return config.generated_dir()

    def pass_dir(self, seq: str) -> Path:
        for g in config.generated_dirs():
            if (g / seq).is_dir():
                return g / seq
        return self.gen / seq

    def refresh(self) -> None:
        with self._lock:
            self._sig_cache.clear()
            self._viol_cache.clear()
            self._ac_cache.clear()
            self._rows_cache = None

    # -------------------------------------------------------------- passes
    def _local_rows(self) -> list[dict]:
        rows, seen = [], set()
        for g in config.generated_dirs():
            if not g.exists():
                continue
            for d in sorted(g.iterdir()):
                mp = d / "meta.json"
                if d.name in seen or not mp.exists():
                    continue
                try:
                    m = json.loads(mp.read_text(encoding="utf-8"))
                    m.setdefault("sequence_name", d.name)
                    rows.append(m)
                    seen.add(d.name)
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
        if self.last_source != "local":
            # mosaico rows are flattened: the pass dir's meta.json is the source of truth for provenance/NDT/media
            merged = []
            for r in raw:
                seq = r.get("sequence_name") or r.get("name")
                mp = self.pass_dir(seq) / "meta.json" if seq else None
                if mp is not None and mp.exists():
                    try:
                        r = {**r, **json.loads(mp.read_text(encoding="utf-8"))}
                    except ValueError:
                        pass
                merged.append(r)
            raw = merged
        rows = [_normalize_row(r) for r in raw if (r.get("sequence_name") or r.get("name"))]
        for r in rows:
            d = self.pass_dir(r["sequence_name"])
            r["has_video"] = (d / "torch_cam.mp4").exists()
            r["has_photo"] = bool(r.get("n_photos")) and any((d / p).exists() for p in self._photo_rel(r["sequence_name"]))
        rows.sort(key=lambda r: (r["synthetic"], r["source_id"] or "", r["spool_id"] or "", r["joint_id"] or "",
                                 r["pass_no"]))
        with self._lock:
            self._rows_cache = (time.time(), rows)
        return rows

    def list_passes(self, process: str | None = None, spool_id: str | None = None, joint_id: str | None = None,
                    pass_type: str | None = None, has_defect: bool | None = None, rt_result: str | None = None,
                    wps_id: str | None = None, with_violations: bool = False, source: str | None = None,
                    defect: str | None = None, **_: Any) -> list[dict]:
        rows = self.all_passes()

        def norm(x):
            return "".join(ch for ch in str(x or "").upper() if ch.isalnum()).replace("TANDEM", "T")

        def ok(r):
            if process and norm(r["process"]) != norm(process):
                return False
            if source and source.lower() not in ((r["source_id"] or "").lower(), "simulated" if r["synthetic"] else "-"):
                if not (source.lower() in ("sim", "synthetic") and r["synthetic"]):
                    return False
            if defect and defect not in (r.get("defect_counts") or {}) and defect not in r["defect_types"]:
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
                rr = rt_result.lower()
                # "reject"/"accept" also match the real-data equivalents
                if not ((rr == "reject" and r["rt_result"] == "indications")
                        or (rr == "accept" and r["rt_result"] == "no_indications")):
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

    def data_mode(self) -> dict:
        """What the loaded passes are: real datasets, simulator, or both (drives wording across the app)."""
        rows = self.all_passes()
        real = [r for r in rows if not r.get("synthetic")]
        ids = list(dict.fromkeys(r["source_id"] for r in real))
        return {"real": bool(real), "simulated": sum(1 for r in rows if r.get("synthetic")), "real_passes": len(real),
                "sources": ids, "short": [next(r["source_short"] for r in real if r["source_id"] == i) for i in ids],
                "choice": config.dataset_choice(), "fixture": config.fixtures_mode() == "real"}

    def sources_summary(self) -> dict:
        """Every dataset behind the loaded passes, with full CC BY attribution and what we use from it."""
        from .provenance import DATASETS, SIMULATOR
        rows = self.all_passes()
        out = []
        ids = list(dict.fromkeys([r["source_id"] for r in rows] + list(DATASETS)))
        for sid in ids:
            rs = [r for r in rows if r["source_id"] == sid]
            meta = self.get_meta(rs[0]["sequence_name"]) if rs else {"source": {"id": sid, **DATASETS.get(sid, SIMULATOR)}}
            src = public_source(source_of(meta))
            src.update({"loaded": bool(rs), "passes": len(rs), "coupons": sorted({r["spool_id"] for r in rs if r["spool_id"]}),
                        "inspected": sum(1 for r in rs if r.get("ndt_inspected")),
                        "with_indications": sum(1 for r in rs if r["rt_result"] == "indications"),
                        "indications": sum(sum((r.get("defect_counts") or {}).values()) for r in rs if not r.get("synthetic")),
                        "videos": sum(1 for r in rs if r.get("has_video")), "photos": sum(r.get("n_photos") or 0 for r in rs),
                        "fixture": any(self.get_meta(r["sequence_name"]).get("fixture") for r in rs[:3])})
            if src["synthetic"] and not rs:
                continue
            out.append(src)
        return {"sources": out, "mode": self.data_mode()}

    def get_pass(self, seq: str) -> dict | None:
        for r in self.all_passes():
            if r["sequence_name"] == seq:
                return r
        return None

    def get_meta(self, seq: str) -> dict:
        mp = self.pass_dir(seq) / "meta.json"
        if mp.exists():
            return json.loads(mp.read_text(encoding="utf-8"))
        r = self.get_pass(seq)
        return dict(r) if r else {}

    def source(self, seq: str) -> dict:
        return public_source(source_of(self.get_meta(seq)))

    def _photo_rel(self, seq: str) -> list[str]:
        try:
            return [str(p) for p in (self.get_meta(seq).get("photos") or []) if isinstance(p, str)]
        except (OSError, ValueError):
            return []

    def photos(self, seq: str) -> list[dict]:
        d = self.pass_dir(seq)
        out = []
        for rel in self._photo_rel(seq):
            p = (d / rel).resolve()
            if p.is_file() and p.is_relative_to(d.resolve()):
                out.append({"name": p.name, "path": str(p), "url": file_url(p)})
        return out

    def ndt(self, seq: str) -> dict:
        """NDT block for a pass: meta.ndt (real data) or synthesised from the RT result (simulator)."""
        meta = self.get_meta(seq)
        n = meta.get("ndt") if isinstance(meta.get("ndt"), dict) else None
        wins = self.defect_windows(seq)
        if n is None:
            rt = meta.get("rt_result")
            return {"methods": ["RT"] if rt in ("accept", "reject") else [], "inspected": rt in ("accept", "reject"),
                    "report_files": [], "simulated": True,
                    "indications": [{"type": w["defect"], "t_start": w["t_start"], "t_end": w["t_end"]} for w in wins]}
        n = dict(n)
        n.setdefault("indications", [{"type": w["defect"], "t_start": w["t_start"], "t_end": w["t_end"]} for w in wins])
        return n

    def wps_limits(self, seq: str) -> dict:
        return _wps_from_meta(self.get_meta(seq))

    # -------------------------------------------------------------- signals
    def _local_signals(self, seq: str) -> pd.DataFrame:
        d = self.pass_dir(seq)
        arc = pd.read_parquet(d / "arc.parquet").sort_values("t_ns")
        t0 = int(self.get_meta(seq).get("t0_ns") or arc["t_ns"].iloc[0])
        base = arc.copy()
        base["t_ns"] = base["t_ns"].astype("int64")
        for name in ("travel", "thermal"):        # optional streams (real datasets may not have them)
            p = d / f"{name}.parquet"
            if p.exists():
                try:
                    other = pd.read_parquet(p).sort_values("t_ns")
                    other["t_ns"] = other["t_ns"].astype("int64")
                    other = other[[c for c in other.columns if c == "t_ns" or c not in base.columns]]
                    base = pd.merge_asof(base, other, on="t_ns", direction="nearest")
                except Exception as e:  # noqa: BLE001 - a broken optional stream must not hide the arc data
                    log.warning("%s: skipping %s.parquet (%s)", seq, name, e)
        for c in ("interpass_c", "temp_left_c", "temp_right_c"):
            if c in base:
                base[c] = base[c].interpolate(limit_direction="both")
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
            df, ac = _fix_ac_channels(df)
            with self._lock:
                self._ac_cache[seq] = ac
            if "heat_input_kj_mm" not in df.columns and {"current_a", "voltage_v", "travel_speed_mmps"} <= set(df.columns):
                meta = self.get_meta(seq)
                eff = 1.0 if str(meta.get("process", "")).upper() == "SAW" else 0.8
                power = df["voltage_v"] * df["current_a"]
                if {"current2_a", "voltage2_v"} <= set(df.columns):
                    power = power + (df["voltage2_v"] * df["current2_a"]).fillna(0)
                df["heat_input_kj_mm"] = eff * power / (1000 * df["travel_speed_mmps"].clip(lower=0.05))
            with self._lock:
                self._sig_cache[seq] = df
        if t0_s is not None:
            df = df[df["t_s"] >= t0_s]
        if t1_s is not None:
            df = df[df["t_s"] <= t1_s]
        return df

    def _transients(self, seq: str) -> tuple[float, float]:
        """Seconds to skip after arc start / before arc stop. Simulator: 1 s / 0.6 s. Real passes: 3 s / 1.5 s -
        the recorded SAW and tandem-GMAW arc-start ramps and crater fills last that long."""
        return (1.0, 0.6) if source_of(self.get_meta(seq)).get("synthetic") else (3.0, 1.5)

    def steady_mask(self, seq: str, df: pd.DataFrame) -> np.ndarray:
        """Arc-on samples without the arc-start transient and the arc-stop crater fill (see _transients)."""
        ev = self.events(seq)
        t = df["t_s"].values
        starts = [e["t_s"] for e in ev if e.get("type") == "arc_start"]
        stops = [e["t_s"] for e in ev if e.get("type") == "arc_stop"]
        skip_on, skip_off = self._transients(seq)
        a = (min(starts) if starts else (t[0] if len(t) else 0)) + skip_on
        b = (max(stops) if stops else (t[-1] if len(t) else 0)) - skip_off
        m = (t >= a) & (t <= b)
        return m if m.sum() >= 10 else np.ones(len(t), bool)

    def available_signals(self, seq: str) -> list[str]:
        """Signals that exist for this pass and carry data (SAW has an all-NaN gas column, BAM L1 no trail)."""
        df = self.get_signals(seq)
        return [p for p in SIGNALS if p in df.columns and df[p].notna().any()]

    def signal_notes(self, seq: str) -> dict:
        meta = self.get_meta(seq)
        avail = set(self.available_signals(seq))
        notes = {}
        if "gas_lpm" not in avail:
            notes["gas_lpm"] = ("flux-shielded (SAW): no shielding-gas channel" if str(meta.get("process", "")).upper() == "SAW"
                                else "no shielding-gas signal recorded for this pass")
        if "travel_speed_mmps" not in avail:
            notes["travel_speed_mmps"] = "no travel-speed stream in this dataset"
        if "interpass_c" not in avail:
            notes["interpass_c"] = "no temperature stream in this dataset"
        for p, kind in (self._ac_cache.get(seq) or {}).items():
            name = SIGNAL_INFO.get(p, {}).get("label", p)
            notes[p] = (f"{name}: AC electrode — RMS shown (10 ms means of an AC signal are meaningless)" if kind == "rms"
                        else f"AC electrode — no RMS in this conversion, so not plotted or checked")
        return notes

    # -------------------------------------------------------------- NDT vs signals
    def indication_check(self, seq: str, params: tuple = ("current_a", "voltage_v", "current2_a", "voltage2_v")) -> dict | None:
        """Do the arc signals look different inside the NDT indication windows than in the clean weld?

        For each signal: mean inside all windows vs the clean arc-on weld, and how many windows have a mean outside
        the 1–99 % band of same-length rolling means of the clean weld. `signature` is true only when ≥30 % of the
        windows fall outside that band - otherwise the honest answer is "no electrical signature"."""
        wins = [w for w in self.defect_windows(seq) if w["t_end"] >= w["t_start"]]
        if not wins:
            return None
        df = self.get_signals(seq)
        t = df["t_s"].values
        if len(t) < 20:
            return None
        hz = 1.0 / max(float(np.median(np.diff(t[:500]))), 1e-4)
        ev = self.events(seq)
        starts = [e["t_s"] for e in ev if e.get("type") == "arc_start"]
        stops = [e["t_s"] for e in ev if e.get("type") == "arc_stop"]
        on = (t >= (min(starts) if starts else t[0]) + 1.0) & (t <= (max(stops) if stops else t[-1]) - 0.6)
        inside = np.zeros(len(t), bool)
        for w in wins:
            inside |= (t >= w["t_start"]) & (t <= w["t_end"])
        L = max(int(round(np.median([w["t_end"] - w["t_start"] for w in wins]) * hz)), 1)
        out: dict[str, Any] = {"n_windows": len(wins), "window_s": round(L / hz, 3), "params": {}}
        for p in params:
            if p not in df.columns or not df[p].notna().any():
                continue
            x = df[p].values.astype(float)
            clean = on & ~inside & np.isfinite(x)
            if clean.sum() < max(3 * L, 20):
                continue
            roll = pd.Series(np.where(clean, x, np.nan)).rolling(L, min_periods=L).mean().dropna().values
            lo, hi = (np.percentile(roll, [1, 99]) if len(roll) > 20 else (np.nan, np.nan))
            flagged, means = 0, []
            for w in wins:
                m = (t >= w["t_start"]) & (t <= w["t_end"]) & np.isfinite(x)
                if not m.any():
                    j = int(np.clip(np.searchsorted(t, w["t_start"]), 0, len(t) - 1))
                    m = np.zeros(len(t), bool)
                    m[j] = np.isfinite(x[j])
                    if not m.any():
                        continue
                mu = float(x[m].mean())
                means.append(mu)
                flagged += int(np.isfinite(lo) and (mu < lo or mu > hi))
            if not means:
                continue
            mean_in = float(np.nanmean(x[inside & np.isfinite(x)])) if (inside & np.isfinite(x)).any() else float(np.mean(means))
            mean_out = float(x[clean].mean())
            out["params"][p] = {
                "mean_in": round(mean_in, 2), "mean_clean": round(mean_out, 2), "std_clean": round(float(x[clean].std()), 2),
                "delta_pct": round(100 * (mean_in - mean_out) / mean_out, 2) if mean_out else None,
                "band_1_99": [round(float(lo), 2), round(float(hi), 2)] if np.isfinite(lo) else None,
                "windows_outside_band": flagged, "windows_checked": len(means),
                "signature": bool(len(means) and flagged / len(means) >= 0.3),
                "unit": SIGNAL_INFO.get(p, {}).get("unit", ""),
            }
        out["signature"] = any(v["signature"] for v in out["params"].values())
        return out

    # -------------------------------------------------------------- WPS
    def _local_violations(self, seq: str) -> list[dict]:
        df = self.get_signals(seq)
        lim = self.wps_limits(seq)
        hz = 1.0 / max(np.median(np.diff(df["t_s"].values[:200])), 1e-3)
        # arc-on window from events (skip 1 s start transient and the arc-stop crater fill)
        ev = self.events(seq)
        starts = [e["t_s"] for e in ev if e.get("type") == "arc_start"]
        stops = [e["t_s"] for e in ev if e.get("type") == "arc_stop"]
        skip_on, skip_off = self._transients(seq)
        t_on = (min(starts) if starts else float(df["t_s"].iloc[0])) + skip_on
        t_off = (max(stops) if stops else float(df["t_s"].iloc[-1])) - skip_off
        out = []
        for param, (lo, hi) in lim.items():
            if param not in df.columns or not df[param].notna().any():
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
        wins = [{"defect": d.get("defect"), "t_start": float(d.get("t_start") or 0), "t_end": float(d.get("t_end") or d.get("t_start") or 0),
                 "severity": d.get("severity")} for d in (meta.get("defects") or [])
                if isinstance(d, dict) and d.get("defect") not in (None, "none") and _finite(d.get("t_start"))]
        if not wins and isinstance(meta.get("ndt"), dict):
            for i in meta["ndt"].get("indications") or []:
                if _finite(i.get("t_start")):
                    wins.append({"defect": i.get("type"), "t_start": float(i["t_start"]),
                                 "t_end": float(i.get("t_end") if _finite(i.get("t_end")) else i["t_start"]), "severity": None})
        if wins:
            ndt = meta.get("ndt") if isinstance(meta.get("ndt"), dict) else {}
            ind = ndt.get("indications") or []
            for k, w in enumerate(wins):     # carry NDT positions (mm) for hover / photo cross-reference
                if k < len(ind) and isinstance(ind[k], dict):
                    for f in ("x_start_mm", "x_end_mm", "method"):
                        if ind[k].get(f) is not None:
                            w[f] = ind[k][f]
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
        meta = self.get_meta(seq)
        src = source_of(meta)
        out: dict[str, Any] = {}
        vinfo = meta.get("video") if isinstance(meta.get("video"), dict) else {}
        vfile = vinfo.get("file") or "torch_cam.mp4"
        for key, rel in (("torch", vfile), ("edge", "control/edge.mp4"), ("depth", "control/depth.mp4")):
            if (d / rel).exists():
                out[key] = file_url(d / rel)
        if "torch" in out:
            real = not src.get("synthetic")
            out["video"] = {"kind": vinfo.get("kind") or ("real_process_camera" if real else "simulated_torch_camera"),
                            "fps": vinfo.get("fps"), "t_offset_s": float(vinfo.get("t_offset_s") or 0.0),
                            "real": real,
                            "caption": (src.get("video_caption") or f"Real process camera — {src.get('short')}, {src.get('license')}")
                            if real else "Torch cam · SIMULATOR"}
        ph = self.photos(seq)
        if ph:
            out["photos"] = [{"name": p["name"], "url": p["url"]} for p in ph]
            out["photo_caption"] = src.get("photo_caption") or f"Bead photo — {src.get('short')}, {src.get('license')}"
        if meta.get("fixture"):   # copilot.fixtures_real stand-ins must never pass for the real dataset
            for k in ("photo_caption",):
                if k in out:
                    out[k] += " · FIXTURE placeholder"
            if out.get("video", {}).get("real"):
                out["video"]["caption"] += " · FIXTURE stand-in"
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
                "rt_reject_clips": sum(1 for r in kept if r.get("rt_result") in BAD_RT)}


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
