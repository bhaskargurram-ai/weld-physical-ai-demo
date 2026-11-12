"""Shared helpers for the real-data adapters (aimen_saw, bam_tgmaw) -> pass directories.

Contract: docs/REAL_DATA.md. Every adapter module exposes
    download(raw_dir: Path, **opts) -> None
    convert(raw_dir: Path, out_dir: Path, **opts) -> list[str]      # sequence names written

STABLE API (other adapters import these; keep signatures backwards compatible):

    ARC_HZ = 100
    grid_mean(t_s, cols, duration_s, hz=100, t_start_s=0.0, fill="interp") -> pd.DataFrame
        Bin samples (t_s seconds, any rate) into 1/hz windows [t_start + k/hz, t_start + (k+1)/hz),
        k = 0..ceil(duration_s*hz)-1, and return the per-window MEAN of every column in `cols`
        (dict name -> array, same length as t_s). Empty windows (low-rate signals) are filled by linear
        interpolation over window centres (fill="interp") or left NaN (fill=None). Column "t_s" = window
        start relative to t_start_s.
    add_t_ns(df, t0_ns, t_col="t_s") -> pd.DataFrame
        Replace the relative-seconds column with int64 "t_ns" = t0_ns + round(t*1e9) (first column).
    write_parquet(df, path) -> Path
    event(t0_ns, t_s, type_, **kw) -> dict          {"t_ns": t0_ns + t_s*1e9, "type": type_, **kw}
    write_events(path, events) -> Path              events.jsonl, sorted by t_ns
    ndt_events(t0_ns, indications) -> list[dict]    defect_on/defect_off per indication, source "NDT"
    source_block(id_, title, authors, doi, url, files, license_="CC BY 4.0") -> dict
    meta_skeleton(sequence_name, spool_id, joint_id, pass_no, pass_type, process, *, source,
                  wps_id, position, material, cell_id, thickness_mm, electrodes, t0_ns) -> dict
        All simulator keys present with neutral defaults + contract extras (source, electrodes,
        wps_basis, ndt, thickness_mm, photos). Fill duration_s / setpoints / wps / ndt afterwards.
    set_indications(meta, indications, inspected=True) -> dict
        Writes ndt.indications, defects (defect,t_start,t_end,severity=None), has_defect, defect_types
        (comma-joined, "none"), max_severity, rt_result ("indications"|"no_indications"|"not_inspected").
    wps_from_nominal(wps_id, process, nominal, tol=0.10, gas_lpm=None, max_interpass_c=None,
                     heat_input_kj_mm=None) -> dict
        nominal: {"current_a","voltage_v","wfs_mpm","travel_speed_mmps"} (None -> key omitted);
        each range = [v*(1-tol), v*(1+tol)]; gas_lpm passed as an explicit (lo, hi) or None (no gas
        -> [0, 0] so WPS checkers see "no shielding gas").
    WPS_BASIS (str)   honest description of how the ranges were made
    heat_input_kj_mm(power_w, travel_mmps, efficiency=1.0) -> np.ndarray   (P*eta/v/1000; NaN if v<=0)
    save_photo(src, dst, max_side=1600, quality=90) -> Path
        Decode (EXIF orientation applied), downscale long side to <= max_side, re-encode as JPEG
        without any metadata (EXIF/GPS stripped).
    signal_stats(df, cols) -> dict                  {col_mean, col_std} rounded
    write_index(generated_dir) -> Path              index.jsonl over every */meta.json
    write_meta(path, meta) -> Path                  json dump (numpy-safe)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ARC_HZ = 100
WPS_BASIS = "published nominal setting ±10 % (demo assumption, not a qualified WPS)"


# ------------------------------------------------------------------ resampling

def grid_mean(t_s: np.ndarray, cols: Mapping[str, np.ndarray], duration_s: float, hz: int = ARC_HZ,
              t_start_s: float = 0.0, fill: Optional[str] = "interp") -> pd.DataFrame:
    t = np.asarray(t_s, dtype=np.float64) - t_start_s
    n = max(int(math.ceil(round(duration_s * hz, 6))), 1)
    k = np.floor(t * hz).astype(np.int64)
    ok = (k >= 0) & (k < n) & np.isfinite(t)
    kk = k[ok]
    out = {"t_s": np.arange(n) / hz}
    centres = (np.arange(n) + 0.5) / hz
    for name, v in cols.items():
        v = np.asarray(v, dtype=np.float64)[ok]
        good = np.isfinite(v)
        s = np.bincount(kk[good], weights=v[good], minlength=n)
        c = np.bincount(kk[good], minlength=n).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / c
        m[c == 0] = np.nan
        if fill == "interp":
            have = np.isfinite(m)
            if have.sum() >= 2:
                m = np.interp(centres, centres[have], m[have])
            elif have.sum() == 1:
                m[:] = m[have][0]
        out[name] = m
    return pd.DataFrame(out)


def add_t_ns(df: pd.DataFrame, t0_ns: int, t_col: str = "t_s") -> pd.DataFrame:
    t_ns = int(t0_ns) + np.round(df[t_col].to_numpy(np.float64) * 1e9).astype(np.int64)
    rest = df.drop(columns=[t_col])
    rest.insert(0, "t_ns", t_ns.astype(np.int64))
    return rest.reset_index(drop=True)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ------------------------------------------------------------------ events

def event(t0_ns: int, t_s: float, type_: str, **kw) -> dict:
    return {"t_ns": int(int(t0_ns) + round(float(t_s) * 1e9)), "type": type_, **kw}


def ndt_events(t0_ns: int, indications: Iterable[Mapping]) -> list[dict]:
    ev = []
    for ind in indications:
        extra = {k: ind[k] for k in ("x_start_mm", "x_end_mm") if k in ind}
        ev.append(event(t0_ns, ind["t_start"], "defect_on", defect=ind["type"], severity=None,
                        source="NDT", **extra))
        ev.append(event(t0_ns, ind["t_end"], "defect_off", defect=ind["type"], severity=None,
                        source="NDT", **extra))
    return ev


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_events(path: Path, events: Iterable[Mapping]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evs = sorted(events, key=lambda e: e["t_ns"])
    with open(path, "w") as f:
        for e in evs:
            f.write(json.dumps(e, default=_json_default) + "\n")
    return path


def write_meta(path: Path, meta: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=_json_default, ensure_ascii=False))
    return path


# ------------------------------------------------------------------ meta

def source_block(id_: str, title: str, authors: str, doi: str, url: str, files: Sequence[str],
                 license_: str = "CC BY 4.0") -> dict:
    return {"id": id_, "title": title, "authors": authors, "doi": doi, "url": url,
            "license": license_, "synthetic": False, "files": list(files)}


def meta_skeleton(sequence_name: str, spool_id: str, joint_id: str, pass_no: int, pass_type: str,
                  process: str, *, source: dict, wps_id: str, position: str, material: str,
                  cell_id: str, thickness_mm: float, electrodes: int, t0_ns: int) -> dict:
    return {
        "sequence_name": sequence_name, "spool_id": spool_id, "joint_id": joint_id,
        "pass_no": int(pass_no), "pass_type": pass_type, "wps_id": wps_id, "process": process,
        "position": position, "material": material, "pipe_od_mm": None, "wall_mm": float(thickness_mm),
        "cell_id": cell_id, "duration_s": 0.0, "defects": [], "rt_result": "not_inspected",
        "has_defect": False, "defect_types": "none", "max_severity": None, "t0_ns": int(t0_ns),
        "start_time_utc": pd.Timestamp(int(t0_ns), unit="ns", tz="UTC").isoformat(),
        "wps": {}, "setpoints": {},
        "source": source, "electrodes": int(electrodes), "wps_basis": WPS_BASIS,
        "ndt": {"methods": [], "inspected": False, "report_files": [], "indications": []},
        "thickness_mm": float(thickness_mm), "photos": [],
        "rates_hz": {"arc": ARC_HZ, "travel": ARC_HZ},
    }


def set_indications(meta: dict, indications: Sequence[Mapping], inspected: bool = True) -> dict:
    inds = [dict(i) for i in indications]
    meta.setdefault("ndt", {})
    meta["ndt"]["inspected"] = bool(inspected)
    meta["ndt"]["indications"] = inds
    meta["defects"] = [{"defect": i["type"], "t_start": i["t_start"], "t_end": i["t_end"], "severity": None}
                       for i in inds]
    meta["has_defect"] = bool(inds)
    meta["defect_types"] = ",".join(sorted({i["type"] for i in inds})) or "none"
    meta["max_severity"] = None
    if not inspected:
        meta["rt_result"] = "not_inspected"
    else:
        meta["rt_result"] = "indications" if inds else "no_indications"
    return meta


def wps_from_nominal(wps_id: str, process: str, nominal: Mapping[str, Optional[float]], tol: float = 0.10,
                     gas_lpm: Optional[Sequence[float]] = None, max_interpass_c: Optional[float] = None,
                     heat_input_kj_mm: Optional[Sequence[float]] = None) -> dict:
    def rng(v):
        return [round(v * (1 - tol), 4), round(v * (1 + tol), 4)]

    w: dict = {"wps_id": wps_id, "process": process}
    for k in ("current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm"):
        v = nominal.get(k)
        if v is not None and np.isfinite(v):
            w[k] = rng(float(v))
    w["gas_lpm"] = [float(gas_lpm[0]), float(gas_lpm[1])] if gas_lpm is not None else [0.0, 0.0]
    w["max_interpass_c"] = max_interpass_c
    if heat_input_kj_mm is not None:
        w["heat_input_kj_mm"] = [float(heat_input_kj_mm[0]), float(heat_input_kj_mm[1])]
    w["tolerance"] = tol
    return w


def heat_input_kj_mm(power_w, travel_mmps, efficiency: float = 1.0):
    p = np.asarray(power_w, dtype=np.float64)
    v = np.asarray(travel_mmps, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        hi = efficiency * p / (v * 1000.0)
    return np.where(v > 0, hi, np.nan)


def signal_stats(df: pd.DataFrame, cols: Iterable[str]) -> dict:
    out = {}
    for c in cols:
        if c in df:
            v = df[c].to_numpy(np.float64)
            v = v[np.isfinite(v)]
            out[f"{c}_mean"] = round(float(v.mean()), 3) if v.size else None
            out[f"{c}_std"] = round(float(v.std()), 3) if v.size else None
    return out


# ------------------------------------------------------------------ photos

def save_photo(src: Path, dst: Path, max_side: int = 1600, quality: int = 90) -> Path:
    """Re-encode a photo: EXIF orientation applied, long side <= max_side, no metadata kept."""
    from PIL import Image, ImageOps

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        s = max_side / float(max(w, h))
        if s < 1.0:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        clean = Image.new("RGB", im.size)
        clean.paste(im)
        clean.save(dst, "JPEG", quality=quality, optimize=True)   # no exif= -> stripped
    return dst


# ------------------------------------------------------------------ index

INDEX_KEYS = ("sequence_name", "spool_id", "joint_id", "pass_no", "pass_type", "process", "wps_id",
              "duration_s", "has_defect", "defect_types", "rt_result", "joint_rt_result", "start_time_utc")


def write_index(generated_dir: Path) -> Path:
    generated_dir = Path(generated_dir)
    rows = []
    for m in sorted(generated_dir.glob("*/meta.json")):
        try:
            meta = json.loads(m.read_text())
        except Exception:
            continue
        row = {k: meta.get(k) for k in INDEX_KEYS}
        row["source"] = (meta.get("source") or {}).get("id", "sim")
        rows.append(row)
    p = generated_dir / "index.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
    return p
