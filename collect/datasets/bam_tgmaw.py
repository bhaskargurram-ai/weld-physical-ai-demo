"""BAM tandem-GMAW open dataset -> WeldWorld pass directories (contract: docs/REAL_DATA.md).

Source (CC BY 4.0, attribution shown in the app):
    Chaurasia P. K., Fabry Ç., Pittner A., Rethmeier M. (BAM) - "Tandem-GMAW: advanced pulsed and
    short-circuiting process mode experimental data". Zenodo, DOI 10.5281/zenodo.21824586.

Layout of one record file ``WID617_L3.zip`` (flat):
    WID617_L3_T1_process.nc                 10 kHz current/voltage leading/trailing, video_frame, video_recording
    WID617_L3_T1_robot.nc                   1 kHz arc_on, gas_flow_leading/trailing [L/min], thermo_1/2 [°C],
                                            tcp_coords [mm] (~125 Hz, NaN between), tcp_angles [deg]
    WID617_L3_T1_power_source_{leading,trailing}.nc   Fronius scope, ~20 Hz (NaN between samples):
                                            ACTUAL_WFS, WFS_COMMANDVALUE, ACTUAL_GASFLOW, GASVALUE, JOBNUMBER,
                                            CURRENT_RECOMMVALUE, VOLTAGE_RECOMMVALUE, ...
    WID617_L3_T1_video.avi                  wmv1 1280x1024 grayscale HDR camera, 3000/167 fps (17.96), BAM overlay
    WID617_L3_T1_scan_pre.nc                pre-weld groove scan (unused; deleted after extraction)

Public API (called by ``collect.datasets.fetch``):
    download(raw_dir, wids=None, **opts) -> None
    convert(raw_dir, out_dir, wids=None, **opts) -> list[str]

Video synchronisation (verified on WID617_L3, see ``sync_video``): the ``video_frame`` channel of
process.nc is a 40.0 Hz pulse train (1516 pulses in a 37.9 s recording window) while the AVI holds
701 frame slots at 17.96 fps, so the pulses are NOT one-per-saved-frame. The AVI frame clock itself is
right (the camera's own overlay clock advances 55.67 ms/frame on average), so the video is anchored
optically: the first frame showing the arc is matched to the electrical arc start and the match is
checked at the other end (first frame without arc vs electrical arc stop). If a future file does have
one pulse per frame (|pulses/frames - 1| <= 3 %), the pulses are used directly.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:  # shared helpers (written alongside this adapter)
    from collect.datasets import common as C
except Exception:  # pragma: no cover - fallback keeps this module usable on its own
    C = None

SOURCE_ID = "bam_tgmaw"
RECORD_ID = 21824586
DOI = "10.5281/zenodo.21824586"
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
FILE_URL = API_URL + "/files/{name}/content"
TITLE = "Tandem-GMAW: advanced pulsed and short-circuiting process mode experimental data"
AUTHORS = "Chaurasia P. K., Fabry Ç., Pittner A., Rethmeier M. (BAM Bundesanstalt für Materialforschung und -prüfung)"
LICENSE = "CC BY 4.0"
ATTRIBUTION = (f"Video and signals: {AUTHORS}, {TITLE}, Zenodo, DOI {DOI}, licensed CC BY 4.0. "
               "Converted (resampled, trimmed, re-encoded) by the WeldWorld demo.")

ALL_WIDS = (615, 617, 630, 631, 632)
DEFAULT_WIDS = (617, 630)
LAYERS = (1, 2, 3, 4, 5, 6)
MATERIAL = "S355J2+N"
THICKNESS_MM = 30.0
JOINT_ID = "V30"
CELL_ID = "BAM tandem-GMAW cell (2x Fronius TPS/i, robot)"
EXTRA_FILES = ("experiment_list.csv", "data_view_example.ipynb")

ARC_HZ = 100
ARC_ON_A = 30.0          # 10 ms mean lead current above this = arc burning
OUTAGE_MIN_S = 0.02      # report arc outages >= 20 ms (two 10 ms bins)
POST_ARC_S = 1.0         # pass covers [arc start, arc stop + 1 s] (matches the video)
TRAVEL_SMOOTH_S = 0.5    # centred moving average on the travel-speed derivative
ARC_EFFICIENCY = 0.8     # EN 1011-1 thermal efficiency k for GMAW (heat input, not arc energy)
GAS_DIP_FRAC = 0.8       # gas below 80 % of GASVALUE for >= 0.2 s while the arc burns -> gas dip
SPIKE_SIGMA = 6.0        # current spikes: 10 ms means > median + 6 robust sigma
SYNC_TOL_FRAMES = 2.0    # optical sync accepted when video and signal arc durations agree within 2 frames
CANNY_LO, CANNY_HI = 30, 90   # control/edge.mp4 (after CLAHE + 5x5 Gaussian blur)


def log(msg: str) -> None:
    print(f"[bam_tgmaw] {msg}", file=sys.stderr, flush=True)


# ================================================================== small helpers (fallbacks)

def _grid_mean(t_s, cols, duration_s, hz=ARC_HZ, fill="interp"):
    if C is not None:
        return C.grid_mean(t_s, cols, duration_s, hz=hz, fill=fill)
    t = np.asarray(t_s, dtype=np.float64)
    n = max(int(math.ceil(round(duration_s * hz, 6))), 1)
    k = np.floor(t * hz).astype(np.int64)
    ok = (k >= 0) & (k < n) & np.isfinite(t)
    out = {"t_s": np.arange(n) / hz}
    centres = (np.arange(n) + 0.5) / hz
    for name, v in cols.items():
        v = np.asarray(v, dtype=np.float64)[ok]
        kk = k[ok]
        g = np.isfinite(v)
        s = np.bincount(kk[g], weights=v[g], minlength=n)
        c = np.bincount(kk[g], minlength=n).astype(np.float64)
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


def _add_t_ns(df: pd.DataFrame, t0_ns: int) -> pd.DataFrame:
    if C is not None:
        return C.add_t_ns(df, t0_ns)
    t_ns = int(t0_ns) + np.round(df["t_s"].to_numpy(np.float64) * 1e9).astype(np.int64)
    rest = df.drop(columns=["t_s"])
    rest.insert(0, "t_ns", t_ns)
    return rest.reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    if C is not None:
        C.write_parquet(df, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _event(t0_ns: int, t_s: float, type_: str, **kw) -> dict:
    return {"t_ns": int(int(t0_ns) + round(float(t_s) * 1e9)), "type": type_, **kw}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _write_events(path: Path, events: Iterable[dict]) -> None:
    if C is not None:
        C.write_events(path, events)
        return
    with open(path, "w") as f:
        for e in sorted(events, key=lambda e: e["t_ns"]):
            f.write(json.dumps(e, default=_json_default) + "\n")


def _write_meta(path: Path, meta: dict) -> None:
    if C is not None:
        C.write_meta(path, meta)
        return
    path.write_text(json.dumps(meta, indent=2, default=_json_default, ensure_ascii=False))


def _source_block(files: Sequence[str]) -> dict:
    if C is not None:
        src = C.source_block(SOURCE_ID, TITLE, AUTHORS, DOI, RECORD_URL, files, license_=LICENSE)
    else:
        src = {"id": SOURCE_ID, "title": TITLE, "authors": AUTHORS, "doi": DOI, "url": RECORD_URL,
               "license": LICENSE, "synthetic": False, "files": list(files)}
    src["attribution"] = ATTRIBUTION
    src["institution"] = "BAM Bundesanstalt für Materialforschung und -prüfung, Berlin"
    return src


def _meta_skeleton(seq, spool, pass_no, pass_type, *, source, wps_id, electrodes, t0_ns) -> dict:
    if C is not None:
        return C.meta_skeleton(seq, spool, JOINT_ID, pass_no, pass_type, "GMAW", source=source, wps_id=wps_id,
                               position="1G", material=MATERIAL, cell_id=CELL_ID, thickness_mm=THICKNESS_MM,
                               electrodes=electrodes, t0_ns=t0_ns)
    return {
        "sequence_name": seq, "spool_id": spool, "joint_id": JOINT_ID, "pass_no": int(pass_no),
        "pass_type": pass_type, "wps_id": wps_id, "process": "GMAW", "position": "1G", "material": MATERIAL,
        "pipe_od_mm": None, "wall_mm": THICKNESS_MM, "cell_id": CELL_ID, "duration_s": 0.0, "defects": [],
        "rt_result": "not_inspected", "has_defect": False, "defect_types": "none", "max_severity": None,
        "t0_ns": int(t0_ns), "start_time_utc": pd.Timestamp(int(t0_ns), unit="ns", tz="UTC").isoformat(),
        "wps": {}, "setpoints": {}, "source": source, "electrodes": int(electrodes),
        "wps_basis": "published nominal setting ±10 % (demo assumption, not a qualified WPS)",
        "ndt": {"methods": [], "inspected": False, "report_files": [], "indications": []},
        "thickness_mm": THICKNESS_MM, "photos": [], "rates_hz": {"arc": ARC_HZ, "travel": ARC_HZ},
    }


def _r(x, nd=3):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as [i0, i1) index pairs."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _movavg(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average that ignores NaN and shrinks at the edges."""
    x = np.asarray(x, dtype=np.float64)
    n = max(int(n), 1)
    g = np.isfinite(x)
    k = np.ones(n)
    s = np.convolve(np.where(g, x, 0.0), k, mode="same")
    c = np.convolve(g.astype(np.float64), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def parse_wids(wids) -> list[int]:
    """None -> default (617, 630); "all" -> all five workpieces; "617,630" / [617, "630"] accepted."""
    if wids is None or wids == "" or wids == []:
        return list(DEFAULT_WIDS)
    if isinstance(wids, str):
        if wids.strip().lower() == "all":
            return list(ALL_WIDS)
        wids = [w for w in wids.replace(";", ",").split(",") if w.strip()]
    out = []
    for w in wids:
        if isinstance(w, str) and w.strip().lower() == "all":
            return list(ALL_WIDS)
        w = int(str(w).strip().upper().replace("WID", ""))
        if w not in ALL_WIDS:
            raise ValueError(f"unknown BAM workpiece WID{w}; available: {ALL_WIDS}")
        out.append(w)
    return sorted(set(out))


def parse_layers(layers) -> list[int]:
    if layers is None or layers == "" or layers == []:
        return list(LAYERS)
    if isinstance(layers, str):
        layers = [x for x in layers.replace(";", ",").split(",") if x.strip()]
    return sorted({int(str(x).strip().upper().lstrip("L")) for x in layers})


# ================================================================== download

def _http_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "weldworld-demo/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dst: Path, size: Optional[int] = None, md5: Optional[str] = None,
                   retries: int = 4, timeout: float = 120.0) -> Path:
    """Stream ``url`` to ``dst`` via ``dst.part`` (resumes with HTTP Range), verify size/md5."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "weldworld-demo/1.0"}
        if have and (size is None or have < size):
            headers["Range"] = f"bytes={have}-"
        elif size is not None and have >= size:
            have = 0
            part.unlink()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                t0, done = time.time(), (have if mode == "ab" else 0)
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                log(f"  {dst.name}: {done / 1e6:.1f} MB in {time.time() - t0:.0f} s")
            if size is not None and part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != {size}")
            if md5 and _md5(part) != md5:
                part.unlink()
                raise IOError("md5 mismatch")
            part.replace(dst)
            return dst
        except Exception as e:  # network hiccup -> retry (resume)
            log(f"  {dst.name}: attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(min(30, 3 * attempt))
    return dst


def _layer_complete(d: Path, stem: str) -> bool:
    need = [f"{stem}_process.nc", f"{stem}_robot.nc", f"{stem}_power_source_leading.nc", f"{stem}_video.avi"]
    return all((d / n).exists() for n in need) and (d / ".extracted").exists()


def download(raw_dir: Path, wids: Optional[list[int]] = None, layers=None, keep_scans: bool = False,
             keep_zips: bool = False, verify: bool = True, **opts) -> None:
    """Fetch WID<w>_L<n>.zip for the chosen workpieces (+ experiment_list.csv) into ``raw_dir``.

    Idempotent: a layer already extracted (``raw_dir/WID617_L3/.extracted``) is skipped. Each zip is
    md5-verified against the Zenodo listing, extracted to ``raw_dir/<WID_Ln>/`` and deleted (unless
    ``keep_zips``); ``*_scan_*.nc`` (pre-weld groove scans, ~50 MB each, unused) are deleted unless
    ``keep_scans``.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    wids = parse_wids(wids if wids is not None else opts.get("bam_wids"))
    layers = parse_layers(layers)
    listing = {}
    try:
        listing = {f["key"]: f for f in _http_json(API_URL).get("files", [])}
    except Exception as e:  # listing is only used for size/md5 checks
        log(f"could not read Zenodo file listing ({e}); downloading without checksum verification")

    def meta_of(name):
        f = listing.get(name) or {}
        md5 = (f.get("checksum") or "").split(":", 1)[-1] if verify else None
        return f.get("size"), (md5 or None)

    for name in EXTRA_FILES:
        dst = raw_dir / name
        if not dst.exists():
            size, md5 = meta_of(name)
            _download_file(FILE_URL.format(name=name), dst, size, md5)

    for w in wids:
        for n in layers:
            stem_dir = f"WID{w}_L{n}"
            d = raw_dir / stem_dir
            if _layer_complete(d, f"{stem_dir}_T1"):
                log(f"{stem_dir}: already extracted")
                continue
            zname = f"{stem_dir}.zip"
            if listing and zname not in listing:
                log(f"{zname}: not in the record listing, skipped")
                continue
            zpath = raw_dir / zname
            size, md5 = meta_of(zname)
            if not (zpath.exists() and (size is None or zpath.stat().st_size == size)):
                log(f"downloading {zname} ({(size or 0) / 1e6:.0f} MB)")
                _download_file(FILE_URL.format(name=zname), zpath, size, md5)
            d.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath) as z:
                for info in z.infolist():
                    base = Path(info.filename).name
                    if not base or info.is_dir():
                        continue
                    if "_scan_" in base and not keep_scans:
                        continue
                    with z.open(info) as src, open(d / base, "wb") as out:   # flatten, no path traversal
                        shutil.copyfileobj(src, out, 1 << 20)
            if not keep_scans:
                for p in d.glob("*_scan_*.nc"):
                    p.unlink()
            (d / ".extracted").write_text(json.dumps({"zip": zname, "md5": md5, "at": time.time()}))
            if not keep_zips:
                zpath.unlink(missing_ok=True)
            log(f"{stem_dir}: extracted")


# ================================================================== experiment list

def read_experiment_list(path: Path) -> pd.DataFrame:
    """experiment_list.csv: row 1 = header, row 2 = units (skipped)."""
    df = pd.read_csv(path, header=0, skiprows=[1])
    for c in ("wid", "layer", "track"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("wts", "wfs_command_leading", "wfs_command_trailing", "wfs_command_combined", "weave_amp",
              "weave_freq", "t85", "X", "Y", "Z", "Rx", "Ry", "Rz"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pass_type_for(layer: int) -> str:
    return "root" if layer == 1 else ("cap" if layer == 6 else "fill")


# ================================================================== loading

def _nc_times_s(ds, t_ref: np.datetime64) -> np.ndarray:
    return (ds["time"].values - t_ref).astype("timedelta64[ns]").astype(np.int64) / 1e9


def _sparse(ds, var: str, t_ref) -> tuple[np.ndarray, np.ndarray]:
    """(t_s, values) of a variable sampled sparsely on a shared time axis (NaN elsewhere)."""
    if var not in ds:
        return np.array([]), np.array([])
    v = ds[var].values.astype(np.float64)
    t = _nc_times_s(ds, t_ref)
    g = np.isfinite(v)
    return t[g], v[g]


def _find(raw_dir: Path, stem: str, suffix: str) -> Optional[Path]:
    p = raw_dir / stem.rsplit("_T", 1)[0] / f"{stem}_{suffix}"
    if p.exists():
        return p
    hits = sorted(raw_dir.rglob(f"{stem}_{suffix}"))
    return hits[0] if hits else None


def _ps_const(ds, var: str):
    if ds is None or var not in ds:
        return None
    v = ds[var].values.astype(np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    vals, cnt = np.unique(v, return_counts=True)
    return float(vals[np.argmax(cnt)])


# ================================================================== arc window & anomalies

def arc_window(t_s: np.ndarray, current: np.ndarray, threshold: float = ARC_ON_A,
               bridge_s: float = 0.5) -> tuple[float, float]:
    """(start, stop) [s] of the main arc from a high-rate current: 10 ms means > threshold; the longest
    burning stretch after bridging gaps < bridge_s."""
    g = _grid_mean(t_s - t_s[0], {"i": current}, float(t_s[-1] - t_s[0]) + 0.01, fill=None)
    on = np.nan_to_num(g["i"].to_numpy(), nan=0.0) > threshold
    if not on.any():
        raise ValueError("no arc found (current never above threshold)")
    runs = _runs(on)
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if (a - merged[-1][1]) / ARC_HZ < bridge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    a, b = max(merged, key=lambda r: r[1] - r[0])
    # refine to raw-sample resolution
    t0, t1 = t_s[0] + a / ARC_HZ, t_s[0] + b / ARC_HZ
    i = np.flatnonzero((t_s >= t0 - 0.01) & (t_s <= t0 + 0.01) & (current > threshold))
    j = np.flatnonzero((t_s >= t1 - 0.02) & (t_s <= t1 + 0.01) & (current > threshold))
    return float(t_s[i[0]] if i.size else t0), float(t_s[j[-1]] if j.size else t1)


def find_anomalies(arc: pd.DataFrame, arc_on_s: float, gas_cmd: Optional[float], electrodes: int,
                   trail_window: Optional[tuple[float, float]] = None,
                   gas_cmd2: Optional[float] = None) -> list[dict]:
    """Honest, signal-only observations inside the arc window (no defect labels are invented)."""
    t = arc["t_s"].to_numpy()
    out: list[dict] = []
    inside = (t >= 0.2) & (t <= arc_on_s - 0.2)

    def runs_of(mask, min_s):
        return [(t[a], t[b - 1] + 1.0 / ARC_HZ) for a, b in _runs(mask) if (b - a) / ARC_HZ >= min_s]

    for col, name in (("current_a", "lead"), ("current2_a", "trail")):
        if col not in arc or (col == "current2_a" and electrodes < 2):
            continue
        x = arc[col].to_numpy()
        win = inside if name == "lead" or trail_window is None else \
            (t >= trail_window[0] + 0.2) & (t <= trail_window[1] - 0.2)
        for a, b in runs_of(win & (np.nan_to_num(x) < ARC_ON_A), OUTAGE_MIN_S):
            out.append({"kind": "arc_outage", "electrode": name, "t_start": _r(a), "t_end": _r(b),
                        "detail": f"{name} 10 ms mean current < {ARC_ON_A:.0f} A for {b - a:.2f} s"})
        xs = x[win]
        if xs.size > 50:
            med = np.nanmedian(xs)
            sig = 1.4826 * np.nanmedian(np.abs(xs - med))
            if sig > 0:
                thr = med + SPIKE_SIGMA * sig
                for a, b in runs_of(win & (x > thr), 0.0):
                    seg = (t >= a) & (t < b)
                    out.append({"kind": "current_spike", "electrode": name, "t_start": _r(a), "t_end": _r(b),
                                "peak_a": _r(np.nanmax(x[seg]), 1),
                                "detail": f"{name} 10 ms mean > median {med:.0f} A + {SPIKE_SIGMA:.0f} σ ({thr:.0f} A)"})
    for col, name, cmd in (("gas_lpm", "lead", gas_cmd), ("gas2_lpm", "trail", gas_cmd2)):
        if not cmd or col not in arc or (name == "trail" and electrodes < 2):
            continue
        g = _movavg(arc[col].to_numpy(), 10)
        lim = GAS_DIP_FRAC * cmd
        for a, b in runs_of(inside & (g < lim), 0.2):
            seg = (t >= a) & (t < b)
            startup = a <= 0.25   # flow still building up after ignition (pre-flow is only ~0.1 s)
            out.append({"kind": "gas_startup_lag" if startup else "gas_dip", "electrode": name,
                        "t_start": _r(a), "t_end": _r(b), "min_lpm": _r(np.nanmin(g[seg]), 2),
                        "detail": (f"{name} gas (100 ms mean) below {lim:.1f} L/min (80 % of GASVALUE {cmd:g}) "
                                   + ("from ignition until " + f"{b:.2f} s" if startup else f"for {b - a:.2f} s"))})
    return sorted(out, key=lambda e: e["t_start"])


# ================================================================== video

def _pulse_edges(t_s: np.ndarray, x: np.ndarray) -> np.ndarray:
    hi = np.asarray(x, dtype=np.float64) > 0.5
    up = np.flatnonzero(~hi[:-1] & hi[1:]) + 1
    return t_s[up]


def read_video_frames(path: Path):
    """Decode all frames as uint8 grayscale; returns (frames [n,h,w], slots [n] (pts in frame periods), fps)."""
    import av

    frames, slots = [], []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 0) or 17.964
        tb = float(s.time_base) if s.time_base else 1.0 / fps
        for i, f in enumerate(c.decode(s)):
            frames.append(f.to_ndarray(format="gray"))
            pts = f.pts if f.pts is not None else i
            slots.append(int(round(pts * tb * fps)))
    return np.stack(frames), np.asarray(slots, dtype=np.int64), fps


def _slot_stack(frames: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """1/8-downscaled frames (overlay band skipped) on the CFR slot grid; dropped slots repeat the previous."""
    small = frames[:, 64::8, ::8].astype(np.float32) if frames.shape[1] > 128 else frames.astype(np.float32)
    if len(small) > 4:
        # some recordings (all of WID630) start with a stale frame left over from the previous weld:
        # it differs strongly from frame 1 while frames 1..3 agree -> replace it by frame 1
        d = [float(np.abs(small[i + 1] - small[i]).mean()) for i in range(3)]
        if d[0] > 3 * max(d[1], d[2], 1.0):
            small = small.copy()
            small[0] = small[1]
    n = int(slots[-1]) + 1
    idx = np.searchsorted(slots, np.arange(n), side="right") - 1
    return small[np.clip(idx, 0, len(small) - 1)]


def change_curve(stack: np.ndarray, w: int = 3) -> np.ndarray:
    """cp[k] = mean |mean(stack[k:k+w]) - mean(stack[k-w:k])|: peaks where the scene changes for good
    (arc ignites / goes out); a single-frame flash or spatter contributes only 1/w."""
    n = len(stack)
    cp = np.zeros(n)
    for k in range(w, n - w + 1):
        cp[k] = float(np.abs(stack[k:k + w].mean(axis=0) - stack[k - w:k].mean(axis=0)).mean())
    return cp


def detect_arc_slots(frames: np.ndarray, slots: np.ndarray, arc_dur_slots: Optional[float] = None) -> dict:
    """Optical arc start / stop slots.

    start = first slot showing the arc, stop = first slot without it. Independent estimates: first strong
    change point in the first third, strongest change point in the last third of the clip. Joint estimate (when the electrical arc duration is
    known, in slots): maximise cp[s] + cp[e] with e - s = duration ± 1 slot - a matched filter that is robust
    to a single bogus change (exposure jumps, the robot moving off after the weld).
    """
    stack = _slot_stack(frames, slots)
    n = len(stack)
    out: dict = {"start_slot": None, "stop_slot": None}
    if n < 12:
        return out
    cp = change_curve(stack)
    third = max(n // 3, 4)
    # start: FIRST strong change (>= 60 % of the strongest in the first third), climbed to its local peak - the
    # weld pool keeps growing after ignition, so the strongest change can come a few frames later
    head = cp[:third]
    s_ind = int(np.flatnonzero(head >= 0.6 * head.max())[0])
    while s_ind + 1 < third and cp[s_ind + 1] > cp[s_ind]:
        s_ind += 1
    e_ind = int(n - third + np.argmax(cp[n - third:]))
    out.update({"start_slot_independent": s_ind, "stop_slot_independent": e_ind,
                "cp_median": _r(np.median(cp[3:n - 3]), 2)})
    if arc_dur_slots is None or arc_dur_slots >= n - 4:
        out.update({"start_slot": s_ind, "stop_slot": e_ind, "method": "independent"})
        return out
    best = (-1.0, None, None)
    for s_ in range(3, int(n - arc_dur_slots)):
        c = int(round(s_ + arc_dur_slots))
        for e_ in (c - 1, c, c + 1):
            if 3 <= e_ < n - 2 and cp[s_] + cp[e_] > best[0]:
                best = (cp[s_] + cp[e_], s_, e_)
    _, s_j, e_j = best
    out.update({"start_slot": s_j, "stop_slot": e_j, "method": "joint",
                "cp_start": _r(cp[s_j], 2), "cp_stop": _r(cp[e_j], 2)})
    return out


def sync_video(process_ds, t_ref, video_path: Path, arc_start_s: float, arc_stop_s: float,
               frames=None, slots=None, fps=None) -> dict:
    """Locate the arc-start frame. Returns a dict with method, frame index, offsets and the checks done."""
    if frames is None:
        frames, slots, fps = read_video_frames(video_path)
    n_slots = int(slots[-1]) + 1
    period = 1.0 / fps
    res: dict = {"frames_decoded": int(len(frames)), "frame_slots": n_slots, "fps": round(fps, 4),
                 "dropped_slots": int(n_slots - len(frames))}
    pulses = rec = None
    if process_ds is not None and "video_frame" in process_ds:
        tp = _nc_times_s(process_ds, t_ref)
        pulses = _pulse_edges(tp, process_ds["video_frame"].values)
        if "video_recording" in process_ds:
            r = process_ds["video_recording"].values > 0.5
            if r.any():
                idx = np.flatnonzero(r)
                rec = (float(tp[idx[0]]), float(tp[idx[-1]]))
                pulses = pulses[(pulses >= rec[0] - 1e-3) & (pulses <= rec[1] + 1e-3)]
        res["sync_pulses"] = int(len(pulses))
        if len(pulses) > 1:
            res["sync_pulse_hz"] = round(1.0 / float(np.median(np.diff(pulses))), 3)
        res["recording_window_s"] = [_r(rec[0]), _r(rec[1])] if rec else None
    # 1) one pulse per frame slot -> direct mapping
    if pulses is not None and len(pulses) and abs(len(pulses) / n_slots - 1) <= 0.03:
        slot_t = pulses[:n_slots] if len(pulses) >= n_slots else np.r_[pulses, pulses[-1] + period *
                                                                           np.arange(1, n_slots - len(pulses) + 1)]
        k = int(np.searchsorted(slot_t, arc_start_s))
        res.update({"method": "video_frame_pulses", "arc_start_slot": k,
                    "t_offset_s": _r(slot_t[k] - arc_start_s, 4), "pulses_match_frames": True})
        return res
    res["pulses_match_frames"] = False
    # 2) optical: the arc lights up / goes out on camera; matched to the electrical arc duration
    dur = arc_stop_s - arc_start_s
    det = detect_arc_slots(frames, slots, dur / period)
    if det.get("start_slot") is None:
        raise ValueError("could not find the arc-start frame optically")
    k = int(det["start_slot"])
    by_slot = {int(s_): i for i, s_ in enumerate(slots)}
    res.update({"method": "optical_arc_start_stop", "arc_start_slot": k, "arc_start_frame": by_slot.get(k),
                # the arc ignited between the previous slot and this one -> mid-point estimate
                "t_offset_s": _r(0.5 * period, 4), "t_offset_uncertainty_s": _r(0.5 * period, 4),
                "optical": det})
    k_stop = det.get("stop_slot")
    if k_stop is not None:
        dur_video = (k_stop - k) * period
        res.update({"arc_stop_slot": int(k_stop), "arc_duration_video_s": _r(dur_video),
                    "arc_duration_signal_s": _r(dur), "duration_error_frames": _r((dur_video - dur) / period, 2)})
    # verified = the strongest start/stop changes found WITHOUT knowing the duration agree with the joint fit
    s_ind, e_ind = det.get("start_slot_independent"), det.get("stop_slot_independent")
    res["independent_agrees"] = bool(s_ind is not None and abs(s_ind - k) <= SYNC_TOL_FRAMES
                                     and k_stop is not None and abs(e_ind - k_stop) <= SYNC_TOL_FRAMES)
    res["verified"] = bool(res["independent_agrees"] or (s_ind is not None and abs(s_ind - k) <= SYNC_TOL_FRAMES))
    # implied DAQ time of slot 0, for the record (relative to arc start)
    res["slot0_vs_arc_start_s"] = _r(-(k - 0.5) * period, 4)
    if rec is not None:
        res["slot0_vs_recording_start_s"] = _r((arc_start_s - (k - 0.5) * period) - rec[0], 4)
    return res


def _ffmpeg_writer(path: Path, w: int, h: int, fps_str: str, crf: int = 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
           "-r", fps_str, "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", fps_str, str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_videos(frames: np.ndarray, slots: np.ndarray, fps: float, start_slot: int, n_out: int,
                 out_cam: Path, out_edge: Optional[Path], stills_dir: Optional[Path] = None,
                 still_fracs=(0.15, 0.5, 0.85), arc_frames: Optional[int] = None,
                 fps_str: Optional[str] = None, max_width: int = 1280) -> dict:
    """CFR re-encode: output frame j = source slot start_slot + j (dropped slots repeat the previous frame)."""
    import cv2

    fps_str = fps_str or f"{fps:.6f}"
    h, w = frames.shape[1:]
    if w > max_width:
        s = max_width / w
        w, h = max_width, int(round(h * s / 2) * 2)
    by_slot = {int(s): i for i, s in enumerate(slots)}
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    procs = [_ffmpeg_writer(out_cam, w, h, fps_str)]
    if out_edge is not None:
        procs.append(_ffmpeg_writer(out_edge, w, h, fps_str))
    last = None
    repeated = 0
    n_written = 0
    still_at = sorted({int(round(f * max((arc_frames or n_out) - 1, 0))) for f in still_fracs})
    stills = []
    try:
        for j in range(n_out):
            i = by_slot.get(start_slot + j)
            if i is None:
                if last is None:   # before the first decoded frame: nearest later frame
                    later = [k for k in range(start_slot + j, start_slot + j + 50) if k in by_slot]
                    if not later:
                        break
                    i = by_slot[later[0]]
                else:
                    repeated += 1
                    i = last
            if start_slot + j > slots[-1]:
                break
            last = i
            img = frames[i]
            if img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            procs[0].stdin.write(np.ascontiguousarray(img).tobytes())
            if out_edge is not None:
                # HDR footage is low-contrast: local contrast equalisation, then fixed Canny thresholds
                edge = cv2.Canny(cv2.GaussianBlur(clahe.apply(img), (5, 5), 0), CANNY_LO, CANNY_HI)
                procs[1].stdin.write(np.ascontiguousarray(edge).tobytes())
            if stills_dir is not None and j in still_at:
                stills_dir.mkdir(parents=True, exist_ok=True)
                p = stills_dir / f"still_{j / fps:05.1f}s.jpg"
                cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                stills.append({"file": p.name, "t_s": round(j / fps, 3), "frame": j})
            n_written += 1
    finally:
        for p in procs:
            p.stdin.close()
            p.wait()
    if any(p.returncode for p in procs):
        raise RuntimeError("ffmpeg failed while encoding the BAM video")
    return {"frames": n_written, "repeated_frames": repeated, "width": w, "height": h, "stills": stills}


# ================================================================== one layer

def convert_layer(raw_dir: Path, out_dir: Path, wid: int, layer: int, exp: Optional[pd.DataFrame] = None,
                  video: bool = True, overwrite: bool = True) -> dict:
    import xarray as xr

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    stem = f"WID{wid}_L{layer}_T1"
    seq = f"WID{wid}_L{layer}_TGMAW"
    f_proc = _find(raw_dir, stem, "process.nc")
    f_rob = _find(raw_dir, stem, "robot.nc")
    f_psl = _find(raw_dir, stem, "power_source_leading.nc")
    f_pst = _find(raw_dir, stem, "power_source_trailing.nc")
    f_vid = _find(raw_dir, stem, "video.avi")
    if f_proc is None or f_rob is None:
        raise FileNotFoundError(f"{stem}: process.nc / robot.nc missing under {raw_dir}")
    d = out_dir / seq
    if d.exists() and not overwrite and (d / "meta.json").exists():
        log(f"{seq}: exists, skipped")
        return json.loads((d / "meta.json").read_text())

    # experiment list row
    row = {}
    if exp is not None:
        m = exp[(exp["wid"] == wid) & (exp["layer"] == layer)]
        if len(m):
            row = {k: (None if (isinstance(v, float) and not np.isfinite(v)) or v is pd.NA else v)
                   for k, v in m.iloc[0].to_dict().items()}
    tandem = (str(row.get("config", "")).lower() == "tandem") if row else (layer > 1)
    electrodes = 2 if tandem else 1

    proc = xr.load_dataset(f_proc)
    rob = xr.load_dataset(f_rob)
    psl = xr.load_dataset(f_psl) if f_psl else None
    pst = xr.load_dataset(f_pst) if (f_pst and tandem) else None
    t_ref = proc["time"].values[0]

    # --- arc window from the 10 kHz lead current; robot arc_on as cross-check
    tp = _nc_times_s(proc, t_ref)
    I1 = proc["current_leading"].values.astype(np.float64)
    U1 = proc["voltage_leading"].values.astype(np.float64)
    a0, a1 = arc_window(tp, I1)
    arc_on_s = a1 - a0
    duration = arc_on_s + POST_ARC_S
    t0_ns = int(np.datetime64(t_ref, "ns").astype(np.int64) + round(a0 * 1e9))

    tr_ao, ao = _sparse(rob, "arc_on", t_ref)
    ao_on = np.flatnonzero(ao > 0.5)
    robot_arc_on = [_r(tr_ao[ao_on[0]] - a0), _r(tr_ao[ao_on[-1]] - a0)] if ao_on.size else None
    trail_window = None
    if tandem:
        I2 = proc["current_trailing"].values.astype(np.float64)
        try:
            b0, b1 = arc_window(tp, I2)
            trail_window = (b0 - a0, b1 - a0)
        except ValueError:
            trail_window = None

    # --- arc @100 Hz
    rel = tp - a0
    cols = {"current_a": I1, "voltage_v": U1, "p1": I1 * U1}
    if tandem:
        U2 = proc["voltage_trailing"].values.astype(np.float64)
        cols.update({"current2_a": I2, "voltage2_v": U2, "p2": I2 * U2})
    g = _grid_mean(rel, cols, duration, fill="interp")
    tw, wfs1 = _sparse(psl, "ACTUAL_WFS", t_ref) if psl is not None else (np.array([]), np.array([]))
    gw = {"wfs_mpm": (tw - a0, wfs1)}
    if tandem and pst is not None:
        tw2, wfs2 = _sparse(pst, "ACTUAL_WFS", t_ref)
        gw["wfs2_mpm"] = (tw2 - a0, wfs2)
    tg, gas1 = _sparse(rob, "gas_flow_leading", t_ref)
    _, gas2 = _sparse(rob, "gas_flow_trailing", t_ref)
    tt1, th1 = _sparse(rob, "thermo_1", t_ref)
    tt2, th2 = _sparse(rob, "thermo_2", t_ref)
    gr = _grid_mean(tg - a0, {"gas_lpm": gas1, "gas2_lpm": gas2}, duration, fill="interp")

    arc = pd.DataFrame({"t_s": g["t_s"], "current_a": g["current_a"], "voltage_v": g["voltage_v"]})
    for k, (tt, vv) in gw.items():
        arc[k] = _grid_mean(tt, {k: vv}, duration, fill="interp")[k] if tt.size else np.nan
    arc["gas_lpm"] = gr["gas_lpm"]
    order = ["t_s", "current_a", "voltage_v", "wfs_mpm", "gas_lpm"]
    if tandem:
        arc["current2_a"] = g["current2_a"]
        arc["voltage2_v"] = g["voltage2_v"]
        if "wfs2_mpm" not in arc:
            arc["wfs2_mpm"] = np.nan
        arc["gas2_lpm"] = gr["gas2_lpm"]
        order += ["current2_a", "voltage2_v", "wfs2_mpm", "gas2_lpm"]
    arc = arc[order].copy()

    # --- travel: TCP path projected on the groove direction (removes the weave)
    tc = rob["tcp_coords"].values.astype(np.float64)
    ttc_all = _nc_times_s(rob, t_ref)
    okc = np.isfinite(tc).all(axis=1)
    ttc, xyz = ttc_all[okc] - a0, tc[okc]
    in_arc = (ttc >= 0) & (ttc <= arc_on_s)
    if in_arc.sum() < 10:
        raise ValueError(f"{stem}: no TCP samples during the arc")
    p_start = np.array([np.interp(0.0, ttc, xyz[:, k]) for k in range(3)])
    p_end = np.array([np.interp(arc_on_s, ttc, xyz[:, k]) for k in range(3)])
    u = p_end - p_start
    seam_len = float(np.linalg.norm(u))
    u = u / seam_len if seam_len > 0 else np.array([1.0, 0.0, 0.0])
    s_raw = (xyz - p_start) @ u
    grob = _grid_mean(ttc, {"s": s_raw, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}, duration, fill="interp")
    pos = grob["s"].to_numpy()
    v = _movavg(np.gradient(pos, 1.0 / ARC_HZ), int(TRAVEL_SMOOTH_S * ARC_HZ))
    wts = row.get("wts")
    power = g["p1"].to_numpy() + (g["p2"].to_numpy() if tandem else 0.0)
    burning = np.nan_to_num(g["current_a"].to_numpy()) > ARC_ON_A
    v_floor = 0.5 * wts if wts else 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        hi = ARC_EFFICIENCY * power / (np.maximum(v, v_floor) * 1000.0)
    hi[~burning] = np.nan
    travel = pd.DataFrame({"t_s": g["t_s"], "travel_speed_mmps": v, "position_mm": pos,
                           "heat_input_kj_mm": hi, "travel_speed_nominal_mmps": float(wts) if wts else np.nan})

    # --- thermal (1 kHz -> 100 Hz means)
    gth = _grid_mean(tt1 - a0, {"interpass_c": th1}, duration, fill="interp")
    gth2 = _grid_mean(tt2 - a0, {"thermo2_c": th2}, duration, fill="interp")
    thermal = pd.DataFrame({"t_s": gth["t_s"], "interpass_c": gth["interpass_c"], "thermo2_c": gth2["thermo2_c"]})

    # --- robot (50 Hz): TCP position + orientation (extrinsic xyz Euler deg -> quaternion)
    ang = rob["tcp_angles"].values.astype(np.float64)[okc]
    ang = np.rad2deg(np.unwrap(np.deg2rad(ang), axis=0))
    rob_hz = 50
    gr50 = _grid_mean(ttc, {"tcp_x": xyz[:, 0], "tcp_y": xyz[:, 1], "tcp_z": xyz[:, 2],
                            "rx": ang[:, 0], "ry": ang[:, 1], "rz": ang[:, 2]}, duration, hz=rob_hz, fill="interp")
    q = euler_xyz_to_quat(gr50[["rx", "ry", "rz"]].to_numpy())
    robot = pd.DataFrame({"t_s": gr50["t_s"], "tcp_x": gr50["tcp_x"], "tcp_y": gr50["tcp_y"], "tcp_z": gr50["tcp_z"],
                          "tcp_qx": q[:, 0], "tcp_qy": q[:, 1], "tcp_qz": q[:, 2], "tcp_qw": q[:, 3],
                          "tcp_rx_deg": gr50["rx"], "tcp_ry_deg": gr50["ry"], "tcp_rz_deg": gr50["rz"]})

    # --- commands / settings
    jobs = {"lead": _ps_const(psl, "JOBNUMBER"), "trail": _ps_const(pst, "JOBNUMBER") if tandem else None}
    gas_cmd = _ps_const(psl, "GASVALUE")
    rec_i1, rec_u1 = _ps_const(psl, "CURRENT_RECOMMVALUE"), _ps_const(psl, "VOLTAGE_RECOMMVALUE")
    rec_i2 = _ps_const(pst, "CURRENT_RECOMMVALUE") if tandem else None
    rec_u2 = _ps_const(pst, "VOLTAGE_RECOMMVALUE") if tandem else None
    wfs_cmd1 = row.get("wfs_command_leading") or _ps_const(psl, "WFS_COMMANDVALUE")
    wfs_cmd2 = (row.get("wfs_command_trailing") or _ps_const(pst, "WFS_COMMANDVALUE")) if tandem else None
    ps_wfs_cmd = {"lead": _ps_const(psl, "WFS_COMMANDVALUE"), "trail": _ps_const(pst, "WFS_COMMANDVALUE")}

    # --- measured (steady part: skip first/last 2 s)
    t = arc["t_s"].to_numpy()
    steady = (t > 2.0) & (t < arc_on_s - 2.0)
    if steady.sum() < 50:
        steady = (t >= 0) & (t <= arc_on_s)

    def mean(df, c, mask=steady):
        if c not in df:
            return None
        x = df.loc[mask, c].to_numpy(np.float64)
        x = x[np.isfinite(x)]
        return _r(x.mean()) if x.size else None

    measured = {"current_a": mean(arc, "current_a"), "voltage_v": mean(arc, "voltage_v"),
                "wfs_mpm": mean(arc, "wfs_mpm"), "gas_lpm": mean(arc, "gas_lpm"),
                "travel_speed_mmps": mean(travel, "travel_speed_mmps"),
                "heat_input_kj_mm": mean(travel, "heat_input_kj_mm"),
                "power_lead_kw": _r(np.nanmean(g["p1"].to_numpy()[steady]) / 1000.0),
                "seam_length_mm": _r(seam_len, 1)}
    if tandem:
        measured.update({"current2_a": mean(arc, "current2_a"), "voltage2_v": mean(arc, "voltage2_v"),
                         "wfs2_mpm": mean(arc, "wfs2_mpm"), "gas2_lpm": mean(arc, "gas2_lpm"),
                         "power_trail_kw": _r(np.nanmean(g["p2"].to_numpy()[steady]) / 1000.0)})
    # power-source side measurements (Fronius, ~20 Hz) for comparison
    for side, ds in (("lead", psl), ("trail", pst)):
        if ds is None:
            continue
        for var, key in (("ACTUAL_CURRENT", "ps_current_a"), ("ACTUAL_VOLTAGE", "ps_voltage_v"),
                         ("ACTUAL_GASFLOW", "ps_gas_lpm")):
            ts_, vs_ = _sparse(ds, var, t_ref)
            ts_ = ts_ - a0
            mm = (ts_ > 2.0) & (ts_ < arc_on_s - 2.0)
            measured[f"{key}_{side}"] = _r(vs_[mm].mean()) if mm.any() else None

    # --- WPS. Published commands: wfs (experiment_list) + wts + GASVALUE. Synergic GMAW has no current/voltage
    # command; the Fronius "recommended" display values sit 10-25 % away from what the arc actually ran at, so
    # current/voltage use this pass's measured steady-state mean (PQR-style: a WPS range around the values
    # recorded during the qualifying weld). Recommended values are kept in meta.nominal.recommended.
    nominal = {"current_a": measured["current_a"], "voltage_v": measured["voltage_v"], "wfs_mpm": wfs_cmd1,
               "travel_speed_mmps": wts}
    if tandem:
        nominal.update({"current2_a": measured.get("current2_a"), "voltage2_v": measured.get("voltage2_v"),
                        "wfs2_mpm": wfs_cmd2})
    gas_rng = [round(gas_cmd * 0.8, 3), round(gas_cmd * 1.2, 3)] if gas_cmd else None
    hi_nom = None
    p_meas = (measured.get("power_lead_kw") or 0) + (measured.get("power_trail_kw") or 0)
    if p_meas and wts:
        hi_nom = ARC_EFFICIENCY * p_meas / wts   # kW / (mm/s) = kJ/mm
    hi_rng = [round(hi_nom * 0.9 ** 2 / 1.1, 4), round(hi_nom * 1.1 ** 2 / 0.9, 4)] if hi_nom else None
    wps_id = f"BAM-WID{wid}-L{layer}"
    if C is not None:
        wps = C.wps_from_nominal(wps_id, "GMAW", nominal, tol=0.10, gas_lpm=gas_rng, heat_input_kj_mm=hi_rng)
    else:
        wps = {"wps_id": wps_id, "process": "GMAW"}
        for k in ("current_a", "voltage_v", "travel_speed_mmps", "wfs_mpm"):
            if nominal.get(k):
                wps[k] = [round(nominal[k] * 0.9, 4), round(nominal[k] * 1.1, 4)]
        wps["gas_lpm"] = gas_rng or [0.0, 0.0]
        if hi_rng:
            wps["heat_input_kj_mm"] = hi_rng
        wps["tolerance"] = 0.10
    if tandem:
        for k in ("current2_a", "voltage2_v", "wfs2_mpm"):
            if nominal.get(k):
                wps[k] = [round(nominal[k] * 0.9, 4), round(nominal[k] * 1.1, 4)]
    if wps.get("max_interpass_c") is None:
        wps.pop("max_interpass_c", None)   # not published by BAM; don't invent a limit
    wps["note"] = ("wfs = experiment_list wfs_command ±10 %; travel = experiment_list wts ±10 %; gas = GASVALUE "
                   "±20 %; current/voltage = this pass's measured steady-state mean ±10 % (PQR-style: no current/"
                   "voltage command exists for synergic GMAW); heat input = η "
                   f"{ARC_EFFICIENCY} × measured mean Σ(U·I) / wts with the ±10 % corners; no interpass limit published")

    gas_cmd2 = _ps_const(pst, "GASVALUE") if tandem else None
    anomalies = find_anomalies(arc, arc_on_s, gas_cmd, electrodes, trail_window, gas_cmd2)

    # thermo_2 reading for the record: plunged pool thermocouple (t8/5) - see signal_notes
    t85_meas = None
    if th2.size:
        k = int(np.nanargmax(th2))
        if th2[k] > 800:
            after_t, after_v = tt2[k:], th2[k:]
            b8 = np.flatnonzero(after_v < 800)
            b5 = np.flatnonzero(after_v < 500)
            if b8.size and b5.size:
                t85_meas = _r(after_t[b5[0]] - after_t[b8[0]], 2)

    # --- meta
    rel_files = [p.relative_to(raw_dir).as_posix() for p in (f_proc, f_rob, f_psl, f_pst, f_vid) if p is not None]
    rel_files.append("experiment_list.csv")
    src = _source_block(rel_files)
    mode = row.get("mode") or ("PMC" if not tandem else "unknown")
    meta = _meta_skeleton(seq, f"WID{wid}", layer, pass_type_for(layer), source=src, wps_id=wps_id,
                          electrodes=electrodes, t0_ns=t0_ns)
    meta.update({
        "duration_s": _r(duration), "arc_on_s": _r(arc_on_s), "mode": mode,
        "mode_leading": row.get("mode_leading"), "mode_trailing": row.get("mode_trailing") if tandem else None,
        "config": row.get("config") or ("tandem" if tandem else "single"),
        "wps": wps, "wps_basis": ("published nominal setting ±10 % (demo assumption, not a qualified WPS): "
                                 "experiment_list wfs/wts commands ±10 %, GASVALUE ±20 %; current/voltage = the "
                                 "pass's own measured mean ±10 % (PQR-style; synergic GMAW has no I/U command)"),
        "setpoints": {k: v for k, v in {**nominal, "gas_lpm": gas_cmd}.items() if v is not None},
        "nominal": {"wts_mmps": wts, "wfs_command_leading": row.get("wfs_command_leading"),
                    "wfs_command_trailing": row.get("wfs_command_trailing"),
                    "wfs_command_combined": row.get("wfs_command_combined"),
                    "power_source_wfs_command": ps_wfs_cmd, "gasvalue_lpm": gas_cmd,
                    "weave_amp_mm": row.get("weave_amp"), "weave_freq_hz": row.get("weave_freq"),
                    "t85_s": row.get("t85"), "job_number": jobs,
                    "recommended": {"current_lead_a": rec_i1, "voltage_lead_v": rec_u1,
                                    "current_trail_a": rec_i2, "voltage_trail_v": rec_u2},
                    "heat_input_kj_mm": _r(hi_nom, 4)},
        "measured": measured,
        "current_mean_a": measured["current_a"], "voltage_mean_v": measured["voltage_v"],
        "travel_mean_mmps": measured["travel_speed_mmps"], "heat_input_mean_kj_mm": measured["heat_input_kj_mm"],
        "travel_speed_mismatch_pct": _r((measured["travel_speed_mmps"] / wts - 1) * 100, 1)
        if (wts and measured["travel_speed_mmps"]) else None,
        "weave_amp_mm": row.get("weave_amp"), "weave_hz": row.get("weave_freq"),
        "interpass_start_c": _r(np.nanmean(thermal["interpass_c"].to_numpy()[:50]), 1),
        "interpass_max_c": _r(np.nanmax(thermal["interpass_c"].to_numpy()), 1),
        "t85_measured_thermo2_s": t85_meas,
        "anomalies": anomalies,
        "robot_arc_on_s": robot_arc_on, "trail_arc_window_s": [_r(x) for x in trail_window] if trail_window else None,
        "consumables": {"shielding_gas": "not stated in the record", "wire": "not stated in the record"},
        "signal_notes": {
            "electrodes": "lead = leading wire (current_a, voltage_v, wfs_mpm, gas_lpm); trail = trailing wire "
                          "(current2_a, voltage2_v, wfs2_mpm, gas2_lpm); L1 root is single-wire",
            "arc_window": f"lead current (process.nc, 10 kHz) 10 ms mean > {ARC_ON_A:.0f} A, gaps < 0.5 s bridged; "
                          "t = 0 at arc start; robot arc_on given as robot_arc_on_s for comparison",
            "current_voltage": "10 ms means of the 10 kHz process.nc signals (measured by BAM's DAQ, not the "
                               "power source); power-source values at ~20 Hz are in measured.ps_*",
            "wfs": "power source ACTUAL_WFS (~20 Hz, linearly interpolated to 100 Hz)",
            "gas": "robot.nc gas_flow_leading / gas_flow_trailing (1 kHz flow meters, L/min)",
            "travel_speed": f"d/dt of the TCP position projected on the seam direction (arc-start -> arc-stop TCP), "
                            f"{TRAVEL_SMOOTH_S} s centred moving average; the weave is perpendicular and drops out",
            "position_mm": "TCP distance along the seam from the arc-start position",
            "heat_input": f"η {ARC_EFFICIENCY} (EN 1011-1, GMAW) × (P_lead + P_trail) / v / 1000 [kJ/mm], P = 10 ms "
                          f"means of instantaneous U·I; v floored at 0.5 × wts; NaN while the lead arc is off",
            "interpass_c": "robot.nc thermo_1 (workpiece thermocouple; smooth 100-290 °C history) - used as the "
                           "interpass / workpiece temperature. Channel meaning is our reading of the data: the "
                           "record's notebook does not document thermo_1/2",
            "thermo2_c": "robot.nc thermo_2 raw: reads noise (≈ -280..+40 °C) until it is plunged into the weld "
                         "pool, then peaks near 1400 °C and cools - consistent with the t8/5 thermocouple "
                         "(experiment_list t85); treat values before the peak as invalid",
            "robot": "tcp_coords [mm] (~125 Hz) and tcp_angles (extrinsic xyz Euler, deg, as in the record's "
                     "notebook WXRotation.from_euler('xyz')) -> 50 Hz means; quaternions derived",
            "time_base": "all .nc files share BAM's DAQ clock (UTC); t0_ns = arc start in that clock",
            "anomalies": "signal-only observations (arc outages, current spikes, gas dips); BAM published no NDT, "
                         "so nothing here is a confirmed defect",
        },
        "rates_hz": {"arc": ARC_HZ, "travel": ARC_HZ, "thermal": ARC_HZ, "robot": rob_hz},
    })
    if C is not None and hasattr(C, "set_indications"):
        C.set_indications(meta, [], inspected=False)
    meta["ndt"].update({"methods": [], "inspected": False, "report_files": [], "indications": [],
                        "note": "BAM published no radiography/UT for these welds"})

    # --- write signals
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    _write_parquet(_add_t_ns(arc, t0_ns), d / "arc.parquet")
    _write_parquet(_add_t_ns(travel, t0_ns), d / "travel.parquet")
    _write_parquet(_add_t_ns(thermal, t0_ns), d / "thermal.parquet")
    _write_parquet(_add_t_ns(robot, t0_ns), d / "robot.parquet")
    events = [_event(t0_ns, 0.0, "arc_start", process="GMAW", mode=mode, electrodes=electrodes),
              _event(t0_ns, arc_on_s, "arc_stop")]
    if trail_window:
        events += [_event(t0_ns, trail_window[0], "trail_arc_start"),
                   _event(t0_ns, trail_window[1], "trail_arc_stop")]
    for a in anomalies:
        events.append(_event(t0_ns, a["t_start"], "signal_anomaly", kind=a["kind"], electrode=a["electrode"],
                             t_end_s=a["t_end"], source="signal", detail=a["detail"]))
    _write_events(d / "events.jsonl", events)

    # --- video
    if video and f_vid is not None:
        try:
            frames, slots, fps = read_video_frames(f_vid)
            sync = sync_video(proc, t_ref, f_vid, a0, a1, frames, slots, fps)
            n_out = int(math.floor(duration * fps)) + 1
            arc_frames = int(math.floor(arc_on_s * fps))
            fps_str = "3000/167" if abs(fps - 3000 / 167) < 1e-3 else f"{fps:.6f}"
            vw = write_videos(frames, slots, fps, sync["arc_start_slot"], n_out, d / "torch_cam.mp4",
                              d / "control" / "edge.mp4", stills_dir=d / "stills", arc_frames=arc_frames,
                              fps_str=fps_str)
            meta["video"] = {"file": "torch_cam.mp4", "kind": "real_process_camera", "fps": round(fps, 4),
                             "fps_rational": fps_str, "t_offset_s": sync["t_offset_s"], "frames": vw["frames"],
                             "width": vw["width"], "height": vw["height"], "grayscale": True,
                             "end": "arc stop + 1 s", "overlay": "BAM camera overlay kept (attribution)",
                             "control": {"edge": "control/edge.mp4"},
                             "stills": [f"stills/{s['file']}" for s in vw["stills"]],
                             "repeated_frames": vw["repeated_frames"], "sync": sync,
                             "attribution": ATTRIBUTION}
            meta["stills"] = meta["video"]["stills"]
        except Exception as e:   # video is optional: keep the signals
            log(f"{seq}: video skipped: {e}")
            meta["video_error"] = str(e)
    _write_meta(d / "meta.json", meta)
    log(f"{seq}: {arc_on_s:.1f} s arc, {len(anomalies)} anomalies"
        + (f", video {meta['video']['frames']} frames ({meta['video']['sync']['method']})" if "video" in meta else ""))
    return meta


def euler_xyz_to_quat(ang_deg: np.ndarray) -> np.ndarray:
    """Extrinsic x-y-z Euler angles [deg] (scipy 'xyz') -> quaternions (x, y, z, w)."""
    a = np.deg2rad(np.asarray(ang_deg, dtype=np.float64)) / 2.0
    cx, sx = np.cos(a[:, 0]), np.sin(a[:, 0])
    cy, sy = np.cos(a[:, 1]), np.sin(a[:, 1])
    cz, sz = np.cos(a[:, 2]), np.sin(a[:, 2])
    # q = qz * qy * qx
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return np.stack([x, y, z, w], axis=1)


# ================================================================== convert + report

def convert(raw_dir: Path, out_dir: Path, wids: Optional[list[int]] = None, layers=None, video: bool = True,
            overwrite: bool = True, report: bool = True, **opts) -> list[str]:
    """Convert every downloaded layer of the chosen workpieces into ``out_dir/WID<w>_L<n>_TGMAW``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    wids = parse_wids(wids if wids is not None else opts.get("bam_wids"))
    layers = parse_layers(layers)
    exp_path = raw_dir / "experiment_list.csv"
    exp = read_experiment_list(exp_path) if exp_path.exists() else None
    if exp is None:
        log("experiment_list.csv missing: nominal travel speed / mode / wfs commands unavailable")
    names, metas = [], []
    for w in wids:
        for n in layers:
            stem = f"WID{w}_L{n}_T1"
            if _find(raw_dir, stem, "process.nc") is None:
                continue
            try:
                m = convert_layer(raw_dir, out_dir, w, n, exp=exp, video=video, overwrite=overwrite)
            except Exception as e:
                log(f"{stem}: FAILED: {e}")
                continue
            names.append(m["sequence_name"])
            metas.append(m)
    if not names:
        log(f"nothing converted (no WID{wids} layers under {raw_dir}; run download first)")
    if report and metas:
        write_report(metas, out_dir / "bam_tgmaw_report.md")
    if C is not None and hasattr(C, "write_index") and names:
        C.write_index(out_dir)
    return names


def _pct(a, b):
    return f"{(a / b - 1) * 100:+.1f} %" if (a is not None and b) else "–"


def write_report(metas: list[dict], path: Path) -> Path:
    L = [f"# BAM tandem-GMAW — conversion report", "",
         f"Source: {AUTHORS}, *{TITLE}*, Zenodo DOI {DOI} ({LICENSE}). Measured = steady part of the arc "
         "(first/last 2 s excluded). Commands = experiment_list (wfs, wts) and GASVALUE (gas); (rec) = the Fronius "
         "synergic *recommended* display value for the job, not a command. Current & voltage from BAM's 10 kHz DAQ "
         "(process.nc). WPS current/voltage ranges = measured mean ±10 % (PQR-style), so they are not checked "
         "against (rec).", "",
         "| pass | mode | arc s | lead I A (rec) | lead U V (rec) | trail I A (rec) | trail U V (rec) | "
         "wfs lead/trail m/min (cmd) | gas lead/trail L/min (cmd) | travel mm/s (wts) | HI kJ/mm | interpass °C | video sync |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in metas:
        me, no = m["measured"], m["nominal"]
        rec = no["recommended"]
        tandem = m["electrodes"] == 2
        v = m.get("video", {}).get("sync", {})
        vs = (f"slot {v.get('arc_start_slot')}, {'verified' if v.get('verified') else 'UNVERIFIED'}"
              if v else "no video")
        L.append("| {} | {} | {:.1f} | {} ({}) | {} ({}) | {} | {} | {} | {} | {} ({}) | {} | {}→{} | {} |".format(
            m["sequence_name"], m.get("mode"), m["arc_on_s"], me["current_a"] and round(me["current_a"]),
            rec["current_lead_a"], me["voltage_v"] and round(me["voltage_v"], 1), rec["voltage_lead_v"],
            f"{round(me['current2_a'])} ({rec['current_trail_a']})" if tandem and me.get("current2_a") else "–",
            f"{round(me['voltage2_v'], 1)} ({rec['voltage_trail_v']})" if tandem and me.get("voltage2_v") else "–",
            f"{me['wfs_mpm']:.2f}/{me['wfs2_mpm']:.2f} ({no['wfs_command_leading']}/{no['wfs_command_trailing']})"
            if tandem and me.get("wfs2_mpm") is not None else f"{me['wfs_mpm']:.2f} ({no['wfs_command_leading']})",
            f"{me['gas_lpm']:.1f}/{me['gas2_lpm']:.1f} ({no['gasvalue_lpm']})" if tandem and me.get("gas2_lpm") is not None
            else f"{me['gas_lpm']:.1f} ({no['gasvalue_lpm']})",
            me["travel_speed_mmps"] and round(me["travel_speed_mmps"], 2), no["wts_mmps"],
            me["heat_input_kj_mm"] and round(me["heat_input_kj_mm"], 2),
            m.get("interpass_start_c"), m.get("interpass_max_c"), vs))
    L += ["", "## Per pass: deviations and anomalies", ""]
    for m in metas:
        me, no = m["measured"], m["nominal"]
        rec = no["recommended"]
        L.append(f"### {m['sequence_name']} ({m.get('mode')}, {m['pass_type']}, {m['electrodes']} wire)")
        L.append(f"- arc {m['arc_on_s']:.2f} s (robot arc_on window {m.get('robot_arc_on_s')} s rel. to electrical "
                 f"arc start); trail arc window {m.get('trail_arc_window_s')}")
        L.append(f"- lead: I {me['current_a']} A vs rec {rec['current_lead_a']} ({_pct(me['current_a'], rec['current_lead_a'])}), "
                 f"U {me['voltage_v']} V vs rec {rec['voltage_lead_v']} ({_pct(me['voltage_v'], rec['voltage_lead_v'])}); "
                 f"power source reads I {me.get('ps_current_a_lead')} A / U {me.get('ps_voltage_v_lead')} V")
        if m["electrodes"] == 2:
            L.append(f"- trail: I {me.get('current2_a')} A vs rec {rec['current_trail_a']} "
                     f"({_pct(me.get('current2_a'), rec['current_trail_a'])}), U {me.get('voltage2_v')} V vs rec "
                     f"{rec['voltage_trail_v']} ({_pct(me.get('voltage2_v'), rec['voltage_trail_v'])}); power source "
                     f"reads I {me.get('ps_current_a_trail')} A / U {me.get('ps_voltage_v_trail')} V")
        L.append(f"- wfs {me['wfs_mpm']} / {me.get('wfs2_mpm')} m/min vs cmd {no['wfs_command_leading']} / "
                 f"{no['wfs_command_trailing']}; gas {me['gas_lpm']} / {me.get('gas2_lpm')} L/min vs GASVALUE "
                 f"{no['gasvalue_lpm']} (power source flow meter {me.get('ps_gas_lpm_lead')})")
        L.append(f"- travel {me['travel_speed_mmps']} mm/s vs wts {no['wts_mmps']} ({m.get('travel_speed_mismatch_pct')} %), "
                 f"seam {me.get('seam_length_mm')} mm; heat input {me['heat_input_kj_mm']} kJ/mm (nominal "
                 f"{no['heat_input_kj_mm']})")
        L.append(f"- thermo_1 (interpass) {m.get('interpass_start_c')} → {m.get('interpass_max_c')} °C; t8/5 from "
                 f"thermo_2 {m.get('t85_measured_thermo2_s')} s vs experiment_list {no.get('t85_s')} s")
        v = m.get("video")
        if v:
            s = v["sync"]
            L.append(f"- video: {s['frames_decoded']} frames / {s['frame_slots']} slots @ {s['fps']} fps "
                     f"({s['dropped_slots']} dropped), {s.get('sync_pulses')} video_frame pulses @ "
                     f"{s.get('sync_pulse_hz')} Hz → pulses match frames: {s['pulses_match_frames']}; method "
                     f"{s['method']}: arc-start slot {s.get('arc_start_slot')}, arc-stop slot {s.get('arc_stop_slot')} "
                     f"(independent estimates {(s.get('optical') or {}).get('start_slot_independent')} / "
                     f"{(s.get('optical') or {}).get('stop_slot_independent')}) → video arc "
                     f"{s.get('arc_duration_video_s')} s vs signal {s.get('arc_duration_signal_s')} s; verified "
                     f"{s.get('verified')}; "
                     f"wrote {v['frames']} frames")
        elif m.get("video_error"):
            L.append(f"- video: FAILED ({m['video_error']})")
        an = m.get("anomalies") or []
        if not an:
            L.append("- anomalies: none found (no arc outage ≥ 20 ms, no current spike > 6 σ, no gas < 80 % GASVALUE ≥ 0.2 s)")
        for a in an:
            L.append(f"- anomaly {a['kind']} ({a['electrode']}) {a['t_start']}–{a['t_end']} s: {a['detail']}"
                     + (f", peak {a['peak_a']} A" if a.get("peak_a") else "")
                     + (f", min {a['min_lpm']} L/min" if a.get("min_lpm") is not None else ""))
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return path


if __name__ == "__main__":   # python -m collect.datasets.bam_tgmaw [download|convert] [WIDS]
    import argparse

    ap = argparse.ArgumentParser(description="BAM tandem-GMAW adapter")
    ap.add_argument("cmd", choices=["download", "convert", "all"])
    ap.add_argument("--wids", default=os.environ.get("BAM_WIDS"))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    raw = Path(a.data_dir) / "raw" / SOURCE_ID
    out = Path(a.data_dir) / "generated"
    if a.cmd in ("download", "all"):
        download(raw, a.wids, layers=a.layers)
    if a.cmd in ("convert", "all"):
        print("\n".join(convert(raw, out, a.wids, layers=a.layers, video=not a.no_video)))
