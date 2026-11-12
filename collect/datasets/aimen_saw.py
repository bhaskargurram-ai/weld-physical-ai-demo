"""AIMEN PENELOPE submerged-arc-welding open repository -> WeldWorld pass directories.

Source: Tapia Suárez E. A., Pérez Couñago I., Eike Precker C., Fernández Montenegro J. M., Ruiz C. —
"Submerged Arc Welding Open Repository" (PENELOPE, H2020 GA 958303), AIMEN. Zenodo,
DOI 10.5281/zenodo.15083865, CC BY 4.0. Contract: docs/REAL_DATA.md.

download(raw_dir, coupons=None, max_photos=2)
    Reads the 12.7 GB Zenodo zip REMOTELY (HTTP range requests) and extracts only, per coupon PP3..PP7:
    per-bead <PPx_NN>.hdf5, up to `max_photos` bead photos, 2. ndt_data/{defects_xlocation.xlsx, RX*.pdf,
    UT.pdf}, Design_Materials_Settings.pdf, README.txt.  Skips *.txt profilometer scans, *profilo*,
    *.nde and pre-welding folders.  Idempotent (existing file with the same size is skipped).
    Layout: raw_dir/<PPx>/<path inside the coupon folder>.

convert(raw_dir, out_dir, coupons=None)
    One pass dir per bead: out_dir/PPx_BNN_SAW/{meta.json, arc.parquet, travel.parquet, thermal.parquet,
    robot.parquet, events.jsonl, photos/*.jpg}.  No video (the dataset has none).

Conventions (stated in meta too):
  * electrode a = DC lead (CC EP), b = AC trail (tandem beads only).
  * arc start/stop: first/last data1 sample of the longest run where the 10 ms-mean lead current >
    ARC_ON_A (100 A) (bridging gaps < 0.5 s); cross-checked with process_status (data0) and reported.
    t = 0 at arc start; t0_ns = unix time of the arc-start data1 sample.
  * travel speed = d(x)/dt of the robot x (data0), 0.5 s centred moving-average then gradient.
  * heat input [kJ/mm] = (V_a·I_a + V_b·I_b) / v_travel / 1000, arc efficiency η = 1 (arc energy,
    EN 1011-1 k=1; SAW thermal efficiency ≈ 1.0 anyway), powers = 10 ms means of instantaneous V·I.
  * NDT indication time = (sample − arc_start_sample) / fs1 where sample = the spreadsheet
    "x_init/x_end [# sample]" (data1 index); x positions kept from the spreadsheet (mm along weld).
  * pass_type: beads 1–2 = root, the last 3 beads of the coupon (22–24) = cap, else fill.
"""
from __future__ import annotations

import io
import json
import math
import pickle
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from collect.datasets import common as C

SOURCE_ID = "aimen_saw"
TITLE = "Submerged Arc Welding Open Repository (PENELOPE, H2020 GA 958303)"
AUTHORS = ("Tapia Suárez E. A., Pérez Couñago I., Eike Precker C., Fernández Montenegro J. M., Ruiz C. "
           "(AIMEN)")
DOI = "10.5281/zenodo.15083865"
RECORD_URL = "https://zenodo.org/records/15083865"
ZIP_URL = "https://zenodo.org/api/records/15083865/files/ZENODO_Penelope_vs2.zip/content"
ZIP_ROOT = "ZENODO_Penelope/"
ALL_COUPONS = ("PP3", "PP4", "PP5", "PP6", "PP7")

ARC_ON_A = 100.0          # lead current threshold for "arc on"
TRAVEL_SMOOTH_S = 0.5
N_CAP = 3                 # last N beads of a coupon = cap
THICKNESS_MM = 52.0
MATERIAL = "SA-516 Gr.70N"
WIRE = "OE-SD3 (Ø3.2 mm)"
FLUX = "OP121TT"

DEFECT_CODE_DEFAULT = {1: "porosity", 2: "lack_of_fusion", 3: "slag_inclusion", 4: "metallic_inclusion",
                       5: "spatter", 6: "crack"}
# Design_Materials_Settings.pdf, "Settings parameters" table (bead ranges -> setting code row order)
BEAD_SETTING_DEFAULT = {1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 7, 11: 7,
                        12: 8, 13: 8, 14: 8, 15: 8, 16: 9, 17: 9, 18: 9, 19: 9, 20: 10, 21: 10,
                        22: 11, 23: 11, 24: 11}


def log(msg: str) -> None:
    print(f"[aimen_saw] {msg}", flush=True)


def _coupons(coupons) -> list[str]:
    if coupons is None or coupons == "" or coupons == "all":
        return list(ALL_COUPONS)
    if isinstance(coupons, str):
        coupons = [c for c in re.split(r"[,\s]+", coupons) if c]
    out = [c.strip().upper() for c in coupons]
    bad = [c for c in out if c not in ALL_COUPONS]
    if bad:
        raise ValueError(f"unknown AIMEN coupon(s) {bad}; valid: {ALL_COUPONS}")
    return out


# ================================================================== download

class _RetryRangeFile(io.RawIOBase):
    """Seekable file over HTTP range requests (same idea as collect/datasets/remotezip.py) + retries."""

    def __init__(self, url: str, retries: int = 6):
        self.retries = retries
        self.url = self._open(urllib.request.Request(url, method="HEAD")).geturl()
        self.size = int(self._open(urllib.request.Request(self.url, method="HEAD")).headers["Content-Length"])
        self.pos = 0
        self.bytes_read = 0

    def _open(self, req):
        delay = 2.0
        for i in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=120)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if i == self.retries - 1:
                    raise
                code = getattr(e, "code", None)
                if code is not None and code not in (429, 500, 502, 503, 504):
                    raise
                log(f"  http retry {i + 1} after {e}")
                time.sleep(delay)
                delay *= 2

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, buf):
        n = len(buf)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        for i in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                b = self._open(req).read()
                break
            except (OSError, urllib.error.URLError) as e:   # incomplete read / reset
                if i == self.retries - 1:
                    raise
                log(f"  range retry {i + 1} after {e}")
                time.sleep(2.0 * (i + 1))
        buf[:len(b)] = b
        self.pos += len(b)
        self.bytes_read += len(b)
        return len(b)


def open_remote_zip(url: str = ZIP_URL) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(_RetryRangeFile(url), buffer_size=4 << 20))


def select_members(names: list[str], coupons: list[str], max_photos: int = 2) -> list[str]:
    """Zip member names to extract for the given coupons (pure function, unit-tested)."""
    want: list[str] = []
    photos: dict[str, list[str]] = {}
    for n in names:
        if n.endswith("/") or not n.startswith(ZIP_ROOT):
            continue
        rel = n[len(ZIP_ROOT):]
        parts = rel.split("/")
        if parts[0] not in coupons:
            continue
        low = rel.lower()
        base = parts[-1]
        if "profilo" in low or low.endswith(".nde") or "pre_welding" in low or "pre-welding" in low:
            continue
        if len(parts) == 2 and base in ("README.txt", "Design_Materials_Settings.pdf"):
            want.append(n)
        elif len(parts) == 3 and parts[1] == "2. ndt_data" and (
                base == "defects_xlocation.xlsx" or re.match(r"(RX|UT)_?\.pdf$", base, re.I)):
            want.append(n)
        elif len(parts) == 4 and parts[1] == "1. process_data":
            bead = parts[2]
            if base == f"{bead}.hdf5":
                want.append(n)
            elif re.search(r"\.jpe?g$", base, re.I):
                photos.setdefault(bead, []).append(n)
    for bead, lst in photos.items():
        want.extend(sorted(lst)[:max_photos])
    return sorted(want)


def download(raw_dir: Path, coupons=None, max_photos: int = 2, url: str = ZIP_URL, workers: int = 6,
             **_opts) -> None:
    """Extract the needed members with `workers` parallel range-request streams (one zip handle each;
    Zenodo serves ~0.8 MB/s per connection)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    raw_dir = Path(raw_dir)
    cps = _coupons(coupons)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"opening remote zip (range requests) {url}")
    z = open_remote_zip(url)
    infos = {i.filename: i for i in z.infolist()}
    members = select_members(list(infos), cps, max_photos=max_photos)
    total = sum(infos[m].compress_size for m in members)
    todo = []
    for m in members:
        dst = raw_dir / m[len(ZIP_ROOT):]
        if not (dst.exists() and dst.stat().st_size == infos[m].file_size):
            todo.append(m)
    todo_b = sum(infos[m].compress_size for m in todo)
    log(f"{len(members)} members for {','.join(cps)} ({total / 1e6:.0f} MB compressed); "
        f"{len(members) - len(todo)} already present, {len(todo)} to fetch ({todo_b / 1e6:.0f} MB)")
    if not todo:
        return
    local = threading.local()
    handles = [z]
    lock = threading.Lock()
    state = {"done_b": 0, "n": 0}
    t0 = time.time()

    def zip_for_thread() -> zipfile.ZipFile:
        if getattr(local, "z", None) is None:
            with lock:
                local.z = handles.pop() if handles else None
            if local.z is None:
                local.z = open_remote_zip(url)
        return local.z

    def fetch(m: str) -> None:
        info = infos[m]
        dst = raw_dir / m[len(ZIP_ROOT):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        for attempt in range(4):
            try:
                with zip_for_thread().open(info) as src, open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
                break
            except (OSError, zipfile.BadZipFile, EOFError) as e:
                if attempt == 3:
                    raise
                log(f"  retry {m}: {e}")
                local.z = None
                time.sleep(3 * (attempt + 1))
        if tmp.stat().st_size != info.file_size:
            raise IOError(f"size mismatch for {m}")
        tmp.replace(dst)
        with lock:
            state["done_b"] += info.compress_size
            state["n"] += 1
            el = time.time() - t0
            log(f"  [{state['n']}/{len(todo)}] {m[len(ZIP_ROOT):]} {info.file_size / 1e6:.1f} MB "
                f"({state['done_b'] / 1e6:.0f}/{todo_b / 1e6:.0f} MB, {state['done_b'] / 1e6 / max(el, 1e-3):.1f} MB/s)")

    # biggest first so the long transfers overlap
    todo.sort(key=lambda m: -infos[m].compress_size)
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for m, fut in [(m, ex.submit(fetch, m)) for m in todo]:
            try:
                fut.result()
            except Exception as e:
                errors.append(f"{m}: {e}")
                log(f"  FAILED {m}: {e}")
    log(f"download done: {state['n']} fetched, {len(members) - len(todo)} already present, "
        f"{len(errors)} failed -> {raw_dir}")
    if errors:
        raise RuntimeError(f"{len(errors)} AIMEN members failed to download (re-run to resume)")


# ================================================================== parsing helpers

class _SafeUnpickler(pickle.Unpickler):
    """process_config attrs are protocol-0 pickled dicts of plain strings: forbid any class lookup."""

    def find_class(self, module, name):  # pragma: no cover - only hit by malicious files
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _attr_dict(v) -> dict:
    try:
        if isinstance(v, (bytes, np.bytes_)):
            return _SafeUnpickler(io.BytesIO(bytes(v))).load()
    except Exception:
        return {}
    return {}


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(str(v).replace(",", "."))
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def read_hdf5(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        d0 = f["data0"][:]
        d1 = f["data1"][:]
        cfg = _attr_dict(f["data0"].attrs.get("process_config"))
        fps0 = f["data0"].attrs.get("fps")
    md = (cfg or {}).get("metadata", {}) or {}
    return {"d0": d0, "d1": d1, "process_config": md,
            "fps0_attr": float(fps0) if fps0 is not None else None}


def read_xlsx(path: Path, coupon: str) -> dict:
    """-> {"defect_code": {int: name}, "settings": {code: {...}}, "rows": [ {defect, setting, bead, ...} ]}"""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    defect_code = dict(DEFECT_CODE_DEFAULT)
    if "defect_code" in wb.sheetnames:
        for r in wb["defect_code"].iter_rows(values_only=True):
            if r and isinstance(r[0], (int, float)) and r[1]:
                defect_code[int(r[0])] = _norm_defect(str(r[1]))
    settings: dict[int, dict] = {}
    if "welding_settings_code" in wb.sheetnames:
        for r in wb["welding_settings_code"].iter_rows(values_only=True):
            if not r or not isinstance(r[0], (int, float)):
                continue
            vals = list(r[1:8]) + [None] * 7
            settings[int(r[0])] = {"Vdc": _f(vals[0]), "Idc": _f(vals[1]), "Vac": _f(vals[2]),
                                   "Iac": _f(vals[3]), "wfs_dc": _f(vals[4]), "wfs_ac": _f(vals[5]),
                                   "ts_cm_min": _f(vals[6])}
    rows = []
    sheet = coupon if coupon in wb.sheetnames else next((s for s in wb.sheetnames if s.upper() == coupon), None)
    if sheet:
        it = wb[sheet].iter_rows(values_only=True)
        header = [str(h).strip().lower() if h is not None else "" for h in next(it, [])]

        def col(*keys):
            for i, h in enumerate(header):
                if all(k in h for k in keys):
                    return i
            return None

        ci = {"defect": col("defect"), "setting": col("welding_setting"), "bead": col("bead"),
              "x0": col("x_init", "mm"), "x1": col("x_end", "mm"),
              "s0": col("x_init", "sample"), "s1": col("x_end", "sample")}
        for r in it:
            if not r or all(v is None for v in r):
                continue
            g = {k: (r[i] if i is not None and i < len(r) else None) for k, i in ci.items()}
            if g["bead"] is None or g["defect"] is None:
                continue
            rows.append({"defect_code": int(g["defect"]), "setting": int(g["setting"]) if g["setting"] else None,
                         "bead": int(g["bead"]), "x0_mm": _f(g["x0"]), "x1_mm": _f(g["x1"]),
                         "s0": int(g["s0"]) if g["s0"] is not None else None,
                         "s1": int(g["s1"]) if g["s1"] is not None else None})
    wb.close()
    return {"defect_code": defect_code, "settings": settings, "rows": rows}


def _norm_defect(s: str) -> str:
    s = s.strip().lower()
    return {"porosity": "porosity", "lack of fusion": "lack_of_fusion", "slag inclusion": "slag_inclusion",
            "metallic inclusion": "metallic_inclusion", "projections": "spatter", "cracks": "crack",
            "crack": "crack"}.get(s, re.sub(r"\W+", "_", s).strip("_"))


def parse_design_pdf(path: Path) -> dict[int, dict]:
    """Settings table of Design_Materials_Settings.pdf -> {bead: {"single": bool, "I_lead", ...}}."""
    from pypdf import PdfReader
    import logging

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return parse_design_text(text)


def parse_design_text(text: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    num = r"(\d+(?:,\d+)?)"
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(\d+)(?:-(\d+))?\s+(single wire|Tandem)\b(.*)$", line, re.I)
        if not m:
            continue
        b0, b1 = int(m.group(1)), int(m.group(2) or m.group(1))
        rest = m.group(4)
        single = m.group(3).lower().startswith("single")
        if single:
            mm = re.search(num + r"\s+(\d+)\s+(\d+)\s+(\d+)\s*$", rest)
            if not mm:
                continue
            rec = {"single": True, "wfs_lead": _f(mm.group(1)), "I_lead": _f(mm.group(2)),
                   "V_lead": _f(mm.group(3)), "ts_cm_min": _f(mm.group(4))}
        else:
            mm = re.search(r"F:\s*" + num + r"\s*B:\s*" + num + r"\s*F:\s*(\d+)\s*B:\s*(\d+)\s*F:\s*(\d+)\s*B:\s*(\d+)"
                           r"\s+(\d+)\s*$", rest)
            if not mm:
                continue
            rec = {"single": False, "wfs_lead": _f(mm.group(1)), "wfs_trail": _f(mm.group(2)),
                   "I_lead": _f(mm.group(3)), "I_trail": _f(mm.group(4)), "V_lead": _f(mm.group(5)),
                   "V_trail": _f(mm.group(6)), "ts_cm_min": _f(mm.group(7))}
        rec["bead_range"] = f"{b0}-{b1}" if b1 != b0 else str(b0)
        for b in range(b0, b1 + 1):
            out[b] = dict(rec)
    return out


def _setting_matches(design: dict, s: dict) -> bool:
    def eq(a, b):
        return (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-6)
    return (eq(design.get("I_lead"), s.get("Idc")) and eq(design.get("V_lead"), s.get("Vdc"))
            and eq(design.get("ts_cm_min"), s.get("ts_cm_min")) and eq(design.get("I_trail"), s.get("Iac"))
            and eq(design.get("V_trail"), s.get("Vac")))


def bead_setting_map(n_beads: int, xls: dict, design: dict[int, dict]) -> tuple[dict[int, int], list[str]]:
    """bead -> welding_settings_code. Priority: per-coupon sheet rows, then the Design PDF row matched to the
    settings-code table, then the documented default mapping. Returns (map, notes)."""
    notes: list[str] = []
    sheet = {}
    for r in xls["rows"]:
        if r["setting"] is not None:
            if r["bead"] in sheet and sheet[r["bead"]] != r["setting"]:
                notes.append(f"bead {r['bead']}: sheet rows disagree on setting ({sheet[r['bead']]} vs {r['setting']})")
            sheet.setdefault(r["bead"], r["setting"])
    out: dict[int, int] = {}
    for b in range(1, n_beads + 1):
        d_code = None
        if b in design and xls["settings"]:
            cands = [c for c, s in xls["settings"].items() if _setting_matches(design[b], s)]
            if len(cands) == 1:
                d_code = cands[0]
        default = BEAD_SETTING_DEFAULT.get(b)
        if b in sheet:
            out[b] = sheet[b]
            if d_code is not None and d_code != sheet[b]:
                notes.append(f"bead {b}: sheet setting {sheet[b]} != design PDF setting {d_code} "
                             f"(resolved by measured lead current in convert)")
        elif d_code is not None:
            out[b] = d_code
        elif default is not None:
            out[b] = default
            notes.append(f"bead {b}: setting from default design mapping")
        if d_code is not None and default is not None and d_code != default:
            notes.append(f"bead {b}: design PDF code {d_code} != built-in default {default}")
    return out, notes


def parse_rx_verdicts(path: Path) -> list[str]:
    from pypdf import PdfReader
    import logging

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    except Exception:
        return []
    v = []
    for w in re.findall(r"\b(Rechazado|Aceptado|Rejected|Accepted)\b", text):
        v.append("reject" if w.lower().startswith(("rech", "rej")) else "accept")
    return v


def pass_type_for(bead: int, n_beads: int) -> str:
    if bead <= 2:
        return "root"
    if bead > n_beads - N_CAP:
        return "cap"
    return "fill"


# ================================================================== signal processing

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def arc_window(t1: np.ndarray, i_lead: np.ndarray, fs1: float, thr: float = ARC_ON_A,
               bridge_s: float = 0.5) -> tuple[int, int]:
    """(first, last) data1 sample index of the main arc-on interval (lead current 10 ms mean > thr)."""
    blk = max(int(round(fs1 * 0.01)), 1)
    nb = len(i_lead) // blk
    if nb == 0:
        raise ValueError("no data1 samples")
    m = np.abs(i_lead[:nb * blk]).reshape(nb, blk).mean(axis=1) > thr
    runs = _runs(m)
    if not runs:
        raise ValueError(f"lead current never exceeds {thr} A")
    gap = int(bridge_s / 0.01)
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    s, e = max(merged, key=lambda r: r[1] - r[0])
    i0 = s * blk
    i1 = min(e * blk, len(i_lead)) - 1
    # refine to the first/last raw sample above threshold inside the edge blocks
    w0 = np.flatnonzero(np.abs(i_lead[i0:i0 + blk]) > thr)
    if w0.size:
        i0 += int(w0[0])
    w1 = np.flatnonzero(np.abs(i_lead[max(i1 - blk + 1, 0):i1 + 1]) > thr)
    if w1.size:
        i1 = max(i1 - blk + 1, 0) + int(w1[-1])
    return int(i0), int(i1)


def sample_to_time(sample, arc_start_sample: int, fs1: float, t1: Optional[np.ndarray] = None):
    """data1 sample index -> seconds from arc start. Uses the recorded data1 timestamps when the index is
    inside the table (exact), else linear extrapolation at the measured data1 rate fs1."""
    s = np.asarray(sample, dtype=np.float64)
    out = (s - arc_start_sample) / fs1
    if t1 is not None and len(t1):
        si = s.astype(np.int64)
        inside = (si >= 0) & (si < len(t1))
        out = np.where(inside, t1[np.clip(si, 0, len(t1) - 1)] - t1[arc_start_sample], out)
    return out


def travel_speed(t0: np.ndarray, x: np.ndarray, smooth_s: float = TRAVEL_SMOOTH_S) -> np.ndarray:
    """|dx/dt| [mm/s] on the data0 grid: centred moving-average of x over smooth_s then gradient."""
    if len(t0) < 3:
        return np.zeros_like(x, dtype=np.float64)
    dt = np.median(np.diff(t0))
    w = max(int(round(smooth_s / dt)), 1) | 1
    pad = w // 2
    xp = np.pad(x.astype(np.float64), pad, mode="edge")
    xs = np.convolve(xp, np.ones(w) / w, mode="valid")
    v = np.gradient(xs, t0)
    return np.abs(v)


# ================================================================== convert one bead

def convert_bead(h5_path: Path, coupon: str, bead: int, n_beads: int, xls: dict, setting_code: Optional[int],
                 out_dir: Path, design_row: Optional[dict] = None, photos: Optional[list[Path]] = None,
                 raw_dir: Optional[Path] = None, coupon_rx: Optional[list[str]] = None,
                 report_files: Optional[list[str]] = None, setting_resolution: Optional[str] = None) -> dict:
    H = read_hdf5(h5_path)
    d0, d1, md = H["d0"], H["d1"], H["process_config"]
    t1 = d1["timestamp"].astype(np.float64)
    fs1 = (len(t1) - 1) / (t1[-1] - t1[0]) if len(t1) > 1 else 5000.0
    t0s = d0["timestamp"].astype(np.float64)
    fs0 = (len(t0s) - 1) / (t0s[-1] - t0s[0]) if len(t0s) > 1 else float("nan")
    ia = d1["current_a"].astype(np.float64)
    i_start, i_stop = arc_window(t1, ia, fs1)
    t_arc0 = float(t1[i_start])
    duration = float(t1[i_stop] - t_arc0)
    t0_ns = int(round(t_arc0 * 1e9))

    # process_status cross-check (data0)
    ps = d0["process_status"].astype(np.float64)
    ps_on = np.flatnonzero(ps > 0)
    ps_window = [float(t0s[ps_on[0]] - t_arc0), float(t0s[ps_on[-1]] - t_arc0)] if ps_on.size else None

    # --- nominal settings
    st = xls["settings"].get(setting_code, {}) if setting_code is not None else {}
    tandem = st.get("Iac") is not None if st else (design_row is not None and not design_row.get("single", True))
    electrodes = 2 if tandem else 1
    ts_mmps = st.get("ts_cm_min") * 10.0 / 60.0 if st.get("ts_cm_min") else None
    nominal = {"current_a": st.get("Idc"), "voltage_v": st.get("Vdc"), "wfs_mpm": st.get("wfs_dc"),
               "travel_speed_mmps": ts_mmps}
    if tandem:
        nominal.update({"current2_a": st.get("Iac"), "voltage2_v": st.get("Vac"), "wfs2_mpm": st.get("wfs_ac")})

    # --- arc @100 Hz (10 ms means of 5 kHz)
    rel1 = t1 - t_arc0
    cols1 = {"current_a": ia, "voltage_v": d1["voltage_a"].astype(np.float64),
             "power_a_w": ia * d1["voltage_a"].astype(np.float64)}
    if tandem:
        ib = d1["current_b"].astype(np.float64)
        vb = d1["voltage_b"].astype(np.float64)
        cols1.update({"current2_a": ib, "voltage2_v": vb, "power_b_w": ib * vb, "current2_rms_a": ib * ib})
    g1 = C.grid_mean(rel1, cols1, duration, fill="interp")
    rel0 = t0s - t_arc0
    cols0 = {"wfs_mpm": d0["wire_speed_a"].astype(np.float64)}
    if tandem:
        cols0["wfs2_mpm"] = d0["wire_speed_b"].astype(np.float64)
    v_tr = travel_speed(t0s, d0["x"].astype(np.float64))
    cols0.update({"travel_speed_mmps": v_tr, "x": d0["x"].astype(np.float64), "y": d0["y"].astype(np.float64),
                  "z": d0["z"].astype(np.float64)})
    for k in ("pyrometer_L", "pyrometer_R", "pyrometer_back"):
        cols0[k] = d0[k].astype(np.float64)
    g0 = C.grid_mean(rel0, cols0, duration, fill="interp")

    arc = pd.DataFrame({"t_s": g1["t_s"], "current_a": g1["current_a"], "voltage_v": g1["voltage_v"],
                        "wfs_mpm": g0["wfs_mpm"], "gas_lpm": np.nan})
    if tandem:
        arc["current2_a"] = g1["current2_a"]
        arc["voltage2_v"] = g1["voltage2_v"]
        arc["wfs2_mpm"] = g0["wfs2_mpm"]
        arc["current2_rms_a"] = np.sqrt(np.maximum(g1["current2_rms_a"], 0.0))
    power = g1["power_a_w"].to_numpy() + (g1["power_b_w"].to_numpy() if tandem else 0.0)
    x_robot = g0["x"].to_numpy()
    travel = pd.DataFrame({"t_s": g1["t_s"], "travel_speed_mmps": g0["travel_speed_mmps"],
                           "position_mm": x_robot - x_robot[0],
                           "heat_input_kj_mm": C.heat_input_kj_mm(power, g0["travel_speed_mmps"].to_numpy())})
    # avoid absurd HI where the carriage is (nearly) stopped
    travel.loc[travel["travel_speed_mmps"] < 0.5, "heat_input_kj_mm"] = np.nan
    thermal = pd.DataFrame({"t_s": g1["t_s"],
                            "interpass_c": np.maximum(g0["pyrometer_L"], g0["pyrometer_R"]),
                            "temp_left_c": g0["pyrometer_L"], "temp_right_c": g0["pyrometer_R"],
                            "temp_back_c": g0["pyrometer_back"]})
    robot = pd.DataFrame({"t_s": g1["t_s"], "tcp_x": g0["x"], "tcp_y": g0["y"], "tcp_z": g0["z"]})

    # --- NDT indications
    # robot-x <-> spreadsheet-x offset: x_robot(at idx_data1 == s0) - x_init_mm
    idx01 = d0["idx_data1"].astype(np.int64)
    inds, offsets = [], []
    for r in [r for r in xls["rows"] if r["bead"] == bead]:
        s0, s1 = r["s0"], r["s1"]
        if s0 is None or s1 is None:
            continue
        a, b = sorted((s0, s1))
        ts0, ts1 = (float(v) for v in sample_to_time([a, b], i_start, fs1, t1))
        xa, xb = (r["x0_mm"], r["x1_mm"]) if s0 <= s1 else (r["x1_mm"], r["x0_mm"])
        j = int(np.clip(np.searchsorted(idx01, s0), 0, len(idx01) - 1))
        if r["x0_mm"] is not None and len(idx01):
            offsets.append(float(d0["x"][j]) - r["x0_mm"])
        inds.append({"type": xls["defect_code"].get(r["defect_code"], f"code_{r['defect_code']}"),
                     "defect_code": r["defect_code"], "t_start": round(ts0, 4), "t_end": round(ts1, 4),
                     "x_start_mm": xa, "x_end_mm": xb, "sample_start": a, "sample_end": b,
                     "outside_arc": bool(ts1 < 0 or ts0 > duration)})
    inds.sort(key=lambda i: i["t_start"])

    # --- measured means over the steady part (exclude first/last 2 s)
    steady = (arc["t_s"] > 2.0) & (arc["t_s"] < duration - 2.0)
    if steady.sum() < 10:
        steady = arc["t_s"] >= 0

    def mean(df, c):
        v = df.loc[steady, c].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        return round(float(v.mean()), 3) if v.size else None

    measured = {"current_a": mean(arc, "current_a"), "voltage_v": mean(arc, "voltage_v"),
                "wfs_mpm": mean(arc, "wfs_mpm"), "travel_speed_mmps": mean(travel, "travel_speed_mmps"),
                "heat_input_kj_mm": mean(travel, "heat_input_kj_mm")}
    if tandem:
        measured.update({"current2_a": mean(arc, "current2_rms_a"), "voltage2_v": None,
                         "wfs2_mpm": mean(arc, "wfs2_mpm")})
        vb_rms = np.sqrt(np.nanmean((d1["voltage_b"].astype(np.float64)[i_start:i_stop + 1]) ** 2))
        measured["voltage2_v"] = round(float(vb_rms), 3)
        measured["current2_mean_a"] = mean(arc, "current2_a")
    ts_meas = measured["travel_speed_mmps"]
    ts_mismatch = (round((ts_meas / ts_mmps - 1) * 100, 1) if ts_meas and ts_mmps else None)

    # nominal HI (arc energy, η=1)
    hi_nom = None
    if st and ts_mmps:
        p_nom = (st["Idc"] or 0) * (st["Vdc"] or 0) + ((st["Iac"] or 0) * (st["Vac"] or 0) if tandem else 0)
        hi_nom = p_nom / ts_mmps / 1000.0

    seq = f"{coupon}_B{bead:02d}_SAW"
    pass_type = pass_type_for(bead, n_beads)
    wps_id = f"AIMEN-SAW-S{setting_code:02d}" if setting_code is not None else "AIMEN-SAW-UNKNOWN"
    rel_files = []
    if raw_dir is not None:
        rel_files.append(h5_path.relative_to(raw_dir).as_posix())
        rel_files += [f"{coupon}/2. ndt_data/defects_xlocation.xlsx", f"{coupon}/Design_Materials_Settings.pdf"]
        rel_files += [f"{coupon}/2. ndt_data/{n}" for n in (report_files or [])]
        rel_files += [p.relative_to(raw_dir).as_posix() for p in (photos or [])]
    src = C.source_block(SOURCE_ID, TITLE, AUTHORS, DOI, RECORD_URL, rel_files)
    meta = C.meta_skeleton(seq, coupon, "BW", bead, pass_type, "SAW", source=src, wps_id=wps_id,
                           position="1G", material=MATERIAL, cell_id="AIMEN-SAW (2x Miller Summit Arc 1250)",
                           thickness_mm=THICKNESS_MM, electrodes=electrodes, t0_ns=t0_ns)
    meta["duration_s"] = round(duration, 3)
    meta["arc_on_s"] = round(duration, 3)
    meta["wps"] = C.wps_from_nominal(wps_id, "SAW", nominal, gas_lpm=None,
                                     heat_input_kj_mm=[round(hi_nom * 0.9, 4), round(hi_nom * 1.1, 4)]
                                     if hi_nom else None)
    if tandem:
        for k in ("current2_a", "voltage2_v", "wfs2_mpm"):
            if nominal.get(k):
                meta["wps"][k] = [round(nominal[k] * 0.9, 4), round(nominal[k] * 1.1, 4)]
    meta["wps"]["note"] = "gas_lpm [0,0]: SAW is flux-shielded (no shielding gas)"
    meta["setpoints"] = {k: v for k, v in nominal.items() if v is not None}
    meta["setpoints"]["gas_lpm"] = None
    meta["nominal"] = {"setting_code": setting_code, "ts_cm_min": st.get("ts_cm_min"),
                       "heat_input_kj_mm": round(hi_nom, 4) if hi_nom else None,
                       "polarity_lead": "DC EP (CC EP)", "polarity_trail": "AC 50/50" if tandem else None,
                       "design_pdf_row": design_row, "hdf5_process_config": {
                           k: md.get(k) for k in ("02 Coupon", "05.1 Lead diameter wire", "06 Flux",
                                                  "20 Travel speed (m/min)", "21.1 Voltage (V)",
                                                  "21.2 Voltage (V)", "22.1 Current (A)", "22.2 Current (A)",
                                                  "23.1 Current and polarity", "23.2 Current and polarity",
                                                  "26 CTWD (mm)")}}
    meta["measured"] = measured
    meta["travel_speed_mismatch_pct"] = ts_mismatch

    def dev(k_meas, k_nom):
        a, b = measured.get(k_meas), nominal.get(k_nom)
        return round((a / b - 1) * 100, 1) if a is not None and b else None

    nc = {"current_dev_pct": dev("current_a", "current_a"), "voltage_dev_pct": dev("voltage_v", "voltage_v"),
          "wfs_dev_pct": dev("wfs_mpm", "wfs_mpm"), "travel_dev_pct": ts_mismatch}
    if tandem:
        nc.update({"current2_dev_pct": dev("current2_a", "current2_a"),
                   "voltage2_dev_pct": dev("voltage2_v", "voltage2_v")})
    nc["outside_10pct"] = sorted(k for k, v in nc.items() if v is not None and abs(v) > 10)
    nc["setting_source"] = setting_resolution or "per-coupon sheet / design PDF (consistent)"
    meta["nominal_check"] = nc
    meta["heat_input_mean_kj_mm"] = measured["heat_input_kj_mm"]
    meta["current_mean_a"] = measured["current_a"]
    meta["voltage_mean_v"] = measured["voltage_v"]
    meta["travel_mean_mmps"] = measured["travel_speed_mmps"]
    ip = thermal["interpass_c"].to_numpy()
    meta["interpass_start_c"] = round(float(np.nanmean(ip[:50])), 1) if ip.size else None
    meta["interpass_max_c"] = round(float(np.nanmax(ip)), 1) if ip.size else None
    meta["consumables"] = {"wire": WIRE, "flux": md.get("06 Flux") or FLUX}
    meta["signal_notes"] = {
        "electrodes": "a = DC lead (CC EP); b = AC trail (tandem beads only)",
        "arc_window": f"lead current 10 ms mean > {ARC_ON_A:.0f} A (gaps < 0.5 s bridged); t=0 at arc start",
        "process_status_window_s": ps_window,
        "travel_speed": f"|d(x)/dt| of robot x (data0), {TRAVEL_SMOOTH_S} s centred moving average",
        "heat_input": "HI [kJ/mm] = (V_a·I_a + V_b·I_b) / v_travel[mm/s] / 1000, η = 1 (arc energy); "
                      "powers = 10 ms means of instantaneous V·I; NaN where v_travel < 0.5 mm/s",
        "interpass_c": "max(pyrometer_L, pyrometer_R); pyrometer_back reads the pool/flux region (temp_back_c)",
        "current2_a": "AC trail: 10 ms mean of signed current (≈0); current2_rms_a = 10 ms RMS",
        "rates": {"data1_hz": round(fs1, 2), "data0_hz": round(fs0, 2), "data0_fps_attr": H["fps0_attr"]},
        "x_offset_robot_minus_sheet_mm": round(float(np.median(offsets)), 2) if offsets else None,
        "pass_type_rule": f"beads 1-2 root, last {N_CAP} beads of the coupon cap, else fill",
    }
    meta["rates_hz"] = {"arc": 100, "travel": 100, "thermal": 100, "robot": 100}
    meta["ndt"].update({"methods": ["RT", "UT"], "inspected": True,
                        "report_files": list(report_files or []),
                        "coupon_rt_verdicts": coupon_rx or [],
                        "time_mapping": "t = data1.timestamp[sample] − data1.timestamp[arc-start sample] "
                                        "(sample = spreadsheet x_init/x_end [# sample], a data1 index)"})
    C.set_indications(meta, inds, inspected=True)

    # --- write
    d = Path(out_dir) / seq
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    C.write_parquet(C.add_t_ns(arc, t0_ns), d / "arc.parquet")
    C.write_parquet(C.add_t_ns(travel, t0_ns), d / "travel.parquet")
    C.write_parquet(C.add_t_ns(thermal, t0_ns), d / "thermal.parquet")
    C.write_parquet(C.add_t_ns(robot, t0_ns), d / "robot.parquet")
    events = [C.event(t0_ns, 0.0, "arc_start", process="SAW", electrodes=electrodes),
              C.event(t0_ns, duration, "arc_stop")] + C.ndt_events(t0_ns, inds)
    C.write_events(d / "events.jsonl", events)
    for k, p in enumerate(photos or [], 1):
        rel = f"photos/{coupon}_{bead:02d}_{k}.jpg"
        try:
            C.save_photo(p, d / rel)
            meta["photos"].append(rel)
        except Exception as e:  # corrupt photo should not kill the pass
            log(f"  photo {p.name}: {e}")
    C.write_meta(d / "meta.json", meta)
    return meta


# ================================================================== convert

def resolve_setting(h5: Path, candidates: list[int], settings: dict) -> tuple[int, str]:
    """Sheet and design PDF disagree: keep the candidate whose nominal lead current is closest to the
    measured mean lead current (arc on)."""
    import h5py

    with h5py.File(h5, "r") as f:
        ia = f["data1"]["current_a"].astype(np.float64)
    meas = float(ia[ia > ARC_ON_A].mean()) if (ia > ARC_ON_A).any() else float("nan")
    scored = [(abs((settings.get(c, {}).get("Idc") or np.inf) - meas), c) for c in candidates]
    best = min(scored)[1]
    return best, (f"sheet/design conflict {candidates} -> setting {best} "
                  f"(measured lead current {meas:.0f} A closest to its nominal {settings.get(best, {}).get('Idc')} A)")

def _find_report_files(ndt_dir: Path) -> list[str]:
    if not ndt_dir.exists():
        return []
    return sorted(p.name for p in ndt_dir.iterdir() if re.match(r"(RX|UT)_?\.pdf$", p.name, re.I))


def convert(raw_dir: Path, out_dir: Path, coupons=None, **_opts) -> list[str]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for cp in _coupons(coupons):
        cdir = raw_dir / cp
        h5s = sorted((cdir / "1. process_data").glob(f"{cp}_*/{cp}_[0-9][0-9].hdf5"))
        if not h5s:
            log(f"{cp}: no hdf5 under {cdir} (run download first) — skipped")
            continue
        xlsx = cdir / "2. ndt_data" / "defects_xlocation.xlsx"
        xls = read_xlsx(xlsx, cp) if xlsx.exists() else {"defect_code": dict(DEFECT_CODE_DEFAULT),
                                                         "settings": {}, "rows": []}
        if not xls["settings"]:
            log(f"{cp}: WARNING no welding_settings_code sheet; nominal settings unavailable")
        dpdf = cdir / "Design_Materials_Settings.pdf"
        design = parse_design_pdf(dpdf) if dpdf.exists() else {}
        beads = sorted(int(p.stem.split("_")[1]) for p in h5s)
        n_beads = max(max(beads), 24)
        smap, notes = bead_setting_map(n_beads, xls, design)
        conflicts = {}
        for n in notes:
            mm = re.match(r"bead (\d+): sheet setting (\d+) != design PDF setting (\d+)", n)
            if mm:
                conflicts[int(mm.group(1))] = [int(mm.group(2)), int(mm.group(3))]
        for n in notes:
            log(f"{cp}: {n}")
        report_files = _find_report_files(cdir / "2. ndt_data")
        rx = next((cdir / "2. ndt_data" / f for f in report_files if f.upper().startswith("RX")), None)
        rx_verdicts = parse_rx_verdicts(rx) if rx else []
        metas = []
        for h5 in h5s:
            bead = int(h5.stem.split("_")[1])
            bdir = h5.parent
            photos = sorted(p for p in bdir.iterdir() if re.search(r"\.jpe?g$", p.name, re.I))
            try:
                code, resolution = smap.get(bead), None
                if bead in conflicts:
                    code, resolution = resolve_setting(h5, conflicts[bead], xls["settings"])
                    log(f"{cp} bead {bead}: {resolution}")
                m = convert_bead(h5, cp, bead, n_beads, xls, code, out_dir, setting_resolution=resolution,
                                 design_row=design.get(bead), photos=photos, raw_dir=raw_dir,
                                 coupon_rx=rx_verdicts, report_files=report_files)
            except Exception as e:
                log(f"{cp} bead {bead}: FAILED {type(e).__name__}: {e}")
                continue
            metas.append(m)
            written.append(m["sequence_name"])
            ms = m["measured"]
            log(f"{m['sequence_name']:<14} {m['pass_type']:<4} e={m['electrodes']} {m['duration_s']:6.1f}s "
                f"I={ms['current_a']:.0f}/{m['setpoints'].get('current_a')} "
                f"U={ms['voltage_v']:.1f}/{m['setpoints'].get('voltage_v')} "
                f"v={ms['travel_speed_mmps']:.2f}/{m['setpoints'].get('travel_speed_mmps', float('nan')):.2f}mm/s "
                f"ind={len(m['ndt']['indications'])} {m['defect_types']}")
        # coupon-level RT verdict
        joint = ("reject" if "reject" in rx_verdicts else "accept") if rx_verdicts else \
            ("indications" if any(m["has_defect"] for m in metas) else "no_indications")
        for m in metas:
            m["joint_rt_result"] = joint
            C.write_meta(out_dir / m["sequence_name"] / "meta.json", m)
        orphan = sorted({r["bead"] for r in xls["rows"]} - set(beads))
        if orphan:
            log(f"{cp}: NDT rows for beads without hdf5: {orphan}")
    if written and _opts.get("report", True):
        try:
            analyze(out_dir)
        except Exception as e:  # analysis must never break the conversion
            log(f"analysis failed: {type(e).__name__}: {e}")
    return written


# ================================================================== honest signal-vs-defect analysis

ANALYSIS_SIGNALS = (("current_a", "arc", "lead current [A]"), ("voltage_v", "arc", "lead voltage [V]"),
                    ("wfs_mpm", "arc", "lead wire speed [m/min]"), ("current2_rms_a", "arc", "trail current RMS [A]"),
                    ("wfs2_mpm", "arc", "trail wire speed [m/min]"),
                    ("travel_speed_mmps", "travel", "travel speed [mm/s]"),
                    ("interpass_c", "thermal", "plate temp max(L,R) [°C]"),
                    ("temp_back_c", "thermal", "pyrometer back [°C]"))
MIN_WIN_S = 0.2
EDGE_S = 2.0
GUARD_S = 1.0
N_NULL = 300


def _load_pass(d: Path) -> dict:
    m = json.loads((d / "meta.json").read_text())
    dfs = {}
    for k in ("arc", "travel", "thermal"):
        p = d / f"{k}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["t"] = (df["t_ns"] - m["t0_ns"]) / 1e9
            dfs[k] = df
    return {"meta": m, **dfs}


def _win_stats(t: np.ndarray, v: np.ndarray, a: float, b: float) -> tuple[float, float]:
    sel = (t >= a) & (t < b) & np.isfinite(v)
    if sel.sum() < 2:
        return float("nan"), float("nan")
    return float(v[sel].mean()), float(v[sel].std())


def analyze(out_dir: Path, seqs: Optional[list[str]] = None, report_path: Optional[Path] = None,
            seed: int = 0) -> str:
    """Compare signals inside NDT indication windows vs the same bead outside them. Writes markdown."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    dirs = sorted(p.parent for p in out_dir.glob("*_SAW/meta.json"))
    if seqs is not None:
        keep = set(seqs)
        dirs = [d for d in dirs if d.name in keep]
    passes = []
    for d in dirs:
        try:
            P = _load_pass(d)
        except Exception:
            continue
        if (P["meta"].get("source") or {}).get("id") == SOURCE_ID:
            passes.append(P)
    # per type: rows of per-indication results
    per_type: dict[str, dict] = {}
    whole_bead: dict[str, list[str]] = {}
    for P in passes:
        m = P["meta"]
        dur = m["duration_s"]
        inds = [i for i in m["ndt"]["indications"] if not i.get("outside_arc")]
        if not inds:
            continue
        # clean reference = steady part, away from every indication (+guard)
        t = P["arc"]["t"].to_numpy()
        clean = (t > EDGE_S) & (t < dur - EDGE_S)
        for i in inds:
            clean &= ~((t >= i["t_start"] - GUARD_S) & (t <= i["t_end"] + GUARD_S))
        for i in inds:
            typ = i["type"]
            a, b = max(i["t_start"], 0.0), min(i["t_end"], dur)
            if b - a < MIN_WIN_S:
                c = 0.5 * (a + b)
                a, b = c - MIN_WIN_S / 2, c + MIN_WIN_S / 2
            T = per_type.setdefault(typ, {"n": 0, "beads": set(), "dur": 0.0, "sig": {}})
            if (b - a) > 0.8 * dur or clean.sum() < 5 * 100:
                whole_bead.setdefault(typ, []).append(f"{m['sequence_name']} ({i['x_start_mm']}–{i['x_end_mm']} mm)")
                continue
            T["n"] += 1
            T["beads"].add(m["sequence_name"])
            T["dur"] += b - a
            L = b - a
            # candidate null-window starts: whole window inside the clean mask
            dt = 0.01
            nwin = max(int(round(L / dt)), 1)
            cm = clean.astype(np.int32)
            cs = np.concatenate([[0], np.cumsum(cm)])
            ok_start = np.flatnonzero((cs[nwin:] - cs[:-nwin]) == nwin) if len(cm) > nwin else np.array([], int)
            starts = t[ok_start] if ok_start.size else np.array([])
            if starts.size > N_NULL:
                starts = rng.choice(starts, N_NULL, replace=False)
            for sig, tab, _lbl in ANALYSIS_SIGNALS:
                df = P.get(tab)
                if df is None or sig not in df:
                    continue
                tt, vv = df["t"].to_numpy(), df[sig].to_numpy(np.float64)
                mi, si = _win_stats(tt, vv, a, b)
                cl = clean if tab == "arc" else np.interp(tt, t, clean.astype(float)) > 0.5
                vo = vv[cl & np.isfinite(vv)]
                if not np.isfinite(mi) or vo.size < 10:
                    continue
                nm = np.array([_win_stats(tt, vv, s0, s0 + L)[0] for s0 in starts]) if starts.size else np.array([])
                ns = np.array([_win_stats(tt, vv, s0, s0 + L)[1] for s0 in starts]) if starts.size else np.array([])
                nm, ns = nm[np.isfinite(nm)], ns[np.isfinite(ns)]
                z_m = (mi - nm.mean()) / nm.std() if nm.size > 20 and nm.std() > 0 else float("nan")
                z_s = (si - ns.mean()) / ns.std() if ns.size > 20 and ns.std() > 0 else float("nan")
                S = T["sig"].setdefault(sig, {"in_m": [], "out_m": [], "in_s": [], "out_s": [], "zm": [], "zs": []})
                S["in_m"].append(mi)
                S["out_m"].append(float(vo.mean()))
                S["in_s"].append(si)
                S["out_s"].append(float(np.mean(ns)) if ns.size else float("nan"))
                S["zm"].append(z_m)
                S["zs"].append(z_s)

    n_ind = sum(len(P["meta"]["ndt"]["indications"]) for P in passes)
    lines = ["# AIMEN SAW — do NDT indications show up in the process signals?", "",
             f"Passes analysed: {len(passes)}; NDT indications: {n_ind}. Source: AIMEN PENELOPE SAW open "
             f"repository (DOI {DOI}, CC BY 4.0).", "",
             "**Method.** For every RT/UT indication (spreadsheet sample index → seconds from arc start) the "
             f"100 Hz signal inside the indication window (min {MIN_WIN_S} s) is compared with the same bead "
             f"outside all indication windows (±{GUARD_S} s guard, first/last {EDGE_S} s excluded). "
             f"`z` = (window value − mean of up to {N_NULL} random same-length clean windows of the same bead) / "
             "their std, for the window mean (`z_mean`) and the window std (`z_std`, i.e. fluctuation). "
             "With no signature, |z| > 2 happens in ≈5 % of windows by chance and the median z is ≈0. "
             "Indications without a clean reference (covering ~the whole bead) are listed separately.",
             ""]
    verdicts = []
    for typ in sorted(per_type):
        T = per_type[typ]
        lines += [f"## {typ}", "", f"{T['n']} windows analysed in {len(T['beads'])} beads "
                  f"(total {T['dur']:.1f} s).", ""]
        if whole_bead.get(typ):
            lines += ["Not window-testable (indication covers >80 % of the bead, or <5 s of clean signal left "
                      f"in the bead): {', '.join(whole_bead[typ])}", ""]
        if T["n"] == 0:
            continue
        lines += ["| signal | mean in | mean out | Δ % | median z_mean | frac \\|z_mean\\|>2 | std in | std out "
                  "(null) | median z_std | frac \\|z_std\\|>2 |", "|---|---|---|---|---|---|---|---|---|---|"]
        flagged = []
        for sig, _tab, lbl in ANALYSIS_SIGNALS:
            S = T["sig"].get(sig)
            if not S:
                continue
            im, om = np.nanmean(S["in_m"]), np.nanmean(S["out_m"])
            zm, zs = np.array(S["zm"], float), np.array(S["zs"], float)
            zm, zs = zm[np.isfinite(zm)], zs[np.isfinite(zs)]
            fm = float(np.mean(np.abs(zm) > 2)) if zm.size else float("nan")
            fs = float(np.mean(np.abs(zs) > 2)) if zs.size else float("nan")
            mzm = float(np.median(zm)) if zm.size else float("nan")
            mzs = float(np.median(zs)) if zs.size else float("nan")
            dpct = (im / om - 1) * 100 if om else float("nan")
            lines.append(f"| {lbl} | {im:.2f} | {om:.2f} | {dpct:+.1f} | {mzm:+.2f} | {fm:.0%} | "
                         f"{np.nanmean(S['in_s']):.2f} | {np.nanmean(S['out_s']):.2f} | {mzs:+.2f} | {fs:.0%} |")
            n = max(zm.size, 1)
            # call it a candidate signature only if consistent: |median z| >= 1 and > 25 % windows beyond 2σ,
            # with at least 5 windows
            if zm.size >= 5 and abs(mzm) >= 1.0 and fm >= 0.25:
                flagged.append(f"{lbl} mean (median z {mzm:+.1f}, {fm:.0%} of {n} windows beyond 2σ)")
            if zs.size >= 5 and abs(mzs) >= 1.0 and fs >= 0.25:
                flagged.append(f"{lbl} fluctuation (median z {mzs:+.1f}, {fs:.0%} beyond 2σ)")
        lines.append("")
        if flagged:
            v = (f"**{typ}: candidate signature** in {'; '.join(flagged)}. Treat as a hypothesis "
                 f"({T['n']} windows, {len(T['beads'])} beads; not a validated detector).")
        else:
            v = (f"**{typ}: no consistent signature.** Signal levels and fluctuation inside the "
                 f"{T['n']} indication windows are within the bead's own clean-window variation.")
        verdicts.append(v)
        lines += [v, ""]
    lines += ["## Bottom line", ""] + [f"- {v}" for v in verdicts] + [
        "- These are radiographic/ultrasonic indications located by x position along the weld; the "
        "x→time mapping relies on the dataset's sample indices and on constant-speed travel, so windows may be "
        "off by a few tenths of a second. Downstream demos must not invent signatures beyond what is "
        "flagged above.", ""]
    md = "\n".join(lines)
    rp = Path(report_path) if report_path else out_dir / "aimen_saw_report.md"
    rp.write_text(md)
    print(md, flush=True)
    return md


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "convert", "both"])
    ap.add_argument("--data-dir", default="data/real")
    ap.add_argument("--coupons", default=None)
    a = ap.parse_args()
    raw = Path(a.data_dir) / "raw" / SOURCE_ID
    if a.cmd in ("download", "both"):
        download(raw, coupons=a.coupons)
    if a.cmd in ("convert", "both"):
        print(convert(raw, Path(a.data_dir) / "generated", coupons=a.coupons))
    sys.exit(0)
