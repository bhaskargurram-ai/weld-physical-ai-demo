"""Tiny pass dirs in the *real-data* contract (docs/REAL_DATA.md), for developing and testing the copilot
before the real AIMEN / BAM conversion has run.

    python -m copilot.fixtures_real --out data/fixtures_real [--with-sim] [--no-video]
    COPILOT_FIXTURES=real uvicorn copilot.app:app --port 7860

Everything here is SYNTHETIC: signal levels follow the published settings, the bead photo is a labelled
placeholder and the "process camera" is a procedurally drawn grey clip stamped FIXTURE. Every meta.json carries
`"fixture": true`. As on the real AIMEN coupon, the porosity windows deliberately leave no trace in the arc signals.

Passes: AIMEN SAW PP4_B05, PP5_B01..B03 (photo, NDT) · BAM tandem-GMAW WID617_L1..L3, WID630_L4 (video, edge,
not inspected) · optional SP-1043_J07_P3_FCAW simulator pass (`source.synthetic: true`).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import DATASETS

HZ = 100
T0_NS = 1_700_000_000_000_000_000
BAM_FPS = 18
VW, VH = 400, 320


def _src(sid: str, files: list[str]) -> dict:
    d = DATASETS[sid]
    return {"id": sid, "title": d["title"], "authors": d["authors"], "doi": d["doi"], "url": d["url"],
            "license": d["license"], "synthetic": False, "files": files}


def _wps(nom: dict, gas: float | None = None) -> dict:
    w = {k: [round(v * 0.9, 3), round(v * 1.1, 3)] for k, v in nom.items()}
    if gas:
        w["gas_lpm"] = [round(gas * 0.8, 2), round(gas * 1.2, 2)]
    return w


def _ramp(t: np.ndarray, dur: float) -> np.ndarray:
    return np.clip(np.minimum(t / 0.6, (dur - t) / 0.4), 0, 1)


def _write(d: Path, meta: dict, arc: pd.DataFrame, travel: pd.DataFrame | None, thermal: pd.DataFrame | None,
           robot: pd.DataFrame | None, events: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    arc.to_parquet(d / "arc.parquet", index=False)
    for name, df in (("travel", travel), ("thermal", thermal), ("robot", robot)):
        if df is not None:
            df.to_parquet(d / f"{name}.parquet", index=False)
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in sorted(events, key=lambda e: e["t_ns"])) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _events(dur: float, indications: list[dict]) -> list[dict]:
    ev = [{"t_ns": T0_NS, "type": "arc_start"}, {"t_ns": T0_NS + int(dur * 1e9) - 10_000_000, "type": "arc_stop"}]
    for i in indications:
        ev.append({"t_ns": T0_NS + int(i["t_start"] * 1e9), "type": "defect_on", "defect": i["type"], "source": "NDT"})
        ev.append({"t_ns": T0_NS + int(i["t_end"] * 1e9), "type": "defect_off", "defect": i["type"], "source": "NDT"})
    return ev


# ------------------------------------------------------------------ AIMEN SAW
def aimen_pass(out: Path, coupon: str, bead: int, dur: float, seed: int, indications=(), travel_excursion=None,
               photo: bool = True) -> str:
    rng = np.random.default_rng(seed)
    seq = f"{coupon}_B{bead:02d}_SAW"
    n = int(dur * HZ)
    t = np.arange(n) / HZ
    r = _ramp(t, dur)
    nom = {"current_a": 535.0, "voltage_v": 31.5, "current2_a": 580.0, "voltage2_v": 35.0, "wfs_mpm": 2.1,
           "wfs2_mpm": 1.9, "travel_speed_mmps": 9.2}
    sig = {k: v * (1 + 0.012 * np.sin(2 * np.pi * t / (7 + i) + rng.uniform(0, 6))) for i, (k, v) in enumerate(nom.items())}
    noise = {"current_a": 7, "voltage_v": 0.35, "current2_a": 9, "voltage2_v": 0.45, "wfs_mpm": 0.015,
             "wfs2_mpm": 0.015, "travel_speed_mmps": 0.05}
    for k in sig:
        sig[k] = sig[k] + rng.normal(0, noise[k], n)
    if travel_excursion:
        a, b, f = travel_excursion
        w = np.clip(np.minimum((t - a) / 0.5, (b - t) / 0.5), 0, 1)
        sig["travel_speed_mmps"] *= 1 + f * w
    for k in ("current_a", "voltage_v", "current2_a", "voltage2_v", "wfs_mpm", "wfs2_mpm"):
        sig[k] *= r
    pos = np.cumsum(sig["travel_speed_mmps"]) / HZ
    heat = (sig["voltage_v"] * sig["current_a"] + sig["voltage2_v"] * sig["current2_a"]) / (1000 * np.clip(sig["travel_speed_mmps"], 0.5, None))
    t_ns = T0_NS + (t * 1e9).astype(np.int64)
    # trail electrode is AC (as on the real coupons): its 10 ms means alias around zero, the RMS carries the level;
    # like the real conversion there is a current RMS column but no voltage RMS
    alias = np.sin(2 * np.pi * 9.7 * t + rng.uniform(0, 6))
    arc = pd.DataFrame({"t_ns": t_ns, "current_a": sig["current_a"].round(2), "voltage_v": sig["voltage_v"].round(3),
                        "wfs_mpm": sig["wfs_mpm"].round(3), "gas_lpm": np.full(n, np.nan),
                        "current2_a": (sig["current2_a"] * 1.1 * alias).round(2),
                        "voltage2_v": (sig["voltage2_v"] * 1.1 * alias).round(3),
                        "wfs2_mpm": sig["wfs2_mpm"].round(3), "current2_rms_a": sig["current2_a"].round(2)})
    travel = pd.DataFrame({"t_ns": t_ns, "travel_speed_mmps": sig["travel_speed_mmps"].round(3),
                           "position_mm": pos.round(2), "heat_input_kj_mm": heat.round(3)})
    tt = np.arange(int(dur * 10)) / 10
    left = 175 + 1.4 * tt + rng.normal(0, 1.5, len(tt))
    right = 170 + 1.5 * tt + rng.normal(0, 1.5, len(tt))
    thermal = pd.DataFrame({"t_ns": T0_NS + (tt * 1e9).astype(np.int64), "interpass_c": np.maximum(left, right).round(1),
                            "temp_left_c": left.round(1), "temp_right_c": right.round(1)})
    rt = np.arange(int(dur * 20)) / 20
    robot = pd.DataFrame({"t_ns": T0_NS + (rt * 1e9).astype(np.int64), "tcp_x": (536.7 + np.interp(rt, t, pos)).round(2),
                          "tcp_y": np.full(len(rt), 12.0), "tcp_z": np.full(len(rt), 48.0 - bead * 1.9)})
    ind = []
    for typ, a, b in indications:
        ind.append({"type": typ, "t_start": a, "t_end": b, "x_start_mm": round(float(np.interp(a, t, pos)), 1),
                    "x_end_mm": round(float(np.interp(b, t, pos)), 1), "method": "RT"})
    d = out / "generated" / seq
    photos = []
    if photo:
        photos = [f"photos/{coupon}_{bead:02d}_1.jpg"]
        placeholder_photo(d / photos[0], f"{coupon} bead {bead}", seed)
    pass_type = "root" if bead == 1 else ("cap" if bead == 24 else "fill")
    meta = {
        "sequence_name": seq, "spool_id": coupon, "joint_id": "BW", "pass_no": bead, "pass_type": pass_type,
        "wps_id": f"AIMEN-{coupon}-S1", "process": "SAW", "position": "1G", "material": "SA-516 Gr.70N",
        "cell_id": "AIMEN SAW cell", "duration_s": dur, "t0_ns": T0_NS, "thickness_mm": 52, "electrodes": 2,
        "rt_result": "indications" if ind else "no_indications", "has_defect": bool(ind),
        "defect_types": sorted({i["type"] for i in ind}),
        "defects": [{"defect": i["type"], "t_start": i["t_start"], "t_end": i["t_end"], "severity": None} for i in ind],
        "wps": _wps(nom), "setpoints": {k: v for k, v in nom.items()},
        "wps_basis": "published nominal setting ±10 % (demo assumption, not a qualified WPS)",
        "ndt": {"methods": ["RT", "UT"], "inspected": True, "report_files": ["RX.pdf", "UT.pdf"], "indications": ind},
        "photos": photos,
        "source": _src("aimen_saw", [f"{coupon}/1. process_data/{coupon}_{bead:02d}/{coupon}_{bead:02d}.hdf5",
                                     f"{coupon}/2. ndt_data/defects_xlocation.xlsx"]),
        "fixture": True,
    }
    _write(d, meta, arc, travel, thermal, robot, _events(dur, ind))
    return seq


def placeholder_photo(path: Path, label: str, seed: int = 0, w: int = 1600, h: int = 420) -> None:
    """Grey steel plate with a slag-free SAW bead, stamped as a placeholder (never a fake defect)."""
    from PIL import Image, ImageDraw, ImageFilter
    rng = np.random.default_rng(seed)
    y = np.arange(h)[:, None].astype(np.float32)
    x = np.arange(w)[None, :].astype(np.float32)
    plate = 92 + 10 * np.sin(x / 210.0) + rng.normal(0, 6, (h, w))
    plate += 18 * np.exp(-((x - w * 0.3) / 500) ** 2)               # mill-scale sheen
    cy, half = h * 0.5, 64 + 4 * np.sin(x / 140.0)
    dy = np.abs(y - cy)
    bead = np.clip(1.15 - dy / half, 0, 1)
    ripple = 0.5 + 0.5 * np.sin(x / 7.0 - (dy / half) ** 2 * 6)
    lum = plate * (1 - bead) + bead * (128 + 70 * np.clip(1 - dy / half, 0, 1) ** 0.7 + 22 * ripple)
    toe = np.exp(-((dy - half) / 3.0) ** 2)
    lum = lum * (1 - 0.35 * toe)
    rgb = np.stack([lum * 1.02, lum * 0.99, lum * 0.95], -1)
    rgb[..., 0] += 25 * bead * np.exp(-((dy / half) ** 2) * 2) * 0.6   # light straw temper colour
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, w, 34], fill=(20, 22, 26))
    dr.text((12, 10), f"FIXTURE PLACEHOLDER - not a real photo · {label}", fill=(255, 210, 120))
    for mm in range(0, w, 100):                                        # ruler
        dr.line([(mm, h - 18), (mm, h - 6)], fill=(230, 230, 230), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=86)


# ------------------------------------------------------------------ BAM tandem-GMAW
def bam_pass(out: Path, wid: str, layer: int, dur: float, seed: int, wfs_excursion=None, video: bool = True) -> str:
    rng = np.random.default_rng(seed)
    seq = f"{wid}_L{layer}_TGMAW"
    tandem = layer > 1
    n = int(dur * HZ)
    t = np.arange(n) / HZ
    r = _ramp(t, dur)
    if tandem:
        nom = {"current_a": 292.0, "voltage_v": 30.2, "current2_a": 255.0, "voltage2_v": 28.4, "wfs_mpm": 11.0,
               "wfs2_mpm": 9.5, "travel_speed_mmps": 10.0}
    else:
        nom = {"current_a": 245.0, "voltage_v": 26.5, "wfs_mpm": 8.0, "travel_speed_mmps": 6.5}
    gas = 18.0
    sig = {k: v * (1 + 0.01 * np.sin(2 * np.pi * t / (5 + i) + rng.uniform(0, 6))) for i, (k, v) in enumerate(nom.items())}
    for k in sig:
        rel = 0.045 if k.startswith("current") else 0.02 if k.startswith("voltage") else 0.006
        sig[k] = sig[k] + rng.normal(0, nom[k] * rel, n)
    if wfs_excursion:
        a, b, f = wfs_excursion
        w = np.clip(np.minimum((t - a) / 0.3, (b - t) / 0.3), 0, 1)
        sig["wfs_mpm"] *= 1 + f * w
        sig["current_a"] *= 1 + 0.6 * f * w
    for k in sig:
        if k != "travel_speed_mmps":
            sig[k] *= r
    gas_l = gas + 0.3 * np.sin(t / 3) + rng.normal(0, 0.12, n)
    t_ns = T0_NS + (t * 1e9).astype(np.int64)
    arc = {"t_ns": t_ns, "current_a": sig["current_a"].round(2), "voltage_v": sig["voltage_v"].round(3),
           "wfs_mpm": sig["wfs_mpm"].round(3), "gas_lpm": gas_l.round(2)}
    if tandem:
        arc.update({"current2_a": sig["current2_a"].round(2), "voltage2_v": sig["voltage2_v"].round(3),
                    "wfs2_mpm": sig["wfs2_mpm"].round(3)})
    power = sig["voltage_v"] * sig["current_a"] + (sig["voltage2_v"] * sig["current2_a"] if tandem else 0)
    travel = pd.DataFrame({"t_ns": t_ns, "travel_speed_mmps": sig["travel_speed_mmps"].round(3),
                           "position_mm": (np.cumsum(sig["travel_speed_mmps"]) / HZ).round(2),
                           "heat_input_kj_mm": (0.8 * power / (1000 * sig["travel_speed_mmps"])).round(3)})
    tt = np.arange(int(dur * 10)) / 10
    th1 = 95 + 35 * layer + 3.2 * tt + rng.normal(0, 0.6, len(tt))
    thermal = pd.DataFrame({"t_ns": T0_NS + (tt * 1e9).astype(np.int64), "interpass_c": th1.round(1),
                            "thermo_2_c": (th1 * 0.82).round(1)})
    rt = np.arange(int(dur * 50)) / 50
    robot = pd.DataFrame({"t_ns": T0_NS + (rt * 1e9).astype(np.int64), "tcp_x": (np.interp(rt, t, travel["position_mm"])).round(2),
                          "tcp_y": np.zeros(len(rt)), "tcp_z": np.full(len(rt), 4.0 * layer)})
    d = out / "generated" / seq
    meta = {
        "sequence_name": seq, "spool_id": wid, "joint_id": "V30", "pass_no": layer,
        "pass_type": "root" if layer == 1 else ("cap" if layer == 6 else "fill"), "wps_id": f"BAM-{wid}-L{layer}",
        "process": "TGMAW" if tandem else "GMAW", "position": "PA", "material": "S355J2+N", "cell_id": "BAM tandem cell",
        "duration_s": dur, "t0_ns": T0_NS, "thickness_mm": 30, "electrodes": 2 if tandem else 1,
        "rt_result": "not_inspected", "has_defect": False, "defect_types": [], "defects": [],
        "wps": _wps(nom, gas), "setpoints": {**nom, "gas_lpm": gas},
        "wps_basis": "published nominal setting ±10 %; gas = GASVALUE ×0.8–1.2 (demo assumption, not a qualified WPS)",
        "ndt": {"methods": [], "inspected": False, "report_files": [], "indications": []},
        "mode": "pulsed" if layer % 2 else "short-circuiting",
        "source": _src("bam_tgmaw", [f"{wid}/{wid}_L{layer}_process.nc", f"{wid}/{wid}_L{layer}_robot.nc",
                                     f"{wid}/{wid}_L{layer}_video.avi"]),
        "fixture": True,
    }
    if video and shutil.which("ffmpeg"):
        render_bam_video(d, dur, tandem, f"{wid} L{layer}", seed)
        meta["video"] = {"file": "torch_cam.mp4", "kind": "real_process_camera", "fps": BAM_FPS, "t_offset_s": 0.0}
    _write(d, meta, pd.DataFrame(arc), travel, thermal, robot, _events(dur, []))
    return seq


def _writer(path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{VW}x{VH}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def render_bam_video(d: Path, dur: float, tandem: bool, label: str, seed: int = 0) -> None:
    """Grey HDR-camera-like clip (side view: wires, arcs, pool, bead) + Canny-style edge video, same timing."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    nf = int(dur * BAM_FPS)
    yy, xx = np.mgrid[:VH, :VW].astype(np.float32)
    plate_y = VH * 0.66
    base = 26 + 10 * (yy > plate_y) + rng.normal(0, 3, (VH, VW))
    base += 18 * np.exp(-((yy - plate_y) / 3.0) ** 2)                 # plate edge highlight
    arcs = [VW * 0.47, VW * 0.60] if tandem else [VW * 0.53]
    wr, we = _writer(d / "torch_cam.mp4", BAM_FPS), _writer(d / "control" / "edge.mp4", BAM_FPS)
    try:
        for i in range(nf):
            t = i / BAM_FPS
            img = base.copy()
            shift = (t * 38) % 24
            bead = (xx < arcs[0] - 8) & (np.abs(yy - plate_y + 7) < 9)
            ripple = 0.5 + 0.5 * np.sin((xx + shift) / 4.0)
            img += bead * (55 + 28 * ripple) * np.clip((xx - 20) / 120, 0.2, 1)
            for k, ax in enumerate(arcs):
                pulse = 0.75 + 0.25 * np.sin(2 * np.pi * (3.1 + k) * t + k) + rng.normal(0, 0.05)
                r = np.hypot((xx - ax) / 1.3, yy - (plate_y - 16))
                img += 255 * pulse * (np.exp(-(r / 16) ** 2) + 0.45 * np.exp(-(r / 46) ** 2))
                wire = (np.abs(xx - ax + (yy - plate_y) * 0.12) < 2.2) & (yy < plate_y - 26)
                img += wire * 120
                pool = np.exp(-(((xx - ax + 10) / 34) ** 2 + ((yy - plate_y + 4) / 7) ** 2))
                img += 150 * pool
            for _ in range(rng.poisson(4)):                              # spatter streaks
                sx, sy = rng.uniform(arcs[0] - 30, arcs[-1] + 30), rng.uniform(plate_y - 90, plate_y - 20)
                img[int(sy):int(sy) + 2, int(sx):int(sx) + 6] += 160
            frame = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            dr = ImageDraw.Draw(frame)
            dr.rectangle([0, 0, VW, 16], fill=0)
            dr.text((6, 3), f"BAM  {label}", fill=235)
            dr.text((VW - 120, 3), f"t = {t:6.2f} s", fill=200)
            dr.text((6, VH - 14), "FIXTURE - synthetic stand-in", fill=150)
            a = np.asarray(frame)
            wr.stdin.write(a.tobytes())
            g = a.astype(np.float32)
            edge = ((np.abs(np.diff(g, axis=1, prepend=g[:, :1])) + np.abs(np.diff(g, axis=0, prepend=g[:1]))) > 26)
            we.stdin.write((edge * 255).astype(np.uint8).tobytes())
    finally:
        for p in (wr, we):
            p.stdin.close()
            p.wait()


# ------------------------------------------------------------------ one SIMULATED pass (DATA_SOURCE=all)
def sim_pass(out: Path) -> str:
    from .fixtures import PassSpec, simulate
    from collect.schema import Defect
    spec = PassSpec("SP-1043", "J07", 3, "fill", 30.0, defects=[(Defect.POROSITY, 11.2, 15.8, 0.8)], rt="reject")
    sim = simulate(spec, 2)
    meta = sim["meta"]
    meta["source"] = {"id": "sim", "title": "WeldCell-sim synthetic passes", "license": "generated", "synthetic": True}
    seq = meta["sequence_name"]
    d = out / "generated" / seq
    _write(d, meta, sim["arc"], sim["travel"], sim["thermal"], sim["robot"],
           [{**e, "t_ns": int(e["t_ns"])} for e in sim["events"]])
    return seq


PP5_B02_POROSITY = [round(12.0 + 1.05 * k + 0.13 * (k % 3), 2) for k in range(14)]


def make_dataset(out: Path | str, videos: bool = True, with_sim: bool = False) -> list[str]:
    out = Path(out)
    seqs = [
        aimen_pass(out, "PP4", 5, 34.0, 4),
        aimen_pass(out, "PP5", 1, 32.0, 5, indications=[("lack_of_fusion", 8.2, 9.6)]),
        aimen_pass(out, "PP5", 2, 38.0, 6, indications=[("porosity", a, round(a + 0.1, 2)) for a in PP5_B02_POROSITY]),
        aimen_pass(out, "PP5", 3, 36.0, 7, travel_excursion=(18.0, 23.5, 0.19)),
        bam_pass(out, "WID617", 1, 12.0, 11, video=videos),
        bam_pass(out, "WID617", 2, 12.0, 12, video=videos),
        bam_pass(out, "WID617", 3, 12.0, 13, video=videos),
        bam_pass(out, "WID630", 4, 12.0, 14, wfs_excursion=(5.0, 7.2, 0.16), video=videos),
    ]
    if with_sim:
        seqs.append(sim_pass(out))
    return seqs


def ensure_fixtures(out: Path, videos: bool = True) -> Path:
    g = out / "generated"
    if not g.exists() or not any(g.iterdir()):
        make_dataset(out, videos=videos)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fixtures_real")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--with-sim", action="store_true")
    a = ap.parse_args()
    s = make_dataset(a.out, videos=not a.no_video, with_sim=a.with_sim)
    print(f"wrote {len(s)} passes to {a.out}")
