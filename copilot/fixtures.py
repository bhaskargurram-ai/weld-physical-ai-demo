"""Mock data that follows the PLAN.md contracts, so the copilot can be developed, tested and
screenshotted without agents A/B/C's modules or a GPU.

    python -m copilot.fixtures --out data/fixtures            # 24 passes + videos + curated manifest
    python -m copilot.fixtures --out /tmp/x --no-video        # fast (tests)

Writes:
  <out>/generated/<seq>/{meta.json, arc.parquet, travel.parquet, robot.parquet, thermal.parquet,
                         events.jsonl, torch_cam.mp4, control/edge.mp4, control/depth.mp4}
  <out>/curated/{manifest.jsonl, dataset_card.md}
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect.schema import (  # noqa: E402
    ARC_HZ, CAMERA_FPS, CLIP_SECONDS, ROBOT_HZ, THERMAL_HZ, WPS_LIBRARY, Defect, DefectEvent,
    Position, Process, WeldPass, heat_input_kj_mm,
)

VIDEO_W, VIDEO_H = 640, 360


@dataclass
class PassSpec:
    spool: str
    joint: str
    pass_no: int
    pass_type: str
    duration: float
    defects: list = field(default_factory=list)   # (Defect, t0, t1, severity)
    rt: str | None = "accept"
    position: str = "5G"
    day: int = 0
    interpass_start: float = 150.0


def _plan() -> list[PassSpec]:
    joints = [("SP-1041", "J03", "5G", 0), ("SP-1042", "J05", "6G", 0), ("SP-1043", "J07", "5G", 1),
              ("SP-1043", "J08", "5G", 1), ("SP-1044", "J02", "6G", 2), ("SP-1045", "J11", "1G", 2)]
    types = [(1, "root"), (2, "hot"), (3, "fill"), (4, "cap")]
    durs = [34, 26, 30, 28]
    specs = []
    for (sp, j, pos, day) in joints:
        for (no, pt), d in zip(types, durs):
            specs.append(PassSpec(sp, j, no, pt, float(d + (hash((sp, j, no)) % 5)), position=pos, day=day))
    by = {(s.spool, s.joint, s.pass_no): s for s in specs}
    # --- injected defects (each mirrors collect.schema.DEFECT_SIGNATURES) ---
    s = by[("SP-1043", "J07", 3)]; s.defects = [(Defect.POROSITY, 11.2, 15.8, 0.8)]; s.rt = "reject"; s.duration = 30.0
    s = by[("SP-1042", "J05", 2)]; s.defects = [(Defect.LACK_OF_FUSION, 6.0, 10.5, 0.7)]; s.rt = "reject"
    s = by[("SP-1044", "J02", 1)]; s.defects = [(Defect.BURN_THROUGH, 14.0, 16.5, 0.9)]; s.rt = "reject"
    s = by[("SP-1045", "J11", 4)]; s.defects = [(Defect.UNDERCUT, 8.0, 12.0, 0.6)]; s.rt = "reject"
    s = by[("SP-1041", "J03", 3)]; s.defects = [(Defect.ARC_INSTABILITY, 18.0, 21.0, 0.35)]; s.rt = "accept"
    s = by[("SP-1043", "J08", 4)]; s.defects = [(Defect.CRACK, 9.0, 24.0, 0.5)]; s.rt = "reject"; s.interpass_start = 238.0
    by[("SP-1045", "J11", 3)].rt = None
    by[("SP-1044", "J02", 4)].rt = None
    return specs


def _seq_name(s: PassSpec, process: str) -> str:
    return f"{s.spool}_{s.joint}_P{s.pass_no}_{process}"


def _nominal(process: str) -> dict:
    if process == "GTAW":
        return dict(current_a=112.0, voltage_v=12.0, travel_speed_mmps=1.45, wfs_mpm=0.8, gas_lpm=15.0)
    return dict(current_a=212.0, voltage_v=25.0, travel_speed_mmps=5.4, wfs_mpm=8.6, gas_lpm=21.0)


def simulate(spec: PassSpec, seed: int = 0) -> dict:
    """Physically-plausible arc/travel/robot/thermal streams + events for one pass."""
    rng = np.random.default_rng(seed)
    process = "GTAW" if spec.pass_type == "root" else "FCAW"
    wps = WPS_LIBRARY["WPS-GTAW-ROOT-01" if process == "GTAW" else "WPS-FCAW-FILL-02"]
    nom = _nominal(process)
    start = datetime(2026, 11, 9, 7, 30, tzinfo=timezone.utc) + timedelta(days=spec.day, minutes=37 * spec.pass_no
                                                                          + 11 * int(spec.joint[1:]))
    t0_ns = int(start.timestamp() * 1e9)
    n = int(spec.duration * ARC_HZ)
    t = np.arange(n) / ARC_HZ

    def drift(scale, period):
        return scale * np.sin(2 * np.pi * t / period + rng.uniform(0, 6.28))

    sig = {
        "current_a": nom["current_a"] + drift(nom["current_a"] * 0.025, 9) + rng.normal(0, nom["current_a"] * 0.012, n),
        "voltage_v": nom["voltage_v"] + drift(0.25, 7) + rng.normal(0, 0.12, n),
        "travel_speed_mmps": nom["travel_speed_mmps"] * (1 + drift(0.03, 11)) + rng.normal(0, nom["travel_speed_mmps"] * 0.01, n),
        "wfs_mpm": nom["wfs_mpm"] + drift(nom["wfs_mpm"] * 0.02, 13) + rng.normal(0, nom["wfs_mpm"] * 0.01, n),
        "gas_lpm": nom["gas_lpm"] + drift(0.35, 17) + rng.normal(0, 0.15, n),
    }
    interpass = spec.interpass_start + 1.3 * t + rng.normal(0, 0.8, n)

    def win(a, b, ramp=0.4):
        return np.clip(np.minimum((t - a) / ramp, (b - t) / ramp), 0, 1)

    for d, a, b, sev in spec.defects:
        w = win(a, b)
        if d == Defect.POROSITY:
            sig["gas_lpm"] = sig["gas_lpm"] * (1 - w) + w * (6.0 + rng.normal(0, 0.6, n))
            spikes = (rng.random(n) < 0.06) * rng.uniform(1.5, 4.0, n)
            sig["voltage_v"] += w * spikes * sev
        elif d == Defect.LACK_OF_FUSION:
            sig["current_a"] *= 1 - 0.27 * w * sev / 0.7
            sig["travel_speed_mmps"] *= 1 + 0.42 * w * sev / 0.7
        elif d == Defect.UNDERCUT:
            sig["voltage_v"] *= 1 + 0.2 * w
            sig["travel_speed_mmps"] *= 1 + 0.22 * w
        elif d == Defect.BURN_THROUGH:
            sig["current_a"] *= 1 + 0.32 * w
            sig["travel_speed_mmps"] *= 1 - 0.42 * w
        elif d == Defect.ARC_INSTABILITY:
            sig["voltage_v"] += w * rng.normal(0, 2.6 * sev / 0.35, n)
            sig["wfs_mpm"] += w * 1.2 * np.sin(2 * np.pi * 6 * t)
        elif d == Defect.CRACK:
            interpass = interpass + w * 18.0

    hi = np.array([heat_input_kj_mm(i, v, s) for i, v, s in zip(sig["current_a"], sig["voltage_v"],
                                                                   sig["travel_speed_mmps"])])
    t_ns = t0_ns + (t * 1e9).astype(np.int64)
    arc = pd.DataFrame({"t_ns": t_ns, **{k: sig[k].round(3) for k in ("current_a", "voltage_v", "wfs_mpm", "gas_lpm")}})
    travel = pd.DataFrame({"t_ns": t_ns, "travel_speed_mmps": sig["travel_speed_mmps"].round(4),
                           "positioner_deg": (360.0 * t / spec.duration).round(3), "heat_input_kj_mm": hi.round(4)})
    rt_ = np.arange(int(spec.duration * ROBOT_HZ)) / ROBOT_HZ
    robot = pd.DataFrame({"t_ns": t0_ns + (rt_ * 1e9).astype(np.int64),
                          **{f"j{i}": np.round(0.2 * i * np.sin(rt_ / (3 + i)), 4) for i in range(1, 7)},
                          "tcp_x": 0.9 + 0.001 * rt_, "tcp_y": 0.05 * np.sin(rt_ / 4), "tcp_z": 0.62 + 0.0,
                          "tcp_qx": 0.0, "tcp_qy": 0.7071, "tcp_qz": 0.0, "tcp_qw": 0.7071})
    tt = np.arange(int(spec.duration * THERMAL_HZ)) / THERMAL_HZ
    thermal = pd.DataFrame({"t_ns": t0_ns + (tt * 1e9).astype(np.int64),
                            "interpass_c": np.interp(tt, t, interpass).round(2)})
    events = [{"t_ns": int(t0_ns), "type": "arc_start"}]
    for d, a, b, sev in spec.defects:
        events.append({"t_ns": int(t0_ns + a * 1e9), "type": "defect_on", "defect": d.value, "severity": sev})
        events.append({"t_ns": int(t0_ns + b * 1e9), "type": "defect_off", "defect": d.value, "severity": sev})
    events.append({"t_ns": int(t_ns[-1]), "type": "arc_stop"})
    events.sort(key=lambda e: e["t_ns"])

    wp = WeldPass(
        sequence_name=_seq_name(spec, process), spool_id=spec.spool, joint_id=spec.joint,
        pass_no=spec.pass_no, pass_type=spec.pass_type, wps_id=wps.wps_id, process=Process(process),
        position=Position(spec.position), material="A106 Gr.B", pipe_od_mm=168.3, wall_mm=10.97,
        cell_id="CELL-2", duration_s=spec.duration,
        defects=[DefectEvent(d, a, b, sev) for d, a, b, sev in spec.defects], rt_result=spec.rt,
    )
    meta = wp.to_metadata()
    meta["wps"] = {"wps_id": wps.wps_id, "process": wps.process.value, "current_a": list(wps.current_a),
                   "voltage_v": list(wps.voltage_v), "travel_speed_mmps": list(wps.travel_speed_mmps),
                   "wfs_mpm": list(wps.wfs_mpm), "gas_lpm": list(wps.gas_lpm),
                   "max_interpass_c": wps.max_interpass_c, "heat_input_kj_mm": list(wps.heat_input_kj_mm)}
    meta["has_defect"] = bool(spec.defects)
    meta["started_at"] = start.isoformat()
    meta["welder_id"] = "ROBOT-R2"
    meta["source"] = "copilot-fixture"
    return dict(meta=meta, arc=arc, travel=travel, robot=robot, thermal=thermal, events=events,
                speed=sig["travel_speed_mmps"], heat=hi)


# ------------------------------------------------------------------ procedural torch-cam render
def _ffmpeg_writer(path: Path, w: int, h: int, fps: int, gray: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def render_videos(pass_dir: Path, sim: dict, seed: int = 0, w: int = VIDEO_W, h: int = VIDEO_H) -> None:
    """Torch-mounted camera view: pipe groove ahead, cooling bead behind, arc + sparks at the pool.
    Travel speed drives scroll, heat input drives bead width, defects leave visible marks.
    Also writes control/edge.mp4 + control/depth.mp4 of the same scene (sim->real conditioning)."""
    rng = np.random.default_rng(seed)
    meta = sim["meta"]
    dur = meta["duration_s"]
    nf = int(dur * CAMERA_FPS)
    px_per_mm = 7.5 if meta["process"] == "FCAW" else 22.0
    tt = np.arange(len(sim["speed"])) / ARC_HZ
    ft = np.arange(nf) / CAMERA_FPS
    speed_f = np.interp(ft, tt, sim["speed"])
    heat_f = np.interp(ft, tt, sim["heat"])
    pos = np.concatenate([[0], np.cumsum(speed_f[:-1] / CAMERA_FPS)]) * px_per_mm  # torch x (px) along strip
    pool_x, pool_y = int(w * 0.56), int(h * 0.55)
    L = int(pos[-1] + w + 40)
    # Strip geometry
    ys = np.arange(h)[:, None].astype(np.float32)
    cyl = 0.55 + 0.45 * np.cos((ys - pool_y) / h * 2.6)          # pipe curvature shading
    noise = rng.normal(0, 1, (h, L)).astype(np.float32)
    noise = (noise + np.roll(noise, 1, 1) + np.roll(noise, 2, 1)) / 3
    groove = np.exp(-((ys - pool_y) / 13.0) ** 2)                   # open bevel
    bevel_hi = np.exp(-((np.abs(ys - pool_y) - 17.0) / 1.8) ** 2)   # bevel edge highlights
    base_clean = (38 + 52 * cyl + 30 * bevel_hi).astype(np.float32)  # steel
    base = (base_clean + 5 * noise).astype(np.float32)
    # bead half-width per strip column from heat input at the time the torch passed it
    col_heat = np.interp(np.arange(L), pos + pool_x, heat_f, left=heat_f[0], right=heat_f[-1])
    nominal_heat = np.median(heat_f)
    half_w = (25 if meta["process"] == "FCAW" else 15) * np.sqrt(np.clip(col_heat / nominal_heat, 0.4, 2.2))
    dy = (ys - pool_y)
    bead_mask = np.clip(1.2 - np.abs(dy) / half_w[None, :], 0, 1)       # (h, L)
    ripple = 0.5 + 0.5 * np.sin(np.arange(L)[None, :] / 3.2 - np.abs(dy) * 0.18)
    bead_lum = (78 + 58 * (1 - np.abs(dy) / np.maximum(half_w[None, :], 1)) + 26 * ripple).astype(np.float32)
    bead_mask = bead_mask.astype(np.float32)
    defect_map = np.zeros((h, L), np.float32)
    for d in meta["defects"]:
        a, b, kind = d["t_start"], d["t_end"], d["defect"]
        xa, xb = np.interp([a, b], ft, pos) + pool_x
        xa, xb = int(xa), int(max(xb, xa + 6))
        if kind == "porosity":
            for _ in range(int((xb - xa) / 4)):
                cx, cy = rng.integers(xa, xb), pool_y + rng.normal(0, 9)
                r = rng.uniform(1.5, 4.0)
                yy, xx = np.ogrid[:h, max(cx - 6, 0):min(cx + 6, L)]
                defect_map[:, max(cx - 6, 0):min(cx + 6, L)] = np.maximum(
                    defect_map[:, max(cx - 6, 0):min(cx + 6, L)], np.clip(1.5 - np.hypot(yy - cy, xx - cx) / r, 0, 1))
        elif kind in ("undercut", "lack_of_fusion"):
            edge = half_w[None, xa:xb] * (0.95 if kind == "undercut" else 0.7)
            band = np.exp(-((np.abs(dy) - edge) / 2.2) ** 2)
            defect_map[:, xa:xb] = np.maximum(defect_map[:, xa:xb], band)
        elif kind == "burn_through":
            yy, xx = np.ogrid[:h, xa:xb]
            defect_map[:, xa:xb] = np.maximum(defect_map[:, xa:xb], np.clip(1.4 - np.abs(yy - pool_y) / 6, 0, 1))
        elif kind == "crack":
            wig = pool_y + 3 * np.sin(np.arange(xa, xb) / 7.0)
            defect_map[:, xa:xb] = np.maximum(defect_map[:, xa:xb], np.exp(-((ys - wig[None, :]) / 1.2) ** 2))
    # Glow sprite
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.hypot((xx - pool_x) / 1.25, yy - pool_y)
    core = np.exp(-(r / 12) ** 2)
    halo = np.exp(-(r / 62) ** 2)
    wide = np.exp(-(r / 240) ** 2)
    glow_rgb = np.stack([255 * (core + halo * 0.8 + wide * 0.10), 255 * (core + halo * 0.42 + wide * 0.04),
                         255 * (core * 1.05 + halo * 0.10 + wide * 0.01)], -1).astype(np.float32)
    vign = (1 - 0.45 * ((xx - w / 2) ** 2 / (w / 2) ** 2 + (yy - h / 2) ** 2 / (h / 2) ** 2)).astype(np.float32)
    nozzle = ((yy < pool_y - 34) & (np.abs(xx - pool_x) < 46 + (pool_y - 34 - yy) * 0.35)).astype(np.float32)
    nozzle_edge = np.clip(1 - np.abs(np.abs(xx - pool_x) - (46 + (pool_y - 34 - yy) * 0.35)) / 3, 0, 1) * (yy < pool_y - 34)
    nozzle_shade = (0.5 + 0.5 * np.cos((xx - pool_x) / 60.0)) * nozzle
    nozzle_rgb = np.stack([120 * nozzle_shade, 78 * nozzle_shade, 50 * nozzle_shade], -1).astype(np.float32)
    defect_frames = [(d["t_start"], d["t_end"], d["defect"]) for d in meta["defects"]]

    wr = _ffmpeg_writer(pass_dir / "torch_cam.mp4", w, h, CAMERA_FPS)
    we = _ffmpeg_writer(pass_dir / "control" / "edge.mp4", w, h, CAMERA_FPS, gray=True)
    wd = _ffmpeg_writer(pass_dir / "control" / "depth.mp4", w, h, CAMERA_FPS, gray=True)
    sparks = np.zeros((0, 4), np.float32)  # x, y, vx, vy
    try:
        for i in range(nf):
            x0 = int(pos[i])
            sl = slice(x0, x0 + w)
            welded = (np.arange(x0, x0 + w) < x0 + pool_x)[None, :].astype(np.float32)
            bm = bead_mask[:, sl] * welded
            dist = np.maximum(x0 + pool_x - np.arange(x0, x0 + w), 0)[None, :].astype(np.float32)
            heat_trail = np.exp(-dist / 55.0) * bm
            temper = np.exp(-dist / 170.0) * bm
            lum = base[:, sl] * (1 - 0.6 * groove * (1 - welded)) * (1 - bm) + bead_lum[:, sl] * bm
            lum = lum * (1 - 0.75 * defect_map[:, sl] * welded)
            clean = base_clean[:, :w] * (1 - 0.6 * groove * (1 - welded)) * (1 - bm) + bead_lum[:, sl] * bm
            clean = clean * (1 - 0.75 * defect_map[:, sl] * welded)
            img = np.stack([lum, lum * 0.98, lum * 1.04], -1)
            img[..., 0] += 45 * temper; img[..., 1] += 18 * temper; img[..., 2] -= 20 * temper   # straw temper
            img[..., 2] += 40 * temper * (dist > 120); img[..., 0] -= 25 * temper * (dist > 120)  # blue temper
            img += np.stack([255 * heat_trail, 120 * heat_trail, 25 * heat_trail], -1)
            t = ft[i]
            unstable = any(a <= t <= b and k in ("arc_instability", "porosity") for a, b, k in defect_frames)
            flick = 1 + rng.normal(0, 0.28 if unstable else 0.05)
            g = np.clip(flick, 0.4, 1.6)
            img += glow_rgb * g
            # sparks
            nnew = rng.poisson(9 if unstable else 3)
            if nnew:
                ang = rng.uniform(-np.pi, 0, nnew)
                spd = rng.uniform(4, 14, nnew)
                sparks = np.vstack([sparks, np.stack([np.full(nnew, pool_x), np.full(nnew, pool_y - 4),
                                                      np.cos(ang) * spd, np.sin(ang) * spd], 1)])
            sparks[:, 0] += sparks[:, 2]; sparks[:, 1] += sparks[:, 3]; sparks[:, 3] += 0.9
            sparks = sparks[(sparks[:, 0] > 0) & (sparks[:, 0] < w - 1) & (sparks[:, 1] > 0) & (sparks[:, 1] < h - 1)]
            for sx, sy, vx, vy in sparks:
                for k in range(4):
                    px, py = int(sx - vx * k * 0.25), int(sy - vy * k * 0.25)
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] += (255, 200, 120)
            img = img * (1 - nozzle[..., None]) + nozzle_rgb + nozzle_edge[..., None] * np.array([110, 70, 40], np.float32) * g
            img = np.clip(img * vign[..., None], 0, 255).astype(np.uint8)
            wr.stdin.write(img.tobytes())
            # control: edges of the (glow-free) scene; depth from geometry
            g0 = clean
            gx = np.abs(np.diff(g0, axis=1, prepend=g0[:, :1])); gy = np.abs(np.diff(g0, axis=0, prepend=g0[:1]))
            edge = ((gx + gy) > 9).astype(np.uint8) * 255
            edge = np.maximum(edge, (nozzle_edge > 0.5).astype(np.uint8) * 255)
            we.stdin.write(edge.tobytes())
            depth = 90 + 80 * cyl[:, :1].repeat(w, 1) - 40 * groove * (1 - welded) + 35 * bm - 30 * defect_map[:, sl] * welded
            depth = np.where(nozzle > 0, 245, depth)
            wd.stdin.write(np.clip(depth, 0, 255).astype(np.uint8).tobytes())
    finally:
        for p in (wr, we, wd):
            p.stdin.close(); p.wait()


# ------------------------------------------------------------------ writer
def write_pass(out: Path, spec: PassSpec, seed: int, videos: bool) -> str:
    sim = simulate(spec, seed)
    seq = sim["meta"]["sequence_name"]
    d = out / "generated" / seq
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(sim["meta"], indent=2))
    for k in ("arc", "travel", "robot", "thermal"):
        sim[k].to_parquet(d / f"{k}.parquet", index=False)
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in sim["events"]) + "\n")
    if videos and shutil.which("ffmpeg"):
        render_videos(d, sim, seed)
    return seq


def _write_pass_args(args):
    return write_pass(*args)


_CAPTIONS = {
    "porosity": "Shielding gas coverage collapses; arc turns erratic and gas pores become visible in the trailing bead.",
    "lack_of_fusion": "Torch travel accelerates while current sags; the bead narrows and does not wet into the bevel sidewalls.",
    "undercut": "Voltage climbs and travel speeds up; a groove is melted into the toe on both sides of the cap bead.",
    "burn_through": "Root pass slows down with high current; the keyhole grows and the root melts through.",
    "arc_instability": "The arc flickers with irregular wire feed; spatter increases and the bead ripples become uneven.",
    "crack": "Interpass temperature is above procedure maximum; a fine longitudinal centerline line appears in the bead.",
}


def write_curated(out: Path, seqs: list[str]) -> None:
    cur = out / "curated"
    cur.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(7)
    for seq in seqs:
        meta = json.loads((out / "generated" / seq / "meta.json").read_text())
        dur = meta["duration_s"]
        n = int(dur // CLIP_SECONDS)
        for i in range(n):
            t0, t1 = i * CLIP_SECONDS, (i + 1) * CLIP_SECONDS
            defs = [d["defect"] for d in meta["defects"] if d["t_start"] < t1 and d["t_end"] > t0]
            proc = meta["process"]
            if defs:
                cap = (f"Close-up torch camera of a robotic {proc} {meta['pass_type']} pass on {meta['material']} pipe "
                       f"({meta['position']}). " + _CAPTIONS[defs[0]])
            else:
                cap = (f"Close-up torch camera of a robotic {proc} {meta['pass_type']} pass on {meta['material']} pipe "
                       f"({meta['position']}). Stable arc, steady travel, uniform bead ripples and clean toe wetting.")
            kept, reason = True, None
            if i == 0:
                kept, reason = False, "arc_start_transient"
            elif rng.random() < 0.08:
                kept, reason = False, "motion_blur"
            elif rng.random() < 0.05:
                kept, reason = False, "near_duplicate"
            clip_id = f"{seq}_c{i:02d}"
            gen_prompt = {
                "subject": f"robotic {proc} weld pass on carbon-steel pipe, torch-mounted camera",
                "scene": "fabrication shop weld cell, pipe on positioner, dark background",
                "action": cap, "camera": "fixed to torch, looking at weld pool, 24 fps",
                "lighting": "bright arc light with orange molten pool glow",
                "style": "photorealistic industrial footage",
            }
            rows.append({
                "clip_id": clip_id, "clip_uri": f"clips/{clip_id}.mp4", "sequence_name": seq,
                "t0_s": float(t0), "t1_s": float(t1), "caption": cap, "caption_model": "fixture-template",
                "sensor_summary": {"process": proc, "pass_type": meta["pass_type"]},
                "wps_violations": [], "defects": defs, "rt_result": meta["rt_result"],
                "quality": {"blur": round(float(rng.uniform(0.02, 0.4 if kept else 0.8)), 3),
                            "motion": round(float(rng.uniform(0.2, 0.9)), 3)},
                "kept": kept, "reject_reason": reason, "generator_prompt": json.dumps(gen_prompt),
            })
    (cur / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    kept = sum(r["kept"] for r in rows)
    (cur / "dataset_card.md").write_text(f"""# WeldWorld curated clips (fixture)

**Clips:** {len(rows)} total, {kept} kept, {len(rows) - kept} rejected
**Source:** WeldCell simulator, robotic GTAW root + FCAW fill/cap on A106 Gr.B pipe
**Clip length:** {CLIP_SECONDS} s @ {CAMERA_FPS} fps

## Intended use
Training and evaluating weld-quality perception models (porosity, lack of fusion, undercut,
burn-through, arc instability, crack) and conditioning Cosmos 3 Generator what-if / sim->real runs.

## Filters
- Arc-start transients removed (first {CLIP_SECONDS} s of each pass)
- Motion blur and near-duplicate filters

## Labels
Per-clip RT (radiography) result inherited from the pass; defect windows from the simulator ground truth.

*This card was generated by `copilot.fixtures` for UI development. The real card comes from `curate/`.*
""")


def make_dataset(out: Path | str, videos: bool = True, curated: bool = True, workers: int = 4,
                 limit: int | None = None) -> list[str]:
    out = Path(out)
    specs = _plan()[:limit] if limit else _plan()
    jobs = [(out, s, i, videos) for i, s in enumerate(specs)]
    if videos and workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            seqs = list(ex.map(_write_pass_args, jobs))
    else:
        seqs = [write_pass(*j) for j in jobs]
    if curated:
        write_curated(out, seqs)
    return seqs


def ensure_fixtures(out: Path, videos: bool = True) -> Path:
    if not (out / "generated").exists() or not any((out / "generated").iterdir()):
        make_dataset(out, videos=videos)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fixtures")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    seqs = make_dataset(a.out, videos=not a.no_video, limit=a.limit, workers=a.workers)
    print(f"wrote {len(seqs)} passes to {a.out}")
