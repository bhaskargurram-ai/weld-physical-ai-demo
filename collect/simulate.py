"""WeldCell simulator: physically plausible robotic weld passes for a pipe-spool /
pressure-vessel fab shop.

Each pass is driven by its WPS (collect/schema.py::WPS_LIBRARY): setpoints are drawn
inside the qualified window, then the process model adds realistic structure:

* GTAW root (CC source): very steady current, arc-length driven voltage, cold-wire dabbing.
* FCAW hot/fill/cap (CV source): rectifier ripple, short-circuit transfer events
  (voltage collapse + current spike + re-ignition overshoot), drive-roll WFS jitter,
  CTWD drift that couples into current.
* SAW vessel seam (CV, flux shielded): smooth high-current arc, no shielding gas.

The pipe turns on a positioner (1G rolled), the robot holds the torch at top-dead-centre
and weaves across the joint (6-axis IK in collect/robot.py). An IR pyrometer tracks the
interpass temperature (arc heating + convective cooling, shop timeline between passes).

Defects are injected with the signatures in DEFECT_SIGNATURES, logged as events, and a
radiography (RT) accept/reject label is derived from defect type and severity.

CLI:
    python -m collect.simulate --n-passes 24 --seed 7 --out data/generated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from collect import robot as rb
from collect.schema import (ARC_HZ, CAMERA_FPS, ROBOT_HZ, THERMAL_HZ, WPS, WPS_LIBRARY, Defect,
                            DefectEvent, Position, Process, WeldPass, heat_input_kj_mm)

AMBIENT_C = 22.0
SHOP_START_UTC = "2026-11-02T12:45:00Z"   # 06:45 CST, Weatherford TX

# ------------------------------------------------------------------ job plan


@dataclass
class Joint:
    spool_id: str
    joint_id: str
    material: str
    pipe_od_mm: float
    wall_mm: float
    cell_id: str
    passes: list[tuple[str, str]]          # (pass_type, wps_id)
    day: int                               # shop day offset


PIPE_PASSES = [("root", "WPS-GTAW-ROOT-01"), ("hot", "WPS-FCAW-FILL-02"),
               ("fill", "WPS-FCAW-FILL-02"), ("cap", "WPS-FCAW-FILL-02")]
SAW_PASSES = [("root", "WPS-SAW-SEAM-03"), ("fill", "WPS-SAW-SEAM-03"),
              ("fill", "WPS-SAW-SEAM-03"), ("cap", "WPS-SAW-SEAM-03")]

BASE_JOBS = [
    # 6" Sch 80 A106 Gr.B process piping spool
    *(Joint("SP-1043", j, "A106 Gr.B", 168.3, 10.97, "WC-02", PIPE_PASSES, 0)
      for j in ("J05", "J06", "J07")),
    # 4" Sch 80 spool
    *(Joint("SP-1044", j, "A106 Gr.B", 114.3, 8.56, "WC-02", PIPE_PASSES, 1)
      for j in ("J02", "J03")),
    # ASME VIII vessel, circumferential shell seam on turning rolls, SAW
    Joint("V-207", "CS2", "SA-516 Gr.70", 1524.0, 25.4, "SAW-01", SAW_PASSES, 2),
]

# Scripted defects for the reference dataset -> stable demo story.
# key = (spool, joint, pass_no) -> (defect, severity)
STORY_DEFECTS: dict[tuple[str, str, int], tuple[Defect, float]] = {
    ("SP-1043", "J05", 3): (Defect.UNDERCUT, 0.35),         # WPS voltage excursion, RT accept
    ("SP-1043", "J06", 1): (Defect.BURN_THROUGH, 0.65),     # root melt-through, RT reject
    ("SP-1043", "J07", 3): (Defect.POROSITY, 0.80),         # gas loss on fill -> J07 fails RT
    ("SP-1043", "J07", 4): (Defect.UNDERCUT, 0.50),         # cap undercut (visual)
    ("SP-1044", "J02", 2): (Defect.ARC_INSTABILITY, 0.60),  # worn contact tip / liner
    ("SP-1044", "J03", 3): (Defect.POROSITY, 0.30),         # scattered porosity, within limits
    ("SP-1044", "J03", 4): (Defect.LACK_OF_FUSION, 0.70),   # travel too fast on cap
    ("V-207", "CS2", 3): (Defect.CRACK, 0.60),              # interpass exceeded on heavy wall
}

DEFECTS_BY_PROCESS = {
    Process.GTAW: [Defect.BURN_THROUGH, Defect.POROSITY, Defect.LACK_OF_FUSION],
    Process.FCAW: [Defect.POROSITY, Defect.UNDERCUT, Defect.LACK_OF_FUSION,
                   Defect.ARC_INSTABILITY, Defect.CRACK],
    Process.SAW: [Defect.CRACK, Defect.LACK_OF_FUSION, Defect.UNDERCUT],
}

# RT reject thresholds by defect type (severity >= threshold -> reject).
RT_REJECT_THRESHOLD = {
    Defect.CRACK: 0.0,             # cracks are never acceptable
    Defect.LACK_OF_FUSION: 0.25,   # incomplete fusion: reject unless trivial
    Defect.BURN_THROUGH: 0.40,
    Defect.POROSITY: 0.50,         # rounded-indication charts: scattered fine porosity OK
    Defect.UNDERCUT: 0.60,         # mostly a visual (VT) finding
    Defect.ARC_INSTABILITY: 0.75,  # only rejectable if it produced inclusions/LOF
}


def rt_result_for(defects: list[DefectEvent]) -> str:
    for d in defects:
        if d.severity >= RT_REJECT_THRESHOLD.get(d.defect, 0.5):
            return "reject"
    return "accept"


@dataclass
class PassPlan:
    joint: Joint
    pass_no: int
    pass_type: str
    wps_id: str
    defect: Optional[tuple[Defect, float]] = None

    @property
    def wps(self) -> WPS:
        return WPS_LIBRARY[self.wps_id]

    @property
    def process(self) -> Process:
        return self.wps.process

    @property
    def sequence_name(self) -> str:
        return f"{self.joint.spool_id}_{self.joint.joint_id}_P{self.pass_no}_{self.process.value}"


def build_plan(n_passes: int, rng: np.random.Generator) -> list[PassPlan]:
    jobs = list(BASE_JOBS)
    k = 0
    while sum(len(j.passes) for j in jobs) < n_passes:  # extend the shop with more spools
        k += 1
        od, wall = [(168.3, 10.97), (114.3, 8.56), (219.1, 12.7)][k % 3]
        jobs += [Joint(f"SP-{1044 + k}", f"J{j:02d}", "A106 Gr.B", od, wall, "WC-02",
                       PIPE_PASSES, 2 + k) for j in (1, 2)]
    plan: list[PassPlan] = []
    for jt in jobs:
        for i, (ptype, wps_id) in enumerate(jt.passes, start=1):
            pp = PassPlan(jt, i, ptype, wps_id)
            key = (jt.spool_id, jt.joint_id, i)
            if key in STORY_DEFECTS:
                d, sev = STORY_DEFECTS[key]
                pp.defect = (d, float(np.clip(sev + rng.normal(0, 0.03), 0.05, 0.98)))
            elif jt.spool_id not in {"SP-1043", "SP-1044", "V-207"} and rng.random() < 0.33:
                d = rng.choice(DEFECTS_BY_PROCESS[pp.process])
                pp.defect = (Defect(d), float(rng.uniform(0.2, 0.9)))
            plan.append(pp)
            if len(plan) >= n_passes:
                return plan
    return plan


# ------------------------------------------------------------------ helpers


def _envelope(t: np.ndarray, t0: float, t1: float, ramp: float = 0.35) -> np.ndarray:
    """Smooth 0..1 window with raised-cosine edges."""
    e = np.zeros_like(t)
    up = (t >= t0) & (t < t0 + ramp)
    e[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - t0) / ramp)
    e[(t >= t0 + ramp) & (t <= t1 - ramp)] = 1.0
    dn = (t > t1 - ramp) & (t <= t1)
    e[dn] = 0.5 + 0.5 * np.cos(np.pi * (t[dn] - (t1 - ramp)) / ramp)
    return e


def _smooth_noise(rng, n: int, hz: float, corner_hz: float, sigma: float) -> np.ndarray:
    """Low-pass (1st order) filtered gaussian noise with given std."""
    a = math.exp(-2 * math.pi * corner_hz / hz)
    w = rng.normal(0, 1, n)
    klen = int(min(n, max(8, 6.0 / max(1 - a, 1e-6))))
    kern = (1 - a) * a ** np.arange(klen)
    y = np.convolve(w, kern)[:n]
    s = y.std()
    return y / s * sigma if s > 0 else y


def _movavg(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def _mid(rng, lo_hi: tuple[float, float], lo_f=0.38, hi_f=0.62) -> float:
    lo, hi = lo_hi
    return lo + (hi - lo) * rng.uniform(lo_f, hi_f)


def _seed_for(name: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


# ------------------------------------------------------------------ pass physics

LAYER_MM = {"root": 3.0, "hot": 2.6, "fill": 3.2, "cap": 2.4}
WEAVE_MM = {("GTAW", "root"): (0.6, 1.0), ("FCAW", "hot"): (2.0, 1.4),
            ("FCAW", "fill"): (3.5, 1.2), ("FCAW", "cap"): (5.5, 1.0)}


def simulate_pass(pp: PassPlan, seed: int, t0_ns: int, interpass_start_c: float,
                  duration_s: Optional[float] = None) -> dict:
    """Simulate one pass. Returns dict with dataframes, events, WeldPass and extras."""
    rng = np.random.default_rng(_seed_for(pp.sequence_name, seed))
    wps, proc = pp.wps, pp.process
    D = float(duration_s if duration_s is not None else round(rng.uniform(22, 36), 2))
    n = int(round(D * ARC_HZ))
    t = np.arange(n) / ARC_HZ

    # ---- setpoints inside the WPS window (robot program values)
    I_set = _mid(rng, wps.current_a)
    V_set = _mid(rng, wps.voltage_v)
    TS_set = _mid(rng, wps.travel_speed_mmps)
    WFS_set = _mid(rng, wps.wfs_mpm)
    GAS_set = _mid(rng, wps.gas_lpm) if proc != Process.SAW else 0.0
    if pp.pass_type == "cap" and proc == Process.FCAW:   # caps run a bit colder & slower
        I_set *= 0.96
        TS_set *= 0.92
    # keep nominal heat input comfortably inside the WPS window
    hi_lo, hi_hi = wps.heat_input_kj_mm
    hi = heat_input_kj_mm(I_set, V_set, TS_set)
    target = hi_lo + (hi_hi - hi_lo) * rng.uniform(0.3, 0.55)
    if not (hi_lo + 0.12 * (hi_hi - hi_lo) < hi < hi_hi - 0.2 * (hi_hi - hi_lo)):
        TS_set = float(np.clip(TS_set * hi / target, *wps.travel_speed_mmps))

    # ---- arc on/off envelope
    arc_off_t = D - 0.4
    ramp_up = np.clip(t / 0.25, 0, 1)
    crater = np.clip((arc_off_t - t) / 0.6, 0.55, 1.0)
    on = t < arc_off_t
    env_arc = ramp_up * crater * on

    # ---- weave + CTWD drift
    weave_amp, weave_hz = WEAVE_MM.get((proc.value, pp.pass_type), (0.0, 0.0))
    weave_amp *= rng.uniform(0.9, 1.1)
    weave = weave_amp * np.sin(2 * np.pi * weave_hz * t)
    ctwd_nom = {Process.GTAW: 3.0, Process.FCAW: 19.0, Process.SAW: 30.0}[proc]
    ctwd = ctwd_nom + _smooth_noise(rng, n, ARC_HZ, 0.3, 0.6) + 0.25 * np.abs(weave) / max(weave_amp, 1e-6) * (weave_amp > 0)

    # ---- base process signals
    if proc == Process.GTAW:
        current = I_set + rng.normal(0, 0.6, n) + _smooth_noise(rng, n, ARC_HZ, 0.5, 0.8)
        voltage = V_set + 0.45 * (ctwd - ctwd_nom) + rng.normal(0, 0.12, n)
        dab = (np.sin(2 * np.pi * 0.9 * t + rng.uniform(0, 6)) > 0.2).astype(float)
        wfs = WFS_set * 1.6 * dab + rng.normal(0, 0.02, n)
        gas = GAS_set + rng.normal(0, 0.15, n) + _smooth_noise(rng, n, ARC_HZ, 0.2, 0.15)
        sc_rate = 0.0
    elif proc == Process.FCAW:
        wfs_jit = _smooth_noise(rng, n, ARC_HZ, 2.0, 0.06) + rng.normal(0, 0.03, n)
        wfs = WFS_set + wfs_jit
        # CV: current follows wire feed, falls with longer stick-out
        current = I_set + 18.0 * (wfs - WFS_set) - 3.2 * (ctwd - ctwd_nom) + rng.normal(0, 2.4, n)
        ripple = 0.22 * np.sin(2 * np.pi * 7.3 * t + rng.uniform(0, 6))  # aliased 360 Hz ripple
        voltage = V_set + ripple + 0.08 * (ctwd - ctwd_nom) + rng.normal(0, 0.28, n)
        gas = GAS_set + rng.normal(0, 0.25, n) + _smooth_noise(rng, n, ARC_HZ, 0.2, 0.2)
        sc_rate = 4.5   # short-circuit events per second visible at 100 Hz
    else:  # SAW
        wfs = WFS_set + _smooth_noise(rng, n, ARC_HZ, 1.0, 0.02) + rng.normal(0, 0.01, n)
        current = I_set + 60.0 * (wfs - WFS_set) + rng.normal(0, 2.5, n) + _smooth_noise(rng, n, ARC_HZ, 0.3, 3.0)
        voltage = V_set + rng.normal(0, 0.12, n) + _smooth_noise(rng, n, ARC_HZ, 0.3, 0.12)
        gas = np.abs(rng.normal(0, 0.005, n))
        sc_rate = 0.0
    travel = TS_set * (1 + rng.normal(0, 0.004, n) + 0.006 * np.sin(2 * np.pi * 0.5 * t))

    # ---- defect injection
    defects: list[DefectEvent] = []
    sc_mult = np.ones(n)
    extra_v_spikes = np.zeros(n)
    if pp.defect is not None:
        d, sev = pp.defect
        dur = float(np.clip(rng.uniform(3.0, 7.0) * (0.7 + 0.6 * sev), 2.5, 9.0))
        ts = float(rng.uniform(0.25 * D, max(0.25 * D + 0.1, 0.85 * D - dur)))
        te = min(ts + dur, arc_off_t - 0.5)
        e = _envelope(t, ts, te)
        if d == Defect.POROSITY:
            if proc != Process.SAW:
                g_low = 7.5 - 4.5 * sev
                gas = gas * (1 - e) + e * (g_low + rng.normal(0, 0.6, n))
            spikes = (rng.random(n) < (8 + 14 * sev) / ARC_HZ) & (e > 0.5)
            extra_v_spikes += spikes * rng.uniform(1.5, 4 + 4 * sev, n)
            voltage = voltage + e * rng.normal(0, 0.4 + 0.9 * sev, n)
            sc_mult += e * 1.5
        elif d == Defect.LACK_OF_FUSION:
            current = current * (1 - e * (0.20 + 0.10 * sev))
            travel = travel * (1 + e * (0.25 + 0.20 * sev))
        elif d == Defect.UNDERCUT:
            voltage = voltage * (1 + e * (0.15 + 0.10 * sev))
            travel = travel * (1 + e * 0.2 * (0.8 + 0.4 * sev))
        elif d == Defect.BURN_THROUGH:
            current = current * (1 + e * (0.22 + 0.12 * sev))
            travel = travel * (1 - e * (0.30 + 0.15 * sev))
        elif d == Defect.ARC_INSTABILITY:
            voltage = voltage + e * rng.normal(0, 1.5 + 2.5 * sev, n)
            osc = (0.8 + 0.8 * sev) * np.sin(2 * np.pi * (4.5 + rng.normal(0, 0.3)) * t)
            wfs = wfs + e * (osc + rng.normal(0, 0.25, n))
            current = current + e * (18.0 * osc + rng.normal(0, 6 + 8 * sev, n))
            sc_mult += e * 4.0
        elif d == Defect.CRACK:
            pass  # thermal: handled below (interpass above WPS max)
        defects.append(DefectEvent(d, round(ts, 3), round(te, 3), round(sev, 3)))

    # ---- short-circuit transfer events (FCAW)
    if sc_rate > 0:
        p = sc_rate * sc_mult / ARC_HZ
        hits = np.flatnonzero((rng.random(n) < p) & (env_arc > 0.9))
        for i in hits:
            voltage[i] *= rng.uniform(0.25, 0.5)
            current[i] *= rng.uniform(1.18, 1.38)
            if i + 1 < n:
                voltage[i + 1] += rng.uniform(1.5, 4.0)   # re-ignition overshoot
    voltage = voltage + extra_v_spikes

    # apply arc envelope (arc off -> 0 A / 0 V, wire stopped; gas post-flow continues)
    current = np.clip(current * env_arc, 0, None)
    voltage = np.where(on, np.clip(voltage * np.clip(ramp_up * 1.3, 0, 1), 0, None), 0.0)
    wfs = np.clip(wfs * env_arc, 0, None)
    gas = np.clip(gas, 0, None)
    travel = np.where(t < arc_off_t + 0.15, travel * np.clip(t / 0.3, 0.2, 1), 0.0)

    # heat input from 0.5 s averages (what a weld monitor reports)
    Im, Vm, Tm = (_movavg(x, ARC_HZ // 2) for x in (current, voltage, travel))
    with np.errstate(divide="ignore", invalid="ignore"):
        hi_series = np.where((Tm > 0.05) & (Im > 5), 0.8 * Vm * Im / (1000.0 * Tm), 0.0)
    hi_series = np.clip(np.nan_to_num(hi_series), 0, 20)

    R_mm = pp.joint.pipe_od_mm / 2.0
    dist = np.cumsum(travel) / ARC_HZ
    positioner = np.degrees(dist / R_mm)

    t_ns_arc = t0_ns + (np.arange(n) * (1_000_000_000 // ARC_HZ)).astype(np.int64)
    arc_df = pd.DataFrame({"t_ns": t_ns_arc, "current_a": current.astype(np.float64),
                           "voltage_v": voltage, "wfs_mpm": wfs, "gas_lpm": gas})
    travel_df = pd.DataFrame({"t_ns": t_ns_arc, "travel_speed_mmps": travel,
                              "positioner_deg": positioner, "heat_input_kj_mm": hi_series})

    # ---- thermal (IR pyrometer ~25 mm from weld toe) @ THERMAL_HZ
    nt = int(round(D * THERMAL_HZ))
    tt = np.arange(nt) / THERMAL_HZ
    power = 0.8 * np.interp(tt, t, current * voltage)       # W into the work
    k_heat = 0.0050 / pp.joint.wall_mm * (114.3 / max(pp.joint.pipe_od_mm, 114.3)) ** 0.15
    tau = 28.0 * pp.joint.wall_mm                             # s
    T = np.empty(nt)
    Tc = interpass_start_c
    for i in range(nt):
        Tc += (k_heat * power[i] - (Tc - AMBIENT_C) / tau) / THERMAL_HZ
        T[i] = Tc
    weave_tt = np.interp(tt, t, weave)
    T_meas = T + rng.normal(0, 0.7, nt) + (0.6 * weave_tt / max(weave_amp, 1) if weave_amp else 0)
    glare = (rng.random(nt) < (0.25 if proc == Process.FCAW else 0.05) / THERMAL_HZ * 10) & (tt < arc_off_t)
    T_meas = T_meas + glare * rng.uniform(4, 14, nt)
    thermal_df = pd.DataFrame({"t_ns": t0_ns + (np.arange(nt) * (1_000_000_000 // THERMAL_HZ)).astype(np.int64),
                               "interpass_c": T_meas})

    # ---- robot @ ROBOT_HZ: torch at TDC, weave across the joint (base y), seam tracking drift
    nr = int(round(D * ROBOT_HZ))
    tr = np.arange(nr) / ROBOT_HZ
    layer_off = sum(LAYER_MM[p] for p, _ in pp.joint.passes[: pp.pass_no - 1])
    track = _smooth_noise(rng, nr, ROBOT_HZ, 0.1, 0.25)        # mm, seam-tracking correction
    ctwd_r = np.interp(tr, t, ctwd)
    pos = np.column_stack([
        np.full(nr, 0.950),
        (np.interp(tr, t, weave) + track) / 1000.0,
        (120.0 + layer_off + (ctwd_r - ctwd_nom)) / 1000.0,
    ])
    push = {Process.GTAW: 12.0, Process.FCAW: 10.0, Process.SAW: 0.0}[proc]
    Q, P, Qt = rb.solve_trajectory(pos, rb.torch_rotation(push_deg=push))
    Q = Q + rng.normal(0, 2e-5, Q.shape)                      # encoder quantisation noise
    robot_df = pd.DataFrame({"t_ns": t0_ns + (np.arange(nr) * (1_000_000_000 // ROBOT_HZ)).astype(np.int64)})
    for j in range(6):
        robot_df[f"j{j + 1}"] = Q[:, j]
    for c, k in zip(("tcp_x", "tcp_y", "tcp_z"), range(3)):
        robot_df[c] = P[:, k]
    for c, k in zip(("tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"), range(4)):
        robot_df[c] = Qt[:, k]

    # ---- crack: driven by interpass above WPS max (window = where it is exceeded)
    if pp.defect is not None and pp.defect[0] == Defect.CRACK:
        over = tt[T > wps.max_interpass_c]
        if len(over):
            ts = float(max(over[0], 1.0))
            te = float(min(ts + rng.uniform(4, 8), arc_off_t - 0.3))
            defects[-1] = DefectEvent(Defect.CRACK, round(ts, 3), round(te, 3), defects[-1].severity)

    # ---- events
    def ev(ts_s, typ, **kw):
        return {"t_ns": int(t0_ns + round(ts_s * 1e9)), "type": typ, **kw}
    events = [ev(0.0, "arc_start", process=proc.value)]
    for de in defects:
        events.append(ev(de.t_start, "defect_on", defect=de.defect.value, severity=de.severity))
        events.append(ev(de.t_end, "defect_off", defect=de.defect.value, severity=de.severity))
    events.append(ev(arc_off_t, "arc_stop"))
    events.sort(key=lambda e: e["t_ns"])

    wp = WeldPass(
        sequence_name=pp.sequence_name, spool_id=pp.joint.spool_id, joint_id=pp.joint.joint_id,
        pass_no=pp.pass_no, pass_type=pp.pass_type, wps_id=pp.wps_id, process=proc,
        position=Position.P1G, material=pp.joint.material, pipe_od_mm=pp.joint.pipe_od_mm,
        wall_mm=pp.joint.wall_mm, cell_id=pp.joint.cell_id, duration_s=D, defects=defects,
        rt_result=rt_result_for(defects))

    arc_mask = current > 10
    extras = {
        "t0_ns": int(t0_ns),
        "start_time_utc": pd.Timestamp(t0_ns, unit="ns", tz="UTC").isoformat(),
        "has_defect": bool(defects),
        "defect_types": ",".join(sorted({d.defect.value for d in defects})) or "none",
        "max_severity": float(max([d.severity for d in defects], default=0.0)),
        "setpoints": {"current_a": round(I_set, 1), "voltage_v": round(V_set, 2),
                      "travel_speed_mmps": round(TS_set, 2), "wfs_mpm": round(WFS_set, 2),
                      "gas_lpm": round(GAS_set, 1)},
        "weave_amp_mm": round(float(weave_amp), 3), "weave_hz": float(weave_hz),
        "ctwd_mm": ctwd_nom, "push_angle_deg": push,
        "layer_offset_mm": layer_off, "layer_mm": LAYER_MM[pp.pass_type],
        "heat_input_mean_kj_mm": round(float(hi_series[arc_mask].mean()), 3),
        "current_mean_a": round(float(current[arc_mask].mean()), 1),
        "voltage_mean_v": round(float(voltage[arc_mask].mean()), 2),
        "travel_mean_mmps": round(float(travel[arc_mask].mean()), 2),
        "interpass_start_c": round(float(T[0]), 1),
        "interpass_max_c": round(float(T.max()), 1),
        "arc_on_s": round(arc_off_t, 2),
        "rates_hz": {"arc": ARC_HZ, "travel": ARC_HZ, "robot": ROBOT_HZ, "thermal": THERMAL_HZ,
                     "camera": CAMERA_FPS},
        "simulator": {"name": "WeldCell-sim", "version": "1.0", "seed": seed},
    }
    return {"arc": arc_df, "travel": travel_df, "robot": robot_df, "thermal": thermal_df,
            "events": events, "pass": wp, "extras": extras, "T_end": float(T[-1])}


def wps_dict(w: WPS) -> dict:
    d = asdict(w)
    d["process"] = w.process.value
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def write_pass_dir(out_root: Path, sim: dict) -> Path:
    wp: WeldPass = sim["pass"]
    d = out_root / wp.sequence_name
    (d / "control").mkdir(parents=True, exist_ok=True)
    meta = wp.to_metadata()
    meta.update(sim["extras"])
    meta["wps"] = wps_dict(WPS_LIBRARY[wp.wps_id])
    for k in ("arc", "travel", "robot", "thermal"):
        sim[k].to_parquet(d / f"{k}.parquet", index=False)
    with open(d / "events.jsonl", "w") as f:
        for e in sim["events"]:
            f.write(json.dumps(e) + "\n")
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return d


# ------------------------------------------------------------------ shop timeline


def simulate_shop(n_passes: int = 24, seed: int = 7, out: str | Path = "data/generated",
                  render: bool = True, workers: Optional[int] = None,
                  duration_s: Optional[float] = None, clean: bool = True,
                  quiet: bool = False) -> list[Path]:
    out = Path(out)
    if clean and out.exists():
        for p in out.iterdir():
            if p.is_dir() and (p / "meta.json").exists():
                shutil.rmtree(p)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    plan = build_plan(n_passes, rng)

    t_day0 = pd.Timestamp(SHOP_START_UTC).value
    clock: dict[int, int] = {}
    dirs: list[Path] = []
    metas: list[dict] = []
    prev_joint, T_end, t_cursor = None, AMBIENT_C, 0
    for pp in plan:
        jt = pp.joint
        if jt.day not in clock:
            clock[jt.day] = t_day0 + jt.day * 86_400 * 10**9
        if prev_joint is not jt:
            t_cursor = clock[jt.day] + int(rng.uniform(5, 25) * 60e9)   # fit-up / setup
            # preheat: torch preheat to WPS minimum (heavier wall -> hotter)
            T_start = 50.0 + rng.uniform(0, 15) if jt.wall_mm < 20 else 95.0 + rng.uniform(0, 10)
        else:
            # wait for interpass: cool toward ambient, CWI checks with temp stick
            tau = 28.0 * jt.wall_mm
            T_target = rng.uniform(100, 160) if jt.wall_mm < 20 else rng.uniform(140, 180)
            if pp.defect and pp.defect[0] == Defect.CRACK:
                # welder did not wait for interpass (and kept the joint hot with the torch)
                T_start = pp.wps.max_interpass_c + 6.0 + 25.0 * pp.defect[1]
                gap = 45.0
            else:
                T_target = min(T_target, T_end - 1.0)
                gap = max(90.0, -tau * math.log(max((T_target - AMBIENT_C) / max(T_end - AMBIENT_C, 1e-3), 1e-3)))
                T_start = AMBIENT_C + (T_end - AMBIENT_C) * math.exp(-gap / tau)
            t_cursor += int((gap + rng.uniform(40, 90)) * 1e9)              # + slag cleaning
        sim = simulate_pass(pp, seed, t_cursor, T_start, duration_s=duration_s)
        T_end = sim["T_end"]
        t_cursor += int(sim["pass"].duration_s * 1e9)
        clock[jt.day] = t_cursor + int(rng.uniform(15, 35) * 60e9)
        prev_joint = jt
        d = write_pass_dir(out, sim)
        dirs.append(d)
        metas.append(json.loads((d / "meta.json").read_text()))
        if not quiet:
            m = metas[-1]
            print(f"  {m['sequence_name']:<22} {m['duration_s']:5.1f}s  HI={m['heat_input_mean_kj_mm']:.2f}  "
                  f"Tip={m['interpass_start_c']:.0f}->{m['interpass_max_c']:.0f}C  "
                  f"defect={m['defect_types']:<16} RT={m['rt_result']}", flush=True)

    # joint-level RT (film shot on the completed joint)
    joint_rt: dict[tuple, str] = {}
    for m in metas:
        k = (m["spool_id"], m["joint_id"])
        joint_rt[k] = "reject" if (m["rt_result"] == "reject" or joint_rt.get(k) == "reject") else "accept"
    for d, m in zip(dirs, metas):
        m["joint_rt_result"] = joint_rt[(m["spool_id"], m["joint_id"])]
        (d / "meta.json").write_text(json.dumps(m, indent=2))
    with open(out / "index.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps({k: m[k] for k in (
                "sequence_name", "spool_id", "joint_id", "pass_no", "pass_type", "process", "wps_id",
                "duration_s", "has_defect", "defect_types", "rt_result", "joint_rt_result",
                "start_time_utc")}) + "\n")

    if render:
        from collect.render import render_many
        render_many(dirs, workers=workers, quiet=quiet)
    return dirs


def main(argv=None):
    ap = argparse.ArgumentParser(description="WeldCell simulator -> data/generated/<sequence>/")
    ap.add_argument("--n-passes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "generated"))
    ap.add_argument("--no-render", action="store_true", help="skip video rendering (signals only)")
    ap.add_argument("--workers", type=int, default=None, help="render worker processes")
    ap.add_argument("--duration", type=float, default=None, help="force pass duration [s]")
    args = ap.parse_args(argv)
    t = time.time()
    print(f"WeldCell sim: {args.n_passes} passes, seed={args.seed} -> {args.out}")
    dirs = simulate_shop(args.n_passes, args.seed, args.out, render=not args.no_render,
                         workers=args.workers, duration_s=args.duration)
    n_def = sum(json.loads((d / "meta.json").read_text())["has_defect"] for d in dirs)
    print(f"done: {len(dirs)} passes ({n_def} with defects) in {time.time() - t:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
