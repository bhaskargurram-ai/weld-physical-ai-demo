"""Procedural torch-camera renderer for WeldCell passes (OpenCV + numpy, CPU).

Renders what a torch-mounted HDR weld camera sees, driven by the simulated signals:

* camera rides on the torch, looking forward/down at the arc on a rolling pipe
  (ray/cylinder intersection -> perspective, curved horizon, surface flowing past
  as the positioner turns, lateral shift from the robot weave)
* groove geometry per pass (open root, bevel faces, previous layer), solidifying bead
  with chevron ripples, slag (FCAW/SAW) or heat tint (GTAW), temperature-driven
  incandescence (blackbody ramp) cooling behind the pool
* weld pool + arc glow with flicker from voltage/current (short circuits dim the arc),
  bloom, fume, spatter particles (rate grows with instability)
* defects: porosity pits + pool bubbling, undercut grooves at the toes, lack-of-fusion
  cold lap, burn-through hole, centreline crack, arc instability (flicker + spatter)

Also writes control videos with identical timing for Cosmos Transfer (sim -> real):
``control/edge.mp4`` (Canny edges of the scene geometry) and ``control/depth.mp4``
(inverse depth, near = bright).

CLI:
    python -m collect.render data/generated/SP-1043_J07_P3_FCAW [--stills docs/img]
    python -m collect.render data/generated --workers 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np
import pandas as pd

from collect.schema import CAMERA_FPS

W, H = 960, 540
F32 = np.float32

# groove fill fraction (of wall) before/after each pass index (4-pass joints)
FILL_FRAC = [0.0, 0.24, 0.47, 0.78, 1.0]
BEVEL_DEG = 37.5
ROOT_GAP_MM = 3.0


# ---------------------------------------------------------------- utilities


def _noise_tile(rng, h, w, scales=((4, 0.5), (16, 0.3), (64, 0.2))) -> np.ndarray:
    """Tileable-ish multi-octave value noise in [0,1]."""
    acc = np.zeros((h, w), F32)
    for cell, amp in scales:
        sh, sw = max(2, h // cell), max(2, w // cell)
        g = rng.random((sh, sw)).astype(F32)
        g = np.pad(g, ((0, 1), (0, 1)), mode="wrap")
        up = cv2.resize(g, (w + w // sw, h + h // sh), interpolation=cv2.INTER_CUBIC)[:h, :w]
        acc += amp * up
    acc -= acc.min()
    return acc / max(acc.max(), 1e-6)


def _lut(values: np.ndarray, x0: float, dx: float, coords: np.ndarray) -> np.ndarray:
    """Fast 1-D lookup (linear, clamped) via cv2.remap on a 1-row image."""
    img = np.ascontiguousarray(values.astype(F32)[None])   # (1,N) or (1,N,C)
    mx = ((coords - x0) / dx).astype(F32)
    my = np.zeros_like(mx)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _sample(tex: np.ndarray, mx: np.ndarray, my: np.ndarray, wrap=False, border=0.0) -> np.ndarray:
    mode = cv2.BORDER_WRAP if wrap else cv2.BORDER_CONSTANT
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=mode, borderValue=border)


def blackbody(T: np.ndarray) -> np.ndarray:
    """Approximate incandescence colour * intensity for temperature in deg C (linear RGB)."""
    x = np.clip((T - 480.0) / 1100.0, 0, 1.4).astype(F32)
    r = np.clip(x * 3.0, 0, 1)
    g = np.clip((x - 0.22) * 1.6, 0, 1) ** 1.3
    b = np.clip((x - 0.62) * 1.9, 0, 1) ** 1.5
    k = (x ** 2.6) * 7.0
    return np.stack([r * k, g * k, b * k], axis=-1)


# ---------------------------------------------------------------- camera / geometry


class Camera:
    def __init__(self, R: float, process: str, w=W, h=H):
        self.R = R
        self.w, self.h = w, h
        fov = 54.0 if process != "SAW" else 60.0
        self.f = (w / 2) / math.tan(math.radians(fov / 2))
        # side-mounted weld camera: behind the torch, yawed off the seam, looking down at the pool
        dist, elev, az = (105.0, 34.0, 28.0) if process != "SAW" else (135.0, 36.0, 26.0)
        self.A = np.array([0.0, 0.0, R])                       # arc point (TDC)
        el, az = math.radians(elev), math.radians(az)
        back = np.array([-math.sin(az), -math.cos(az), 0.0])
        self.C = self.A + dist * (math.cos(el) * back + np.array([0, 0, math.sin(el)]))
        target = self.A + np.array([4.0, 9.0 if process != "SAW" else 22.0, -1.0])
        fwd = target - self.C
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self.fwd, [0, 0, 1.0])
        self.right = right / np.linalg.norm(right)
        self.down = np.cross(self.fwd, self.right)

    def project(self, P: np.ndarray) -> np.ndarray:
        """World points (...,3) -> pixel coords (...,2) and depth."""
        d = P - self.C
        z = d @ self.fwd
        x = d @ self.right
        y = d @ self.down
        return np.stack([self.w / 2 + self.f * x / z, self.h / 2 + self.f * y / z], -1), z

    def surface_maps(self):
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float64)
        dc = np.stack([(xs - self.w / 2) / self.f, (ys - self.h / 2) / self.f, np.ones_like(xs)], -1)
        dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
        d = dc[..., 0:1] * self.right + dc[..., 1:2] * self.down + dc[..., 2:3] * self.fwd
        C, R = self.C, self.R
        a = d[..., 1] ** 2 + d[..., 2] ** 2
        b = 2 * (C[1] * d[..., 1] + C[2] * d[..., 2])
        c = C[1] ** 2 + C[2] ** 2 - R ** 2
        disc = b * b - 4 * a * c
        hit = disc > 0
        t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), 0)
        hit &= t > 0
        P = C + t[..., None] * d
        u = R * np.arctan2(P[..., 1], P[..., 2])
        v = P[..., 0]
        n = np.stack([np.zeros_like(u), P[..., 1] / R, P[..., 2] / R], -1)
        tu = np.stack([np.zeros_like(u), P[..., 2] / R, -P[..., 1] / R], -1)
        tv = np.zeros_like(n)
        tv[..., 0] = 1.0
        zdepth = t * (d @ self.fwd)
        cos_view = np.clip(-(d * n).sum(-1), 0.05, 1)
        return dict(hit=hit, u=u, v=v, n=n, tu=tu, tv=tv, P=P, d=d, z=zdepth, cos_view=cos_view)


def _light_terms(m: dict, L_pos: np.ndarray, r0: float):
    """Per-pixel light direction in the local (tu, tv, n) frame + attenuation + half vector."""
    Lv = L_pos - m["P"]
    r = np.linalg.norm(Lv, axis=-1)
    l = Lv / np.maximum(r[..., None], 1e-6)
    view = -m["d"]
    hv = l + view
    hv /= np.maximum(np.linalg.norm(hv, axis=-1, keepdims=True), 1e-6)
    comp = lambda vec: [(vec * m[k]).sum(-1).astype(F32) for k in ("tu", "tv", "n")]
    att = (1.0 / (1.0 + (r / r0) ** 2)).astype(F32)
    return comp(l), comp(hv), att


def _dir_terms(m: dict, direction: np.ndarray):
    l = np.broadcast_to(direction / np.linalg.norm(direction), m["P"].shape)
    view = -m["d"]
    hv = l + view
    hv = hv / np.maximum(np.linalg.norm(hv, axis=-1, keepdims=True), 1e-6)
    comp = lambda vec: [(vec * m[k]).sum(-1).astype(F32) for k in ("tu", "tv", "n")]
    return comp(l), comp(hv)


# ---------------------------------------------------------------- pass model


class PassScene:
    def __init__(self, pass_dir: Path, seed: Optional[int] = None):
        self.dir = Path(pass_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        m = self.meta
        self.process = m["process"]
        self.pass_type = m["pass_type"]
        if seed is None:
            seed = int.from_bytes(hashlib.sha256(m["sequence_name"].encode()).digest()[:8], "little")
        self.rng = np.random.default_rng(seed)
        arc = pd.read_parquet(self.dir / "arc.parquet")
        trv = pd.read_parquet(self.dir / "travel.parquet")
        self.t = ((arc["t_ns"].to_numpy() - arc["t_ns"].iloc[0]) / 1e9).astype(np.float64)
        self.I = arc["current_a"].to_numpy()
        self.V = arc["voltage_v"].to_numpy()
        self.WFS = arc["wfs_mpm"].to_numpy()
        self.GAS = arc["gas_lpm"].to_numpy()
        self.TS = trv["travel_speed_mmps"].to_numpy()
        self.HI = trv["heat_input_kj_mm"].to_numpy()
        self.s = np.cumsum(self.TS) / 100.0                  # mm travelled
        self.D = float(m["duration_s"])
        sp = m.get("setpoints", {})
        self.I_nom = float(sp.get("current_a", np.median(self.I[self.I > 10])))
        self.V_nom = float(sp.get("voltage_v", np.median(self.V[self.V > 1])))
        self.P_nom = self.I_nom * self.V_nom
        self.weave_amp = float(m.get("weave_amp_mm", 0.0))
        self.weave_hz = float(m.get("weave_hz", 0.0))
        self.defects = m.get("defects", [])
        self.R = float(m["pipe_od_mm"]) / 2.0
        self.wall = float(m["wall_mm"])
        k = int(m["pass_no"])
        n_passes = 4
        self.f_prev = FILL_FRAC[min(k - 1, n_passes)]
        self.f_now = FILL_FRAC[min(k, n_passes)]
        self.is_cap = self.pass_type == "cap"
        self.is_root = self.pass_type == "root"
        self.tan_b = math.tan(math.radians(BEVEL_DEG))
        vis_wall = min(self.wall, 14.0)                      # visual depth cap (parallax-free shading)
        self.depth_scale = vis_wall / self.wall
        self.Wt = ROOT_GAP_MM / 2 + self.wall * self.tan_b   # groove top half width (real mm)
        self.T_ip = float(m.get("interpass_start_c", 100.0))
        self._build_profiles()
        self._build_detail_texture()

    # ---- bead / groove 1-D profiles -------------------------------------------------
    def _half_width_at_depth(self, y):
        return ROOT_GAP_MM / 2 + np.maximum(self.wall - y, 0) * self.tan_b

    def _build_profiles(self):
        # groove profile before this pass, as a function of |v| (height mm, <= 0)
        y_p = self.wall * (1 - self.f_prev)
        hw_p = self._half_width_at_depth(y_p)
        self.vgrid = np.linspace(0, self.Wt + 25, 900)
        vg = self.vgrid
        h = np.zeros_like(vg)
        wall_zone = (vg > hw_p) & (vg < self.Wt)
        h[wall_zone] = -(self.Wt - vg[wall_zone]) / self.tan_b
        inner = vg <= hw_p
        if self.is_root:
            # open root: root faces then the gap (deep)
            h[inner] = -self.wall
            h[vg <= ROOT_GAP_MM / 2] = -self.wall - 6.0
        else:
            prev_crown = 0.5
            h[inner] = -y_p + prev_crown * (1 - (vg[inner] / max(hw_p, 1e-3)) ** 2)
        self.h_prev_v = (h * self.depth_scale).astype(F32)
        self.bevel_v = wall_zone.astype(F32)
        self.prev_bead_v = (inner & ~((vg <= ROOT_GAP_MM / 2) & self.is_root)).astype(F32) * (0 if self.is_root else 1)
        self.gap_v = ((vg <= ROOT_GAP_MM / 2) & self.is_root).astype(F32)
        self.hw_p = hw_p

        # bead after this pass, as a function of u (deposition conditions)
        y_n = self.wall * (1 - self.f_now)
        b_nom = self._half_width_at_depth(y_n) + (1.4 if self.is_cap else 0.5)
        if self.is_root:
            b_nom = ROOT_GAP_MM / 2 + 1.6
        self.y_n = y_n
        HI_nom = np.median(self.HI[self.I > 10]) if np.any(self.I > 10) else 1.0
        s_u = self.s
        self.u_grid = np.linspace(-60, s_u[-1] + 60, int((s_u[-1] + 120) / 0.1))
        tdep = np.interp(self.u_grid, s_u, self.t, left=-1e3, right=1e9)
        self.tdep_u = tdep
        hi_u = np.interp(tdep, self.t, self.HI)
        ratio = np.clip(hi_u / max(HI_nom, 1e-3), 0.55, 1.6)
        b_u = b_nom * np.sqrt(ratio) if self.is_cap else b_nom * (0.85 + 0.15 * ratio)
        crown_u = (1.2 if self.is_cap else 0.7) * np.clip(1 / np.sqrt(ratio), 0.7, 1.6)
        crown_u = np.full_like(self.u_grid, 1.0) * crown_u
        level_u = np.full_like(self.u_grid, -y_n)
        # defect modifiers along u
        self.uc_u = np.zeros_like(self.u_grid)     # undercut depth
        self.lof_u = np.zeros_like(self.u_grid)    # cold-lap crevice
        self.bt_u = np.zeros_like(self.u_grid)     # burn-through
        self.cr_u = np.zeros_like(self.u_grid)     # crack
        self.por_u = np.zeros_like(self.u_grid)
        for d in self.defects:
            env = np.clip(np.minimum(tdep - d["t_start"], d["t_end"] - tdep) / 0.4, 0, 1)
            sev = float(d["severity"])
            kind = d["defect"]
            if kind == "undercut":
                self.uc_u = np.maximum(self.uc_u, env * (0.5 + 0.9 * sev))
            elif kind == "lack_of_fusion":
                self.lof_u = np.maximum(self.lof_u, env * (0.6 + 0.8 * sev))
                b_u = b_u * (1 - 0.28 * env)
                crown_u = crown_u * (1 + 0.6 * env)
            elif kind == "burn_through":
                self.bt_u = np.maximum(self.bt_u, env * sev)
                level_u = level_u - 2.0 * env * sev
            elif kind == "crack":
                self.cr_u = np.maximum(self.cr_u, env)
            elif kind == "porosity":
                self.por_u = np.maximum(self.por_u, env * sev)
        self.b_u = b_u.astype(F32)
        self.crown_u = crown_u.astype(F32)
        self.level_u = (level_u * self.depth_scale).astype(F32)
        self.u0, self.du = float(self.u_grid[0]), float(self.u_grid[1] - self.u_grid[0])
        self.ripple_pitch = {"GTAW": max(0.8, float(np.median(self.TS)) / 0.9),
                             "FCAW": 1.05, "SAW": 0.85}[self.process]

    # ---- sparse detail texture (pores, crack, stuck spatter) in absolute coords ----
    def _build_detail_texture(self):
        rng = self.rng
        res = 0.08
        self.tex_res = res
        self.tex_v0 = -(self.Wt + 12.0)
        nv = int(2 * (self.Wt + 12.0) / res)
        nu = int((self.u_grid[-1] - self.u0) / res)
        nu = min(nu, 9000)
        dh = np.zeros((nv, nu), F32)
        dark = np.zeros((nv, nu), F32)
        crack = np.zeros((nv, nu), F32)
        to_px = lambda u, v: (int((u - self.u0) / res), int((v - self.tex_v0) / res))
        # pores
        for i in np.flatnonzero(self.por_u > 0.05)[:: max(1, int(0.6 / self.du))]:
            u = self.u_grid[i]
            sev = self.por_u[i]
            if rng.random() < 0.55 * sev:
                b = float(self.b_u[i])
                v = rng.uniform(-0.8 * b, 0.8 * b)
                r = rng.uniform(0.18, 0.45 + 0.6 * sev)
                cx, cy = to_px(u, v)
                rp = max(1, int(r / res))
                cv2.circle(dh, (cx, cy), rp, -0.6 - r, -1, lineType=cv2.LINE_AA)
                cv2.circle(dark, (cx, cy), rp, 1.0, -1, lineType=cv2.LINE_AA)
                cv2.circle(dh, (cx, cy), rp + 2, 0.12, 2, lineType=cv2.LINE_AA)
        # crack: jagged centreline
        idx = np.flatnonzero(self.cr_u > 0.2)
        if len(idx):
            us = self.u_grid[idx[0]: idx[-1]: 8]
            v = 0.0
            pts = []
            for u in us:
                v = float(np.clip(v + rng.normal(0, 0.12), -1.2, 1.2))
                pts.append(to_px(u, v))
            pts = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(crack, [pts], False, 1.0, max(1, int(0.22 / res)), lineType=cv2.LINE_AA)
            for _ in range(len(us) // 25):     # transverse branches
                j = rng.integers(0, len(pts))
                p0 = pts[j, 0]
                p1 = p0 + np.array([rng.integers(-25, 25), rng.integers(-40, 40)])
                cv2.line(crack, tuple(int(x) for x in p0), tuple(int(x) for x in p1), 0.7, 1, cv2.LINE_AA)
        # stuck spatter globules beside the bead (FCAW)
        if self.process == "FCAW":
            inst = 1.0 + 3.0 * float(any(d["defect"] == "arc_instability" for d in self.defects))
            n_sp = int((self.u_grid[-1] - self.u0) * 0.12 * inst)
            us = rng.uniform(self.u0, self.u_grid[-1], n_sp)
            sides = rng.choice([-1, 1], n_sp)
            vs = sides * (self.Wt + rng.exponential(3.0, n_sp) + 0.5)
            for u, v in zip(us, vs):
                cx, cy = to_px(u, v)
                rp = max(1, int(rng.uniform(0.15, 0.55) / res))
                cv2.circle(dh, (cx, cy), rp, 0.35, -1, lineType=cv2.LINE_AA)
        k = 3
        dh = cv2.GaussianBlur(dh, (k, k), 0)
        self.tex_dh = dh
        gy, gx = np.gradient(dh, res)
        self.tex_gu, self.tex_gv = gx.astype(F32), gy.astype(F32)
        self.tex_dark = cv2.GaussianBlur(dark, (3, 3), 0)
        self.tex_crack = crack

    # ---- per-frame signal summary --------------------------------------------------
    def frame_signals(self, tf: float, fps: int):
        i0 = int(max(0, (tf - 0.5 / fps) * 100))
        i1 = int(min(len(self.t), (tf + 0.5 / fps) * 100 + 1))
        sl = slice(i0, max(i1, i0 + 1))
        I = float(self.I[sl].mean())
        V = float(self.V[sl].mean())
        sc = int(np.sum((self.V[sl] < 0.6 * self.V_nom) & (self.I[sl] > 10)))
        vstd = float(self.V[sl].std()) if i1 - i0 > 1 else 0.0
        s = float(np.interp(tf, self.t, self.s))
        w = self.weave_amp * math.sin(2 * math.pi * self.weave_hz * tf)
        active = [d for d in self.defects if d["t_start"] <= tf <= d["t_end"]]
        return dict(I=I, V=V, sc=sc, vstd=vstd, s=s, w=w, gas=float(self.GAS[sl].mean()),
                    wfs=float(self.WFS[sl].mean()), active=active, P=I * V)


# ---------------------------------------------------------------- torch hardware sprites


def _torch_layer(cam: Camera, process: str, push_deg: float, ctwd: float):
    """Static screen-space torch hardware: returns (alpha, albedo rgb, rim mask, depth, wire_tip_px)."""
    h, w = cam.h, cam.w
    alpha = np.zeros((h, w), F32)
    albedo = np.zeros((h, w, 3), F32)
    rim = np.zeros((h, w), F32)
    depth = np.full((h, w), np.inf, F32)
    ph = math.radians(push_deg)
    axis = np.array([0.0, -math.sin(ph), math.cos(ph)])       # from arc up the torch (leaning back)
    A = cam.A + np.array([0, 0, 0.5])

    def cone(z0, z1, r0, r1, color, rim_strength=1.0, n=48):
        pts = []
        for z, r in ((z0, r0), (z1, r1)):
            c = A + axis * z
            for k in range(n):
                a = 2 * math.pi * k / n
                off = np.array([math.cos(a), math.sin(a) * math.cos(ph), math.sin(a) * math.sin(ph)]) * r
                pts.append(c + off)
        pts = np.array(pts)
        px, z = cam.project(pts)
        if np.any(z <= 1):
            return
        hull = cv2.convexHull(px.astype(np.float32)).astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, hull, 1, lineType=cv2.LINE_AA)
        mm = m.astype(bool)
        # cylindrical shading across the silhouette
        xs = np.arange(w, dtype=F32)
        xmin, xmax = hull[:, 0, 0].min(), hull[:, 0, 0].max()
        xn = np.clip((xs - xmin) / max(1, xmax - xmin), 0, 1)
        shade = (0.35 + 0.65 * np.sin(np.pi * xn) ** 0.8)[None, :].repeat(h, 0)
        ys = np.arange(h, dtype=F32)[:, None].repeat(w, 1)
        ymax = hull[:, 0, 1].max()
        rim_m = np.exp(-np.maximum(ymax - ys, 0) / (18.0 * w / 960)) * rim_strength
        sel = mm & (z.mean() < depth)
        alpha[sel] = 1.0
        albedo[sel] = np.array(color, F32) * shade[sel, None]
        rim[sel] = rim_m[sel]
        depth[sel] = z.min()

    if process == "FCAW":
        tip = ctwd - 3.0
        cone(tip + 2, tip + 24, 7.5, 9.5, (0.55, 0.32, 0.16))           # copper nozzle
        cone(tip + 24, tip + 70, 9.5, 12.0, (0.10, 0.10, 0.11), 0.4)     # insulated neck
        cone(tip - 0.5, tip + 2.5, 3.2, 3.6, (0.6, 0.4, 0.25))           # contact tip
    elif process == "GTAW":
        cone(4.5, 22, 5.5, 8.0, (0.75, 0.55, 0.48))                      # alumina cup
        cone(22, 65, 9.0, 11.0, (0.09, 0.09, 0.10), 0.4)                 # torch body
        cone(1.2, 4.6, 0.35, 1.2, (0.55, 0.55, 0.6), 1.5)                # tungsten
    else:  # SAW
        cone(6, 75, 2.6, 3.2, (0.55, 0.35, 0.2), 0.3)                    # contact tube
        cone(40, 90, 6.5, 7.5, (0.15, 0.16, 0.17), 0.1)                  # flux hose / nozzle
    wire_top, _ = cam.project(A + axis * (ctwd - 3.0))
    wire_bot, _ = cam.project(A)
    return alpha, albedo, rim, depth, (wire_top, wire_bot)


# ---------------------------------------------------------------- renderer


class Renderer:
    def __init__(self, scene: PassScene, w=W, h=H, fps=CAMERA_FPS, scale: float = 2 / 3):
        self.sc = sc = scene
        self.ow, self.oh = w, h                      # output size
        w, h = int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2
        self.w, self.h, self.fps = w, h, fps        # internal render size
        self.px = w / 960.0                          # pixel-size scale for screen-space effects
        proc = sc.process
        # the camera rides at the current layer level so the arc stays in the same place
        R_eff = sc.R
        self.cam = cam = Camera(R_eff, proc, w, h)
        m = cam.surface_maps()
        self.hit = m["hit"]
        self.hitf = m["hit"].astype(F32)
        self.U = m["u"].astype(F32)
        self.V = m["v"].astype(F32)
        self.Z = np.where(m["hit"], m["z"], np.inf).astype(F32)
        self.cos_view = m["cos_view"].astype(F32)
        L_arc = cam.A + np.array([0, 0, 2.5])
        (self.lu, self.lv, self.ln), (self.hu, self.hvv, self.hn), self.att = _light_terms(m, L_arc, 22.0)
        (self.au, self.av, self.an), (self.ahu, self.ahv, self.ahn) = _dir_terms(m, np.array([0.25, -0.5, 1.0]))
        rng = sc.rng
        # textures
        self.base_tex = _noise_tile(rng, 512, 512)
        self.scale_tex = _noise_tile(rng, 256, 256, ((3, 0.6), (12, 0.4)))
        self.grain_tex = _noise_tile(rng, 256, 256, ((1, 0.7), (3, 0.3)))
        self.fume_tex = cv2.GaussianBlur(_noise_tile(rng, h, w, ((24, 0.5), (60, 0.35), (140, 0.15))), (0, 0), 6 * w / 960)
        self.pool_tex = _noise_tile(rng, 128, 128, ((4, 0.6), (16, 0.4)))
        self.sensor_noise = [rng.normal(0, 1, (self.oh, self.ow)).astype(F32) for _ in range(4)]
        # static radial maps
        self.dist_arc = np.sqrt(self.U ** 2 + self.V ** 2).astype(F32)
        oys, oxs = np.mgrid[0:self.oh, 0:self.ow].astype(F32)
        self.vignette = (1.0 - 0.5 * (((oxs - self.ow / 2) / (self.ow / 2)) ** 2 +
                                      ((oys - self.oh / 2) / (self.oh / 2)) ** 2) ** 1.3).clip(0.25, 1)
        ys, xs = np.mgrid[0:h, 0:w].astype(F32)
        self.vignette_in = cv2.resize(self.vignette, (w, h), interpolation=cv2.INTER_AREA)
        self.arc_px = cam.project(cam.A + np.array([0, 0, 1.0]))[0]
        ax, ay = self.arc_px
        self.r_arc_px = np.sqrt((xs - ax) ** 2 + (ys - ay) ** 2).astype(F32)
        above = np.clip((ay - ys) / (0.6 * h), 0, 1)
        self.fume_mask = (np.exp(-np.abs(xs - ax) / (0.22 * w)) * above ** 0.7 * np.exp(-self.r_arc_px / (0.45 * w))).astype(F32)
        # torch hardware
        push = float(sc.meta.get("push_angle_deg", 10.0))
        ctwd = float(sc.meta.get("ctwd_mm", 15.0))
        self.t_alpha, self.t_alb, self.t_rim, self.t_depth, self.wire = _torch_layer(cam, proc, push, ctwd)
        # background (dark shop with a few defocused lights)
        bg = np.zeros((h, w, 3), F32)
        for _ in range(7):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h * 0.45)
            col = np.array([rng.uniform(0.6, 1.0), rng.uniform(0.5, 0.8), rng.uniform(0.3, 0.7)], F32)
            cv2.circle(bg, (int(cx), int(cy)), int(rng.uniform(10, 34)), tuple(float(c) for c in col * 0.05), -1)
        bg = cv2.GaussianBlur(bg, (0, 0), 9)
        bg += (np.linspace(0.012, 0.0, h, dtype=F32)[:, None, None] * np.array([0.7, 0.8, 1.0], F32))
        self.bg = bg * (6.0 if proc == "SAW" else 1.0)
        # spatter particles
        self.parts = np.zeros((0, 6), F32)   # x, y, vx, vy, age, life
        self.ambient = 0.55 if proc == "SAW" else 0.025
        self.exposure = 1.25 if proc == "SAW" else 1.0
        # flux pile (SAW) in torch frame
        if proc == "SAW":
            fu, fv = self.U.astype(np.float64), self.V.astype(np.float64)
            sw = 16.0                                            # ridge half-width scale [mm]
            ahead = 1 / (1 + np.exp(-(fu + 32.0) / 3.0))         # flux recovered ~32 mm behind arc
            prof = np.exp(-(fv / sw) ** 2)
            hgt = 7.0 * ahead * prof
            self.flux_h = hgt.astype(F32)
            self.flux_gu = (7.0 * prof * ahead * (1 - ahead) / 3.0).astype(F32)
            self.flux_gv = (hgt * (-2 * fv / sw ** 2)).astype(F32)
        self.frame_idx = 0
        self.steel_tint = np.array([1.0, 1.02, 1.07], F32)
        self.spec_tint = np.array([1.0, 0.92, 0.82], F32)
        self.pool_hot = np.array([1.0, 0.80, 0.45], F32)
        self.pool_rim = np.array([1.0, 0.45, 0.10], F32)
        Tg = np.arange(0, 2000, 2.0, dtype=F32)
        self.bb_lut = blackbody(Tg)                      # (N, 3)
        # static ROI that contains the largest possible pool
        Lmax = {"GTAW": 5.5, "FCAW": 9.0, "SAW": 18.0}[proc] * 1.5 * 1.4 + 2
        pwm = max(3.0, float(np.max(sc.b_u))) * 1.6 * 1.3 + 2
        roi = (np.abs(self.V) < pwm) & (self.U > -Lmax) & (self.U < Lmax) & self.hit
        ys_, xs_ = np.nonzero(roi)
        if len(ys_):
            self.pool_roi = (int(ys_.min()), int(ys_.max()) + 1, int(xs_.min()), int(xs_.max()) + 1)
        else:
            self.pool_roi = (0, h, 0, w)
        pos = np.clip(self.sensor_noise[0] * 2.6, 0, 255).astype(np.uint8)
        neg = np.clip(-self.sensor_noise[0] * 2.6, 0, 255).astype(np.uint8)
        self.noise8 = [(np.roll(pos, k * 7, axis=1)[..., None].repeat(3, -1),
                        np.roll(neg, k * 7, axis=1)[..., None].repeat(3, -1)) for k in range(4)]

    # --------------------------------------------------------------------------
    def render(self, tf: float):
        sc, rng = self.sc, self.sc.rng
        fs = sc.frame_signals(tf, self.fps)
        proc = sc.process
        arc_on = fs["I"] > 8
        P_rel = fs["P"] / max(sc.P_nom, 1)
        ua = self.U + F32(fs["s"])           # absolute u
        va = self.V + F32(fs["w"])           # absolute v (bead centred at 0)
        av_ = np.abs(va)

        # ---- geometry: groove before pass, deposited bead after the torch
        h_prev = _lut(sc.h_prev_v, 0.0, sc.vgrid[1] - sc.vgrid[0], av_)
        dvg = sc.vgrid[1] - sc.vgrid[0]
        dh_prev = np.gradient(sc.h_prev_v, dvg).astype(F32)
        gv_prev = _lut(dh_prev, 0.0, dvg, av_) * np.sign(va)
        bevel = _lut(sc.bevel_v, 0.0, dvg, av_)
        prevb = _lut(sc.prev_bead_v, 0.0, dvg, av_)
        gap = _lut(sc.gap_v, 0.0, dvg, av_)

        b = _lut(sc.b_u, sc.u0, sc.du, ua)
        crown = _lut(sc.crown_u, sc.u0, sc.du, ua)
        level = _lut(sc.level_u, sc.u0, sc.du, ua)
        tdep = _lut(sc.tdep_u.astype(F32), sc.u0, sc.du, ua)
        uc = _lut(sc.uc_u.astype(F32), sc.u0, sc.du, ua)
        lof = _lut(sc.lof_u.astype(F32), sc.u0, sc.du, ua)
        bt = _lut(sc.bt_u.astype(F32), sc.u0, sc.du, ua)
        crk_u = _lut(sc.cr_u.astype(F32), sc.u0, sc.du, ua)

        q = np.clip(va / np.maximum(b, 0.5), -1.5, 1.5)
        prof = np.clip(1 - q * q, 0, 1)
        pitch = sc.ripple_pitch
        phase = (ua + 0.035 * va * va + 0.25 * np.sin(ua * 0.41)) * (2 * np.pi / pitch)
        rip = np.sin(phase)
        rip_amp = (0.035 if proc != "SAW" else 0.02) * (0.6 + 0.8 * np.sin(ua * 0.23) ** 2)
        h_bead = level + crown * prof + rip_amp * rip * prof
        gv_bead = crown * (-2 * q / np.maximum(b, 0.5)) * (np.abs(q) < 1) + rip_amp * np.cos(phase) * 0.07 * va * (2 * np.pi / pitch) * prof
        gu_bead = rip_amp * np.cos(phase) * (2 * np.pi / pitch) * prof
        in_bead = (np.abs(q) < 1.0).astype(F32)
        ts_now = max(float(np.interp(tf, sc.t, sc.TS)), 0.3)
        dep = np.clip((tf - tdep) * F32(ts_now / 2.5), 0, 1) * (tdep > -100).astype(F32)  # deposited behind torch
        bead_m = dep * np.clip((1.0 - np.abs(q)) * 4, 0, 1)
        # undercut notch at the toes / cold-lap crevice on one side
        toe = np.exp(-(((av_ - b - 0.35) / 0.45) ** 2)).astype(F32)
        h_bead_full = np.maximum(h_bead, h_prev)
        notch = dep * (uc * toe * 1.1 + lof * np.exp(-(((va - b - 0.2) / 0.35) ** 2)) * 1.3)
        h = h_prev * (1 - bead_m) + h_bead_full * bead_m - notch
        gu = gu_bead * bead_m
        gv = gv_prev * (1 - bead_m) + gv_bead * bead_m + dep * (uc * toe * 2.0 * np.sign(va) * -1)

        # detail texture (pores, crack, spatter balls)
        mx = (ua - F32(sc.u0)) / F32(sc.tex_res)
        my = (va - F32(sc.tex_v0)) / F32(sc.tex_res)
        dth = _sample(sc.tex_dh, mx, my)
        gate = np.maximum(dep, (av_ > sc.Wt).astype(F32))
        h = h + dth * gate
        gu = gu + _sample(sc.tex_gu, mx, my) * gate
        gv = gv + _sample(sc.tex_gv, mx, my) * gate
        pore_dark = _sample(sc.tex_dark, mx, my) * dep
        t_since = np.maximum(tf - tdep, 0)
        crack = _sample(sc.tex_crack, mx, my) * dep * np.clip((t_since - 1.2) / 1.0, 0, 1) * (crk_u > 0)
        # burn-through hole
        bt_hole = dep * bt * np.clip(1 - (va / np.maximum(b * 0.9, 0.5)) ** 2, 0, 1)
        bt_hole = np.clip((bt_hole - 0.25) * 3.0, 0, 1)

        # ---- temperature field
        tau_c = {"GTAW": 1.8, "FCAW": 3.2, "SAW": 4.5}[proc]
        t_pool = {"GTAW": 5.5, "FCAW": 9.0, "SAW": 18.0}[proc] / max(float(np.median(sc.TS)), 0.5)
        T_bead = sc.T_ip + (1480 - sc.T_ip) * np.exp(-np.maximum(t_since - 0.6 * t_pool, 0) / tau_c) * \
            (1 - 0.45 * q * q) * np.clip(t_since * ts_now / 5.0, 0.5, 1)
        T = sc.T_ip + (T_bead - sc.T_ip) * bead_m
        T = T.astype(F32)
        if arc_on:
            haz = 950.0 * min(P_rel, 1.6) * np.exp(-self.dist_arc / (5.0 + 2.5 * min(P_rel, 1.5)))
            T = np.maximum(T, haz.astype(F32))

        # ---- weld pool (torch frame)
        Lb = {"GTAW": 5.5, "FCAW": 9.0, "SAW": 18.0}[proc] * (0.75 + 0.35 * min(P_rel, 2.0))
        La = Lb * 0.35
        pw = max(2.2, float(np.median(sc.b_u)) - 0.4 * sc.weave_amp) * (1.0 + 0.6 * float(np.mean(bt)))
        if any(d["defect"] == "burn_through" for d in fs["active"]):
            Lb *= 1.5
            pw *= 1.3
        ux = np.where(self.U < 0, self.U / Lb, self.U / La)
        pool_r = ux * ux + (self.V / pw) ** 2
        pool = np.clip((1.0 - pool_r) * 4.0, 0, 1) * (1.0 if arc_on else 0.0)
        if not arc_on and tf > 0.5:   # crater solidifying
            pool *= 0

        # ---- shading
        norm = np.sqrt(1 + gu * gu + gv * gv)
        ndl = np.clip((self.ln - gu * self.lu - gv * self.lv) / norm, 0, 1)
        ndh = np.clip((self.hn - gu * self.hu - gv * self.hvv) / norm, 0, 1)
        ndla = np.clip((self.an - gu * self.au - gv * self.av) / norm, 0, 1)
        ndha = np.clip((self.ahn - gu * self.ahu - gv * self.ahv) / norm, 0, 1)

        # albedo: neutral steel luminance + coloured bead layer
        base_n = _sample(self.base_tex, ua * F32(1.6), va * F32(1.6), wrap=True)
        scl = _sample(self.scale_tex, ua * F32(0.9), va * F32(0.9), wrap=True)
        ground_zone = np.clip((sc.Wt + 9.0 - av_) / 3.0, 0, 1)
        A = (0.10 + 0.05 * base_n) * (0.7 + 0.5 * scl) * (1 - ground_zone) + \
            (0.24 + 0.04 * np.sin(va * 9.0 + base_n * 9.0) + 0.06 * base_n) * ground_zone
        A = A * (1 - bevel) + (0.20 + 0.08 * base_n) * bevel
        A = A * (1 - prevb) + (0.20 + 0.05 * np.sin(phase)) * prevb
        A = A * (1 - gap) * (1 - bead_m)
        dark = (1 - 0.92 * pore_dark) * (1 - 0.95 * crack) * (1 - bt_hole)
        spec_k = 0.15 * ground_zone + 0.15 * bevel + 0.08

        if proc == "FCAW":
            bead_c = np.array([0.085, 0.07, 0.06], F32)[None, None] * (0.8 + 0.4 * base_n)[..., None]
            bead_spec = 0.55
        elif proc == "GTAW":
            aq = np.abs(q)
            tint = np.clip(1 - aq, 0, 1)[..., None]
            band = np.clip(aq * 1.4, 0, 1)[..., None]
            straw = np.array([0.55, 0.42, 0.18], F32)
            blue = np.array([0.18, 0.22, 0.45], F32)
            bead_c = np.array([0.55, 0.55, 0.58], F32) * tint + (straw + (blue - straw) * band) * (1 - tint)
            bead_spec = 0.9
        else:
            peel = np.clip((-self.U - 55) / 10, 0, 1)[..., None]
            slag = np.array([0.11, 0.09, 0.08], F32) * (0.8 + 0.4 * base_n[..., None])
            bead_c = slag + (np.array([0.42, 0.42, 0.45], F32) - slag) * peel
            bead_spec = 0.6
        alb = (A * dark)[..., None] * self.steel_tint + bead_c * (bead_m * dark)[..., None]
        spec_k = spec_k * (1 - bead_m) + bead_spec * bead_m

        arc_I = 0.0
        if arc_on:
            flick = 1.0 + 0.10 * rng.normal() + 0.4 * (fs["vstd"] / max(sc.V_nom, 1)) * rng.normal()
            if fs["sc"] > 0:
                flick *= 0.55 ** min(fs["sc"], 2)
            arc_I = max(0.15, P_rel * flick) * (3.2 if proc != "SAW" else 0.0)
        light = self.att * (ndl * arc_I) + self.ambient * ndla
        spec = (self.att * self.att * arc_I * 1.6 * ndh ** 40 + self.ambient * 0.6 * ndha ** 30) * spec_k
        img = alb * light[..., None] + spec[..., None] * self.spec_tint

        # incandescence (blackbody LUT)
        img += _lut(self.bb_lut, 0.0, 2.0, T)
        # weld pool (only inside its static ROI)
        if arc_on:
            y0, y1, x0, x1 = self.pool_roi
            U, Vv = self.U[y0:y1, x0:x1], self.V[y0:y1, x0:x1]
            pr = pool_r[y0:y1, x0:x1]
            pm = pool[y0:y1, x0:x1][..., None]
            pt = _sample(self.pool_tex, U * F32(3) + F32(tf * 60 % 128), Vv * F32(3) + F32(tf * 23 % 128), wrap=True)
            edge_f = np.clip(pr, 0, 1)[..., None]
            hot = self.pool_hot + (self.pool_rim - self.pool_hot) * edge_f
            refl = np.exp(-((U + 1.0) ** 2 + Vv ** 2) / 2.0)[..., None] * 3.0
            pool_col = hot * (1.1 + 1.1 * pt[..., None]) + refl
            if any(d["defect"] == "porosity" for d in fs["active"]):
                pool_col = pool_col * (1 - (pt > 0.62).astype(F32) * 0.85)[..., None]
            sub = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = sub * (1 - pm) + pool_col * pm * min(1.6, 0.6 + 0.5 * P_rel)

        if proc == "SAW":
            img = self._flux(img, ua, va, tf, fs)

        # cylinder hit mask -> background
        hm = self.hitf[..., None]
        img = img * hm + self.bg * (1 - hm)

        # torch hardware (lit by the arc from below)
        ta = self.t_alpha[..., None]
        torch_col = self.t_alb * (self.ambient * 1.2 + 0.03) + self.t_alb * self.t_rim[..., None] * (arc_I * 0.55)
        img = img * (1 - ta) + torch_col * ta

        # arc glow, wire, fume, spatter
        if arc_on and proc != "SAW":
            self._wire(img, fs)
            ra = self.r_arc_px
            px = self.px
            core = (4.0 + 1.5 * P_rel) * px
            glow = (np.exp(-(ra / core) ** 2) * 9.0 + np.exp(-ra / ((12 + 5 * P_rel) * px)) * 0.9 +
                    np.exp(-ra / (110.0 * px)) * 0.10) * arc_I / 3.2
            tint = np.array([0.95, 0.95, 1.0], F32) if proc == "GTAW" else np.array([1.0, 0.9, 0.75], F32)
            img += glow[..., None] * tint
            if proc == "FCAW":
                self._fume(img, tf, arc_I)
        self._spatter(img, fs, arc_on)

        # bloom
        img = self._bloom(img)
        # tonemap + camera
        out = (1.0 - np.exp(-img * self.exposure)) * self.vignette_in[..., None]
        out = cv2.pow(np.clip(out, 0, 1), 0.68) * F32(255)     # display gamma (keeps shadows dark)
        rgb = cv2.resize(out.astype(np.uint8), (self.ow, self.oh), interpolation=cv2.INTER_CUBIC)
        npos, nneg = self.noise8[self.frame_idx % 4]
        rgb = cv2.subtract(cv2.add(rgb, npos), nneg)

        # control: depth (inverse depth, near = bright)
        z = self.Z - h * 2.5 / self.cos_view          # relief exaggerated for the control signal
        if proc == "SAW":
            z = z - self.flux_h * 0.8 / self.cos_view
        z = np.minimum(z, self.t_depth)
        zn, zf = 30.0, 450.0
        inv = (1 / np.clip(z, zn, zf) - 1 / zf) / (1 / zn - 1 / zf)
        depth8 = (np.clip(inv, 0, 1) ** 0.6 * 255).astype(np.uint8)
        depth8 = cv2.resize(depth8, (self.ow, self.oh), interpolation=cv2.INTER_LINEAR)

        # control: edges from a flat-lit geometry render (no glow / noise)
        lum = alb.mean(-1)
        geo = np.clip(0.12 + 1.6 * lum, 0, 1) * (0.25 + 0.75 * ndla) * self.hitf
        geo = geo + 0.25 * np.clip(-h, 0, 12) / 12 * self.hitf          # groove depth cue
        geo = geo * (1 - pool) + 0.95 * pool
        if proc == "SAW":
            geo = geo * (1 - self.flux_m) + self.flux_m * (0.35 + 0.4 * self.flux_shade)
        geo = geo * (1 - self.t_alpha) + self.t_alpha * (0.5 + 0.4 * self.t_rim)
        geo = cv2.resize(geo.astype(F32), (self.ow, self.oh), interpolation=cv2.INTER_LINEAR)
        geo8 = (np.clip(geo, 0, 1) * 255).astype(np.uint8)
        geo8 = cv2.GaussianBlur(geo8, (5, 5), 0)
        edge8 = cv2.Canny(geo8, 12, 36)
        self.frame_idx += 1
        return rgb, edge8, depth8

    # -------------------------------------------------------------------- effects
    def _flux(self, img, ua, va, tf, fs):
        fh = self.flux_h
        grain = _sample(self.grain_tex, ua * F32(4), va * F32(4), wrap=True)
        gu, gv = self.flux_gu, self.flux_gv
        norm = np.sqrt(1 + gu * gu + gv * gv)
        nd = np.clip((self.an - gu * self.au - gv * self.av) / norm, 0, 1)
        shade = (0.25 + 0.75 * nd) * (0.55 + 0.6 * grain)
        col = np.array([0.36, 0.31, 0.25], F32) * shade[..., None] * self.ambient * 1.4
        m = np.clip((fh + 0.9 * (grain - 0.5)) / 1.2, 0, 1)
        # glow seeping through thin flux near the arc
        glow = np.exp(-(self.dist_arc / 9.0) ** 2) * (fs["I"] > 8) * (0.6 + 0.4 * grain)
        col += (glow * 0.9)[..., None] * np.array([1.0, 0.45, 0.1], F32)
        self.flux_m = m
        self.flux_shade = shade
        return img * (1 - m[..., None]) + col * m[..., None]

    def _wire(self, img, fs):
        (x0, y0), (x1, y1) = self.wire
        sc = self.sc
        if sc.process == "GTAW":
            # filler rod dabbed in from the leading edge when wire feed is on
            if fs["wfs"] > 0.2:
                ax, ay = self.arc_px
                p = self.px
                cv2.line(img, (int(ax - 230 * p), int(ay - 150 * p)), (int(ax - 6 * p), int(ay - 14 * p)),
                         (0.35, 0.33, 0.3), max(2, int(5 * p)), cv2.LINE_AA)
                cv2.circle(img, (int(ax - 7 * p), int(ay - 14 * p)), max(2, int(4 * p)), (3.0, 2.0, 1.0), -1, cv2.LINE_AA)
            return
        th = max(2, int(round(3 * self.px)))
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0.9, 0.75, 0.6), th, cv2.LINE_AA)
        cv2.line(img, (int(x0 + (x1 - x0) * 0.6), int(y0 + (y1 - y0) * 0.6)), (int(x1), int(y1)),
                 (3.5, 2.2, 1.0), th, cv2.LINE_AA)

    def _fume(self, img, tf, arc_I):
        h, w = self.h, self.w
        off = int(tf * 45) % h
        f = np.roll(self.fume_tex, -off, axis=0)
        f = np.roll(f, int(12 * math.sin(tf * 0.7)), axis=1)
        dens = np.clip(f - 0.45, 0, 1) * self.fume_mask * 0.22 * arc_I / 3.2
        img += dens[..., None] * np.array([1.0, 0.72, 0.45], F32)

    def _spatter(self, img, fs, arc_on):
        sc, rng = self.sc, self.sc.rng
        dt = 1.0 / self.fps
        rate = {"GTAW": 0.0, "FCAW": 2.2, "SAW": 0.0}[sc.process]   # particles per frame
        if arc_on and rate:
            rate *= 1 + 1.2 * fs["sc"] + 8.0 * fs["vstd"] / max(sc.V_nom, 1)
            if any(d["defect"] in ("arc_instability", "porosity") for d in fs["active"]):
                rate *= 2.5
            n = rng.poisson(rate)
            if n:
                ang = rng.uniform(-math.pi * 0.95, -math.pi * 0.05, n)
                spd = rng.uniform(250, 1400, n) * self.px
                ax, ay = self.arc_px
                new = np.column_stack([ax + rng.normal(0, 3, n), ay + rng.normal(0, 3, n),
                                       np.cos(ang) * spd, np.sin(ang) * spd,
                                       np.zeros(n), rng.uniform(0.06, 0.3, n)]).astype(F32)
                self.parts = np.vstack([self.parts, new])
        if len(self.parts) == 0:
            return
        p = self.parts
        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt
        p[:, 3] += 1800 * dt * self.px
        p[:, 4] += dt
        p = p[(p[:, 4] < p[:, 5]) & (p[:, 0] > -50) & (p[:, 0] < self.w + 50) & (p[:, 1] < self.h + 50)]
        self.parts = p
        for x, y, vx, vy, age, life in p:
            k = max(0.0, 1 - age / life)
            x0, y0 = x - vx * dt * 0.3, y - vy * dt * 0.3
            k = float(k)
            col = (3.0 * k, 2.0 * k ** 1.3, 0.8 * k ** 2)
            cv2.line(img, (int(x0), int(y0)), (int(x), int(y)), col, 1, cv2.LINE_AA)

    def _bloom(self, img):
        bright = np.clip(img - 0.9, 0, None)
        small = cv2.resize(bright, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(small, (0, 0), 3)
        b2 = cv2.GaussianBlur(small, (0, 0), 12)
        bl = cv2.resize(0.6 * b1 + 0.5 * b2, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return img + bl


# ---------------------------------------------------------------- encoding


class _Encoder:
    def __init__(self, path: Path, w: int, h: int, fps: int, crf: int = 23):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(path), mode="w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height = w, h
        self.s.pix_fmt = "yuv420p"
        self.s.options = {"crf": str(crf), "preset": "veryfast", "tune": "film",
                          "movflags": "+faststart"}
        self.s.thread_count = 1

    def write(self, arr: np.ndarray, fmt: str):
        fr = av.VideoFrame.from_ndarray(arr, format=fmt)
        for pkt in self.s.encode(fr):
            self.c.mux(pkt)

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()


def render_pass(pass_dir: str | Path, stills_dir: Optional[str | Path] = None,
                still_times: Optional[list[float]] = None, max_seconds: Optional[float] = None,
                w: int = W, h: int = H, fps: int = CAMERA_FPS) -> dict:
    pass_dir = Path(pass_dir)
    scene = PassScene(pass_dir)
    r = Renderer(scene, w, h, fps)
    D = scene.D if max_seconds is None else min(scene.D, max_seconds)
    n = int(round(D * fps))
    encs = {"rgb": _Encoder(pass_dir / "torch_cam.mp4", w, h, fps, 22),
            "edge": _Encoder(pass_dir / "control" / "edge.mp4", w, h, fps, 25),
            "depth": _Encoder(pass_dir / "control" / "depth.mp4", w, h, fps, 25)}
    still_frames = {int(round(t * fps)): t for t in (still_times or [])}
    t0 = time.time()
    try:
        for k in range(n):
            tf = (k + 0.5) / fps
            rgb, edge, depth = r.render(tf)
            encs["rgb"].write(rgb, "rgb24")
            encs["edge"].write(edge, "gray")
            encs["depth"].write(depth, "gray")
            if stills_dir is not None and k in still_frames:
                Path(stills_dir).mkdir(parents=True, exist_ok=True)
                stem = f"{scene.meta['sequence_name']}_t{still_frames[k]:05.1f}s"
                cv2.imwrite(str(Path(stills_dir) / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(Path(stills_dir) / f"{stem}_edge.png"), edge)
                cv2.imwrite(str(Path(stills_dir) / f"{stem}_depth.png"), depth)
    finally:
        for e in encs.values():
            e.close()
    return {"sequence_name": scene.meta["sequence_name"], "frames": n, "seconds": round(time.time() - t0, 1)}


def _render_job(args):
    d, stills = args
    return render_pass(d, stills_dir=stills[0] if stills else None, still_times=stills[1] if stills else None)


def render_many(dirs, workers: Optional[int] = None, quiet: bool = False, stills: Optional[dict] = None):
    dirs = [Path(d) for d in dirs]
    if workers is None:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    jobs = [(d, (stills or {}).get(d.name)) for d in dirs]
    t0 = time.time()
    if workers <= 1:
        res = [_render_job(j) for j in jobs]
        for r_ in res:
            if not quiet:
                print(f"  rendered {r_['sequence_name']}: {r_['frames']} frames in {r_['seconds']}s", flush=True)
    else:
        res = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r_ in ex.map(_render_job, jobs):
                res.append(r_)
                if not quiet:
                    print(f"  rendered {r_['sequence_name']}: {r_['frames']} frames in {r_['seconds']}s", flush=True)
    if not quiet:
        print(f"render: {len(res)} passes in {time.time() - t0:.1f}s ({workers} workers)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render torch camera + control videos for pass dirs")
    ap.add_argument("paths", nargs="+", help="pass dirs, or a parent dir (e.g. data/generated)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stills", default=None, help="write PNG stills to this dir")
    ap.add_argument("--still-times", default="6,14", help="comma separated seconds for stills")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)
    dirs = []
    for p in map(Path, args.paths):
        if (p / "meta.json").exists():
            dirs.append(p)
        else:
            dirs += sorted(d for d in p.iterdir() if (d / "meta.json").exists())
    if args.max_seconds or (args.stills and len(dirs) == 1):
        times = [float(x) for x in args.still_times.split(",")]
        for d in dirs:
            print(render_pass(d, stills_dir=args.stills, still_times=times, max_seconds=args.max_seconds))
        return
    stills = None
    if args.stills:
        times = [float(x) for x in args.still_times.split(",")]
        stills = {d.name: (args.stills, times) for d in dirs}
    render_many(dirs, workers=args.workers, stills=stills)


if __name__ == "__main__":
    sys.exit(main())
