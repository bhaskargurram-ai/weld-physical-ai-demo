"""Tiny synthetic torch-camera clips + sidecars (for tests and offline smoke runs)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def make_clip(path: str | Path, *, seconds: float = 5.0, fps: int = 24, size: tuple[int, int] = (320, 180),
              kind: str = "good", seed: int = 0) -> Path:
    """Write an mp4 resembling a weld camera view.

    kind: good (textured pipe scrolling + bright arc), static (frozen frame), blurry (heavy blur,
    no texture), glare (mostly saturated), dark (black).
    """
    w, h = size
    rng = np.random.default_rng(seed)
    coarse = rng.integers(30, 150, size=(max(2, h // 20), max(2, w * 4 // 20)), dtype=np.uint8)
    tex = cv2.resize(coarse, (w * 4, h), interpolation=cv2.INTER_CUBIC).astype(np.int16)
    tex = np.clip(tex + rng.integers(-25, 25, size=tex.shape), 0, 255).astype(np.uint8)  # fine grain
    n = int(round(seconds * fps))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        if kind == "dark":
            frame = np.zeros((h, w, 3), np.uint8)
        elif kind == "glare":
            frame = np.full((h, w, 3), 255, np.uint8)
            frame[:, : w // 8] = 120
        else:
            off = 0 if kind == "static" else (i * 3) % (w * 3)
            g = tex[:, off: off + w]
            if kind == "blurry":
                g = np.full_like(g, 90)
            frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            # weld bead line + arc glow
            cv2.line(frame, (0, h // 2), (w, h // 2), (60, 70, 90), 6)
            r = 10 + (0 if kind == "static" else int(3 * np.sin(i / 2)))
            cv2.circle(frame, (w // 2, h // 2), r + 10, (0, 140, 255), -1)
            cv2.circle(frame, (w // 2, h // 2), r, (255, 255, 255), -1)
            if kind == "blurry":
                frame = cv2.GaussianBlur(frame, (31, 31), 12)
        vw.write(frame)
    vw.release()
    return Path(path)


def make_sidecar(path: str | Path, clip_id: str, *, seq: str = "SP-1043_J07_P2_FCAW", t0: float = 0.0,
                 t1: float = 5.0, defect: str | None = None, gas_low: bool = False,
                 rt: str = "accept") -> Path:
    meta = {
        "sequence_name": seq, "spool_id": "SP-1043", "joint_id": "J07", "pass_no": 2, "pass_type": "fill",
        "wps_id": "WPS-FCAW-FILL-02", "process": "FCAW", "position": "1G", "material": "A106 Gr.B",
        "pipe_od_mm": 168.3, "wall_mm": 7.1, "cell_id": "CELL-2", "duration_s": 60.0,
        "defects": ([{"defect": defect, "t_start": t0 + 1.0, "t_end": t0 + 3.5, "severity": 0.7}]
                    if defect else []),
        "rt_result": rt,
        "wps": {"current_a": [180, 240], "voltage_v": [23, 27], "gas_lpm": [18, 25]},
    }
    gas = 6.5 if gas_low else 21.0
    sc = {
        "clip_id": clip_id, "sequence_name": seq, "t0_s": t0, "t1_s": t1, "meta": meta,
        "signal_stats": {
            "current_a": {"mean": 212.0, "std": 5.1, "min": 199.0, "max": 224.0},
            "voltage_v": {"mean": 24.8, "std": 0.4, "min": 23.9, "max": 25.9},
            "wfs_mpm": {"mean": 8.6, "std": 0.1, "min": 8.3, "max": 8.9},
            "gas_lpm": {"mean": gas, "std": 0.5, "min": gas - 1, "max": gas + 1},
            "travel_speed_mmps": {"mean": 5.2, "std": 0.2, "min": 4.8, "max": 5.6},
            "heat_input_kj_mm": {"mean": 0.81, "std": 0.03, "min": 0.75, "max": 0.88},
            "interpass_c": {"mean": 180.0, "std": 2.0, "min": 176.0, "max": 184.0},
        },
        "wps_violations": ([{"param": "gas_lpm", "t_start_s": t0 + 0.8, "t_end_s": t0 + 3.6, "value": gas - 1,
                             "limit_lo": 18, "limit_hi": 25}] if gas_low else []),
        "defects_in_window": meta["defects"],
        "rt_result": rt,
    }
    Path(path).write_text(json.dumps(sc, indent=1))
    return Path(path)
