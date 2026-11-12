"""Tiny pass directories that follow the real-data contract (docs/REAL_DATA.md).

Used by the tests and for offline smoke runs before the real datasets are downloaded::

    python -m mosaico_io.testdata /tmp/realfix      # -> /tmp/realfix/generated/{PP5_B02_SAW,WID617_L3_TGMAW}

* ``PP5_B02_SAW``     AIMEN-style SAW bead: 2 electrodes (lead DC + trail AC), no gas (NaN), no video,
                      one bead photo, one NDT porosity indication, ``rt_result = "indications"``.
* ``WID617_L3_TGMAW`` BAM-style tandem GMAW layer: gas, real-camera-like grayscale video at 17.96 fps,
                      ``control/edge.mp4``, no defects, ``rt_result = "not_inspected"``.

The signals are synthetic but shaped like the real files (rates, units, columns); the electrical
signals inside the porosity window are deliberately *normal* (as on the real AIMEN data).
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

T0_NS = 1_700_000_000_000_000_000
HZ = 100

AIMEN_SOURCE = {
    "id": "aimen_saw", "title": "Submerged Arc Welding Open Repository (PENELOPE)",
    "authors": "Tapia Suárez E. A., Pérez Couñago I., Eike Precker C., Fernández Montenegro J. M., Ruiz C. (AIMEN)",
    "doi": "10.5281/zenodo.15083865", "url": "https://zenodo.org/records/15083865", "license": "CC BY 4.0",
    "synthetic": False, "files": ["PP5/1. process_data/PP5_02/PP5_02.hdf5"],
}
BAM_SOURCE = {
    "id": "bam_tgmaw", "title": "Tandem-GMAW: advanced pulsed and short-circuiting process mode experimental data",
    "authors": "Chaurasia P. K., Fabry Ç., Pittner A., Rethmeier M. (BAM)",
    "doi": "10.5281/zenodo.21824586", "url": "https://zenodo.org/records/21824586", "license": "CC BY 4.0",
    "synthetic": False, "files": ["WID617_L3_process.nc", "WID617_L3_video.avi"],
}


def _pm10(x: float) -> list[float]:
    return [round(x * 0.9, 3), round(x * 1.1, 3)]


def _write_common(d: Path, meta: dict, arc: pd.DataFrame, travel: pd.DataFrame, thermal: pd.DataFrame,
                  robot: pd.DataFrame | None, events: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    arc.to_parquet(d / "arc.parquet", index=False)
    travel.to_parquet(d / "travel.parquet", index=False)
    thermal.to_parquet(d / "thermal.parquet", index=False)
    if robot is not None:
        robot.to_parquet(d / "robot.parquet", index=False)
    with open(d / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def _events(duration_s: float, indications: list[dict]) -> list[dict]:
    ev = [{"t_ns": T0_NS, "type": "arc_start"}]
    for ind in indications:
        ev.append({"t_ns": T0_NS + int(ind["t_start"] * 1e9), "type": "defect_on", "defect": ind["type"],
                   "severity": None, "source": "NDT"})
        ev.append({"t_ns": T0_NS + int(ind["t_end"] * 1e9), "type": "defect_off", "defect": ind["type"],
                   "severity": None, "source": "NDT"})
    ev.append({"t_ns": T0_NS + int(duration_s * 1e9), "type": "arc_stop"})
    return sorted(ev, key=lambda e: e["t_ns"])


def make_saw_pass(root: Path, duration_s: float = 30.0, seed: int = 0, prefix: str = "") -> Path:
    rng = np.random.default_rng(seed)
    seq = f"{prefix}PP5_B02_SAW"
    d = root / seq
    n = int(duration_s * HZ)
    t = np.arange(n) / HZ
    t_ns = T0_NS + (t * 1e9).astype(np.int64)
    I1, V1, F1 = 535 + rng.normal(0, 6, n), 31.0 + rng.normal(0, 0.3, n), np.full(n, 2.0) + rng.normal(0, 0.01, n)
    I2, V2, F2 = 600 + rng.normal(0, 8, n), 34.0 + rng.normal(0, 0.4, n), np.full(n, 2.2) + rng.normal(0, 0.01, n)
    arc = pd.DataFrame({"t_ns": t_ns, "current_a": I1, "voltage_v": V1, "wfs_mpm": F1, "gas_lpm": np.nan,
                        "current2_a": I2, "voltage2_v": V2, "wfs2_mpm": F2})
    v = np.full(n, 9.17) + rng.normal(0, 0.02, n)
    hi = 0.8 * (I1 * V1 + I2 * V2) / (1000 * v)
    travel = pd.DataFrame({"t_ns": t_ns, "travel_speed_mmps": v, "position_mm": 30.0 + np.cumsum(v) / HZ,
                           "heat_input_kj_mm": hi})
    tt = np.arange(0, duration_s, 0.1)
    left, right = 150 + 0.5 * tt + rng.normal(0, 1, len(tt)), 148 + 0.5 * tt + rng.normal(0, 1, len(tt))
    thermal = pd.DataFrame({"t_ns": T0_NS + (tt * 1e9).astype(np.int64), "interpass_c": np.maximum(left, right),
                            "temp_left_c": left, "temp_right_c": right, "temp_back_c": 900 + rng.normal(0, 20, len(tt))})
    robot = pd.DataFrame({"t_ns": t_ns, "tcp_x": 566.7 + np.cumsum(v) / HZ, "tcp_y": np.zeros(n), "tcp_z": np.full(n, 52.0)})
    ind = [{"type": "porosity", "t_start": 17.0, "t_end": 17.1, "x_start_mm": 185.9, "x_end_mm": 186.8}]
    photos = d / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    _bead_photo(photos / "PP5_02_1.jpg", seed)
    meta = {
        "sequence_name": seq, "spool_id": "PP5", "joint_id": "BW", "pass_no": 2, "pass_type": "fill",
        "wps_id": "AIMEN-SAW-S1", "process": "SAW", "position": "1G", "material": "SA-516 Gr.70N",
        "cell_id": "AIMEN-SAW", "duration_s": duration_s, "arc_on_s": duration_s,
        "defects": [{"defect": "porosity", "t_start": 17.0, "t_end": 17.1, "severity": None}],
        "rt_result": "indications", "has_defect": True, "defect_types": "porosity", "t0_ns": T0_NS,
        "wps": {"wps_id": "AIMEN-SAW-S1", "process": "SAW", "current_a": _pm10(535), "voltage_v": _pm10(31),
                "wfs_mpm": _pm10(2.0), "travel_speed_mmps": _pm10(9.17), "current2_a": _pm10(600),
                "voltage2_v": _pm10(34), "wfs2_mpm": _pm10(2.2), "max_interpass_c": 250},
        "setpoints": {"current_a": 535, "voltage_v": 31, "wfs_mpm": 2.0, "travel_speed_mmps": 9.17},
        "source": AIMEN_SOURCE, "electrodes": 2,
        "wps_basis": "published nominal setting ±10 % (demo assumption, not a qualified WPS)",
        "ndt": {"methods": ["RT", "UT"], "inspected": True, "report_files": ["RX.pdf", "UT.pdf"], "indications": ind},
        "thickness_mm": 52, "photos": ["photos/PP5_02_1.jpg"],
        "current_mean_a": round(float(I1.mean()), 1), "voltage_mean_v": round(float(V1.mean()), 2),
        "travel_mean_mmps": round(float(v.mean()), 2), "heat_input_mean_kj_mm": round(float(hi.mean()), 3),
    }
    _write_common(d, meta, arc, travel, thermal, robot, _events(duration_s, ind))
    return d


def make_gmaw_pass(root: Path, duration_s: float = 12.0, fps: float = 17.96, seed: int = 1,
                   size: tuple[int, int] = (320, 256), prefix: str = "") -> Path:
    rng = np.random.default_rng(seed)
    seq = f"{prefix}WID617_L3_TGMAW"
    d = root / seq
    n = int(duration_s * HZ)
    t = np.arange(n) / HZ
    t_ns = T0_NS + 10**12 + (t * 1e9).astype(np.int64)
    t0 = int(t_ns[0])
    I1, V1, F1 = 280 + rng.normal(0, 10, n), 28.5 + rng.normal(0, 0.5, n), np.full(n, 9.0) + rng.normal(0, 0.05, n)
    I2, V2, F2 = 250 + rng.normal(0, 10, n), 27.0 + rng.normal(0, 0.5, n), np.full(n, 8.0) + rng.normal(0, 0.05, n)
    G = 18.0 + rng.normal(0, 0.2, n)
    arc = pd.DataFrame({"t_ns": t_ns, "current_a": I1, "voltage_v": V1, "wfs_mpm": F1, "gas_lpm": G,
                        "current2_a": I2, "voltage2_v": V2, "wfs2_mpm": F2})
    v = np.full(n, 8.0) + rng.normal(0, 0.02, n)
    hi = 0.8 * (I1 * V1 + I2 * V2) / (1000 * v)
    travel = pd.DataFrame({"t_ns": t_ns, "travel_speed_mmps": v, "position_mm": np.cumsum(v) / HZ,
                           "heat_input_kj_mm": hi})
    tt = np.arange(0, duration_s, 0.01)
    thermal = pd.DataFrame({"t_ns": t0 + (tt * 1e9).astype(np.int64), "interpass_c": 120 + 2 * tt,
                            "thermo2_c": 95 + 1.5 * tt})
    robot = pd.DataFrame({"t_ns": t_ns, "tcp_x": np.cumsum(v) / HZ, "tcp_y": np.zeros(n), "tcp_z": np.full(n, 20.0)})
    d.mkdir(parents=True, exist_ok=True)
    _process_video(d / "torch_cam.mp4", d / "control" / "edge.mp4", duration_s, fps, size, seed)
    meta = {
        "sequence_name": seq, "spool_id": "WID617", "joint_id": "V30", "pass_no": 3, "pass_type": "fill",
        "wps_id": "BAM-TGMAW-WID617-L3", "process": "GMAW", "position": "1G", "material": "S355J2+N",
        "cell_id": "BAM-TGMAW", "duration_s": duration_s, "arc_on_s": duration_s,
        "defects": [], "rt_result": "not_inspected", "has_defect": False, "defect_types": "none", "t0_ns": t0,
        "wps": {"wps_id": "BAM-TGMAW-WID617-L3", "process": "GMAW", "current_a": _pm10(280),
                "voltage_v": _pm10(28.5), "wfs_mpm": _pm10(9.0), "travel_speed_mmps": _pm10(8.0),
                "gas_lpm": [14.4, 21.6]},
        "setpoints": {"current_a": 280, "voltage_v": 28.5, "wfs_mpm": 9.0, "travel_speed_mmps": 8.0, "gas_lpm": 18},
        "source": BAM_SOURCE, "electrodes": 2,
        "wps_basis": "published nominal setting ±10 % (demo assumption, not a qualified WPS)",
        "ndt": {"methods": [], "inspected": False, "report_files": [], "indications": []},
        "thickness_mm": 30, "photos": [],
        "video": {"file": "torch_cam.mp4", "kind": "real_process_camera", "fps": fps, "t_offset_s": 0.0},
        "current_mean_a": round(float(I1.mean()), 1), "voltage_mean_v": round(float(V1.mean()), 2),
        "travel_mean_mmps": round(float(v.mean()), 2), "heat_input_mean_kj_mm": round(float(hi.mean()), 3),
    }
    _write_common(d, meta, arc, travel, thermal, robot, [])
    ev = [{"t_ns": t0, "type": "arc_start"}, {"t_ns": t0 + int(duration_s * 1e9), "type": "arc_stop"}]
    with open(d / "events.jsonl", "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    return d


def _bead_photo(path: Path, seed: int, size: tuple[int, int] = (640, 360)) -> None:
    import cv2
    rng = np.random.default_rng(seed)
    w, h = size
    coarse = rng.integers(60, 140, size=(h // 12, w // 12), dtype=np.uint8)
    img = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x in range(0, w, 9):     # bead ripples
        cv2.ellipse(img, (x, h // 2), (14, 40), 0, -90, 90, (150, 160, 170), 2)
    cv2.circle(img, (int(w * 0.6), h // 2), 4, (20, 20, 20), -1)    # a pore
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _process_video(rgb: Path, edge: Path, duration_s: float, fps: float, size: tuple[int, int], seed: int) -> None:
    """Grayscale HDR-camera-like clip (bright arc spot over a scrolling groove) + its Canny edge map."""
    import av
    import cv2
    rng = np.random.default_rng(seed)
    w, h = size
    coarse = rng.integers(20, 110, size=(max(2, h // 16), max(2, w * 4 // 16)), dtype=np.uint8)
    tex = cv2.resize(coarse, (w * 4, h), interpolation=cv2.INTER_CUBIC)
    rate = Fraction(fps).limit_denominator(1000)
    edge.parent.mkdir(parents=True, exist_ok=True)
    outs = []
    for p in (rgb, edge):
        c = av.open(str(p), "w")
        s = c.add_stream("libx264", rate=rate)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "28", "preset": "veryfast"}
        outs.append((c, s))
    for i in range(int(round(duration_s * fps))):
        off = (i * 4) % (w * 3)
        g = tex[:, off: off + w].copy()
        cv2.line(g, (0, h // 2), (w, h // 2), 150, 8)
        cv2.circle(g, (w // 2, h // 2), 18 + int(4 * np.sin(i / 2)), 255, -1)
        cv2.putText(g, "BAM", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        e = cv2.Canny(g, 60, 160)
        for (c, s), img in zip(outs, (g, e)):
            fr = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), format="rgb24")
            for pkt in s.encode(fr.reformat(format="yuv420p")):
                c.mux(pkt)
    for c, s in outs:
        for pkt in s.encode():
            c.mux(pkt)
        c.close()


def make_real_fixture(root: str | Path, prefix: str = "") -> Path:
    """Write both passes to ``<root>/generated`` and return that directory. ``prefix`` renames the sequences
    (e.g. ``PYTEST_``) so a live-mosaicod test never overwrites the real PP5_B02_SAW / WID617_L3_TGMAW."""
    gen = Path(root) / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    make_saw_pass(gen, prefix=prefix)
    make_gmaw_pass(gen, prefix=prefix)
    return gen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    a = ap.parse_args(argv)
    print(make_real_fixture(a.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
