"""BAM tandem-GMAW adapter tests: tiny synthetic NetCDF/AVI fixtures, no network (fast, CPU)."""
from __future__ import annotations

import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("av")

from collect.datasets import bam_tgmaw as B  # noqa: E402

T0 = np.datetime64("2025-07-04T08:48:37.000", "ns")
ARC0, ARC1 = 1.0, 4.0          # arc window [s] in the process.nc clock
FPS = 10.0
REC0 = 0.5                     # video_recording rises here; frame k is exposed at REC0 + k / FPS
N_FRAMES = 52                  # 5.2 s of video


def _times(rate_hz: float, dur_s: float) -> np.ndarray:
    return T0 + (np.arange(int(round(dur_s * rate_hz))) * (1e9 / rate_hz)).astype("timedelta64[ns]")


def _write_nc(ds, path):
    ds.to_netcdf(path, engine="h5netcdf")


def make_layer(root, wid=617, layer=3, pulses_per_frame=True, tandem=True, gas_dip=None, outage=None):
    """Synthetic WID<wid>_L<layer> raw dir (same variables/sampling layout as the BAM record)."""
    stem = f"WID{wid}_L{layer}_T1"
    d = root / f"WID{wid}_L{layer}"
    d.mkdir(parents=True, exist_ok=True)
    # --- process.nc 10 kHz
    tp = _times(10_000, 6.0)
    ts = (tp - T0).astype(np.int64) / 1e9
    on = (ts >= ARC0) & (ts < ARC1)
    pulse = 300 + 150 * (np.sin(2 * np.pi * 250 * ts) > 0.6)     # pulsed lead current
    I1 = np.where(on, pulse, 0.0)
    U1 = np.where(on, 32.0 + 3 * (pulse > 300), 0.0)
    I2 = np.where(on, 330.0, 0.0)
    U2 = np.where(on, 31.0, 0.0)
    if outage:
        m = (ts >= outage[0]) & (ts < outage[1])
        I1[m], U1[m] = 0.0, 0.0
    rec = (ts >= REC0) & (ts < REC0 + N_FRAMES / FPS)
    vf = np.zeros_like(ts)
    rate = FPS if pulses_per_frame else 40.0
    k = np.floor((ts - REC0) * rate + 1e-9)
    first = np.r_[True, np.diff(k) != 0] & rec
    for i in np.flatnonzero(first):
        vf[i:i + 10] = 1.0                                         # 1 ms high
    proc = xr.Dataset({"current_leading": ("time", I1), "current_trailing": ("time", I2 if tandem else 0 * I2),
                       "voltage_leading": ("time", U1), "voltage_trailing": ("time", U2 if tandem else 0 * U2),
                       "video_frame": ("time", vf), "video_recording": ("time", rec.astype(float))},
                      coords={"time": tp})
    _write_nc(proc, d / f"{stem}_process.nc")
    # --- robot.nc 1 kHz (tcp every 8th sample)
    tr = _times(1000, 8.0)
    tsr = (tr - T0).astype(np.int64) / 1e9
    onr = (tsr >= ARC0 + 0.01) & (tsr < ARC1)
    gas = np.where(tsr >= ARC0 - 0.2, 20.5, 0.2)
    if gas_dip:
        gas[(tsr >= gas_dip[0]) & (tsr < gas_dip[1])] = 6.0
    x = 30 + 7.0 * np.clip(tsr - ARC0, 0, ARC1 - ARC0)              # 7 mm/s along +x
    y = 175 + 2.0 * np.sin(2 * np.pi * 0.8 * tsr)                   # weave across the seam
    z = np.full_like(tsr, -16.0)
    xyz = np.stack([x, y, z], axis=1)
    ang = np.tile([0.0, -8.0, 0.0], (len(tsr), 1))
    keep = (np.arange(len(tsr)) % 8) == 0
    xyz[~keep] = np.nan
    ang[~keep] = np.nan
    rob = xr.Dataset({"arc_on": ("time", onr.astype(float)), "gas_flow_leading": ("time", gas),
                      "gas_flow_trailing": ("time", gas - 1.0),
                      "thermo_1": ("time", 100 + 20 * np.clip(tsr - ARC0, 0, None)),
                      "thermo_2": ("time", np.where(tsr > 3.0, 1400.0 - 100 * (tsr - 3.0), -50.0)),
                      "tcp_coords": (("time", "c"), xyz), "tcp_angles": (("time", "c"), ang)},
                     coords={"time": tr, "c": ["x", "y", "z"]})
    _write_nc(rob, d / f"{stem}_robot.nc")
    # --- power sources: 1 kHz axis, values every 50th sample (20 Hz), NaN elsewhere
    for side, wfs_cmd, irec, urec in (("leading", 20.0, 392.5, 32.25), ("trailing", 15.0, 350.0, 27.5)):
        if side == "trailing" and not tandem:
            continue
        tps = _times(1000, 6.0)
        tss = (tps - T0).astype(np.int64) / 1e9
        sparse = (np.arange(len(tss)) % 50) == 0

        def sp(v):
            v = np.broadcast_to(np.asarray(v, float), tss.shape).copy()
            v[~sparse] = np.nan
            return ("time", v)

        ps_on = (tss >= ARC0) & (tss < ARC1)
        ps = xr.Dataset({"ACTUAL_WFS": sp(np.where(ps_on, wfs_cmd, 0.0)), "WFS_COMMANDVALUE": sp(wfs_cmd),
                         "ACTUAL_GASFLOW": sp(np.where(ps_on, 20.0, 0.0)), "GASVALUE": sp(20.0),
                         "JOBNUMBER": sp(220.0 if side == "leading" else 215.0),
                         "CURRENT_RECOMMVALUE": sp(irec), "VOLTAGE_RECOMMVALUE": sp(urec),
                         "ACTUAL_CURRENT": sp(np.where(ps_on, irec, 0.0)),
                         "ACTUAL_VOLTAGE": sp(np.where(ps_on, urec, 0.0))},
                        coords={"time": tps})
        _write_nc(ps, d / f"{stem}_power_source_{side}.nc")
    # --- video: dark until the arc, bright moving blob while it burns, static glow after
    vw = cv2.VideoWriter(str(d / f"{stem}_video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 128))
    rng = np.random.default_rng(0)
    for k in range(N_FRAMES):
        t = REC0 + k / FPS
        img = np.full((128, 160), 30, np.uint8)
        cv2.putText(img, f"{k}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)   # overlay counter
        if ARC0 <= t < ARC1:
            cx = int(40 + 20 * (t - ARC0))
            cv2.circle(img, (cx, 80), 18, 250, -1)
            img[70:128] = np.clip(img[70:128].astype(int) + rng.integers(0, 60, (58, 160)), 0, 255)
        elif t >= ARC1:
            cv2.line(img, (20, 80), (120, 80), 120, 3)
        vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    vw.release()
    return d


def write_experiment_list(root, rows=None):
    rows = rows or [(617, 1, "single", 6.67, "PMC", "PMC", "", 12.5, 0.0, 12.5, 0.65, 2.5, 8.8),
                    (617, 3, "tandem", 7.0, "PMC+PMC", "PMC", "PMC", 20.0, 15.0, 35.0, 5.5, 0.8, 21.2)]
    head = ("wid,layer,track,config,wts,mode,mode_leading,mode_trailing,wfs_command_leading,wfs_command_trailing,"
            "wfs_command_combined,weave_amp,weave_freq,t85")
    units = "No Unit,No Unit,No Unit,No Unit,mm / s,No Unit,No Unit,No Unit,m / min,m / min,m / min,mm,Hz,s"
    lines = [head, units] + [",".join(str(v) for v in (r[0], r[1], 1) + r[2:]) for r in rows]
    (root / "experiment_list.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("gen")
    make_layer(raw, pulses_per_frame=True, gas_dip=(2.0, 2.5))
    write_experiment_list(raw)
    names = B.convert(raw, out, wids=[617])
    return raw, out, names


# ------------------------------------------------------------------ unit pieces

def test_parse_wids():
    assert B.parse_wids(None) == [617, 630]
    assert B.parse_wids("all") == [615, 617, 630, 631, 632]
    assert B.parse_wids("WID630, 617") == [617, 630]
    assert B.parse_layers("L1,3") == [1, 3]
    with pytest.raises(ValueError):
        B.parse_wids([999])


def test_experiment_list_skips_units_row(tmp_path):
    write_experiment_list(tmp_path)
    df = B.read_experiment_list(tmp_path / "experiment_list.csv")
    assert len(df) == 2 and df["wts"].dtype.kind == "f"
    assert df.loc[df["layer"] == 3, "mode"].item() == "PMC+PMC"
    assert [B.pass_type_for(n) for n in (1, 3, 6)] == ["root", "fill", "cap"]


def test_arc_window_from_current():
    t = np.arange(0, 5, 1e-4)
    i = np.where((t >= 1.2345) & (t < 3.5), 350.0, 0.0)
    i[(t >= 2.0) & (t < 2.1)] = 0.0       # short outage is bridged, not a new arc
    a0, a1 = B.arc_window(t, i)
    assert a0 == pytest.approx(1.2345, abs=2e-4)
    assert a1 == pytest.approx(3.5, abs=2e-4)


def test_euler_to_quaternion():
    q = B.euler_xyz_to_quat(np.array([[0.0, 0.0, 90.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(q[0], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    assert np.allclose(q[1], [0, 0, 0, 1])


def test_sync_pulses_one_per_frame(tmp_path):
    d = make_layer(tmp_path, pulses_per_frame=True)
    proc = xr.load_dataset(next(d.glob("*_process.nc")))
    res = B.sync_video(proc, proc["time"].values[0], next(d.glob("*_video.avi")), ARC0, ARC1)
    assert res["method"] == "video_frame_pulses"
    assert res["sync_pulses"] == N_FRAMES and res["pulses_match_frames"]
    # frame k at REC0 + k/FPS -> the arc (t = 1.0 s) is first seen on frame 5
    assert res["arc_start_slot"] == round((ARC0 - REC0) * FPS)
    assert 0 <= res["t_offset_s"] < 1 / FPS


def test_sync_falls_back_to_optical_when_pulses_do_not_match(tmp_path):
    d = make_layer(tmp_path, pulses_per_frame=False)          # 40 Hz pulses vs 10 fps video (as on BAM files)
    proc = xr.load_dataset(next(d.glob("*_process.nc")))
    res = B.sync_video(proc, proc["time"].values[0], next(d.glob("*_video.avi")), ARC0, ARC1)
    assert res["sync_pulse_hz"] == pytest.approx(40.0, rel=1e-3)
    assert not res["pulses_match_frames"]
    assert res["method"] == "optical_arc_start_stop"
    assert res["arc_start_slot"] == round((ARC0 - REC0) * FPS)          # first frame showing the arc
    assert res["arc_stop_slot"] == round((ARC1 - REC0) * FPS)           # first frame without it
    assert res["optical"]["start_slot_independent"] == res["arc_start_slot"]
    assert res["verified"] and res["independent_agrees"] and abs(res["duration_error_frames"]) <= 1


# ------------------------------------------------------------------ full convert

def test_convert_writes_contract(converted):
    _, out, names = converted
    assert names == ["WID617_L3_TGMAW"]
    d = out / names[0]
    for f in ("meta.json", "arc.parquet", "events.jsonl", "travel.parquet", "thermal.parquet", "robot.parquet",
              "torch_cam.mp4", "control/edge.mp4"):
        assert (d / f).exists(), f
    m = json.loads((d / "meta.json").read_text())
    assert m["process"] == "GMAW" and m["mode"] == "PMC+PMC" and m["electrodes"] == 2
    assert m["material"] == "S355J2+N" and m["thickness_mm"] == 30 and m["pass_type"] == "fill"
    assert m["wps_id"] == "BAM-WID617-L3" and m["rt_result"] == "not_inspected" and not m["has_defect"]
    assert m["defects"] == [] and m["spool_id"] == "WID617" and m["pass_no"] == 3
    assert m["source"]["id"] == "bam_tgmaw" and m["source"]["license"] == "CC BY 4.0"
    assert m["source"]["doi"] == "10.5281/zenodo.21824586" and m["source"]["synthetic"] is False
    assert "Chaurasia" in m["source"]["authors"]
    w = m["wps"]
    assert w["wfs_mpm"] == pytest.approx([18.0, 22.0]) and w["travel_speed_mmps"] == pytest.approx([6.3, 7.7])
    assert w["gas_lpm"] == pytest.approx([16.0, 24.0]) and w["current_a"] == pytest.approx([353.25, 431.75])
    assert w["wfs2_mpm"] == pytest.approx([13.5, 16.5])
    assert m["arc_on_s"] == pytest.approx(ARC1 - ARC0, abs=0.01)
    assert m["duration_s"] == pytest.approx(ARC1 - ARC0 + 1.0, abs=0.01)
    assert (out / "bam_tgmaw_report.md").exists()


def test_convert_signals(converted):
    _, out, names = converted
    d = out / names[0]
    m = json.loads((d / "meta.json").read_text())
    arc = pd.read_parquet(d / "arc.parquet")
    assert list(arc.columns)[:5] == ["t_ns", "current_a", "voltage_v", "wfs_mpm", "gas_lpm"]
    assert {"current2_a", "voltage2_v", "wfs2_mpm"} <= set(arc.columns)
    assert arc["t_ns"].iloc[0] == m["t0_ns"] and np.allclose(np.diff(arc["t_ns"]), 1e7)
    t = (arc["t_ns"] - m["t0_ns"]) / 1e9
    steady = (t > 0.2) & (t < 2.8)
    assert arc.loc[steady, "current_a"].mean() == pytest.approx(300 + 150 * 0.295, rel=0.05)
    assert arc.loc[steady, "current2_a"].mean() == pytest.approx(330, rel=0.01)
    assert arc.loc[steady, "wfs_mpm"].mean() == pytest.approx(20.0, rel=0.02)
    assert arc.loc[t > 3.2, "current_a"].max() < 1
    trv = pd.read_parquet(d / "travel.parquet")
    assert trv.loc[steady, "travel_speed_mmps"].median() == pytest.approx(7.0, rel=0.03)   # weave removed
    assert trv["position_mm"].iloc[int(2.0 * 100)] == pytest.approx(14.0, abs=0.5)
    assert trv.loc[steady, "heat_input_kj_mm"].notna().all()
    thm = pd.read_parquet(d / "thermal.parquet")
    assert list(thm.columns) == ["t_ns", "interpass_c", "thermo2_c"]
    assert thm["interpass_c"].iloc[0] == pytest.approx(100, abs=1)
    rob = pd.read_parquet(d / "robot.parquet")
    assert {"tcp_x", "tcp_y", "tcp_z", "tcp_qw"} <= set(rob.columns)
    ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]
    types = [e["type"] for e in ev]
    assert "arc_start" in types and "arc_stop" in types
    gas = [a for a in m["anomalies"] if a["kind"] == "gas_dip" and a["electrode"] == "lead"]
    assert len(gas) == 1 and gas[0]["t_start"] == pytest.approx(1.0, abs=0.15)


def test_convert_video_trimmed_to_arc_start(converted):
    _, out, names = converted
    d = out / names[0]
    m = json.loads((d / "meta.json").read_text())
    v = m["video"]
    assert v["kind"] == "real_process_camera" and v["file"] == "torch_cam.mp4"
    assert v["sync"]["method"] == "video_frame_pulses"
    assert v["fps"] == pytest.approx(FPS, rel=1e-3)
    assert abs(v["frames"] - m["duration_s"] * FPS) <= 1          # arc start .. arc stop + 1 s
    assert len(v["stills"]) == 3 and all((d / s).exists() for s in v["stills"])
    cap = cv2.VideoCapture(str(d / "torch_cam.mp4"))
    ok, f0 = cap.read()
    assert ok and f0.shape[:2] == (128, 160)
    assert f0[60:100, 20:60].mean() > 120          # frame 0 already shows the arc (bright blob)
    n = 1
    while cap.read()[0]:
        n += 1
    assert n == v["frames"]
    e = cv2.VideoCapture(str(d / "control" / "edge.mp4"))
    assert int(e.get(cv2.CAP_PROP_FRAME_COUNT)) == v["frames"]


def test_arc_outage_is_reported(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "gen"
    make_layer(raw, outage=(2.0, 2.05))
    write_experiment_list(raw)
    names = B.convert(raw, out, wids=[617], video=False)
    m = json.loads((out / names[0] / "meta.json").read_text())
    outs = [a for a in m["anomalies"] if a["kind"] == "arc_outage"]
    assert len(outs) == 1 and outs[0]["t_start"] == pytest.approx(1.0, abs=0.02)
    assert "video" not in m


# ------------------------------------------------------------------ download (offline)

def test_download_extracts_and_is_idempotent(tmp_path, monkeypatch):
    src = tmp_path / "src"
    d = make_layer(src, wid=630, layer=2)
    (d / "WID630_L2_T1_scan_pre.nc").write_bytes(b"x" * 100)
    zpath = tmp_path / "WID630_L2.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in d.iterdir():
            z.write(p, p.name)
    write_experiment_list(src)
    calls = []

    def fake_download(url, dst, size=None, md5=None, **kw):
        calls.append(dst.name)
        shutil.copy(zpath if dst.suffix == ".zip" else src / "experiment_list.csv", dst)
        return dst

    def no_network(url, timeout=60.0):
        raise OSError("offline")

    monkeypatch.setattr(B, "_download_file", fake_download)
    monkeypatch.setattr(B, "_http_json", no_network)
    monkeypatch.setattr(B, "EXTRA_FILES", ("experiment_list.csv",))
    raw = tmp_path / "raw"
    B.download(raw, wids=[630], layers=[2])
    assert calls == ["experiment_list.csv", "WID630_L2.zip"]
    out = raw / "WID630_L2"
    assert (out / "WID630_L2_T1_process.nc").exists() and (out / "WID630_L2_T1_video.avi").exists()
    assert not list(out.glob("*_scan_*")) and not (raw / "WID630_L2.zip").exists()
    B.download(raw, wids=[630], layers=[2])            # second run: nothing fetched
    assert calls == ["experiment_list.csv", "WID630_L2.zip"]
