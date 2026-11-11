"""WeldCell simulator + renderer tests (fast, CPU)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from collect import simulate as sim
from collect.schema import ARC_HZ, ROBOT_HZ, THERMAL_HZ, WPS_LIBRARY, Defect, heat_input_kj_mm

CONTRACT_FILES = ["meta.json", "arc.parquet", "travel.parquet", "robot.parquet", "thermal.parquet",
                  "events.jsonl"]


def _plan(spool_idx: int, pass_no: int, defect=None) -> sim.PassPlan:
    jt = sim.BASE_JOBS[spool_idx]
    ptype, wps_id = jt.passes[pass_no - 1]
    return sim.PassPlan(jt, pass_no, ptype, wps_id, defect)


def _window(df, t0_ns, d, col):
    t = (df["t_ns"] - t0_ns) / 1e9
    inside = (t >= d.t_start + 0.4) & (t <= d.t_end - 0.4)
    outside = (t >= 1.0) & (t <= d.t_start - 1.0)
    return df.loc[inside, col].to_numpy(), df.loc[outside, col].to_numpy()


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    dirs = sim.simulate_shop(n_passes=12, seed=7, out=out, render=False, duration_s=16.0, quiet=True)
    return out, dirs


def test_contract_files(shop):
    out, dirs = shop
    assert len(dirs) == 12
    for d in dirs:
        for f in CONTRACT_FILES:
            assert (d / f).exists(), f"{d.name}/{f} missing"
        assert (d / "control").is_dir()
        meta = json.loads((d / "meta.json").read_text())
        for k in ("sequence_name", "spool_id", "joint_id", "pass_no", "pass_type", "wps_id", "process",
                  "position", "material", "pipe_od_mm", "wall_mm", "cell_id", "duration_s", "defects",
                  "rt_result", "wps", "has_defect", "t0_ns"):
            assert k in meta
        assert meta["sequence_name"] == d.name
        assert meta["rt_result"] in ("accept", "reject")
        arc = pd.read_parquet(d / "arc.parquet")
        assert list(arc.columns) == ["t_ns", "current_a", "voltage_v", "wfs_mpm", "gas_lpm"]
        assert np.allclose(np.diff(arc["t_ns"]), 1e9 / ARC_HZ)
        trv = pd.read_parquet(d / "travel.parquet")
        assert list(trv.columns) == ["t_ns", "travel_speed_mmps", "positioner_deg", "heat_input_kj_mm"]
        rob = pd.read_parquet(d / "robot.parquet")
        assert list(rob.columns) == ["t_ns", "j1", "j2", "j3", "j4", "j5", "j6", "tcp_x", "tcp_y", "tcp_z",
                                     "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]
        assert np.allclose(np.diff(rob["t_ns"]), 1e9 / ROBOT_HZ)
        q = rob[["tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]].to_numpy()
        assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)
        thm = pd.read_parquet(d / "thermal.parquet")
        assert list(thm.columns) == ["t_ns", "interpass_c"]
        assert np.allclose(np.diff(thm["t_ns"]), 1e9 / THERMAL_HZ)
        ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]
        assert ev[0]["type"] == "arc_start" and ev[-1]["type"] == "arc_stop"
        assert sum(e["type"] == "defect_on" for e in ev) == len(meta["defects"])
    assert (out / "index.jsonl").exists()


def test_determinism(tmp_path):
    a = sim.simulate_shop(n_passes=3, seed=11, out=tmp_path / "a", render=False, duration_s=10, quiet=True)
    b = sim.simulate_shop(n_passes=3, seed=11, out=tmp_path / "b", render=False, duration_s=10, quiet=True)
    for da, db in zip(a, b):
        for f in ("arc", "travel", "robot", "thermal"):
            pd.testing.assert_frame_equal(pd.read_parquet(da / f"{f}.parquet"), pd.read_parquet(db / f"{f}.parquet"))
        assert (da / "meta.json").read_text() == (db / "meta.json").read_text()
        assert (da / "events.jsonl").read_text() == (db / "events.jsonl").read_text()
    c = sim.simulate_shop(n_passes=3, seed=12, out=tmp_path / "c", render=False, duration_s=10, quiet=True)
    assert not pd.read_parquet(a[0] / "arc.parquet").equals(pd.read_parquet(c[0] / "arc.parquet"))


def test_reference_dataset_mix():
    rng = np.random.default_rng(7)
    plan = sim.build_plan(24, rng)
    assert len(plan) == 24
    n_def = sum(p.defect is not None for p in plan)
    assert 0.28 <= n_def / 24 <= 0.42
    assert {p.process.value for p in plan} == {"GTAW", "FCAW", "SAW"}
    assert any(p.sequence_name == "SP-1043_J07_P3_FCAW" and p.defect[0] == Defect.POROSITY for p in plan)


def test_heat_input_and_wps_window_clean_pass():
    out = sim.simulate_pass(_plan(0, 3), seed=7, t0_ns=0, interpass_start_c=120.0, duration_s=20)
    arc, trv = out["arc"], out["travel"]
    wps = WPS_LIBRARY["WPS-FCAW-FILL-02"]
    on = (arc["current_a"] > 10).to_numpy()
    mid = slice(int(1.0 * ARC_HZ), int(18.0 * ARC_HZ))
    I = arc["current_a"].to_numpy()[mid].mean()
    V = arc["voltage_v"].to_numpy()[mid].mean()
    S = trv["travel_speed_mmps"].to_numpy()[mid].mean()
    assert wps.current_a[0] < I < wps.current_a[1]
    assert wps.voltage_v[0] < V < wps.voltage_v[1]
    assert wps.travel_speed_mmps[0] < S < wps.travel_speed_mmps[1]
    hi = trv["heat_input_kj_mm"].to_numpy()[mid].mean()
    assert abs(hi - heat_input_kj_mm(I, V, S)) / hi < 0.05
    assert wps.heat_input_kj_mm[0] < hi < wps.heat_input_kj_mm[1]
    assert on[:int(19.0 * ARC_HZ)][int(0.3 * ARC_HZ):].all()
    # FCAW short-circuit transfer: voltage collapses with current spikes
    v = arc["voltage_v"].to_numpy()[mid]
    assert (v < 0.6 * V).sum() > 10
    # positioner turns the pipe: degrees = travel / radius
    deg = trv["positioner_deg"].to_numpy()
    assert deg[-1] > 30 and np.all(np.diff(deg) >= 0)
    assert out["pass"].rt_result == "accept" and not out["pass"].defects


@pytest.mark.parametrize("defect,sev,spool,pass_no", [
    (Defect.POROSITY, 0.8, 0, 3), (Defect.LACK_OF_FUSION, 0.7, 0, 4), (Defect.UNDERCUT, 0.6, 0, 4),
    (Defect.BURN_THROUGH, 0.65, 0, 1), (Defect.ARC_INSTABILITY, 0.6, 0, 2), (Defect.CRACK, 0.6, 5, 3)])
def test_defect_signatures(defect, sev, spool, pass_no):
    pp = _plan(spool, pass_no, (defect, sev))
    T0 = pp.wps.max_interpass_c + 20 if defect == Defect.CRACK else 120.0
    out = sim.simulate_pass(pp, seed=3, t0_ns=0, interpass_start_c=T0, duration_s=30)
    d = out["pass"].defects[0]
    assert d.defect == defect and d.t_end > d.t_start
    arc, trv, thm = out["arc"], out["travel"], out["thermal"]
    ev = {e["type"] for e in out["events"]}
    assert {"defect_on", "defect_off"} <= ev
    if defect == Defect.POROSITY:
        gi, go = _window(arc, 0, d, "gas_lpm")
        assert gi.mean() < 8.0 < go.mean()
        vi, vo = _window(arc, 0, d, "voltage_v")
        assert vi.std() > 1.5 * vo.std()
        assert out["pass"].rt_result == "reject"
    elif defect == Defect.LACK_OF_FUSION:
        ci, co = _window(arc, 0, d, "current_a")
        si, so = _window(trv, 0, d, "travel_speed_mmps")
        assert ci.mean() < 0.85 * co.mean() and si.mean() > 1.25 * so.mean()
        assert out["pass"].rt_result == "reject"
    elif defect == Defect.UNDERCUT:
        vi, vo = _window(arc, 0, d, "voltage_v")
        si, so = _window(trv, 0, d, "travel_speed_mmps")
        assert vi.mean() > 1.12 * vo.mean() and si.mean() > 1.12 * so.mean()
    elif defect == Defect.BURN_THROUGH:
        ci, co = _window(arc, 0, d, "current_a")
        si, so = _window(trv, 0, d, "travel_speed_mmps")
        assert ci.mean() > 1.2 * co.mean() and si.mean() < 0.75 * so.mean()
        hi_i, hi_o = _window(trv, 0, d, "heat_input_kj_mm")
        assert hi_i.mean() > 1.5 * hi_o.mean()
        assert out["pass"].rt_result == "reject"
    elif defect == Defect.ARC_INSTABILITY:
        vi, vo = _window(arc, 0, d, "voltage_v")
        wi, wo = _window(arc, 0, d, "wfs_mpm")
        mad = lambda x: np.median(np.abs(x - np.median(x)))   # robust to short-circuit dips
        assert mad(vi) > 2 * mad(vo) and wi.std() > 3 * wo.std()
    elif defect == Defect.CRACK:
        assert thm["interpass_c"].max() > pp.wps.max_interpass_c
        assert out["pass"].rt_result == "reject"


def test_rt_rules():
    from collect.schema import DefectEvent
    assert sim.rt_result_for([]) == "accept"
    assert sim.rt_result_for([DefectEvent(Defect.CRACK, 1, 2, 0.1)]) == "reject"
    assert sim.rt_result_for([DefectEvent(Defect.POROSITY, 1, 2, 0.3)]) == "accept"
    assert sim.rt_result_for([DefectEvent(Defect.POROSITY, 1, 2, 0.7)]) == "reject"
    assert sim.rt_result_for([DefectEvent(Defect.UNDERCUT, 1, 2, 0.4)]) == "accept"


def test_robot_ik_tracks_weave():
    from collect import robot as rb
    n = 200
    t = np.arange(n) / ROBOT_HZ
    pos = np.column_stack([np.full(n, 0.95), 0.004 * np.sin(2 * np.pi * 1.2 * t), np.full(n, 0.12)])
    Q, P, Qt = rb.solve_trajectory(pos, rb.torch_rotation(10))
    assert np.abs(P - pos).max() < 1e-4
    assert np.ptp(Q[:, 0]) > 1e-3        # base axis weaves


def test_render_small(tmp_path):
    """Render 1 s of a porosity pass at low resolution: contract videos exist with identical timing."""
    import av
    from collect.render import render_pass
    pp = _plan(0, 3, (Defect.POROSITY, 0.8))
    out = sim.simulate_pass(pp, seed=7, t0_ns=0, interpass_start_c=120.0, duration_s=8)
    d = sim.write_pass_dir(tmp_path, out)
    res = render_pass(d, max_seconds=1.0, w=320, h=180, stills_dir=tmp_path / "img", still_times=[0.5])
    assert res["frames"] == 24
    counts = {}
    for f in ("torch_cam.mp4", "control/edge.mp4", "control/depth.mp4"):
        with av.open(str(d / f)) as c:
            s = c.streams.video[0]
            assert s.codec_context.name == "h264"
            frames = list(c.decode(s))
            counts[f] = len(frames)
            assert frames[0].width == 320 and frames[0].height == 180
    assert len(set(counts.values())) == 1 and counts["torch_cam.mp4"] == 24
    assert len(list((tmp_path / "img").glob("*.png"))) == 3
