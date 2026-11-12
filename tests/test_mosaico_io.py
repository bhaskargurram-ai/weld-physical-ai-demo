"""mosaico_io tests: offline mode (always) + live mosaicod integration (marked `mosaico`)."""
from __future__ import annotations

import json
import os
import socket

import numpy as np
import pytest

from collect import simulate as sim
from collect.schema import Defect


def _mosaicod_up() -> bool:
    host = os.environ.get("MOSAICO_HOST", "localhost")
    port = int(os.environ.get("MOSAICO_PORT", "6726"))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    """12 SP-1043 passes (J05..J07; J05_P3 undercut, J06_P1 burn-through, J07_P3 porosity)."""
    root = tmp_path_factory.mktemp("data")
    out = root / "generated"
    sim.simulate_shop(n_passes=12, seed=7, out=out, render=False, duration_s=16.0, quiet=True)
    return root


@pytest.fixture
def offline(gen, monkeypatch):
    monkeypatch.setenv("MOSAICO_OFFLINE", "1")
    monkeypatch.setenv("DATA_DIR", str(gen))
    monkeypatch.delenv("WELD_GENERATED_DIR", raising=False)
    return gen


def test_list_passes_filters(offline):
    from mosaico_io import list_passes
    rows = list_passes()
    assert len(rows) == 12
    assert {"sequence_name", "process", "spool_id", "joint_id", "pass_type", "rt_result", "has_defect",
            "defects", "wps"} <= set(rows[0])
    fc = list_passes(process="FCAW", pass_type=["fill", "cap"])
    assert len(fc) == 6 and all(r["process"] == "FCAW" for r in fc)
    assert [r["sequence_name"] for r in list_passes(joint_id="J07", pass_type="fill", has_defect=True)] == ["SP-1043_J07_P3_FCAW"]
    assert all(r["sequence_name"].startswith("SP-1043_J06") for r in list_passes(sequence_name="SP-1043_J06_*"))
    assert all(r["heat_input_mean_kj_mm"] > 0.9 for r in list_passes(heat_input_mean_kj_mm__gt=0.9))


def test_get_signals_shape(offline):
    from mosaico_io import get_signals
    df = get_signals("SP-1043_J07_P3_FCAW")
    assert list(df.columns) == ["t_s", "current_a", "voltage_v", "wfs_mpm", "gas_lpm", "travel_speed_mmps",
                                "positioner_deg", "heat_input_kj_mm", "interpass_c"]
    assert abs(df["t_s"].iloc[0]) < 1e-9 and np.allclose(np.diff(df["t_s"]), 0.01)
    w = get_signals("SP-1043_J07_P3_FCAW", 2.0, 4.0)
    assert len(w) == 200 and w["t_s"].min() >= 2.0 and w["t_s"].max() < 4.0
    assert not df.isna().any().any()


def test_wps_violations(offline):
    from mosaico_io import list_passes, wps_violations
    for r in list_passes(has_defect=False):
        assert wps_violations(r["sequence_name"]) == [], r["sequence_name"]
    v = wps_violations("SP-1043_J07_P3_FCAW")
    gas = [x for x in v if x["param"] == "gas_lpm"]
    assert gas and gas[0]["kind"] == "low" and gas[0]["value"] < 8
    assert set(gas[0]) >= {"param", "t_start_s", "t_end_s", "value", "limit_lo", "limit_hi"}
    bt = {x["param"] for x in wps_violations("SP-1043_J06_P1_GTAW")}
    assert {"current_a", "travel_speed_mmps"} <= bt


def test_find_anomalies(offline):
    from mosaico_io import find_anomalies
    gas = find_anomalies("gas_lpm", lt=12)
    assert [a["sequence_name"] for a in gas] == ["SP-1043_J07_P3_FCAW"]
    assert gas[0]["duration_s"] > 1.0 and gas[0]["peak"] < 8
    hv = find_anomalies("voltage_v", gt=27.5, process="FCAW")
    assert "SP-1043_J05_P3_FCAW" in {a["sequence_name"] for a in hv}
    with pytest.raises(ValueError):
        find_anomalies("current_a")


def test_export_clips_offline(offline, tmp_path):
    import av
    from collect.render import render_pass
    from mosaico_io import export_clips
    seq = "SP-1043_J07_P3_FCAW"
    render_pass(offline / "generated" / seq, w=320, h=180)
    res = export_clips(tmp_path, sequences=[seq], clip_s=2.0)
    assert any(r["label"] == "defect" for r in res) and any(r["label"] == "clean" for r in res)
    for r in res:
        sc = json.loads((tmp_path / f"{r['clip_id']}.json").read_text())
        assert set(sc) >= {"clip_id", "sequence_name", "t0_s", "t1_s", "meta", "signal_stats",
                           "wps_violations", "defects_in_window", "rt_result"}
        assert set(sc["signal_stats"]["current_a"]) == {"mean", "std", "min", "max"}
        assert abs(sc["t1_s"] - sc["t0_s"] - 2.0) < 1e-6
        with av.open(str(tmp_path / f"{r['clip_id']}.mp4")) as c:
            st = c.streams.video[0]
            assert st.average_rate == 24
            assert sum(1 for _ in c.decode(st)) == 48
            assert abs(float(st.duration * st.time_base) - 2.0) < 0.05
        assert (tmp_path / sc["control"]["edge"]).exists() and (tmp_path / sc["control"]["depth"]).exists()
    d = [r for r in res if r["label"] == "defect"][0]
    assert d["defects_in_window"][0]["defect"] == "porosity"
    assert d["signal_stats"]["gas_lpm"]["min"] < 8
    assert (tmp_path / "clips.jsonl").exists()


@pytest.mark.mosaico
@pytest.mark.skipif(not _mosaicod_up(), reason="mosaicod not reachable on MOSAICO_HOST:MOSAICO_PORT")
def test_mosaico_roundtrip(tmp_path, monkeypatch):
    """Ingest one pass into the live mosaicod, read it back, query it, delete it."""
    import mosaico_io.common as common
    from mosaico_io import connect, find_anomalies, get_signals, ingest_pass_dir, list_passes, wps_violations

    jt = sim.Joint("TEST-9001", "J99", "A106 Gr.B", 168.3, 10.97, "WC-TEST", sim.PIPE_PASSES, 0)
    pp = sim.PassPlan(jt, 3, "fill", "WPS-FCAW-FILL-02", (Defect.POROSITY, 0.8))
    out = sim.simulate_pass(pp, seed=1, t0_ns=1_800_000_000_000_000_000, interpass_start_c=120.0, duration_s=12)
    d = sim.write_pass_dir(tmp_path / "generated", out)
    from collect.render import render_pass
    render_pass(d, max_seconds=2.0, w=320, h=180)

    monkeypatch.setenv("MOSAICO_OFFLINE", "0")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    client = connect()
    seq = ingest_pass_dir(d, client, camera_fps=12)
    try:
        assert seq == "TEST-9001_J99_P3_FCAW" and client.sequence_exists(seq)
        h = client.sequence_handler(seq)
        assert {"/weld/arc", "/weld/travel", "/weld/thermal", "/robot/joint_states", "/robot/tcp_pose",
                "/camera/torch/image", "/weld/events"} <= set(h.topics)
        assert h.user_metadata["spool_id"] == "TEST-9001"
        rows = list_passes(spool_id="TEST-9001")
        assert [r["sequence_name"] for r in rows] == [seq]
        on = get_signals(seq)
        v_on = wps_violations(seq)
        an = find_anomalies("gas_lpm", lt=12, spool_id="TEST-9001")
        assert an and an[0]["sequence_name"] == seq and an[0]["source"] == "mosaico"
        monkeypatch.setenv("MOSAICO_OFFLINE", "1")
        off = get_signals(seq)
        assert on.shape == off.shape
        assert np.allclose(on.to_numpy(), off.to_numpy())
        assert v_on == wps_violations(seq)
        assert common.is_offline()
    finally:
        client.sequence_delete(seq)


def test_real_data_wps_check_ignores_ramps_blips_and_unchecked_channels():
    import numpy as np
    import pandas as pd
    from mosaico_io.query import violations_from_signals
    t = np.arange(0, 60, 0.01)
    cur = np.full_like(t, 500.0)
    cur[t < 1.5] = 300.0                        # start ramp
    cur[(t > 20) & (t < 20.4)] = 420.0          # 0.4 s blip (normal SAW wander)
    cur[(t > 30) & (t < 33)] = 420.0            # 3 s sustained excursion -> the only one reported
    cur[t > 58.5] = 300.0                       # stop ramp
    df = pd.DataFrame({"t_s": t, "current_a": cur, "voltage_v": 27.0, "wfs_mpm": 1.2,
                       "travel_speed_mmps": 12.0, "heat_input_kj_mm": 1.1, "gas_lpm": np.nan})
    meta = {"source": {"id": "aimen_saw", "synthetic": False}, "arc_on_s": 60.0,
            "wps": {"current_a": [450, 550], "voltage_v": [24.3, 29.7], "wfs_mpm": [1.62, 1.98],
                    "unchecked": ["wfs_mpm"]}}
    v = violations_from_signals(df, meta)
    assert [(x["param"], x["kind"]) for x in v] == [("current_a", "low")]
    assert 29.5 < v[0]["t_start_s"] < 31 and v[0]["t_end_s"] > 32
