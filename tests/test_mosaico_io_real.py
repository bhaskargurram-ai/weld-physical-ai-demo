"""mosaico_io on the real-data contract (docs/REAL_DATA.md): optional streams, tandem fields, NaN gas,
photos, real-video export. Offline always; live mosaicod round trip when one is reachable (marked `mosaico`)."""
from __future__ import annotations

import json
import os
import socket

import numpy as np
import pandas as pd
import pytest

from mosaico_io.testdata import make_real_fixture

SAW, GMAW = "PP5_B02_SAW", "WID617_L3_TGMAW"


def _mosaicod_up() -> bool:
    try:
        with socket.create_connection((os.environ.get("MOSAICO_HOST", "localhost"),
                                       int(os.environ.get("MOSAICO_PORT", "6726"))), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def real_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("real")
    make_real_fixture(root)
    return root


@pytest.fixture
def offline(real_root, monkeypatch):
    monkeypatch.setenv("MOSAICO_OFFLINE", "1")
    monkeypatch.setenv("DATA_DIR", str(real_root))
    monkeypatch.delenv("WELD_GENERATED_DIR", raising=False)
    return real_root


def test_list_passes_provenance(offline):
    from mosaico_io import list_passes
    rows = {r["sequence_name"]: r for r in list_passes()}
    assert set(rows) == {SAW, GMAW}
    s, g = rows[SAW], rows[GMAW]
    assert s["source_id"] == "aimen_saw" and s["source_license"] == "CC BY 4.0"
    assert s["source_doi"] == "10.5281/zenodo.15083865" and s["electrodes"] == 2
    assert s["rt_result"] == "indications" and s["n_photos"] == 1 and not s["has_video"]
    assert "Zenodo, DOI 10.5281/zenodo.15083865" in s["source"]["attribution"]
    assert g["rt_result"] == "not_inspected" and g["has_video"] and g["source"]["id"] == "bam_tgmaw"
    assert [r["sequence_name"] for r in list_passes(source_id="aimen_saw")] == [SAW]
    assert {r["sequence_name"] for r in list_passes(**{"source.license": "CC BY 4.0", "electrodes": 2})} == {SAW, GMAW}


def test_signals_optional_and_tandem(offline):
    from mosaico_io import get_signals
    s = get_signals(SAW)
    core = ["t_s", "current_a", "voltage_v", "wfs_mpm", "gas_lpm", "travel_speed_mmps", "heat_input_kj_mm", "interpass_c"]
    assert list(s.columns[:8]) == core and "positioner_deg" not in s
    assert {"current2_a", "voltage2_v", "wfs2_mpm", "temp_left_c", "temp_back_c", "position_mm"} <= set(s.columns)
    assert s["gas_lpm"].isna().all() and s["current2_a"].notna().all()       # SAW: flux, no gas
    g = get_signals(GMAW, 2.0, 4.0)
    assert g["gas_lpm"].between(15, 21).all() and "thermo2_c" in g
    assert abs(g["t_s"].iloc[0] - 2.0) < 1e-6 and len(g) == 200


def test_signals_without_travel_or_thermal(offline):
    from mosaico_io import get_signals, wps_violations
    d = offline / "generated" / SAW
    for f in ("travel.parquet", "thermal.parquet", "robot.parquet"):
        (d / f).rename(d / f"{f}.bak")
    try:
        df = get_signals(SAW)
        assert df["travel_speed_mmps"].isna().all() and df["interpass_c"].isna().all()
        assert wps_violations(SAW) == []               # missing signals are skipped, not reported
    finally:
        for f in ("travel.parquet", "thermal.parquet", "robot.parquet"):
            (d / f"{f}.bak").rename(d / f)


def test_wps_violations_trail_electrode_and_missing_ranges(offline, monkeypatch):
    from mosaico_io import find_anomalies, query
    from mosaico_io.common import load_meta, offline_signals
    meta, df = load_meta(SAW), offline_signals(SAW)
    assert query.violations_from_signals(df, meta) == []
    df2 = df.copy()
    m = (df2["t_s"] > 10) & (df2["t_s"] < 12)
    df2.loc[m, "current2_a"] = 720.0                   # trail electrode above 540-660 A
    v = query.violations_from_signals(df2, meta)
    assert [x["param"] for x in v] == ["current2_a"] and v[0]["kind"] == "high" and v[0]["limit_hi"] == 660.0
    # no range for a param -> skipped (gas on SAW has no range; remove trail ranges too)
    meta2 = {**meta, "wps": {k: w for k, w in meta["wps"].items() if not k.endswith("2_a")}}
    assert query.violations_from_signals(df2, meta2) == []
    meta3 = {**meta, "wps": {**meta["wps"], "gas_lpm": [0.0, 0.0], "max_interpass_c": None}}
    assert query.violations_from_signals(df, meta3) == []   # NaN gas vs a [0,0] range: not a violation
    assert find_anomalies("gas_lpm", lt=10) == []           # SAW has no gas signal; BAM gas is normal
    hits = find_anomalies("current2_a", gt=580)
    assert hits and {h["sequence_name"] for h in hits} == {SAW} and hits[0]["field"] == "current2_avg_a"


def test_export_video_and_photo_windows(offline, tmp_path):
    import av
    from mosaico_io import export_clips
    out = tmp_path / "export"
    res = {r["clip_id"]: r for r in export_clips(out)}
    saw = [r for r in res.values() if r["sequence_name"] == SAW]
    bam = [r for r in res.values() if r["sequence_name"] == GMAW]
    assert len(saw) == 1 and saw[0]["media_type"] == "image" and saw[0]["video"] is None
    s = saw[0]
    assert s["t0_s"] <= 17.0 and s["t1_s"] >= 17.1 and s["label"] == "defect"
    assert s["photos"] == ["photos/PP5_B02_SAW/PP5_02_1.jpg"] and (out / s["photos"][0]).exists()
    assert "x=185.9-186.8 mm" in s["photo_note"] and s["ndt"]["indications_in_window"][0]["x_start_mm"] == 185.9
    assert s["source"]["license"] == "CC BY 4.0" and s["electrodes"] == 2 and "gas_lpm" not in s["signal_stats"]
    assert "current2_a" in s["signal_stats"] and not (out / f"{s['clip_id']}.mp4").exists()
    assert bam and all(r["media_type"] == "video" and r["label"] == "clean" for r in bam)
    b = bam[0]
    assert b["fps"] == pytest.approx(17.96) and b["video_kind"] == "real_process_camera"
    with av.open(str(out / b["video"])) as c:
        st = c.streams.video[0]
        n = sum(1 for _ in c.decode(st))
        assert float(st.average_rate) == pytest.approx(17.96, rel=1e-3)
    assert n == pytest.approx(5 * 17.96, abs=2)
    assert (out / b["control"]["edge"]).exists() and "depth" not in b["control"]
    side = json.loads((out / f"{s['clip_id']}.json").read_text())
    assert side["media_type"] == "image" and side["read_from"] == "local"
    idx = [json.loads(x) for x in (out / "clips.jsonl").read_text().splitlines()]
    assert {r["media_type"] for r in idx} == {"image", "video"}


def test_choose_windows_merges_and_image_rules():
    from mosaico_io.export import choose_windows
    meta = {"duration_s": 60.0, "defects": [{"defect": "porosity", "t_start": 20.0, "t_end": 20.2},
                                            {"defect": "porosity", "t_start": 20.5, "t_end": 20.7},
                                            {"defect": "lack_of_fusion", "t_start": 40.0, "t_end": 41.0}]}
    w = choose_windows(meta, 5.0, media="video")
    assert [x["kind"] for x in w].count("defect") == 2 and any(x["kind"] == "clean" for x in w)
    wi = choose_windows(meta, 5.0, media="image")
    assert [x["kind"] for x in wi] == ["defect", "defect"]                  # no clean window w/ a defective photo
    assert [x["kind"] for x in choose_windows({"duration_s": 60.0, "defects": []}, 5.0, media="image")] == ["clean"]


@pytest.mark.mosaico
@pytest.mark.skipif(not _mosaicod_up(), reason="mosaicod not reachable")
def test_live_ingest_roundtrip(monkeypatch, tmp_path):
    from mosaico_io import common, export_clips, get_meta, get_signals, ingest_pass_dir, list_passes, wps_violations
    root = tmp_path / "live"
    gen = make_real_fixture(root, prefix="PYTEST_")       # never overwrite the real sequences in mosaicod
    SAW, GMAW = "PYTEST_PP5_B02_SAW", "PYTEST_WID617_L3_TGMAW"   # noqa: N806
    monkeypatch.setenv("DATA_DIR", str(root))
    monkeypatch.setenv("MOSAICO_OFFLINE", "0")
    for seq in (SAW, GMAW):
        assert ingest_pass_dir(gen / seq) == seq
    meta = get_meta(SAW)
    assert meta["source_id"] == "aimen_saw" and meta["source"]["license"] == "CC BY 4.0"
    assert meta["electrodes"] == 2 and meta["rt_result"] == "indications"
    assert any(t["topic"] == common.TOPIC_BEAD_PHOTO for t in meta["topics"])
    assert SAW in [r["sequence_name"] for r in list_passes(source_id="aimen_saw")]
    on = get_signals(SAW)
    monkeypatch.setenv("MOSAICO_OFFLINE", "1")
    off = get_signals(SAW)
    pd.testing.assert_index_equal(on.columns, off.columns)
    assert np.allclose(on["current2_a"], off["current2_a"]) and on["gas_lpm"].isna().all()
    # export reading only from Mosaico (no local pass dirs): photos come from /camera/bead/photo, video from
    # /camera/torch/image
    monkeypatch.setenv("MOSAICO_OFFLINE", "0")
    monkeypatch.setenv("WELD_GENERATED_DIR", str(tmp_path / "nothing"))
    assert wps_violations(GMAW) == []
    res = {r["sequence_name"]: r for r in export_clips(tmp_path / "exp", sequences=[SAW, GMAW])}
    assert res[SAW]["media_type"] == "image" and (tmp_path / "exp" / res[SAW]["photos"][0]).stat().st_size > 1000
    assert res[GMAW]["media_type"] == "video" and res[GMAW]["video_source"].startswith("mosaico:")
    client = common.connect()
    for seq in (SAW, GMAW):
        client.sequence_delete(seq)


def test_ac_trail_electrode_uses_rms_and_real_excursions_merge(offline):
    from mosaico_io import query
    from mosaico_io.common import load_meta, offline_signals
    meta, df = load_meta(SAW), offline_signals(SAW)
    df = df.copy()
    rms = df["current2_a"].to_numpy().copy()
    df["current2_a"] = rms * np.sqrt(2) * np.sin(2 * np.pi * 50 * df["t_s"].to_numpy() + 0.3)   # AC: mean ~0
    df["current2_rms_a"] = rms
    assert query.violations_from_signals(df, meta) == []          # the ~0 mean is not a "low current" excursion
    df.loc[(df["t_s"] > 10) & (df["t_s"] < 12), "current2_rms_a"] = 720.0
    v = query.violations_from_signals(df, meta)
    assert [(x["param"], x.get("basis")) for x in v] == [("current2_a", "rms")]
    df2 = offline_signals(SAW).drop(columns=[])
    df2["current2_a"] = df["current2_a"]                          # AC without an RMS column -> skipped
    assert query.violations_from_signals(df2, meta) == []
    # real data: excursions < 1 s apart become one record; the arc-start ramp (first 2 s) is ignored
    df3 = offline_signals(SAW).copy()
    t = df3["t_s"]
    df3.loc[t < 1.5, "current_a"] = 200.0
    for a in (10.0, 10.6, 11.2):
        df3.loc[(t > a) & (t < a + 0.4), "current_a"] = 400.0
    v3 = query.violations_from_signals(df3, meta)
    assert len(v3) == 1 and v3[0]["param"] == "current_a" and v3[0]["t_start_s"] < 10.5 < 11.2 < v3[0]["t_end_s"]
