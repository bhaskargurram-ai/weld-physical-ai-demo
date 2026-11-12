"""AIMEN SAW adapter: synthetic HDF5 + xlsx fixtures (no network)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
openpyxl = pytest.importorskip("openpyxl")

from collect.datasets import aimen_saw as A  # noqa: E402
from collect.datasets import common as C  # noqa: E402

FS1, FS0 = 5000.0, 120.0
T_FILE, ARC_ON, ARC_OFF = 12.0, 2.0, 10.0
UNIX0 = 1_722_594_480.0

D0 = np.dtype([("process_status", "<f8"), ("pyrometer_L", "<f8"), ("pyrometer_R", "<f8"),
               ("pyrometer_back", "<f8"), ("wire_speed_a", "<f8"), ("wire_speed_b", "<f8"), ("x", "<f8"),
               ("y", "<f8"), ("z", "<f8"), ("timestamp", "<f8"), ("idx_data1", "<u4")])
D1 = np.dtype([("current_a", "<f8"), ("current_b", "<f8"), ("voltage_a", "<f8"), ("voltage_b", "<f8"),
               ("timestamp", "<f8")])

CONTRACT_KEYS = ("sequence_name", "spool_id", "joint_id", "pass_no", "pass_type", "wps_id", "process",
                 "position", "material", "cell_id", "duration_s", "defects", "rt_result", "has_defect",
                 "defect_types", "t0_ns", "wps", "setpoints", "source", "electrodes", "wps_basis", "ndt",
                 "thickness_mm", "photos")


def make_h5(path: Path, tandem: bool, I=500.0, V=27.0, Ib=450.0, ts_mmps=12.0, seed=0) -> None:
    rng = np.random.default_rng(seed)
    n1 = int(T_FILE * FS1)
    t1 = UNIX0 + np.arange(n1) / FS1
    on1 = (t1 - UNIX0 >= ARC_ON) & (t1 - UNIX0 < ARC_OFF)
    d1 = np.zeros(n1, D1)
    d1["current_a"] = np.where(on1, I + rng.normal(0, 10, n1), 0.0)
    d1["voltage_a"] = np.where(on1, V + rng.normal(0, 0.5, n1), 0.0)
    if tandem:
        ph = 2 * np.pi * 50 * (t1 - UNIX0)
        d1["current_b"] = np.where(on1, Ib * np.sqrt(2) * np.sin(ph), 0.0)
        d1["voltage_b"] = np.where(on1, 35 * np.sqrt(2) * np.sin(ph), 0.0)
    d1["timestamp"] = t1
    n0 = int(T_FILE * FS0)
    t0 = UNIX0 + np.arange(n0) / FS0
    rel = t0 - UNIX0
    on0 = (rel >= ARC_ON) & (rel < ARC_OFF)
    d0 = np.zeros(n0, D0)
    d0["process_status"] = np.where(on0, 255.0, 0.0)
    d0["pyrometer_L"] = 40 + 5 * rel
    d0["pyrometer_R"] = 30 + 6 * rel
    d0["pyrometer_back"] = 99.0
    d0["wire_speed_a"] = np.where(on0, 1.8, 0.0)
    d0["wire_speed_b"] = np.where(on0 & tandem, 1.95, 0.0)
    d0["x"] = 500.0 + ts_mmps * np.clip(rel - ARC_ON, 0, ARC_OFF - ARC_ON)
    d0["y"], d0["z"] = 155.0, 88.0
    d0["timestamp"] = t0
    d0["idx_data1"] = np.clip(np.round(rel * FS1), 0, n1 - 1).astype(np.uint32)
    cfg = {"metadata": {"02 Coupon": path.stem, "06 Flux": "OP121TT", "22.1 Current (A)": str(int(I))},
           "name": "Submerged Arc Welding", "parameters": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ds0 = f.create_dataset("data0", data=d0)
        ds0.attrs["process_config"] = np.bytes_(pickle.dumps(cfg, protocol=0))
        ds0.attrs["fps"] = 120.0
        f.create_dataset("data1", data=d1)


def make_xlsx(path: Path, coupon: str, rows: list[tuple]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "defect_code"
    ws.append(["Code", "Defect"])
    for r in [(1, "Porosity"), (2, "Lack of fusion"), (3, "Slag inclusion"), (4, "Metallic inclusion"),
              (5, "Projections"), (6, "Cracks")]:
        ws.append(r)
    ws = wb.create_sheet("welding_settings_code")
    ws.append(["Code", "welding settings"])
    ws.append([None, "Vdc (V)", "Idc (A)", "Vac (V)", "Iac (A)", "wfs (dc)", "wfs (ac)", "ts (cm/min)"])
    ws.append([1, 27, 450, None, None, 1.6, None, 48])
    ws.append([2, 27, 550, None, None, 2.05, None, 48])
    ws.append([3, 27, 500, 35, 450, 1.8, 1.95, 72])
    ws = wb.create_sheet(coupon)
    ws.append(["defect", "welding_setting", "bead", "x_init [mm]", "x_end [mm]", "x_init [# sample]",
               "x_end [# sample]"])
    for r in rows:
        ws.append(list(r))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


@pytest.fixture()
def raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    pd_ = raw / "PP9" / "1. process_data"
    make_h5(pd_ / "PP9_02" / "PP9_02.hdf5", tandem=False, I=550, ts_mmps=8.0)
    make_h5(pd_ / "PP9_03" / "PP9_03.hdf5", tandem=True, I=500, ts_mmps=12.0)
    make_h5(pd_ / "PP9_24" / "PP9_24.hdf5", tandem=False, I=550, ts_mmps=8.0)
    # bead 2 porosity: data1 samples 25000..26000 -> 5.0..5.2 s file time -> 3.0..3.2 s from arc start
    # bead 3 metallic inclusion, x_end < x_init in the sheet (as in the real PP5 file)
    make_xlsx(raw / "PP9" / "2. ndt_data" / "defects_xlocation.xlsx", "PP9",
              [(1, 2, 2, 100, 101, 25000, 26000), (4, 3, 3, 177, 160, 40000, 35000)])
    return raw


@pytest.fixture()
def converted(raw: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(A, "ALL_COUPONS", A.ALL_COUPONS + ("PP9",))
    out = tmp_path / "generated"
    seqs = A.convert(raw, out, coupons="PP9", report=True)
    return out, seqs


def test_convert_writes_contract_pass_dirs(converted):
    out, seqs = converted
    assert seqs == ["PP9_B02_SAW", "PP9_B03_SAW", "PP9_B24_SAW"]
    for s in seqs:
        d = out / s
        for f in ("meta.json", "arc.parquet", "events.jsonl", "travel.parquet", "thermal.parquet"):
            assert (d / f).exists(), f
        assert not (d / "torch_cam.mp4").exists()          # no faked video for SAW
        m = json.loads((d / "meta.json").read_text())
        for k in CONTRACT_KEYS:
            assert k in m, k
        assert m["source"]["id"] == "aimen_saw" and m["source"]["synthetic"] is False
        assert m["source"]["doi"] == "10.5281/zenodo.15083865" and m["source"]["license"] == "CC BY 4.0"
        assert m["process"] == "SAW" and m["spool_id"] == "PP9" and m["joint_id"] == "BW"
        assert m["wps_basis"].startswith("published nominal setting ±10 %")
        assert m["ndt"]["inspected"] is True and set(m["ndt"]["methods"]) == {"RT", "UT"}
        assert m["duration_s"] == pytest.approx(ARC_OFF - ARC_ON, abs=0.02)
    assert (out / "aimen_saw_report.md").exists()


def test_arc_parquet_100hz_t0_at_arc_start(converted):
    out, _ = converted
    m = json.loads((out / "PP9_B02_SAW" / "meta.json").read_text())
    arc = pd.read_parquet(out / "PP9_B02_SAW" / "arc.parquet")
    assert list(arc.columns[:5]) == ["t_ns", "current_a", "voltage_v", "wfs_mpm", "gas_lpm"]
    assert arc["t_ns"].dtype == np.int64
    assert int(arc["t_ns"].iloc[0]) == m["t0_ns"]
    assert np.all(np.diff(arc["t_ns"].to_numpy()) == 10_000_000)
    assert m["t0_ns"] == pytest.approx((UNIX0 + ARC_ON) * 1e9, abs=1e6)
    assert arc["gas_lpm"].isna().all()
    assert arc["current_a"].iloc[10:-10].mean() == pytest.approx(550, rel=0.02)
    assert "current2_a" not in arc
    ev = [json.loads(l) for l in (out / "PP9_B02_SAW" / "events.jsonl").read_text().splitlines()]
    assert ev[0]["type"] == "arc_start" and ev[0]["t_ns"] == m["t0_ns"]
    assert ev[-1]["type"] == "arc_stop"
    assert [e["type"] for e in ev].count("defect_on") == 1
    assert all(e["source"] == "NDT" for e in ev if e["type"].startswith("defect"))


def test_index_to_time_mapping(converted):
    out, _ = converted
    m = json.loads((out / "PP9_B02_SAW" / "meta.json").read_text())
    (ind,) = m["ndt"]["indications"]
    assert ind["type"] == "porosity"
    assert ind["t_start"] == pytest.approx(25000 / FS1 - ARC_ON, abs=1e-3)
    assert ind["t_end"] == pytest.approx(26000 / FS1 - ARC_ON, abs=1e-3)
    assert m["defects"] == [{"defect": "porosity", "t_start": ind["t_start"], "t_end": ind["t_end"],
                             "severity": None}]
    assert m["rt_result"] == "indications" and m["has_defect"] is True and m["defect_types"] == "porosity"
    # reversed sheet row is normalised (start < end), type from defect_code sheet
    m3 = json.loads((out / "PP9_B03_SAW" / "meta.json").read_text())
    (i3,) = m3["ndt"]["indications"]
    assert i3["type"] == "metallic_inclusion"
    assert i3["t_start"] == pytest.approx(35000 / FS1 - ARC_ON, abs=1e-3)
    assert i3["t_end"] == pytest.approx(40000 / FS1 - ARC_ON, abs=1e-3)
    assert (i3["x_start_mm"], i3["x_end_mm"]) == (160, 177)
    # pure function, extrapolating beyond the table
    assert A.sample_to_time(15000, 10000, 5000.0) == pytest.approx(1.0)


def test_wps_plus_minus_10_percent_and_tandem(converted):
    out, _ = converted
    m2 = json.loads((out / "PP9_B02_SAW" / "meta.json").read_text())
    assert m2["electrodes"] == 1 and m2["wps_id"] == "AIMEN-SAW-S02"
    assert m2["wps"]["current_a"] == pytest.approx([495.0, 605.0])
    assert m2["wps"]["voltage_v"] == pytest.approx([24.3, 29.7])
    assert m2["wps"]["travel_speed_mmps"] == pytest.approx([7.2, 8.8])     # 48 cm/min = 8 mm/s
    assert m2["wps"]["gas_lpm"] == [0.0, 0.0]
    assert m2["setpoints"]["current_a"] == 550
    assert m2["measured"]["travel_speed_mmps"] == pytest.approx(8.0, rel=0.03)
    assert abs(m2["travel_speed_mismatch_pct"]) < 3
    m3 = json.loads((out / "PP9_B03_SAW" / "meta.json").read_text())
    assert m3["electrodes"] == 2
    assert m3["wps"]["current2_a"] == pytest.approx([405.0, 495.0])
    arc3 = pd.read_parquet(out / "PP9_B03_SAW" / "arc.parquet")
    for c in ("current2_a", "voltage2_v", "wfs2_mpm"):
        assert c in arc3
    assert m3["measured"]["current2_a"] == pytest.approx(450, rel=0.05)      # RMS of the AC trail
    tr = pd.read_parquet(out / "PP9_B03_SAW" / "travel.parquet")
    assert list(tr.columns) == ["t_ns", "travel_speed_mmps", "position_mm", "heat_input_kj_mm"]
    # HI = (500*27 + 450*35) / 12 / 1000 = 2.4375 kJ/mm in steady state
    assert np.nanmedian(tr["heat_input_kj_mm"].to_numpy()[100:-100]) == pytest.approx(2.4375, rel=0.05)
    th = pd.read_parquet(out / "PP9_B03_SAW" / "thermal.parquet")
    assert np.allclose(th["interpass_c"], np.maximum(th["temp_left_c"], th["temp_right_c"]))


def test_clean_bead_and_pass_type(converted):
    out, _ = converted
    m = json.loads((out / "PP9_B24_SAW" / "meta.json").read_text())
    assert m["rt_result"] == "no_indications" and m["has_defect"] is False and m["defects"] == []
    assert m["defect_types"] == "none"
    assert m["pass_type"] == "cap"
    assert m["wps_id"] == "AIMEN-SAW-S11"        # from the design default mapping (no sheet row)
    assert [A.pass_type_for(b, 24) for b in (1, 2, 3, 21, 22, 24)] == ["root", "root", "fill", "fill", "cap", "cap"]


def test_wps_from_nominal_helper():
    w = C.wps_from_nominal("X", "SAW", {"current_a": 600, "voltage_v": 30, "wfs_mpm": None,
                                        "travel_speed_mmps": 10.0})
    assert w["current_a"] == [540.0, 660.0] and w["voltage_v"] == [27.0, 33.0]
    assert w["travel_speed_mmps"] == [9.0, 11.0] and "wfs_mpm" not in w
    g = C.wps_from_nominal("Y", "GMAW", {"current_a": 200}, gas_lpm=(16, 24))
    assert g["gas_lpm"] == [16.0, 24.0]


def test_grid_mean_10ms():
    t = np.arange(0, 1, 1 / 5000)
    df = C.grid_mean(t, {"v": np.where(t < 0.5, 1.0, 3.0)}, 1.0)
    assert len(df) == 100 and df["v"].iloc[0] == 1.0 and df["v"].iloc[-1] == 3.0
    t0 = np.arange(0, 1, 1 / 120)                                   # low-rate -> interpolated, no NaN
    assert C.grid_mean(t0, {"x": t0}, 1.0)["x"].notna().all()


def test_design_text_parse_and_setting_map():
    text = """1 single wire CC EP
(continuous current electrode to positive)
1,6 450 27 48
2 single wire CC EP 2,05 550 27 48
3-4 Tandem Forward: CC EP Back: AC (50/50%) F: 1,8 B: 1,95 F: 500 B: 450 F: 27 B: 35 72
22-24 single wire CC EP 2,05 550 30 50"""
    d = A.parse_design_text(text)
    assert d[2] == {"single": True, "wfs_lead": 2.05, "I_lead": 550.0, "V_lead": 27.0, "ts_cm_min": 48.0,
                    "bead_range": "2"}
    assert d[4]["I_trail"] == 450 and d[4]["ts_cm_min"] == 72 and not d[4]["single"]
    assert 1 not in d                                                  # wrapped line -> default mapping
    xls = {"settings": {2: {"Vdc": 27, "Idc": 550, "Vac": None, "Iac": None, "ts_cm_min": 48},
                        3: {"Vdc": 27, "Idc": 500, "Vac": 35, "Iac": 450, "ts_cm_min": 72}},
           "rows": [{"bead": 5, "setting": 4}]}
    smap, _ = A.bead_setting_map(24, xls, d)
    assert smap[1] == 1 and smap[2] == 2 and smap[3] == 3 and smap[4] == 3 and smap[5] == 4
    assert smap[12] == 8 and smap[24] == 11


def test_select_members_filters():
    root = "ZENODO_Penelope/"
    names = [root + n for n in (
        "PP5/README.txt", "PP5/Design_Materials_Settings.pdf", "PP5/2. ndt_data/defects_xlocation.xlsx",
        "PP5/2. ndt_data/RX.pdf", "PP4/2. ndt_data/RX_.pdf", "PP5/2. ndt_data/UT.pdf",
        "PP5/1. process_data/PP5_01/PP5_01.hdf5", "PP5/1. process_data/PP5_01/PP5_01.txt",
        "PP5/1. process_data/PP5_01/PP5_01_1.jpg", "PP5/1. process_data/PP5_01/PP5_01_2.jpg",
        "PP5/1. process_data/PP5_01/PP5_01_3.jpg", "PP5/1. process_data/PP5_01/pre-welding/PP5_01_pre1.jpg",
        "PP5/1. process_data/PP5_01/PP5_01_profilo.hdf5", "PP5/3. pics/PP5_15_faceA_1.jpg",
        "PP6/1. process_data/PP6_01/PP6_01.hdf5", "PP5/x.nde")]
    got = A.select_members(names, ["PP5", "PP4"])
    rel = sorted(g[len(root):] for g in got)
    assert rel == sorted(["PP5/README.txt", "PP5/Design_Materials_Settings.pdf",
                          "PP5/2. ndt_data/defects_xlocation.xlsx", "PP5/2. ndt_data/RX.pdf",
                          "PP4/2. ndt_data/RX_.pdf", "PP5/2. ndt_data/UT.pdf",
                          "PP5/1. process_data/PP5_01/PP5_01.hdf5", "PP5/1. process_data/PP5_01/PP5_01_1.jpg",
                          "PP5/1. process_data/PP5_01/PP5_01_2.jpg"])


def test_save_photo_strips_exif_and_resizes(tmp_path):
    from PIL import Image

    src = tmp_path / "in.jpg"
    im = Image.new("RGB", (3200, 1000), (200, 100, 50))
    exif = Image.Exif()
    exif[0x010F] = "SecretCamMaker"   # Make
    im.save(src, exif=exif.tobytes())
    dst = C.save_photo(src, tmp_path / "out" / "p.jpg")
    with Image.open(dst) as o:
        assert max(o.size) == 1600
        assert len(o.getexif()) == 0
    assert b"SecretCamMaker" not in dst.read_bytes()


def test_fetch_missing_adapter_is_skipped(tmp_path, capsys):
    from collect.datasets import fetch

    rc = fetch.main(["--sources", "no_such_dataset_xyz", "--data-dir", str(tmp_path), "--skip-download"])
    assert rc == 0
    assert "skipped" in capsys.readouterr().out
    assert (tmp_path / "generated" / "index.jsonl").exists()
