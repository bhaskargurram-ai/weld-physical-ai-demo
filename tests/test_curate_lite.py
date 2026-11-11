"""End-to-end test of the CPU ``lite`` curation backend on tiny synthetic clips."""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curate.lite import LiteOptions, run_lite  # noqa: E402
from curate.manifest import MANIFEST_KEYS, read_manifest  # noqa: E402
from curate.synth import make_clip, make_sidecar  # noqa: E402

COSMOS3_PROMPT_KEYS = {"subjects", "background_setting", "lighting", "aesthetics", "cinematography",
                       "style_medium", "artistic_style", "context", "actions", "segments",
                       "temporal_caption", "audio_description", "resolution", "aspect_ratio", "duration", "fps"}


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("export")
    kinds = [("good", None, False), ("good", "porosity", True), ("static", None, False),
             ("blurry", None, False), ("glare", None, False), ("dark", None, False)]
    for i, (kind, defect, gas_low) in enumerate(kinds):
        cid = f"SP-1043_J07_P2_FCAW_{i:03d}"
        make_clip(d / f"{cid}.mp4", kind=kind, seed=i)
        make_sidecar(d / f"{cid}.json", cid, t0=5.0 * i, t1=5.0 * i + 5, defect=defect, gas_low=gas_low)
    make_clip(d / "LONG.mp4", seconds=12, seed=9)
    make_sidecar(d / "LONG.json", "LONG", t0=100.0, t1=112.0)
    shutil.copy(d / "SP-1043_J07_P2_FCAW_000.mp4", d / "DUP.mp4")
    make_sidecar(d / "DUP.json", "DUP")
    make_clip(d / "control_preview.mp4", seconds=1)  # no sidecar -> ignored
    return d


@pytest.fixture(scope="module")
def curated(export_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("curated")
    run_lite(export_dir, out, LiteOptions(captioner="template", workers=2))
    from curate.stats import write_card
    write_card(out, "lite")
    return out


def _rows(out):
    return {r["clip_id"]: r for r in read_manifest(out / "manifest.jsonl")}


def test_manifest_contract(curated):
    rows = _rows(curated)
    assert "control_preview" not in rows
    assert len(rows) == 7 + 3  # 7 single clips + LONG split into 3 (5 s, 5 s, 2 s)
    for r in rows.values():
        assert tuple(r) == MANIFEST_KEYS
        assert isinstance(r["kept"], bool)
        assert set(r["quality"]) >= {"blur", "motion"}
        assert (r["reject_reason"] is None) == r["kept"]
        assert r["caption_model"] == "weld_template" and r["caption"]
        prompt = json.loads(r["generator_prompt"])
        assert COSMOS3_PROMPT_KEYS <= set(prompt)
        assert prompt["resolution"] == {"W": 1280, "H": 720} and prompt["aspect_ratio"] == "16,9"


def test_filters_and_dedup(curated):
    rows = _rows(curated)
    assert rows["SP-1043_J07_P2_FCAW_002"]["reject_reason"] == "static"
    assert rows["SP-1043_J07_P2_FCAW_003"]["reject_reason"] == "blurry"
    assert rows["SP-1043_J07_P2_FCAW_004"]["reject_reason"] == "overexposed_glare"
    assert rows["SP-1043_J07_P2_FCAW_005"]["reject_reason"] == "underexposed"
    # byte-identical copy: exactly one of the pair survives
    pair = [rows["DUP"], rows["SP-1043_J07_P2_FCAW_000"]]
    assert sorted(r["kept"] for r in pair) == [False, True]
    assert {r["reject_reason"] for r in pair} == {None, "near_duplicate"}
    assert rows["SP-1043_J07_P2_FCAW_001"]["kept"]
    assert all(rows[f"LONG_{k:02d}"]["kept"] for k in range(3))


def test_split_times_and_labels(curated):
    rows = _rows(curated)
    assert (rows["LONG_00"]["t0_s"], rows["LONG_00"]["t1_s"]) == (100.0, 105.0)
    assert (rows["LONG_02"]["t0_s"], rows["LONG_02"]["t1_s"]) == (110.0, 112.0)
    por = rows["SP-1043_J07_P2_FCAW_001"]
    assert [d["defect"] for d in por["defects"]] == ["porosity"]
    assert por["wps_violations"][0]["param"] == "gas_lpm"
    assert "porosity" in por["caption"] and "shielding gas" in por["caption"]
    assert por["sensor_summary"]["gas_lpm"]["mean"] == 6.5
    assert por["rt_result"] == "accept"


def test_clips_are_h264(curated):
    import av
    rows = _rows(curated)
    for r in rows.values():
        p = Path(r["clip_uri"])
        p = p if p.is_absolute() else ROOT / p
        assert p.exists(), p
        with av.open(str(p)) as c:
            assert c.streams.video[0].codec_context.name == "h264"
    assert len(list((curated / "clips").glob("*.mp4"))) == len(rows)


def test_lance_curator_next_columns(curated):
    import lance
    t = lance.dataset(str(curated / "lance")).to_table()
    for col in ("clip_id", "clip_uri", "start_ns", "end_ns", "source_uri", "span_id",
                "caption_weld_template_v1", "embedding_weldlite_v1"):
        assert col in t.column_names
    assert t.num_rows == 10
    row = {r["clip_id"]: r for r in t.to_pylist()}["LONG_01"]
    assert (row["start_ns"], row["end_ns"]) == (5_000_000_000, 10_000_000_000)
    from curate.video import EMBED_DIM
    assert len(row["embedding_weldlite_v1"]) == EMBED_DIM


def test_prompt_variants_and_card(curated):
    v = json.loads((curated / "prompts" / "SP-1043_J07_P2_FCAW_001.json").read_text())
    assert set(v) == {"photoreal_transfer", "what_if_travel_speed", "what_if_gas_loss", "defect_synthesis"}
    assert "20%" in v["what_if_travel_speed"]["subjects"][0]["state_changes"]
    assert "porosity" in v["defect_synthesis"]["context"]
    card = (curated / "dataset_card.md").read_text()
    assert "near_duplicate" in card and "porosity" in card and "hours kept" in card
    s = json.loads((curated / "dataset_card.json").read_text())
    assert s["clips_total"] == 10 and s["clips_kept"] == 5
    assert s["rejected_by_reason"]["near_duplicate"] == 1


def test_reasoner_captioner_with_fallback(export_dir, tmp_path, monkeypatch):
    """worldmodels.reason.chat is used when importable; failures fall back to the template."""
    calls = []

    def chat(messages, video_path=None):
        calls.append(video_path)
        if "002" in str(video_path):
            raise RuntimeError("cache miss")
        assert messages[0]["role"] == "system" and "Certified Welding Inspector" in messages[0]["content"]
        return "The torch tracks the joint with a stable FCAW arc and moderate spatter."

    pkg = types.ModuleType("worldmodels")
    reason = types.ModuleType("worldmodels.reason")
    reason.chat = chat
    pkg.reason = reason
    monkeypatch.setitem(sys.modules, "worldmodels", pkg)
    monkeypatch.setitem(sys.modules, "worldmodels.reason", reason)
    monkeypatch.setenv("COSMOS_BACKEND", "replay")

    src = tmp_path / "in"
    src.mkdir()
    for cid in ("SP-1043_J07_P2_FCAW_000", "SP-1043_J07_P2_FCAW_002"):
        for ext in (".mp4", ".json"):
            shutil.copy(export_dir / f"{cid}{ext}", src / f"{cid}{ext}")
    out = tmp_path / "out"
    run_lite(src, out, LiteOptions(captioner="auto", embeddings=False))
    rows = _rows(out)
    assert rows["SP-1043_J07_P2_FCAW_000"]["caption_model"] == "cosmos3_reasoner"
    assert rows["SP-1043_J07_P2_FCAW_002"]["caption_model"] == "weld_template"
    assert len(calls) == 2


def test_cli_lite(export_dir, tmp_path):
    from curate.run import main
    out = tmp_path / "cli"
    assert main(["--input", str(export_dir), "--out", str(out), "--backend", "lite",
                 "--captioner", "template", "--no-dedup"]) == 0
    rows = read_manifest(out / "manifest.jsonl")
    assert not any(r["reject_reason"] == "near_duplicate" for r in rows)
    assert (out / "dataset_card.md").exists()
