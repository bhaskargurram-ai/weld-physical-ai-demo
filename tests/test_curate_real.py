"""curate on real-data exports (docs/REAL_DATA.md): photo-only items, real-video clips, honest captions,
media_type / source in the manifest, sources + licenses on the dataset card."""
from __future__ import annotations

import json

import pytest

from curate.manifest import MANIFEST_KEYS, read_manifest
from mosaico_io.testdata import make_real_fixture


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("real")
    make_real_fixture(root)
    mp = pytest.MonkeyPatch()
    mp.setenv("MOSAICO_OFFLINE", "1")
    mp.setenv("DATA_DIR", str(root))
    mp.delenv("WELD_GENERATED_DIR", raising=False)
    try:
        from mosaico_io import export_clips
        export_clips(root / "export")
    finally:
        mp.undo()
    return root / "export"


@pytest.fixture(scope="module")
def curated(export_dir, tmp_path_factory):
    from curate.lite import LiteOptions, run_lite
    from curate.stats import write_card
    out = tmp_path_factory.mktemp("curated")
    run_lite(export_dir, out, LiteOptions(captioner="template", workers=2))
    write_card(out, "lite")
    return out


def test_discover_photo_items(export_dir):
    from curate.common import discover_inputs
    items = discover_inputs(export_dir)
    img = [i for i in items if i.is_image]
    vid = [i for i in items if not i.is_image]
    assert len(img) == 1 and img[0].video.suffix == ".jpg" and img[0].clip_id.startswith("PP5_B02_SAW_t")
    assert vid and all(i.video.suffix == ".mp4" and i.sequence_name == "WID617_L3_TGMAW" for i in vid)
    assert img[0].source_info["license"] == "CC BY 4.0" and "Zenodo" in img[0].source_info["attribution"]


def test_manifest_rows_media_and_source(curated):
    rows = {r["clip_id"]: r for r in read_manifest(curated)}
    for r in rows.values():
        assert tuple(r) == MANIFEST_KEYS and r["media_type"] in ("video", "image")
        assert set(r["source"]) >= {"id", "license", "attribution"}
    img = next(r for r in rows.values() if r["media_type"] == "image")
    assert img["clip_uri"].endswith(".jpg") and (curated / "clips" / f"{img['clip_id']}.jpg").exists()
    assert img["kept"] and img["quality"]["motion"] is None and img["quality"]["blur"] > 5
    assert [d["defect"] for d in img["defects"]] == ["porosity"] and img["rt_result"] == "indications"
    assert img["sensor_summary"]["electrodes"] == 2 and "current2_a" in img["sensor_summary"]
    assert img["source"]["id"] == "aimen_saw"
    vid = [r for r in rows.values() if r["media_type"] == "video"]
    assert vid and all(r["source"]["id"] == "bam_tgmaw" and r["quality"]["fps"] == pytest.approx(17.96, abs=0.05)
                       for r in vid)
    gp = json.loads(vid[0]["generator_prompt"])
    assert "pipe-spool" not in gp["background_setting"] and "plate" in gp["background_setting"]
    assert "simulated weld" not in gp["context"]


def test_captions_describe_the_data_not_invented_visuals(curated):
    rows = read_manifest(curated)
    img = next(r for r in rows if r["media_type"] == "image")
    vid = next(r for r in rows if r["media_type"] == "video")
    for r in (img, vid):
        c = r["caption"]
        assert "Torch-camera view" not in c and "Process cues" not in c and "spatter" not in c.lower()
        assert "CC BY 4.0" in c
    assert vid["caption"].startswith("HDR process camera footage, grayscale")
    assert "not NDT-inspected" in vid["caption"] and "trail current" in vid["caption"]
    assert img["caption"].startswith("Post-weld photo of the complete SAW bead")
    assert "no signature" in img["caption"] and "x=185.9-186.8 mm" in img["caption"]
    assert "no shielding gas" in img["caption"]


def test_dataset_card_sources_and_licenses(curated):
    card = (curated / "dataset_card.md").read_text()
    s = json.loads((curated / "dataset_card.json").read_text())
    assert "## Sources, licenses and attribution" in card and "CC BY 4.0" in card
    assert "DOI 10.5281/zenodo.15083865" in card and "DOI 10.5281/zenodo.21824586" in card
    assert set(s["sources"]) == {"aimen_saw", "bam_tgmaw"} and s["sources"]["aimen_saw"]["image"] == 1
    assert s["media_type_all"]["image"] == 1


def test_curator_next_sends_only_video(export_dir, tmp_path, monkeypatch):
    from curate.curator_next import run_curator_next
    monkeypatch.setenv("S3_ENDPOINT", "http://localhost:9000")
    assert run_curator_next(export_dir, tmp_path / "cn", configs_only=True) is None
    cut = (tmp_path / "cn" / "configs" / "01-video-cut.yaml").read_text()
    assert "WID617_L3_TGMAW" in cut and "PP5_B02_SAW" not in cut and ".jpg" not in cut


def test_reasoner_caption_for_photo_uses_image(export_dir, tmp_path, monkeypatch):
    """captioner -> worldmodels.reason.chat_meta(image_path=...) for photo items."""
    from curate import captions
    from curate.lite import LiteOptions, run_lite
    seen = []

    class FakeReason:
        @staticmethod
        def chat_meta(messages, video_path=None, image_path=None):
            seen.append((video_path, image_path, messages[0]["content"][:160]))
            return {"text": "Visual: a wide SAW bead with regular ripples.", "source": "cosmos3-local"}

    monkeypatch.setattr(captions, "_reasoner_reachable", lambda timeout=2.0: True)
    import worldmodels
    monkeypatch.setattr(worldmodels, "reason", FakeReason, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "worldmodels.reason", FakeReason)
    rows = run_lite(export_dir, tmp_path / "c", LiteOptions(captioner="reason", embeddings=False, workers=1))
    img = next(r for r in rows if r.source.is_image)
    assert img.caption_model == "cosmos3_reasoner" and "SAW bead" in img.caption
    photo_calls = [s for s in seen if s[1]]
    assert len(photo_calls) == 1 and photo_calls[0][1].endswith(".jpg") and "post-weld photo" in photo_calls[0][2]
    assert any(s[0] and s[0].endswith(".mp4") for s in seen)
