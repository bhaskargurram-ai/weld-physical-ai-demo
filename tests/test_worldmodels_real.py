"""worldmodels on real data: photo assessment (image_url), file-hash cache keys, honest heuristic fallback,
real-data prompts, warm-cache picking real video frames and BAM edge controls."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from worldmodels import cache, config, prompts, reason
from worldmodels.http import set_transport
from worldmodels.video import write_mp4

from test_worldmodels import GOOD_JSON, FakeCosmos  # noqa: F401  (shared fakes)

AIMEN = {"id": "aimen_saw", "license": "CC BY 4.0", "doi": "10.5281/zenodo.15083865", "synthetic": False,
         "attribution": "Tapia Suárez et al. (AIMEN) — SAW Open Repository. Zenodo, DOI 10.5281/zenodo.15083865."}
BAM = {"id": "bam_tgmaw", "license": "CC BY 4.0", "synthetic": False}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COSMOS_CACHE_DIR", str(tmp_path / "cache"))
    for k in ("COSMOS_BACKEND", "COSMOS_REASONER_BACKEND", "COSMOS_GENERATOR_BACKEND", "NVIDIA_API_KEY", "NGC_API_KEY",
              "COSMOS_CACHE", "COSMOS_REASONER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    config.clear_probe_cache()
    reason._MODEL_MEMO.clear()
    yield
    set_transport(None)
    config.clear_probe_cache()


@pytest.fixture
def saw_ctx() -> dict:
    return {
        "clip_id": "PP5_B02_SAW_t0146", "sequence_name": "PP5_B02_SAW", "t0_s": 14.6, "t1_s": 19.6,
        "media_type": "image",
        "meta": {"spool_id": "PP5", "joint_id": "BW", "pass_no": 2, "pass_type": "fill", "process": "SAW",
                 "material": "SA-516 Gr.70N", "thickness_mm": 52, "electrodes": 2, "rt_result": "indications",
                 "wps_basis": "published nominal setting ±10 % (demo assumption, not a qualified WPS)",
                 "wps": {"wps_id": "AIMEN-SAW-S1", "process": "SAW", "current_a": [481.5, 588.5],
                         "voltage_v": [27.9, 34.1], "current2_a": [540, 660]}},
        "signal_stats": {"current_a": {"mean": 535.0, "std": 6.0, "min": 516.0, "max": 551.0},
                         "current2_a": {"mean": 600.0, "std": 8.0, "min": 577.0, "max": 631.0}},
        "wps_violations": [],
        "defects_in_window": [{"defect": "porosity", "severity": None, "t_start_s": 17.0, "t_end_s": 17.1,
                               "x_start_mm": 185.9, "x_end_mm": 186.8, "source": "NDT"}],
        "ndt": {"methods": ["RT", "UT"], "inspected": True,
                "indications_in_window": [{"defect": "porosity", "t_start_s": 17.0, "t_end_s": 17.1,
                                           "x_start_mm": 185.9, "x_end_mm": 186.8, "source": "NDT"}]},
        "photo_note": "Post-weld photo(s) of the complete bead; not time-aligned to this window.",
        "rt_result": "indications", "source": AIMEN,
    }


def _photo(path: Path, seed: int = 0) -> Path:
    from PIL import Image
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)).save(path, quality=90)
    return path


def test_real_prompt_context(saw_ctx):
    text = prompts.build_assess_text(saw_ctx, think=False, media="image")
    assert text.startswith("Inspect this post-weld photo of the finished bead")
    assert "material SA-516 Gr.70N, 52 mm thick" in text and "process tandem SAW (2 electrodes" in text
    assert "trail-electrode current 540-660 A" in text and "no shielding gas (flux-shielded SAW)" in text
    assert "NDT record (RT/UT of the finished weld)" in text and "x=185.9-186.8 mm" in text
    assert "no electrical signature" in text and "published nominal setting" in text
    assert "SA-516 Gr.70" in prompts.CWI_SYSTEM_PROMPT and "S355" in prompts.CWI_SYSTEM_PROMPT
    assert "tandem" in prompts.CWI_SYSTEM_PROMPT and "SA-106" not in prompts.CWI_SYSTEM_PROMPT
    bam = {**saw_ctx, "media_type": "video", "source": BAM, "rt_result": "not_inspected", "defects_in_window": [],
           "ndt": {"inspected": False}, "video_kind": "real_process_camera",
           "meta": {**saw_ctx["meta"], "process": "GMAW", "material": "S355J2+N", "thickness_mm": 30}}
    t2 = prompts.build_assess_text(bam, think=False)
    assert "grayscale high-dynamic-range process camera" in t2 and "not NDT-inspected" in t2
    assert "torch-camera clip" not in t2


def test_assess_image_heuristic_is_honest(tmp_path, saw_ctx, monkeypatch):
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    p = _photo(tmp_path / "clips" / "PP5_B02_SAW_t0146.jpg")
    r = reason.assess_image(str(p), saw_ctx)
    assert r["source"] == "heuristic-fallback" and r["media_type"] == "image"
    assert r["verdict"] == "review"                       # porosity of unknown size: needs the RT/UT report
    assert r["defects"][0]["type"] == "porosity"
    assert "no electrical signature" in r["root_cause"] and "NDT reported porosity" in r["root_cause"]
    assert "flux" in r["root_cause"] and "gas flow at the nozzle" not in r["recommended_action"]
    clean = {**saw_ctx, "defects_in_window": [], "ndt": {"inspected": True, "indications_in_window": []},
             "rt_result": "no_indications"}
    r2 = reason.assess_clip(str(p), clean)
    assert r2["verdict"] == "accept" and "no electrical signature" in r2["root_cause"]
    lof = {**saw_ctx, "defects_in_window": [{"defect": "lack_of_fusion", "severity": None, "t_start_s": 15,
                                             "t_end_s": 16, "source": "NDT"}]}
    assert reason.assess_clip(str(p), lof)["verdict"] == "reject"   # LOF rejectable at any size
    ex = reason.explain_signals(saw_ctx)
    assert "no electrical signature" in ex


def test_assess_image_local_sends_image_url_and_caches_on_file_hash(tmp_path, saw_ctx, monkeypatch):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos()
    set_transport(httpx.MockTransport(fake))
    p = _photo(tmp_path / "a.jpg")
    r = reason.assess_clip(str(p), saw_ctx)
    assert r["source"] == "cosmos3-local" and r["media_type"] == "image"
    body = json.loads([q for q in fake.requests if q.url.path.endswith("/chat/completions")][0].content)
    parts = body["messages"][1]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith(("file://", "data:image/jpeg"))
    assert "media_io_kwargs" not in body                  # video sampling only for clips
    # same bytes under another name -> same cache key; different bytes -> different key
    q = tmp_path / "b.jpg"
    q.write_bytes(p.read_bytes())
    k1, _, _ = reason.assess_inputs(str(p), saw_ctx)
    k2, _, _ = reason.assess_inputs(str(q), saw_ctx)
    assert cache.make_key("assess_clip", k1) == cache.make_key("assess_clip", k2)
    _photo(q, seed=3)
    assert cache.make_key("assess_clip", reason.assess_inputs(str(q), saw_ctx)[0]) != cache.make_key("assess_clip", k1)
    assert "image" in k1 and "video" not in k1
    # replay: the cached Cosmos answer comes back
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    set_transport(None)
    again = reason.assess_clip(str(p), saw_ctx)
    assert again["source"] == "cache" and again["origin_source"] == "cosmos3-local"


def test_chat_with_image(tmp_path, monkeypatch):
    monkeypatch.setenv("COSMOS_BACKEND", "local")
    fake = FakeCosmos(chat_replies=["A wide bead with even ripples."])
    set_transport(httpx.MockTransport(fake))
    p = _photo(tmp_path / "bead.jpg")
    out = reason.chat_meta([{"role": "user", "content": "describe"}], image_path=str(p))
    assert out["text"].startswith("A wide bead")
    body = json.loads(fake.requests[-1].content)
    assert body["messages"][-1]["content"][0]["type"] == "image_url"


def test_real_scene_generator_prompt():
    p = prompts.weld_scene_prompt("travel speed drops 20 %", process="GMAW", source="bam_tgmaw")
    assert "pipe-spool" not in p["background_setting"] and "S355" in p["background_setting"]
    assert "grayscale" in p["artistic_style"].lower() and "tandem" in p["subjects"][0]["description"].lower()
    sim = prompts.weld_scene_prompt("x")
    assert "pipe-spool" in sim["background_setting"]


def _mk_video(path: Path, n: int = 24) -> Path:
    frames = [np.full((96, 128, 3), (i * 9) % 255, np.uint8) for i in range(n)]
    return write_mp4(frames, path, fps=18)


def test_warm_cache_real_rows(tmp_path, monkeypatch, saw_ctx):
    """photos + videos are assessed (separate limits); what-if uses a real BAM video frame; transfer uses the
    BAM edge control."""
    from worldmodels.showcase import pick_clip, pick_pass_dir, warm_cache
    monkeypatch.setenv("COSMOS_BACKEND", "replay")
    data = tmp_path / "data"
    clips = data / "curated" / "clips"
    _mk_video(clips / "SIM_c1.mp4")
    _mk_video(clips / "WID617_L3_TGMAW_t0025.mp4")
    _photo(clips / "PP5_B02_SAW_t0146.jpg")
    _photo(clips / "PP5_B03_SAW_t0100.jpg", seed=2)
    (data / "export").mkdir(parents=True)
    (data / "export" / "PP5_B02_SAW_t0146.json").write_text(json.dumps(saw_ctx))
    rows = [
        {"clip_id": "SIM_c1", "clip_uri": "clips/SIM_c1.mp4", "kept": True, "media_type": "video",
         "source": {"id": "weldcell_sim", "synthetic": True}, "defects": []},
        {"clip_id": "WID617_L3_TGMAW_t0025", "clip_uri": "clips/WID617_L3_TGMAW_t0025.mp4", "kept": True,
         "media_type": "video", "source": BAM, "defects": []},
        {"clip_id": "PP5_B02_SAW_t0146", "clip_uri": "clips/PP5_B02_SAW_t0146.jpg", "kept": True,
         "media_type": "image", "source": AIMEN, "defects": [{"defect": "porosity"}]},
        {"clip_id": "PP5_B03_SAW_t0100", "clip_uri": "clips/PP5_B03_SAW_t0100.jpg", "kept": True,
         "media_type": "image", "source": AIMEN, "defects": []},
    ]
    manifest = data / "curated" / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for seq, src in (("WID617_L3_TGMAW", BAM), ("SP-1_J1_P1_FCAW", None)):
        d = data / "generated" / seq
        _mk_video(d / "control" / "edge.mp4", n=12)
        (d / "meta.json").write_text(json.dumps({"sequence_name": seq, "defects": [],
                                                 **({"source": src} if src else {"simulator": {}})}))
    picked = pick_clip(rows, "auto", manifest)
    assert picked[0]["clip_id"] == "WID617_L3_TGMAW_t0025"            # real video before simulator, never a photo
    assert pick_pass_dir("auto", rows, ["edge"]).name == "WID617_L3_TGMAW"
    assert pick_pass_dir("auto", rows, ["depth"]) is None
    rep = warm_cache(manifest, preset="smoke", limit=1, photo_limit=1, do_explain=False, log=lambda m: None)
    assessed = {a["clip_id"]: a for a in rep["assess"]}
    assert set(assessed) == {"SIM_c1", "PP5_B02_SAW_t0146"}
    assert assessed["PP5_B02_SAW_t0146"]["media_type"] == "image"
    wi = rep["what_if"]["travel_speed_-20"]
    assert wi["image"].endswith("WID617_L3_TGMAW_t0025_middle.png") and Path(wi["image"]).exists()
    tr = rep["transfer"]["transfer-edge-bam"]
    assert tr["sequence_name"] == "WID617_L3_TGMAW" and tr["controls"]["edge"].endswith("control/edge.mp4")
    assert tr["data_source"] == "bam_tgmaw"
    rep2 = warm_cache(manifest, preset="smoke", do_generate=False, do_transfer=False, do_explain=False,
                      log=lambda m: None)
    assert len(rep2["assess"]) == 4
