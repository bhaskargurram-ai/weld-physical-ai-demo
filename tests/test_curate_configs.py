"""Curator Next config generation is validated against the real Curator pydantic models.

The recipe config modules are imported from a cosmos-curator checkout (``CURATOR_HOME`` or
/home/user/ref/cosmos-curator). video-transcode / video-caption / video-embed /
multimodal-caption import with only pydantic + PyYAML; video-cut, mp4-index and
multimodal-split also need ``attrs``, ``loguru`` and ``smart_open``. When those are
missing, the config is checked structurally against the field names copied from the source.
"""
from __future__ import annotations

import configparser
import importlib
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curate import curator_next as cn  # noqa: E402
from curate.curator_legacy import build_command  # noqa: E402
from curate.prompts import WELD_CAPTION_PROMPT  # noqa: E402

CURATOR_SRC = Path(os.environ.get("CURATOR_HOME") or "/home/user/ref/cosmos-curator")
KIND_MODULE = {"video-cut": "video_cut", "video-transcode": "video_transcode", "video-caption": "video_caption",
               "video-embed": "video_embed", "mp4-index": "mp4_index", "multimodal-split": "multimodal_split",
               "multimodal-caption": "multimodal_caption"}

# Field names per section, transcribed from cosmos_curator/next/recipes/*/config.py (extra="forbid").
STRUCTURE = {
    "video-cut": {"input": {"uris", "root_uri"},
                  "split": {"method", "duration_s", "stride_s", "min_duration_s"},
                  "output": {"media_root", "spans_lance_uri", "staging_root_uri"},
                  "execution": {"storage_profile", "max_items", "rows_per_fragment", "probe_timeout_s",
                                "rows_per_checkpoint", "commit_attempts", "progress", "fixed_stride",
                                "transnetv2"}},
    "mp4-index": {"input": {"uris", "path_prefix", "globs", "exclude_globs", "limit", "container_kind"},
                  "output": {"path"},
                  "execution": {"storage_profile", "session_id_path_depth", "min_actors", "max_actors",
                                "progress"}},
    "multimodal-split": {"input": {"input_path_prefix", "session_id_list_path", "mp4_index_uri", "limit"},
                         "clip": {"duration_s", "sampling_fps", "whole_session"},
                         "video": {"max_delta_s", "encoder"},
                         "output": {"path"},
                         "execution": {"storage_profile", "progress", "read_buffer_bytes", "span_planner",
                                       "split_session"}},
}


def _curator_module(kind: str):
    if not (CURATOR_SRC / "cosmos_curator").is_dir():
        return None
    if str(CURATOR_SRC) not in sys.path:
        sys.path.append(str(CURATOR_SRC))
    try:
        return importlib.import_module(f"cosmos_curator.next.recipes.{KIND_MODULE[kind]}.config")
    except ImportError:
        return None


def _resolved_model(mod):
    for name in dir(mod):
        obj = getattr(mod, name)
        if (name.startswith("Resolved") and name.endswith("Config") and hasattr(obj, "model_validate")
                and "kind" in getattr(obj, "model_fields", {}) and "Run" not in name):
            return obj
    raise AssertionError(f"no Resolved*Config in {mod}")


def _validate(cfg: dict):
    kind = cfg["kind"]
    mod = _curator_module(kind)
    if mod is not None:
        return _resolved_model(mod).model_validate(cfg)
    spec = STRUCTURE.get(kind)
    if spec is None:
        pytest.fail(f"{kind} config model should import with pydantic+yaml only")
    assert cfg["schema_version"] == 1
    for section, allowed in spec.items():
        if section in cfg:
            assert set(cfg[section]) <= allowed, (kind, section, set(cfg[section]) - allowed)
    return None


@pytest.fixture
def cfgs_qwen(tmp_path):
    return cn.build_configs(media_root=str(tmp_path / "curated"),
                            input_uris=["s3://raw/weldworld/b.mp4", "s3://raw/weldworld/a.mp4"])


@pytest.mark.parametrize("whole_session", [True, False])
def test_cosmos3_chain_validates(tmp_path, whole_session):
    cfgs = cn.build_configs(media_root=str(tmp_path / "curated"), input_uris=["s3://raw/weldworld/a.mp4"],
                            captioner="cosmos3_nano", whole_session=whole_session)
    assert list(cfgs) == ["01-video-cut", "02-video-transcode", "04-video-embed", "05-mp4-index",
                          "06-multimodal-split", "07-multimodal-caption"]
    paths = cn.write_configs(cfgs, tmp_path / "configs")
    for p in paths:
        cfg = yaml.safe_load(p.read_text())
        _validate(cfg)
    cap = yaml.safe_load((tmp_path / "configs" / "07-multimodal-caption.yaml").read_text())
    assert cap["model"]["variant"] == "cosmos3_nano"
    assert cap["inference"]["prompt"] == {"mode": "custom", "template": WELD_CAPTION_PROMPT}  # exact round-trip
    assert "{ego_trajectory_context}" not in WELD_CAPTION_PROMPT
    resolved = _validate(cap)
    if resolved is not None:
        assert resolved.effective_result_key == "weld_cwi"


def test_qwen_chain_validates_and_matches_contract(cfgs_qwen, tmp_path):
    assert list(cfgs_qwen) == ["01-video-cut", "02-video-transcode", "03-video-caption", "04-video-embed"]
    paths = cn.write_configs(cfgs_qwen, tmp_path / "configs")
    loaded = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    for cfg in loaded.values():
        _validate(cfg)
    cut = loaded["01-video-cut"]
    assert cut["split"] == {"method": "fixed-stride", "duration_s": 5.0, "stride_s": 5.0, "min_duration_s": 2.0}
    assert cut["input"]["uris"] == ["s3://raw/weldworld/a.mp4", "s3://raw/weldworld/b.mp4"]
    assert loaded["02-video-transcode"]["transcode"]["video_encoder"] == "libopenh264"
    assert loaded["03-video-caption"]["model"]["variant"] == "qwen3_8_27b_fp8"
    assert loaded["04-video-embed"]["model"]["variant"] == "336p"
    roots = {c["input"].get("media_root") or c["output"].get("media_root") for c in loaded.values()}
    assert len(roots) == 1  # one shared media_root for the whole chain
    assert (tmp_path / "configs" / "weld_caption_prompt.txt").read_text().strip() == WELD_CAPTION_PROMPT


def test_real_models_reject_bad_fields(cfgs_qwen):
    """Sanity: the curator models are strict, so the positive tests above mean something."""
    mod = _curator_module("video-transcode")
    if mod is None:
        pytest.skip("cosmos-curator source not available")
    bad = {**cfgs_qwen["02-video-transcode"], "transcode": {"video_encoder": "libx265"}}
    with pytest.raises(ValueError):  # pydantic ValidationError subclasses ValueError
        _resolved_model(mod).model_validate(bad)
    cap = _curator_module("video-caption")
    bad = {**cfgs_qwen["03-video-caption"], "model": {"variant": "cosmos3_nano"}}  # not a video-caption variant
    with pytest.raises(ValueError):  # pydantic ValidationError subclasses ValueError
        _resolved_model(cap).model_validate(bad)


def test_cosmos3_caption_column_matches_curator():
    try:
        sys.path.append(str(CURATOR_SRC))
        from cosmos_curator.next.recipes.multimodal_caption.contracts import result_field_names
    except Exception:  # noqa: BLE001
        pytest.skip("multimodal_caption.contracts not importable here")
    assert result_field_names("video_only", cn.RESULT_KEY, "cosmos3_nano")[0] == \
        cn.cosmos3_caption_column("cosmos3_nano")


def test_aws_profile_written_and_preserves_others(tmp_path, monkeypatch):
    creds = tmp_path / "credentials"
    creds.write_text("[other]\naws_access_key_id = X\naws_secret_access_key = Y\n")
    monkeypatch.setenv("S3_ENDPOINT", "http://minio:9000")
    s3 = cn.s3_settings()
    cn.write_aws_profile(s3, creds)
    cp = configparser.ConfigParser()
    cp.read(creds)
    assert cp["other"]["aws_access_key_id"] == "X"
    for section in ("weldworld", "default"):
        assert cp[section]["endpoint_url"] == "http://minio:9000"
        assert cp[section]["aws_access_key_id"] == "minioadmin"
        assert cp[section]["region"] == "us-east-1"


def test_configs_only_cli(tmp_path):
    from curate.run import main
    from curate.synth import make_clip, make_sidecar
    src = tmp_path / "export"
    src.mkdir()
    make_clip(src / "c1.mp4", seconds=2)
    make_sidecar(src / "c1.json", "c1")
    out = tmp_path / "curated"
    assert main(["--input", str(src), "--out", str(out), "--backend", "curator-next", "--configs-only",
                 "--captioner", "cosmos3_nano"]) == 0
    split = yaml.safe_load((out / "configs" / "06-multimodal-split.yaml").read_text())
    assert split["clip"]["whole_session"] is True  # 2 s input -> one episode per exported clip
    cut = yaml.safe_load((out / "configs" / "01-video-cut.yaml").read_text())
    assert cut["input"]["uris"] == ["s3://raw/weldworld/c1.mp4"]


def test_legacy_command_flags_exist_in_curator():
    cmd = build_command(Path("/data/export"), Path("/data/curated/legacy"), captioner="cosmos3_nano")
    assert cmd[:5] == ["pixi", "run", "--as-is", "video-pipeline", "split"]
    assert "--generate-cosmos-predict-dataset" in cmd
    assert cmd[cmd.index("--captioning-prompt-text") + 1] == WELD_CAPTION_PROMPT
    src = CURATOR_SRC / "cosmos_curator" / "pipelines" / "video" / "splitting_pipeline.py"
    if not src.exists():
        pytest.skip("cosmos-curator source not available")
    text = src.read_text()
    for flag in (c for c in cmd[5:] if c.startswith("--")):
        assert f'"{flag}"' in text, flag
    assert '"cosmos3_nano"' in text


def _fake_inputs(tmp_path):
    from curate.common import discover_inputs
    from curate.synth import make_clip, make_sidecar
    src = tmp_path / "export"
    src.mkdir()
    make_clip(src / "c1.mp4", seconds=2)
    make_sidecar(src / "c1.json", "c1", t0=20.0, t1=30.0, defect="porosity", gas_low=True)
    return discover_inputs(src)


def test_curator_next_read_back_joins_sidecars(tmp_path):
    """A Lance table shaped like video-cut/transcode/caption/embed output joins back to sidecars."""
    import lance
    import pyarrow as pa
    inputs = _fake_inputs(tmp_path)
    out = tmp_path / "curated"
    clip = out / "clips" / "abc.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"")
    table = pa.table({
        "source_uri": ["s3://raw/weldworld/c1.mp4", "s3://raw/weldworld/c1.mp4"],
        "span_id": ["s1", None], "start_ns": [5_000_000_000, None], "end_ns": [10_000_000_000, None],
        "clip_id": ["abc", None], "clip_uri": [str(clip), None],
        "caption_qwen3_8_27b_fp8_v1": ["A FCAW pass with porosity.", None],
        "embedding_cosmos_embed1_336p_v1": pa.array([[0.1] * 768, None], pa.list_(pa.float32(), 768)),
    })
    lance.write_dataset(table, str(out / "lance"))
    recs = cn.read_back(out, inputs, "qwen3_8_27b_fp8")
    assert len(recs) == 1  # error/empty-source placeholder rows (clip_id NULL) are skipped
    r = recs[0]
    assert (r.t0_s, r.t1_s) == (25.0, 30.0)
    assert r.caption_model == "qwen3_8_27b_fp8" and r.embedding.shape == (768,)
    assert r.embedding_name == "cosmos_embed1_336p"
    ctx = r.context()
    # sub-clip of the export (offset 5 s): porosity at 21-23.5 s and the gas deviation at 20.8-23.6 s
    # lie outside the 25-30 s window, so they must not be attached to this clip
    assert ctx["defects"] == [] and ctx["wps_violations"] == []
