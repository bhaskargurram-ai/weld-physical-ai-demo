"""``curator-next`` backend: NVIDIA Cosmos Curator Next recipe chain.

Chain (one shared ``media_root`` = the curate output dir, so Curator itself writes
``<out>/lance`` and ``<out>/clips/<clip_id>.mp4``):

  01-video-cut        fixed-stride 5 s spans; input MUST be S3 -> MinIO bucket ``raw``
  02-video-transcode  libopenh264 (Curator's LGPL ffmpeg has no libx264)
  03-video-caption    Qwen3.8-27B (FP8) with Curator's code-owned default prompt
  04-video-embed      Cosmos-Embed1 336p (768-d)
  optional Cosmos 3 captioning with the weld CWI prompt (``--captioner cosmos3_nano``):
  05-mp4-index -> 06-multimodal-split -> 07-multimodal-caption
     (video-caption has no prompt setting; multimodal-caption supports
      ``model.variant: cosmos3_nano|cosmos3_super`` and ``inference.prompt.mode: custom``)

Field names follow the pydantic models in
cosmos_curator/next/recipes/{video_cut,video_transcode,video_caption,video_embed,
mp4_index,multimodal_split,multimodal_caption}/config.py (all ``extra="forbid"``).
"""
from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from curate.common import CLIP_DEFAULTS, ClipInput, discover_inputs, display_path, env, log

CAPTION_VARIANTS_VIDEO = ("qwen3_8_27b_fp8", "qwen3_8_27b")          # video-caption ModelVariant
CAPTION_VARIANTS_COSMOS3 = ("cosmos3_nano", "cosmos3_super")          # multimodal-caption only
EMBED_VARIANTS = ("224p", "336p", "448p")
CAMERA_KEY = "torch_cam"
RESULT_KEY = "weld_cwi"


# ---------------------------------------------------------------- config builders
def build_configs(*, media_root: str, input_uris: list[str], clip_s: float = CLIP_DEFAULTS["clip_s"],
                  stride_s: float = CLIP_DEFAULTS["stride_s"], min_s: float = CLIP_DEFAULTS["min_s"],
                  captioner: str = "qwen3_8_27b_fp8", embed_variant: str = "336p",
                  storage_profile: str = "weldworld", max_model_len: int = 32768,
                  transcode_cpus: float = 2.0, caption_prompt: str | None = None,
                  whole_session: bool = True, fps: int = 24,
                  skip_caption: bool = False, skip_embed: bool = False) -> dict[str, dict[str, Any]]:
    """Return ``{"01-video-cut": {...}, ...}`` in run order."""
    root = media_root.rstrip("/")
    cfgs: dict[str, dict[str, Any]] = {}
    cfgs["01-video-cut"] = {
        "schema_version": 1,
        "kind": "video-cut",
        "input": {"uris": sorted(input_uris)},
        "split": {"method": "fixed-stride", "duration_s": float(clip_s), "stride_s": float(stride_s),
                  "min_duration_s": float(min(min_s, clip_s))},
        "output": {"media_root": root},
        "execution": {"storage_profile": storage_profile},
    }
    cfgs["02-video-transcode"] = {
        "schema_version": 1,
        "kind": "video-transcode",
        "input": {"media_root": root},
        "transcode": {"video_encoder": "libopenh264", "video_bitrate": "4M"},
        "execution": {"storage_profile": storage_profile, "transcode_cpus": float(transcode_cpus)},
    }
    if not skip_caption and captioner in CAPTION_VARIANTS_VIDEO:
        cfgs["03-video-caption"] = {
            "schema_version": 1,
            "kind": "video-caption",
            "input": {"media_root": root},
            "model": {"variant": captioner},
            "execution": {"storage_profile": storage_profile, "inference_concurrency": 1,
                          "max_model_len": int(max_model_len)},
        }
    if not skip_embed:
        cfgs["04-video-embed"] = {
            "schema_version": 1,
            "kind": "video-embed",
            "input": {"media_root": root},
            "model": {"variant": embed_variant},
            "execution": {"storage_profile": storage_profile, "inference_concurrency": 1},
        }
    if not skip_caption and captioner in CAPTION_VARIANTS_COSMOS3:
        from curate.prompts import WELD_CAPTION_PROMPT

        sessions = f"{root}/sessions"
        cfgs["05-mp4-index"] = {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {"path_prefix": sessions, "globs": [f"*/*/{CAMERA_KEY}.mp4"],
                      "container_kind": "nonfragmented"},
            "output": {"path": f"{root}/mp4-index"},
            "execution": {"storage_profile": storage_profile},
        }
        clip_cfg: dict[str, Any] = {"whole_session": True} if whole_session else {
            "whole_session": False, "duration_s": float(clip_s), "sampling_fps": int(fps)}
        if whole_session:
            clip_cfg["sampling_fps"] = int(fps)
        cfgs["06-multimodal-split"] = {
            "schema_version": 1,
            "kind": "multimodal-split",
            "input": {"input_path_prefix": sessions, "mp4_index_uri": f"{root}/mp4-index/manifest.lance"},
            "sensors": {CAMERA_KEY: {"sensor_type": "camera", "uri_pattern": f"*/{CAMERA_KEY}.mp4",
                                     "required": True, "camera_alignment_fields": ["pts_ns"]}},
            "output": {"path": f"{root}/episodes"},
            "clip": clip_cfg,
            "video": {"max_delta_s": 0.05,
                      "encoder": {"video_encoder": "libopenh264", "video_bitrate": "4M",
                                  "gop_size": int(fps), "faststart": True}},
            "execution": {"storage_profile": storage_profile},
        }
        cfgs["07-multimodal-caption"] = {
            "schema_version": 1,
            "kind": "multimodal-caption",
            "input": {"episodes_lance_uri": f"{root}/episodes/episodes.lance"},
            "context": {"variant": "video_only", "camera_key": CAMERA_KEY},
            "model": {"variant": captioner},
            "inference": {"prompt": {"mode": "custom", "template": caption_prompt or WELD_CAPTION_PROMPT}},
            "output": {"result_key": RESULT_KEY, "tag_camera_key": False},
            "execution": {"storage_profile": storage_profile, "inference_concurrency": 1,
                          "inference_batch_size": 4},
        }
    return cfgs


def cosmos3_caption_column(captioner: str) -> str:
    """multimodal_caption.contracts.result_field_names(video_only, weld_cwi, model)."""
    return f"caption_video_only_{RESULT_KEY}_{captioner}_v1"


def dump_yaml(cfg: dict[str, Any]) -> str:
    import yaml
    head = ("# Generated by curate.curator_next - WeldWorld Curator Next recipe config.\n"
            "# Validate (in $CURATOR_HOME): pixi run -e dev cosmos-curator pipeline validate <this file>\n")
    return head + yaml.safe_dump(cfg, sort_keys=False, width=1_000_000, allow_unicode=True)


def write_configs(cfgs: dict[str, dict[str, Any]], config_dir: Path) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    for old in config_dir.glob("*.yaml"):
        old.unlink()
    paths = []
    for name, cfg in cfgs.items():
        p = config_dir / f"{name}.yaml"
        p.write_text(dump_yaml(cfg))
        paths.append(p)
    from curate.prompts import WELD_CAPTION_PROMPT
    (config_dir / "weld_caption_prompt.txt").write_text(WELD_CAPTION_PROMPT + "\n")
    return paths


# ---------------------------------------------------------------- S3 / MinIO
def s3_settings() -> dict[str, str]:
    return {
        "endpoint_url": env("S3_ENDPOINT", "http://localhost:9000"),
        "aws_access_key_id": env("MINIO_ROOT_USER", "minioadmin"),
        "aws_secret_access_key": env("MINIO_ROOT_PASSWORD", "minioadmin"),
        "region": env("S3_REGION", "us-east-1"),
        "bucket": env("S3_BUCKET", "raw"),
        "prefix": env("S3_PREFIX", "weldworld").strip("/"),
        "profile": env("CURATOR_S3_PROFILE", "weldworld"),
    }


def write_aws_profile(s3: dict[str, str], path: Path | None = None) -> Path:
    """Add/replace the profile in ~/.aws/credentials (Curator reads endpoint_url/region there too).

    Other sections are preserved; ``[default]`` is only written when absent.
    """
    path = Path(path or os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or Path.home() / ".aws" / "credentials")
    cp = configparser.ConfigParser()
    if path.exists():
        cp.read(path)
    body = {k: s3[k] for k in ("aws_access_key_id", "aws_secret_access_key", "region", "endpoint_url")}
    cp[s3["profile"]] = body
    if "default" not in cp:
        cp["default"] = body
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        cp.write(f)
    path.chmod(0o600)
    return path


_UPLOAD_SCRIPT = r"""
import json, os, sys
import boto3
from botocore.exceptions import ClientError
cfg = json.loads(os.environ["WELD_UPLOAD_SPEC"])
s3 = boto3.client("s3", endpoint_url=cfg["endpoint_url"], aws_access_key_id=cfg["aws_access_key_id"],
                  aws_secret_access_key=cfg["aws_secret_access_key"], region_name=cfg["region"])
try:
    s3.head_bucket(Bucket=cfg["bucket"])
except ClientError:
    s3.create_bucket(Bucket=cfg["bucket"])
for src, key in cfg["files"]:
    s3.upload_file(src, cfg["bucket"], key)
    print("uploaded", src, "->", f"s3://{cfg['bucket']}/{key}")
"""


def s3_uri_for(inp: ClipInput, s3: dict[str, str]) -> str:
    return f"s3://{s3['bucket']}/{s3['prefix']}/{inp.video.name}"


def upload_inputs(inputs: list[ClipInput], s3: dict[str, str], curator_home: Path | None) -> list[str]:
    files = [[str(i.video.resolve()), f"{s3['prefix']}/{i.video.name}"] for i in inputs]
    spec = {**s3, "files": files}
    e = {**os.environ, "WELD_UPLOAD_SPEC": json.dumps(spec)}
    try:
        import boto3  # noqa: F401
        cmd, cwd = [sys.executable, "-c", _UPLOAD_SCRIPT], None
    except ImportError:
        if not curator_home:
            raise RuntimeError("boto3 not installed here and CURATOR_HOME not set (Curator's pixi env has boto3)")
        cmd, cwd = ["pixi", "run", "--as-is", "python", "-c", _UPLOAD_SCRIPT], str(curator_home)
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"S3 upload failed: {r.stderr[-2000:]}")
    log.info("uploaded %d clips to s3://%s/%s/", len(files), s3["bucket"], s3["prefix"])
    return [f"s3://{s3['bucket']}/{k}" for _, k in files]


def stage_sessions(inputs: list[ClipInput], root: Path) -> Path:
    """Local session layout for mp4-index/multimodal-split: sessions/<clip_id>/cam0/torch_cam.mp4."""
    sess = root / "sessions"
    for i in inputs:
        d = sess / i.clip_id / "cam0"
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{CAMERA_KEY}.mp4"
        if not dst.exists():
            shutil.copy2(i.video, dst)
    return sess


# ---------------------------------------------------------------- run
def run_stage(yaml_path: Path, curator_home: Path, s3_profile_file: Path, attempts: int = 5) -> dict:
    """``pixi run --as-is run-pipeline <yaml> --json`` inside CURATOR_HOME.

    A run that published work but still owes more exits non-zero; rerun it (Curator resumes).
    """
    e = {**os.environ, "COSMOS_S3_PROFILE_PATH": str(s3_profile_file)}
    cmd = ["pixi", "run", "--as-is", "run-pipeline", str(yaml_path.resolve()), "--json"]
    last = ""
    for k in range(attempts):
        log.info("[%s] %s (attempt %d)", yaml_path.stem, " ".join(cmd), k + 1)
        r = subprocess.run(cmd, cwd=curator_home, env=e, capture_output=True, text=True)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                return {"stdout": r.stdout[-4000:]}
        last = (r.stderr or r.stdout)[-4000:]
        if r.returncode != 2 or "remain" not in last.lower():
            break
    raise RuntimeError(f"Curator stage {yaml_path.name} failed:\n{last}")


# ---------------------------------------------------------------- read back
def _source_index(inputs: list[ClipInput]) -> dict[str, ClipInput]:
    return {i.video.name: i for i in inputs}


def read_back(out: Path, inputs: list[ClipInput], captioner: str):
    """Rows of ``<out>/lance`` joined to sidecars -> ClipRecords (+ Cosmos 3 episode captions)."""
    import lance
    import numpy as np

    from curate.manifest import ClipRecord

    ds = lance.dataset(str(out / "lance"))
    table = ds.to_table(filter="clip_id IS NOT NULL")
    rows = table.to_pylist()
    by_name = _source_index(inputs)
    cap_cols = [c for c in table.column_names if c.startswith("caption_") and not c.endswith("_metadata")]
    emb_cols = [c for c in table.column_names if c.startswith("embedding_") and not c.endswith("_metadata")]
    records = []
    for row in rows:
        src = by_name.get(Path(str(row.get("source_uri", ""))).name)
        if src is None:
            log.warning("no sidecar for source %s", row.get("source_uri"))
            src = ClipInput(clip_id=Path(str(row.get("source_uri"))).stem, video=Path(str(row.get("source_uri"))))
        clip_uri = str(row.get("clip_uri") or "")
        rec = ClipRecord(clip_id=str(row["clip_id"]),
                         clip_uri=display_path(clip_uri) if clip_uri and "://" not in clip_uri else clip_uri,
                         source=src, start_s=row["start_ns"] / 1e9, end_s=row["end_ns"] / 1e9)
        for c in cap_cols:
            if row.get(c):
                rec.caption, rec.caption_model = row[c], c[len("caption_"):-len("_v1")]
                break
        for c in emb_cols:
            if row.get(c) is not None:
                rec.embedding = np.asarray(row[c], dtype=np.float32)
                rec.embedding_name = c[len("embedding_"):-len("_v1")]
                break
        records.append(rec)

    if captioner in CAPTION_VARIANTS_COSMOS3:
        _attach_episode_captions(out, records, cosmos3_caption_column(captioner), captioner)
    return records


def _attach_episode_captions(out: Path, records, column: str, model: str) -> None:
    import lance
    p = out / "episodes" / "episodes.lance"
    if not p.exists():
        log.warning("no episodes table at %s", p)
        return
    t = lance.dataset(str(p)).to_table()
    if column not in t.column_names:
        log.warning("episodes table has no %s", column)
        return
    eps: dict[str, list[tuple[int, str]]] = {}
    for r in t.select(["source_session_id", "start_ns", column]).to_pylist():
        if r[column]:
            eps.setdefault(str(r["source_session_id"]), []).append((int(r["start_ns"]), r[column]))
    for sid, lst in eps.items():
        lst.sort()
        base = lst[0][0]
        clips = [rec for rec in records if rec.source.clip_id == sid or sid.endswith(rec.source.clip_id)]
        for rec in clips:
            off = int(rec.start_s * 1e9)
            _, cap = min(lst, key=lambda e: abs((e[0] - base) - off))
            rec.caption, rec.caption_model = cap, model


# ---------------------------------------------------------------- entry
def run_curator_next(input_dir: str | Path, out_dir: str | Path, *, captioner: str = "qwen3_8_27b_fp8",
                     embed_variant: str = "336p", clip_s: float = CLIP_DEFAULTS["clip_s"],
                     stride_s: float = CLIP_DEFAULTS["stride_s"], min_s: float = CLIP_DEFAULTS["min_s"],
                     configs_only: bool = False, skip_upload: bool = False, thresholds=None,
                     dedup_enabled: bool = True):
    from curate.captions import TEMPLATE_MODEL, template_caption
    from curate.manifest import Thresholds, apply_filters, dedup, write_manifest
    from curate.video import analyze, probe

    out = Path(out_dir).resolve()
    inputs = discover_inputs(input_dir)
    if not inputs:
        raise SystemExit(f"no *.mp4 clips found in {input_dir}")
    s3 = s3_settings()
    uris = [s3_uri_for(i, s3) for i in inputs]
    durations = []
    for i in inputs:
        try:
            durations.append(probe(i.video).duration_s)
        except Exception:  # noqa: BLE001
            pass
    whole = bool(durations) and max(durations) <= clip_s + 0.6
    cfgs = build_configs(media_root=str(out), input_uris=uris, clip_s=clip_s, stride_s=stride_s, min_s=min_s,
                         captioner=captioner, embed_variant=embed_variant, storage_profile=s3["profile"],
                         whole_session=whole)
    paths = write_configs(cfgs, out / "configs")
    log.info("wrote %d Curator Next configs to %s", len(paths), out / "configs")
    if configs_only:
        return None

    home = os.environ.get("CURATOR_HOME")
    if not home or not Path(home).is_dir():
        raise SystemExit("CURATOR_HOME must point at a cosmos-curator checkout with `pixi install` done "
                         "(configs were written; run them manually or use --backend lite)")
    curator_home = Path(home)
    if (out / "lance").exists() and not (out / "staging").exists():
        raise SystemExit(f"{out}/lance exists but was not written by Curator (lite run?). Curator Next appends "
                         "to the table at <media_root>/lance - use a fresh --out directory.")
    profile_file = write_aws_profile(s3)
    if not skip_upload:
        upload_inputs(inputs, s3, curator_home)
    if captioner in CAPTION_VARIANTS_COSMOS3:
        stage_sessions(inputs, out)
    for p in paths:
        res = run_stage(p, curator_home, profile_file)
        (out / "configs" / f"{p.stem}.result.json").write_text(json.dumps(res, indent=2, default=str))

    records = read_back(out, inputs, captioner)
    for r in records:
        local = Path(r.clip_uri)
        if local.exists():
            try:
                r.quality, _ = analyze(local)
            except Exception as exc:  # noqa: BLE001
                r.quality = {"error": str(exc)}
        if not r.caption:
            r.caption, r.caption_model = template_caption(r.context(), r.quality), TEMPLATE_MODEL
    th = thresholds or Thresholds()
    apply_filters(records, th)
    if dedup_enabled:
        dedup(records, th.dedup_cosine)
    write_manifest(records, out)
    return records
