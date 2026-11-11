# curate — stage 3: weld clips → curated, captioned dataset

Input is agent A's export: `data/export/<clip_id>.mp4` + `<clip_id>.json` sidecar
(`{clip_id, sequence_name, t0_s, t1_s, meta, signal_stats, wps_violations, defects_in_window, rt_result}`).
If a directory has any sidecars, mp4s without one (previews, control videos) are skipped.

```bash
. .venv/bin/activate
python -m curate.run --input data/export --out data/curated --backend lite             # CPU, works anywhere
python -m curate.run --input data/export --out data/curated_next --backend curator-next # GPU + Cosmos Curator
python -m curate.run --input data/export --out data/curated_legacy --backend curator-legacy
python -m curate.stats data/curated                                                    # rebuild dataset card
pytest -q tests/test_curate_lite.py tests/test_curate_configs.py
```

## Output (`data/curated/`, all backends)

| path | content |
|---|---|
| `manifest.jsonl` | one row per clip, exactly the PLAN.md contract: `clip_id, clip_uri, sequence_name, t0_s, t1_s, caption, caption_model, sensor_summary, wps_violations, defects, rt_result, quality{blur, motion, ...}, kept, reject_reason, generator_prompt` |
| `clips/<clip_id>.mp4` | H.264 (libx264, yuv420p) clips |
| `lance/` | Curator Next schema: `clip_id, clip_uri, source_uri, span_id, start_ns, end_ns, clip_duration_ns, clip_width/height/frame_rate, caption_<model>_v1, embedding_<name>_v1` (+ lite extras `sequence_name, t0_s, t1_s, kept, reject_reason, quality_*`) |
| `prompts/<clip_id>.json` | Cosmos 3 JSON prompts: `photoreal_transfer`, `what_if_travel_speed` (−20 %), `what_if_gas_loss`, `defect_synthesis` |
| `dataset_card.md` / `.json` | kept/rejected by reason, defect distribution, hours, WPS deviations, caption models, quality stats |
| `configs/` | curator backends: generated YAML / legacy command, `weld_caption_prompt.txt` |

Notes on manifest fields:
- `t0_s/t1_s` are pass time (sidecar `t0_s` + offset inside the exported clip). `start_ns/end_ns` in Lance are offsets inside the source mp4 (Curator semantics).
- `generator_prompt` is a **JSON string** (the `photoreal_transfer` variant) — pass it straight to
  `worldmodels.generate.what_if(prompt)` / `transfer.sim2real(..., prompt)`. Other variants:
  `curate.prompts.build_generator_prompt(row, "what_if_gas_loss" | "what_if_travel_speed" | "defect_synthesis", travel_speed_factor=0.8, defect="porosity")`,
  `curate.prompts.prompt_text(d)` to serialise, `curate.prompts.NEGATIVE_PROMPT` for the negative prompt.
  The schema follows `cosmos_framework/inference/prompting_templates/external_api/t2v_i2v_video_json_schema.json`
  (subjects, background_setting, lighting, aesthetics, cinematography, actions, segments, temporal_caption, audio_description, resolution 1280x720, aspect_ratio "16,9", duration, fps 24) plus `physical_realism`.
- `quality`: `blur` (median Laplacian variance, 320-px grey; higher = sharper), `motion` (mean |frame diff| per frame, 0–255), `glare_frac` (saturated-pixel fraction — arc glare), `dark_frac`, `mean_luma`, `arc_on_frac`, `duration_s`, `width`, `height`, `fps`.
- `reject_reason` ∈ `decode_error, too_short, underexposed, overexposed_glare, blurry, static, arc_off, near_duplicate, curator_invalid`. Thresholds: `--min-blur 5 --min-motion 0.05 --max-glare 0.35 --dedup-cosine 0.98 --min-arc-on 0` (`arc_off` is opt-in, e.g. `--min-arc-on 0.5`).
- Near-duplicate removal (cosine on mean-centred embeddings — raw cosines of weld-camera clips are all ≈0.999) is **label-aware**: only clips with the same defect set, violated WPS params and RT result can be duplicates (a look-alike clip with different labels is a different training sample). Sharpest member survives.

## Backends

### `lite` (CPU)
fixed-stride cut (5 s, whole clip if the export is ≈ one clip long) → libx264 transcode (ffmpeg CLI, PyAV fallback) →
quality metrics → filters → embedding `embedding_weldlite_v1` (4 frames @16×9 standardised + luma histogram + WPS-scaled sensor
mean/std, 606-d, L2) → near-dup → caption.
Captioner `--captioner auto|reason|template`: `auto` uses `worldmodels.reason.chat_meta/chat` (Cosmos 3 Reasoner) with
`WELD_CAPTION_PROMPT` + the clip's logged facts when a backend resolves (local/nim/replay); replay-mode placeholders and
failures fall back per clip to the deterministic CWI template built from the sidecar (process cues, arc data, WPS deviations,
defects + likely cause, RT result, visual glare/motion). `caption_model` is `cosmos3_reasoner` or `weld_template`.

### `curator-next` (GPU, Cosmos Curator Next)
Writes `configs/01-video-cut.yaml … 04-video-embed.yaml` (+ `05-07` for Cosmos 3 captioning), uploads inputs to MinIO,
then runs `pixi run --as-is run-pipeline <yaml> --json` in `$CURATOR_HOME` for each, re-running a stage that exits
reporting remaining work. `media_root` is the `--out` dir, so Curator writes `<out>/lance` and `<out>/clips/` itself; the
Lance table is then read back (`clip_id IS NOT NULL`), joined to sidecars by source file name, quality-scored locally,
filtered/de-duplicated (Cosmos-Embed1 vectors) and written to `manifest.jsonl`.

| stage | key settings (verified against `cosmos_curator/next/recipes/*/config.py`, all `extra="forbid"`) |
|---|---|
| video-cut | `input.uris: [s3://raw/weldworld/<clip>.mp4]` (S3 only), `split: {method: fixed-stride, duration_s: 5, stride_s: 5, min_duration_s: 2}`, `output.media_root` |
| video-transcode | `transcode: {video_encoder: libx264, video_bitrate: 4M}` |
| video-caption | `model.variant: qwen3_8_27b_fp8` (only `qwen3_8_27b[_fp8]` allowed; **no prompt setting** — prompt is code-owned), `execution: {inference_concurrency: 1, max_model_len: 32768}` |
| video-embed | `model.variant: 336p` → `embedding_cosmos_embed1_336p_v1` |
| Cosmos 3 + weld prompt (`--captioner cosmos3_nano\|cosmos3_super`) | inputs staged locally as `sessions/<clip_id>/cam0/torch_cam.mp4` → `mp4-index` (`container_kind: nonfragmented`) → `multimodal-split` (one `torch_cam` camera, `camera_alignment_fields: [pts_ns]`, `whole_session: true` for ≈5 s exports) → `multimodal-caption` (`context: video_only`, `model.variant: cosmos3_nano`, `inference.prompt: {mode: custom, template: <WELD_CAPTION_PROMPT>}`, `output.result_key: weld_cwi`) → column `caption_video_only_weld_cwi_cosmos3_nano_v1`, mapped back to cut clips by session + start offset |

Env: `CURATOR_HOME` (cosmos-curator checkout, `pixi install` done, models staged), `S3_ENDPOINT` (http://localhost:9000),
`MINIO_ROOT_USER/PASSWORD` (minioadmin), `S3_BUCKET` (raw), `S3_PREFIX` (weldworld), `CURATOR_S3_PROFILE` (weldworld).
Credentials: profile `[weldworld]` (and `[default]` if absent) with `aws_access_key_id, aws_secret_access_key, region,
endpoint_url` is added to `~/.aws/credentials` (other sections preserved); stages get `COSMOS_S3_PROFILE_PATH` pointing at
it and `execution.storage_profile: weldworld`. Upload uses boto3 from this venv or, if missing, Curator's pixi env.
`--configs-only` just writes the YAML (validate with `pixi run -e dev cosmos-curator pipeline validate <yaml>`),
`--skip-upload` if the mp4s are already in the bucket. Use a **fresh `--out`** per cut setting (Curator appends to
`<media_root>/lance`; the runner refuses a dir that holds a lite table).

### `curator-legacy` (GPU, best effort)
Runs, in `$CURATOR_HOME` (command also saved to `configs/legacy_command.json`):
```bash
pixi run --as-is video-pipeline split --input-video-path /abs/data/export --output-clip-path /abs/<out>/legacy \
  --splitting-algorithm fixed-stride --fixed-stride-split-duration 5 --fixed-stride-min-clip-length-s 2.0 \
  --transcode-encoder libopenh264 --motion-filter score-only --embedding-algorithm cosmos-embed1-336p \
  --captioning-algorithm cosmos3_nano --captioning-prompt-text "<WELD_CAPTION_PROMPT>" --generate-cosmos-predict-dataset
```
Accepts local input, supports custom prompts and `cosmos3_nano|cosmos3_super|cosmos_r2|qwen…` captioners, and emits a
Cosmos-Predict2 post-training set under `<out>/legacy/cosmos_predict2_video2world_dataset/`. Read back from
`<out>/legacy/metas/v0/*.json` (`source_video, start_ns, end_ns, clip_location, windows[].<model>_caption, embedding`).

## Tests
`tests/test_curate_lite.py` builds synthetic weld-like clips (good / porosity+gas-loss / static / blurry / glare / dark /
12 s long / byte-identical duplicate) and checks the manifest contract, each reject reason, splitting + pass times, H.264
output, Lance columns, prompt variants, dataset card, reasoner captioning with per-clip fallback, and the CLI.
`tests/test_curate_configs.py` validates every generated Curator config with the real pydantic models imported from the
Curator source (`CURATOR_HOME` or `/home/user/ref/cosmos-curator`); video-cut, mp4-index and multimodal-split additionally need
`attrs loguru smart_open` — without them those three fall back to a structural field-name check.
