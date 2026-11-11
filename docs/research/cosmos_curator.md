# Cosmos Curator notes (NVIDIA cosmos-curator, Curator Next)

Source checkout for reading: /home/user/ref/cosmos-curator

## Install (single GPU box)
- Python 3.13, pixi; CUDA 13.0.3 runtime -> host driver must support CUDA 13 (>= 580 series). CHECK on RunPod.
- `git clone --recurse-submodules https://github.com/nvidia-cosmos/cosmos-curator.git` (needs cosmos-xenna submodule)
- `pixi install --frozen -e default` (vLLM 0.33, torch 2.14, ffmpeg 8.1.2, av 17.0.0)
- RunPod pods are containers -> no docker-in-docker; use native pixi path.
- Models: registry `cosmos_curator/configs/all_models.json`; `pixi run --as-is model-download --models transnetv2,qwen3_8_27b_fp8,cosmos_embed1_336p`
  HF token in `~/.config/cosmos_curator/config.yaml` -> `huggingface: {user, api_key}`; weights at `/config/models/<model_id>` (symlink ok).
  Next recipes never download at runtime.

## Next recipe chain (same media_root for all)
```
cosmos-curator pipeline template <kind> > x.yaml      # validate | render | schema
pixi run --as-is run-pipeline x.yaml [--set path=value] [--json]
```
- video-cut: INPUT MUST BE S3 (`input: {root_uri: s3://bucket/raw/}`); creds/endpoint_url in ~/.aws/credentials. Output local ok.
  `split: {method: fixed-stride, duration_s: 10, stride_s: 10, min_duration_s: 2}` or transnetv2.
- video-transcode: `transcode: {video_encoder: libopenh264}` -> `<media_root>/clips/<clip_id>.mp4`
- video-caption: `model: {variant: qwen3_8_27b_fp8}`, `execution: {inference_concurrency: 1, max_model_len: 32768}`
- video-embed: `model: {variant: 336p}` (Cosmos-Embed1, 768-d)
Example configs: examples/ci/vnext_fixed_stride_caption/

## Output: Lance table at <media_root>/lance
cols: source_uri, span_id, start_ns, end_ns, clip_id, clip_uri, clip_duration_ns, clip_width/height/frame_rate,
caption_qwen3_8_27b_fp8_v1, caption_qwen3_8_27b_fp8_v1_metadata{status,...}, embedding_cosmos_embed1_336p_v1 (768 f32)
```python
import lance
t = lance.dataset("/data/out/lance").to_table(filter="clip_id IS NOT NULL", columns=[...])
```

## Legacy pipeline (accepts local input; filters + Predict dataset export)
`pixi run --as-is video-pipeline split --input-video-path DIR --output-clip-path OUT --limit 1 --generate-cosmos-predict-dataset
 --transnetv2-min-length-frames 120 --motion-filter ... --aesthetic-threshold ... --embedding-algorithm cosmos-embed1-336p --captioning-algorithm cosmos_r2`
-> OUT/cosmos_predict2_video2world_dataset/{metas/*.txt, videos/*.mp4, t5_xxl/*.pickle}

## Gotchas
- Caption/embed refuse until transcode done. Changing cut settings needs a fresh media_root.
- Next has no motion/aesthetic filter; dedup only in robot `curate` recipe (cuML).
- Captioner alternatives: Cosmos-Reason2-8B (`cosmos_r2`) in legacy pipeline.
- Needs >=48GB GPU for reference pipelines.
