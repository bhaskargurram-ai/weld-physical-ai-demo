#!/usr/bin/env bash
# Cosmos3-Nano Generator (text/image/video -> video, and control Transfer) on vLLM-Omni - port 8001.
#
#   bash worldmodels/serve/generator.sh                              # BF16 (dual-GPU default)
#   GENERATOR_PRECISION=fp8 OFFLOAD=1 bash worldmodels/serve/generator.sh   # single-GPU budget
#   DAEMON=1 bash worldmodels/serve/generator.sh                     # background
#
# Command verified 2026-11-11 against the cosmos cookbook "vLLM-Omni" setup section and the
# vllm-omni recipes/cosmos3/Cosmos3-Nano.md:
#   vllm serve nvidia/Cosmos3-Nano --omni --model-class-name Cosmos3OmniDiffusersPipeline \
#     --allowed-local-media-path / --port 8000 --init-timeout 1800 [--no-guardrails]
#   VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 raises /v1/videos/sync from its 600 s default (first
#   request includes a one-time warm-up; Nano edge transfer measured 700-810 s on 1x H100).
#
# Knobs:
#   GENERATOR_GPU        CUDA_VISIBLE_DEVICES (default 1 when >=2 GPUs, else 0)
#   GENERATOR_PRECISION  bf16 (default) | fp8 (online --quantization fp8, vllm-omni recipe:
#                        "cuts peak VRAM for 720p video generation from ~50 GB to ~36 GB")
#                        | fp8-ckpt (published nvidia/Cosmos3-Nano-FP8, served from a local dir)
#   OFFLOAD=1            --enable-layerwise-offload (less VRAM, slower, more host RAM)
#   COSMOS_GUARDRAILS=1  keep the gated nvidia/Cosmos-1.0-Guardrail on (needs HF_TOKEN access +
#                        cosmos-guardrail pip); default OFF -> --no-guardrails (demo; you are
#                        responsible for license compliance, see README)
#   GENERATOR_EXTRA_ARGS appended verbatim (e.g. "--cache-backend sea_cache", "--vae-fast-path channels_last")
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

NGPU="$(gpu_count)"
PORT="${GENERATOR_PORT:-8001}"
if [ -z "${GENERATOR_GPU:-}" ]; then
  if [ "${NGPU:-0}" -ge 2 ]; then GENERATOR_GPU=1; else GENERATOR_GPU=0; fi
fi
export CUDA_VISIBLE_DEVICES="$GENERATOR_GPU"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-3600}"
export PATH="$OMNI_VENV/bin:$PATH"

MODEL_ARG="$COSMOS_MODEL"
ARGS=()
case "${GENERATOR_PRECISION:-bf16}" in
  bf16) ;;
  fp8) ARGS+=(--quantization fp8) ;;
  fp8-ckpt)
    FP8_DIR="${FP8_MODEL_DIR:-$COSMOS_HOME/models/cosmos3-nano-fp8}"
    [ -f "$FP8_DIR/hf_quant_config.json" ] || uvx --no-config hf@latest download nvidia/Cosmos3-Nano-FP8 --local-dir "$FP8_DIR"
    # cookbook: "Do not select FP8 with --revision alone: the Cosmos3 vLLM-Omni pipeline must
    # receive a materialized local checkpoint." Keep the served name so clients do not change.
    MODEL_ARG="$FP8_DIR"; ARGS+=(--served-model-name "$COSMOS_MODEL") ;;
  *) echo "unknown GENERATOR_PRECISION=${GENERATOR_PRECISION}" >&2; exit 2 ;;
esac
[ "${OFFLOAD:-0}" = "1" ] && ARGS+=(--enable-layerwise-offload)
[ "${COSMOS_GUARDRAILS:-0}" = "1" ] || ARGS+=(--no-guardrails)
# shellcheck disable=SC2206
[ -n "${GENERATOR_EXTRA_ARGS:-}" ] && ARGS+=(${GENERATOR_EXTRA_ARGS})

echo "Generator: GPU=$CUDA_VISIBLE_DEVICES precision=${GENERATOR_PRECISION:-bf16} offload=${OFFLOAD:-0} port=$PORT"
echo "Ready when 'curl localhost:$PORT/health' returns 200 (/v1/models answers 200 with an empty list while loading)"
launch generator "$OMNI_VENV/bin/vllm" serve "$MODEL_ARG" \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --host 0.0.0.0 --port "$PORT" \
  --init-timeout 1800 \
  "${ARGS[@]}"
