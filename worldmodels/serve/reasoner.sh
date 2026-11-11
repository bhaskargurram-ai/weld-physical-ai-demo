#!/usr/bin/env bash
# Cosmos3-Nano Reasoner on vLLM (OpenAI-compatible) - port 8000.
#
#   bash worldmodels/serve/reasoner.sh             # foreground
#   DAEMON=1 bash worldmodels/serve/reasoner.sh    # background, log in $COSMOS_LOG_DIR/reasoner.log
#
# Flags follow the Cosmos3-Nano model card "vLLM" section (verified 2026-11-11):
#   vllm serve nvidia/Cosmos3-Nano --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'
#     --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling
#     --allowed-local-media-path / --media-io-kwargs '{"video": {"num_frames": -1}}' --port 8000
# (--hf-overrides only with the vllm-cosmos3 plugin route; vLLM >= 0.23 loads Cosmos3 natively.)
#
# Knobs:
#   REASONER_GPU            CUDA_VISIBLE_DEVICES (default 0)
#   REASONER_GPU_UTIL       vLLM --gpu-memory-utilization. vLLM pre-allocates this fraction of the GPU!
#                           default 0.45 with >=2 GPUs (GPU0 also runs Curator captioning),
#                           0.32 on a single GPU shared with the generator (~25.6 GB of 80).
#   REASONER_MAX_MODEL_LEN  default 32768 (a 5 s clip at 4 fps is far below this)
#   REASONER_ROUTE          plugin|native (must match install.sh)
#   REASONER_EXTRA_ARGS     appended verbatim (e.g. "--trust-request-mm-kwargs")
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

NGPU="$(gpu_count)"
PORT="${REASONER_PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${REASONER_GPU:-0}"
if [ -z "${REASONER_GPU_UTIL:-}" ]; then
  if [ "${NGPU:-0}" -ge 2 ]; then REASONER_GPU_UTIL=0.45; else REASONER_GPU_UTIL=0.32; fi
fi
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"   # per the plugin README
export PATH="$REASONER_VENV/bin:$PATH"                 # FlashInfer JIT needs the venv's ninja on PATH

ARGS=(serve "$COSMOS_MODEL"
  --tensor-parallel-size "${REASONER_TP:-1}"
  --mm-encoder-tp-mode data
  --async-scheduling
  --allowed-local-media-path /
  --media-io-kwargs '{"video": {"num_frames": -1}}'
  --max-model-len "${REASONER_MAX_MODEL_LEN:-32768}"
  --gpu-memory-utilization "$REASONER_GPU_UTIL"
  --host 0.0.0.0 --port "$PORT")
if [ "${REASONER_ROUTE:-plugin}" = "plugin" ]; then
  ARGS+=(--hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}')
fi
# shellcheck disable=SC2206
[ -n "${REASONER_EXTRA_ARGS:-}" ] && ARGS+=(${REASONER_EXTRA_ARGS})

echo "Reasoner: GPU=$CUDA_VISIBLE_DEVICES util=$REASONER_GPU_UTIL port=$PORT route=${REASONER_ROUTE:-plugin}"
echo "Ready when 'Application startup complete.' is logged (~5 min first start). Check: curl localhost:$PORT/v1/models"
launch reasoner "$REASONER_VENV/bin/vllm" "${ARGS[@]}"
