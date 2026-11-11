#!/usr/bin/env bash
# Cosmos3 Transfer (control video -> photoreal video).
#
# Default path: Transfer is served by the SAME vLLM-Omni generator (generator.sh) - the Python
# client sends the control hint in extra_params, nothing extra to start:
#     python -m worldmodels.cli transfer data/generated/<seq>/control/edge.mp4 --kind edge --prompt "..."
#
# Offline / batch path (this script): cosmos-framework native inference on a spec JSON, as in the
# cosmos3-inference-transfer skill README quickstart:
#     .venv/bin/python -m cosmos_framework.scripts.inference --parallelism-preset=latency \
#       -i spec.json -o outputs/ --checkpoint-path Cosmos3-Nano --seed 2026 [--no-guardrails]
# Output: <out_dir>/<spec name>/vision.mp4. The framework loads the full omni model: do not run it
# on a GPU that also hosts the vLLM-Omni generator.
#
#   bash worldmodels/serve/transfer.sh <spec.json> <out_dir> [seed]
#   (the Python client writes specs itself when COSMOS_TRANSFER_BACKEND=framework)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

SPEC="${1:?usage: transfer.sh <spec.json> <out_dir> [seed]}"
OUT="${2:?usage: transfer.sh <spec.json> <out_dir> [seed]}"
SEED="${3:-2026}"
[ -x "$FRAMEWORK_DIR/.venv/bin/python" ] || { echo "cosmos-framework venv missing: INSTALL_FRAMEWORK=1 bash $HERE/install.sh" >&2; exit 1; }

NGPU="$(gpu_count)"
if [ -z "${TRANSFER_GPU:-}" ]; then
  if [ "${NGPU:-0}" -ge 2 ]; then TRANSFER_GPU=1; else TRANSFER_GPU=0; fi
fi
# drop inherited NGC library dirs that conflict with the venv's torch (skill note), keep cuda-compat
LD_CLEAN="$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v -E 'torch|cudnn|nccl|tensorrt|triton|site-packages|dist-packages' || true; } | paste -sd: -)"
export LD_LIBRARY_PATH="$LD_CLEAN"
GUARD=(--no-guardrails); [ "${COSMOS_GUARDRAILS:-0}" = "1" ] && GUARD=()

cd "$FRAMEWORK_DIR"
CUDA_VISIBLE_DEVICES="$TRANSFER_GPU" .venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$(realpath "$SPEC")" \
  -o "$(realpath -m "$OUT")" \
  --checkpoint-path "${COSMOS_FRAMEWORK_CHECKPOINT:-Cosmos3-Nano}" \
  --seed "$SEED" \
  "${GUARD[@]}"
