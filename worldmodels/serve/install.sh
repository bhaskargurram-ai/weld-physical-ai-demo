#!/usr/bin/env bash
# Install the Cosmos 3 serving stacks natively (RunPod pods have no docker-in-docker).
#
#   bash worldmodels/serve/install.sh                # reasoner + generator venvs, prefetch weights
#   INSTALL_FRAMEWORK=1 bash worldmodels/serve/install.sh     # + cosmos-framework (Transfer via spec JSON)
#   INSTALL_ANOMALYGEN=1 bash worldmodels/serve/install.sh    # + cdf-anomalygen + AnomalyGen-Metal LoRA
#
# Versions (verified 2026-11-11 against huggingface.co/nvidia/Cosmos3-Nano model card and
# github.com/NVIDIA/cosmos-framework .agents/skills/cookbooks/cosmos3/README.md):
#   Reasoner, REASONER_ROUTE=plugin (default, "release-tested" per model card):
#       CUDA 13 driver: --torch-backend=cu130 vllm==0.21.0 + vllm-cosmos3 plugin
#       CUDA 12.8     : --torch-backend=cu128 vllm==0.19.1 + vllm-cosmos3 plugin
#   Reasoner, REASONER_ROUTE=native (cookbook: "Native Cosmos3 Reasoner support first appears in
#       the vLLM v0.23.0 stable release"): vllm>=0.23.0, no plugin, no --hf-overrides.
#   Generator: published matching pair vllm==0.32.0 + vllm-omni==0.32.0 (vllm-omni docs, "pre-built wheels";
#       v0.32.0 ships Cosmos3OmniDiffusersPipeline). vllm-omni is a vLLM plugin: vllm must be installed with
#       the SAME major.minor, it is not pulled in automatically. Override: VLLM_OMNI_VERSION, or
#       VLLM_OMNI_REF=<git ref> to build vllm-omni from source on top of vllm==$VLLM_OMNI_VERSION.
#   Framework: uv 0.11.29 (pinned by the framework's pyproject), uv sync --group cu130-train|cu128-train.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

CUDA_TAG="$(detect_cuda_tag)"
REASONER_ROUTE="${REASONER_ROUTE:-plugin}"
VLLM_OMNI_VERSION="${VLLM_OMNI_VERSION:-0.32.0}"
VLLM_OMNI_REF="${VLLM_OMNI_REF:-}"
COSMOS_FRAMEWORK_REF="${COSMOS_FRAMEWORK_REF:-main}"
PY="${COSMOS_PYTHON:-3.13}"
echo "== Cosmos 3 install: CUDA tag=$CUDA_TAG reasoner route=$REASONER_ROUTE home=$COSMOS_HOME"

# --- uv -------------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
fi
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv --version

# --- Reasoner: vLLM (+ vllm-cosmos3 plugin) ---------------------------------------------
if [ "${INSTALL_REASONER:-1}" = "1" ]; then
  echo "== Reasoner venv: $REASONER_VENV"
  ( cd /tmp && uv venv "$REASONER_VENV" --python "$PY" --seed --managed-python --allow-existing --no-config )
  if [ "$REASONER_ROUTE" = "native" ]; then
    uv pip install --no-config --python "$REASONER_VENV/bin/python" --torch-backend="$CUDA_TAG" "vllm>=0.23.0" openai
  else
    if [ "$CUDA_TAG" = "cu130" ]; then VLLM_PIN="vllm==0.21.0"; else VLLM_PIN="vllm==0.19.1"; fi
    uv pip install --no-config --python "$REASONER_VENV/bin/python" --torch-backend="$CUDA_TAG" "$VLLM_PIN" \
      "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3" \
      openai
  fi
  "$REASONER_VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
fi

# --- Generator: vLLM-Omni ---------------------------------------------------------------
if [ "${INSTALL_GENERATOR:-1}" = "1" ]; then
  if [ -n "$VLLM_OMNI_REF" ]; then OMNI_SPEC="vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@${VLLM_OMNI_REF}"
  else OMNI_SPEC="vllm-omni==${VLLM_OMNI_VERSION}"; fi
  echo "== Generator venv: $OMNI_VENV (vllm==${VLLM_OMNI_VERSION} + ${OMNI_SPEC})"
  ( cd /tmp && uv venv "$OMNI_VENV" --python "$PY" --seed --managed-python --allow-existing --no-config )
  # one resolve for both, so pip cannot pick a vllm-omni that needs a different vllm
  uv pip install --no-config --python "$OMNI_VENV/bin/python" --torch-backend="$CUDA_TAG" \
    "vllm==${VLLM_OMNI_VERSION}" "$OMNI_SPEC"
  if [ "${COSMOS_GUARDRAILS:-0}" = "1" ]; then
    # only needed with guardrails ON (gated nvidia/Cosmos-1.0-Guardrail + HF_TOKEN with access)
    uv pip install --no-config --python "$OMNI_VENV/bin/python" cosmos-guardrail
  fi
  "$OMNI_VENV/bin/python" - <<'PY'
import vllm, vllm_omni
from importlib.metadata import version
v, o = version("vllm"), version("vllm-omni")
assert v.split(".")[:2] == o.split(".")[:2], f"vllm {v} / vllm-omni {o} major.minor mismatch"
print(f"vllm {v} + vllm-omni {o} ok")
PY
fi

# --- Weights (shared by reasoner and generator) --------------------------------------------
if [ "${PREFETCH_WEIGHTS:-1}" = "1" ]; then
  echo "== Prefetch $COSMOS_MODEL into $HF_HOME (~33 GB)"
  uvx --no-config hf@latest download "$COSMOS_MODEL" >/dev/null   # honours HF_HOME / HF_TOKEN
  if [ "${GENERATOR_PRECISION:-bf16}" = "fp8-ckpt" ]; then
    uvx --no-config hf@latest download nvidia/Cosmos3-Nano-FP8 --local-dir "$COSMOS_HOME/models/cosmos3-nano-fp8" >/dev/null
  fi
fi

# --- cosmos-framework (Transfer via spec JSON; also the reference for specs) ----------------
if [ "${INSTALL_FRAMEWORK:-0}" = "1" ]; then
  echo "== cosmos-framework in $FRAMEWORK_DIR"
  if [ ! -d "$FRAMEWORK_DIR/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$COSMOS_FRAMEWORK_REF" https://github.com/NVIDIA/cosmos-framework.git "$FRAMEWORK_DIR"
  fi
  # the framework pins uv ==0.11.29 in its pyproject
  if [ "$(uv --version | awk '{print $2}')" != "0.11.29" ]; then
    curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
  fi
  ( cd "$FRAMEWORK_DIR" && GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --group="${CUDA_TAG}-train" )
  echo "framework ready: export COSMOS_FRAMEWORK_DIR=$FRAMEWORK_DIR"
fi

# --- cdf-anomalygen (stretch) -----------------------------------------------------------
if [ "${INSTALL_ANOMALYGEN:-0}" = "1" ]; then
  echo "== cdf-anomalygen in $ANOMALYGEN_DIR (needs a visible GPU while building)"
  [ -d "$ANOMALYGEN_DIR/.git" ] || git clone --depth 1 https://github.com/NVIDIA/cdf-anomalygen.git "$ANOMALYGEN_DIR"
  ( cd "$ANOMALYGEN_DIR" && bash scripts/env_setup.sh "${MAX_JOBS:-16}" 1 )
  # Checkpoint groups: `scripts/download_checkpoints.sh --list`. A full download is ~140 GB (60 GB of it
  # the Gemma VLM for agentic mask generation, which this demo does not use) -> name the groups you need
  # in ANOMALYGEN_CKPT_GROUPS (README example for an edge run: "base-edge wan-vae edge-processor guardrail").
  (
    cd "$ANOMALYGEN_DIR"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    bash scripts/download_checkpoints.sh --list || true
    # shellcheck disable=SC2086
    bash scripts/download_checkpoints.sh ${ANOMALYGEN_CKPT_GROUPS:-}
  )
  uvx --no-config hf@latest download nvidia/Cosmos-AnomalyGen-Metal-Nano \
    --local-dir "$ANOMALYGEN_DIR/checkpoints/Cosmos-AnomalyGen-Metal-Nano"
  echo "anomalygen ready: export COSMOS_ANOMALYGEN_DIR=$ANOMALYGEN_DIR"
fi

echo "== done. Next: bash worldmodels/serve/reasoner.sh  /  bash worldmodels/serve/generator.sh"
