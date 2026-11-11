# shellcheck shell=bash
# Shared settings for the Cosmos 3 serve scripts (sourced, not executed).
#
# COSMOS_HOME   where venvs / checkouts / logs live (RunPod: persistent /workspace volume)
# HF_HOME       Hugging Face cache (Cosmos3-Nano is ~33 GB; reasoner + generator share it)

export COSMOS_HOME="${COSMOS_HOME:-/workspace/cosmos}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export COSMOS_LOG_DIR="${COSMOS_LOG_DIR:-$COSMOS_HOME/logs}"
export COSMOS_MODEL="${COSMOS_MODEL:-nvidia/Cosmos3-Nano}"
mkdir -p "$COSMOS_HOME" "$HF_HOME" "$COSMOS_LOG_DIR"

REASONER_VENV="${REASONER_VENV:-$COSMOS_HOME/venv-reasoner}"
OMNI_VENV="${OMNI_VENV:-$COSMOS_HOME/venv-omni}"
FRAMEWORK_DIR="${COSMOS_FRAMEWORK_DIR:-$COSMOS_HOME/cosmos-framework}"
ANOMALYGEN_DIR="${COSMOS_ANOMALYGEN_DIR:-$COSMOS_HOME/cdf-anomalygen}"
export REASONER_VENV OMNI_VENV FRAMEWORK_DIR ANOMALYGEN_DIR

gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

# "cu130" for a CUDA 13.x driver, "cu128" for 12.x (cosmos cookbook: pick the wheel matching the driver).
detect_cuda_tag() {
  if [ -n "${COSMOS_CUDA_TAG:-}" ]; then echo "$COSMOS_CUDA_TAG"; return; fi
  local v
  v="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\.\([0-9]*\).*/\1/p' | head -n1)"
  if [ -z "$v" ]; then
    echo "WARN: nvidia-smi not found; assuming CUDA 13 driver (cu130). Override with COSMOS_CUDA_TAG." >&2
    echo cu130
  elif [ "$v" -ge 13 ]; then
    echo cu130
  else
    echo cu128
  fi
}

# Run in the foreground, or (DAEMON=1) in the background with a log + pid file.
launch() {
  local name="$1"; shift
  if [ "${DAEMON:-0}" = "1" ]; then
    local log="$COSMOS_LOG_DIR/$name.log"
    nohup "$@" >"$log" 2>&1 &
    echo $! >"$COSMOS_LOG_DIR/$name.pid"
    echo "$name started (pid $(cat "$COSMOS_LOG_DIR/$name.pid")), log: $log"
  else
    exec "$@"
  fi
}

wait_ready() {  # wait_ready <url> <timeout_s>
  local url="$1" t="${2:-1800}" start
  start=$(date +%s)
  until curl -fsS "$url" >/dev/null 2>&1; do
    if [ $(( $(date +%s) - start )) -ge "$t" ]; then echo "not ready after ${t}s: $url" >&2; return 1; fi
    sleep 5
  done
  echo "ready after $(( $(date +%s) - start ))s: $url"
}
