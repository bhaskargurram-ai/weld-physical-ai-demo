#!/usr/bin/env bash
# Runs INSIDE the Runpod pod, started by runpod/launch.py (the pod's start command clones the repo
# and execs this script). It runs the whole demo unattended and then shuts the pod down.
#
#   RUN_MODE=job    bootstrap -> infra -> pipeline -> Cosmos 3 servers -> warm cache -> results -> terminate
#   RUN_MODE=serve  same, then keeps the Weld Copilot up for SERVE_HOURS -> terminate
#
# Safety: a watchdog terminates the pod after MAX_HOURS whatever happens; any failure also
# ends in a shutdown. Persistent data (models, venvs, data/, results/) lives on the network
# volume at /workspace, so the next run is fast.
#
# Progress lines start with "WELDWORLD " so launch.py can follow them in the log stream.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
WS="${WORKSPACE:-/workspace}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS="${WS}/results/${RUN_ID}"
mkdir -p "${WS}/logs" "$RESULTS"
LOG="${WS}/logs/run-${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

RUN_MODE="${RUN_MODE:-job}"
MAX_HOURS="${MAX_HOURS:-6}"
SERVE_HOURS="${SERVE_HOURS:-2}"
CURATE_BACKEND="${CURATE_BACKEND:-curator-next}"
COPILOT_PORT="${COPILOT_PORT:-7860}"
export RUN_MODE CURATE_BACKEND COPILOT_PORT
export DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"

say() { printf 'WELDWORLD %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

API="https://api.runpod.io/v2"
_pod_api() {  # _pod_api METHOD PATH [JSON]  -> HTTP status code
  curl -sS -o /dev/null -w '%{http_code}' -X "$1" "${API}$2" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" -H 'Content-Type: application/json' \
    ${3:+--data "$3"} --max-time 30 2>/dev/null || echo 000
}

SHUTTING_DOWN=0
shutdown_pod() {  # shutdown_pod REASON RC
  [[ $SHUTTING_DOWN -eq 1 ]] && return
  SHUTTING_DOWN=1
  say "shutdown reason=$1 rc=${2:-?}"
  bash runpod/stop_all.sh >/dev/null 2>&1 || true   # clean Postgres/MinIO shutdown before the volume detaches
  sync
  if [[ -z "${RUNPOD_POD_ID:-}" || -z "${RUNPOD_API_KEY:-}" ]]; then
    say "no RUNPOD_POD_ID/RUNPOD_API_KEY in the pod env -> leaving termination to launch.py"
  else
    local code
    code="$(_pod_api POST "/pods/${RUNPOD_POD_ID}/action" '{"action":"terminate"}')"
    [[ "$code" == 2* ]] || code="$(_pod_api DELETE "/pods/${RUNPOD_POD_ID}")"
    if [[ "$code" == 2* ]]; then
      say "terminate requested (HTTP $code)"
    else
      # pod-scoped keys may only be allowed to stop: a stopped pod bills only for its disk
      code="$(_pod_api POST "/pods/${RUNPOD_POD_ID}/action" '{"action":"stop"}')"
      say "terminate refused; stop requested (HTTP $code) — launch.py will terminate it"
    fi
  fi
  say "DONE rc=${2:-1}"
  sleep infinity   # never return: if the container exited Runpod would restart it
}

trap 'shutdown_pod signal 143' TERM INT

# ---------------------------------------------------------------- hard cap
# (runs the shutdown itself: a TERM trap in the main shell would wait for the current stage to finish)
( sleep "$(awk -v h="$MAX_HOURS" 'BEGIN{printf "%d", h*3600}')"; shutdown_pod "max-hours(${MAX_HOURS})" 124 ) &

STATUS=0
stage() {  # stage NAME CMD... ; records failures but keeps going where it makes sense
  local name="$1"; shift
  local t0=$SECONDS
  say "stage=${name} start"
  "$@"
  local rc=$?
  say "stage=${name} rc=${rc} secs=$(( SECONDS - t0 ))"
  [[ $rc -eq 0 ]] || STATUS=$rc
  return $rc
}

say "START run=${RUN_ID} mode=${RUN_MODE} pod=${RUNPOD_POD_ID:-local} commit=$(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/WELDWORLD gpu /'

# ---------------------------------------------------------------- 1. install (idempotent, cached on the volume)
BOOT_FLAGS=()
[[ "$CURATE_BACKEND" == curator-* ]] || BOOT_FLAGS+=(--skip-curator)
stage bootstrap bash runpod/bootstrap.sh "${BOOT_FLAGS[@]}" || shutdown_pod bootstrap-failed 10

# shellcheck source=runpod/lib.sh
. runpod/lib.sh && load_env
PY="${REPO_DIR}/.venv/bin/python"

# Curator needs a CUDA 13 driver; bootstrap skips it otherwise -> fall back to the CPU curation
if [[ "$CURATE_BACKEND" == curator-* && ! -d "${CURATOR_HOME:-${WS}/cosmos-curator}/.pixi" ]]; then
  say "Cosmos Curator not installed on this host -> CURATE_BACKEND=lite"
  CURATE_BACKEND=lite
fi

# ---------------------------------------------------------------- 2. data services
stage infra bash runpod/start_all.sh infra || shutdown_pod infra-failed 11

# ---------------------------------------------------------------- 3. collect -> Mosaico -> export -> curate
stage pipeline "$PY" scripts/run_pipeline.py --data-dir "$DATA_DIR" --n-passes "${N_PASSES:-24}" \
  --backend "$CURATE_BACKEND" --skip warm-cache --keep-going
if [[ ! -s "${DATA_DIR}/curated/manifest.jsonl" && "$CURATE_BACKEND" != lite ]]; then
  say "curation with ${CURATE_BACKEND} produced no manifest -> retrying with lite"
  # a successful fallback recovers the run: the earlier pipeline rc should not fail it
  stage curate-lite "$PY" scripts/run_pipeline.py --data-dir "$DATA_DIR" --only curate --backend lite \
    && STATUS=0 && say "recovered: curated with lite (Cosmos Curator failed, see above)"
fi

# ---------------------------------------------------------------- 4. Cosmos 3 servers + warm cache
if stage models bash runpod/start_all.sh models --wait; then
  stage warm-cache "$PY" scripts/run_pipeline.py --data-dir "$DATA_DIR" --only warm-cache \
    --warm-preset "${COSMOS_GEN_PRESET:-live}"
else
  say "Cosmos 3 servers did not come up -> copilot will use nim/replay"
fi

# ---------------------------------------------------------------- 5. results (kept on the volume)
{
  cp -f "${DATA_DIR}/curated/dataset_card.md" "${DATA_DIR}/curated/manifest.jsonl" "$RESULTS/" 2>/dev/null
  find "${DATA_DIR}/cache" -maxdepth 2 -name '*report*.json' -exec cp -f {} "$RESULTS/" \; 2>/dev/null
  cp -f "$LOG" "$RESULTS/run.log" 2>/dev/null
  tar -czf "${WS}/results/${RUN_ID}.tar.gz" -C "${WS}/results" "$RUN_ID" 2>/dev/null
} || true
say "results=${RESULTS} (tarball ${WS}/results/${RUN_ID}.tar.gz on the volume)"
[[ -f "${DATA_DIR}/curated/dataset_card.md" ]] && sed -n '1,25p' "${DATA_DIR}/curated/dataset_card.md" | sed 's/^/WELDWORLD card | /'

# ---------------------------------------------------------------- 6. serve (optional) then shut down
if [[ "$RUN_MODE" == serve ]]; then
  if stage copilot bash runpod/start_all.sh copilot; then
    say "COPILOT_URL=https://${RUNPOD_POD_ID:-localhost}-${COPILOT_PORT}.proxy.runpod.net serving_for_hours=${SERVE_HOURS}"
    sleep "$(awk -v h="$SERVE_HOURS" 'BEGIN{printf "%d", h*3600}')" & wait $!
  fi
fi

shutdown_pod finished "$STATUS"
