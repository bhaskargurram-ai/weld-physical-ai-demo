#!/usr/bin/env bash
# Start the WeldWorld stack on a RunPod pod (idempotent: running services are left alone).
#
#   bash runpod/start_all.sh            # infra + models + copilot
#   bash runpod/start_all.sh infra      # postgres, minio, mosaicod
#   bash runpod/start_all.sh models     # Cosmos3 Reasoner (vLLM) + Generator (vLLM-Omni)
#   bash runpod/start_all.sh copilot    # FastAPI copilot on :7860
#   bash runpod/start_all.sh --wait ... # block until the model servers answer (up to WAIT_S=1800)
#
# GPU layout (auto from GPU count; override REASONER_GPU, REASONER_GPU_UTIL, GENERATOR_GPU,
# GENERATOR_PRECISION=bf16|fp8|fp8-ckpt, OFFLOAD=0|1 — passed to worldmodels/serve/*.sh):
#   2+ GPUs : GPU0 = Reasoner (0.45, + Curator captioning batches), GPU1 = Generator BF16
#   1 GPU   : GPU0 = Reasoner (0.32 of memory) + Generator FP8 with layerwise offload
#   0 GPUs  : no model servers; copilot uses COSMOS_BACKEND=auto -> nim (NVIDIA_API_KEY) or replay
# Logs: /workspace/logs/<service>.log   PIDs: /workspace/run/<service>.pid
set -Eeuo pipefail
# shellcheck source=runpod/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WAIT=0; WHAT=()
for a in "$@"; do
  case "$a" in
    --wait) WAIT=1 ;;
    infra|models|copilot|all) WHAT+=("$a") ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) die "unknown argument: $a" ;;
  esac
done
[[ ${#WHAT[@]} -gt 0 ]] || WHAT=(all)
want() { local w; for w in "${WHAT[@]}"; do [[ "$w" == all || "$w" == "$1" ]] && return 0; done; return 1; }

load_env
mkdir -p "$LOG_DIR" "$RUN_DIR"
cd "$REPO_DIR"

# ================================================================== infra
start_postgres() {
  if ! ls -d /usr/lib/postgresql/*/bin >/dev/null 2>&1; then
    warn "postgres binaries missing (container disk reset after pod restart) -> re-running apt step"
    bash "${RUNPOD_DIR}/bootstrap.sh" --apt-only
  fi
  # shellcheck disable=SC1091
  [[ -f "${STATE_DIR}/pg.env" ]] && . "${STATE_DIR}/pg.env"
  if [[ -z "${PGDATA:-}" || ! -s "${PGDATA}/PG_VERSION" ]]; then
    warn "no Postgres cluster found -> initialising (Mosaico catalog will be empty: run 'make ingest')"
    bash "${RUNPOD_DIR}/bootstrap.sh" --skip-apt --postgres-only
    # shellcheck disable=SC1091
    . "${STATE_DIR}/pg.env"
  fi
  if su postgres -c "${PGBIN}/pg_ctl -D '${PGDATA}' status" >/dev/null 2>&1; then
    ok "postgres already running"
  else
    # after a pod restart the re-created `postgres` user may have a different uid than the files on /workspace
    chown -R postgres:postgres "$PGDATA" 2>/dev/null || true
    pg_log_prepare
    su postgres -c "${PGBIN}/pg_ctl -D '${PGDATA}' -l '${PG_LOG}' -w -t 60 start" >/dev/null
    ok "postgres started (${PGDATA})"
  fi
}

start_minio() {
  if [[ ! -x "${BIN_DIR}/minio" ]]; then warn "minio not installed — skipping (Curator Next needs it; lite does not)"; return; fi
  local redirect=()
  [[ -n "${RUNPOD_POD_ID:-}" ]] && redirect=(env "MINIO_BROWSER_REDIRECT_URL=$(proxy_url "$MINIO_CONSOLE_PORT")")
  MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}" MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}" \
    start_bg minio "${redirect[@]}" "${BIN_DIR}/minio" server "${MINIO_DATA}" \
      --address ":${MINIO_API_PORT}" --console-address ":${MINIO_CONSOLE_PORT}"
  wait_port "$MINIO_API_PORT" 30 minio || true
}

start_mosaicod() {
  [[ -x "${BIN_DIR}/mosaicod" ]] || die "mosaicod missing — run runpod/bootstrap.sh"
  mkdir -p "${MOSAICO_STORE_DIR}"
  # env per docs/research/mosaico.md (MOSAICOD_DB_URL has no credentials; user/password separate)
  MOSAICOD_DB_URL="${MOSAICOD_DB_URL:-postgresql://127.0.0.1:${PG_PORT}/mosaico}" \
  MOSAICOD_DB_USER="${MOSAICOD_DB_USER:-postgres}" \
  MOSAICOD_DB_PASSWORD="${MOSAICOD_DB_PASSWORD:?run bootstrap.sh first}" \
  MOSAICOD_STORE_ENDPOINT="${MOSAICOD_STORE_ENDPOINT:-file://${MOSAICO_STORE_DIR}}" \
  MOSAICOD_STORE_BUCKET="${MOSAICOD_STORE_BUCKET:-mosaico}" \
  MIMALLOC_PURGE_DELAY=0 \
    start_bg mosaicod "${BIN_DIR}/mosaicod" server --host "${MOSAICOD_BIND:-0.0.0.0}" --port "$MOSAICO_PORT" \
      --log-level "${MOSAICOD_LOG_LEVEL:-info}"
  wait_port "$MOSAICO_PORT" 60 mosaicod || { tail -n 20 "${LOG_DIR}/mosaicod.log" >&2; die "mosaicod failed to start"; }
}

# ================================================================== models
# Agent C owns worldmodels/serve/*.sh; we only pick GPUs/ports and call them.
find_serve() {  # find_serve <reasoner|generator>
  local d="${REPO_DIR}/worldmodels/serve" c
  for c in "$d/$1.sh" "$d/serve_$1.sh" "$d/start_$1.sh"; do [[ -f "$c" ]] && { echo "$c"; return 0; }; done
  c="$(find "$d" -maxdepth 1 -name "*${1:0:5}*.sh" ! -name install.sh 2>/dev/null | sort | head -1)"
  [[ -n "$c" ]] && echo "$c"
}

start_models() {
  local n; n="$(gpu_count)"
  if [[ "$n" -eq 0 ]]; then
    warn "no GPU visible — skipping model servers (copilot: COSMOS_BACKEND=auto -> nim/replay)"
    return
  fi
  local rs gs; rs="$(find_serve reasoner || true)"; gs="$(find_serve generator || true)"
  [[ -n "$rs" && -n "$gs" ]] || { warn "worldmodels/serve/{reasoner,generator} scripts not found — skipping"; return; }

  # Variable names are the ones worldmodels/serve/{reasoner,generator}.sh read (agent C).
  local r_gpu g_gpu prec offload r_util
  if [[ "$n" -ge 2 ]]; then
    r_gpu="${REASONER_GPU:-0}"; g_gpu="${GENERATOR_GPU:-1}"
    prec="${GENERATOR_PRECISION:-bf16}"; offload="${OFFLOAD:-${GENERATOR_OFFLOAD:-0}}"
    r_util="${REASONER_GPU_UTIL:-${REASONER_GPU_MEM_UTIL:-0.45}}"
    log "layout: ${n} GPUs -> reasoner GPU${r_gpu} (util ${r_util}), generator GPU${g_gpu} (${prec})"
  else
    r_gpu="${REASONER_GPU:-0}"; g_gpu="${GENERATOR_GPU:-0}"
    prec="${GENERATOR_PRECISION:-fp8}"; offload="${OFFLOAD:-${GENERATOR_OFFLOAD:-1}}"
    r_util="${REASONER_GPU_UTIL:-${REASONER_GPU_MEM_UTIL:-0.32}}"
    log "layout: 1 GPU -> reasoner (util ${r_util}) + generator ${prec} offload=${offload} share GPU0"
    if pgrep -f cosmos_curator >/dev/null 2>&1; then
      warn "Cosmos Curator is running on the only GPU — start models after curation finishes"
    fi
  fi

  # Foreground mode (no DAEMON=1): the serve scripts exec vllm, so our pid file tracks the server itself.
  local common=(env -u DAEMON "COSMOS_HOME=${COSMOS_HOME:-${WS}/cosmos}"
                "HF_HOME=${HF_HOME:-${WS}/.cache/huggingface}" "COSMOS_CUDA_TAG=$(cuda_flavor)")
  start_bg reasoner "${common[@]}" REASONER_GPU="$r_gpu" REASONER_GPU_UTIL="$r_util" \
    REASONER_PORT="$REASONER_PORT" bash "$rs"
  # On one GPU let the reasoner claim its share first, otherwise both size their pools off free memory.
  if [[ "$r_gpu" == "$g_gpu" ]]; then
    log "waiting for reasoner before starting the generator on the same GPU"
    wait_http "http://localhost:${REASONER_PORT}/v1/models" "${WAIT_S:-1800}" reasoner || true
  fi
  # Runpod base images run nginx on 8001 -> 8000 (reasoner); never share a port with something else
  if port_open "$GENERATOR_PORT" && ! is_running generator; then
    warn "port ${GENERATOR_PORT} is already taken by another process - not starting the generator:"
    (ss -ltnp 2>/dev/null | grep ":${GENERATOR_PORT} " >&2) || true
    return
  fi
  start_bg generator "${common[@]}" GENERATOR_GPU="$g_gpu" GENERATOR_PRECISION="$prec" OFFLOAD="$offload" \
    GENERATOR_PORT="$GENERATOR_PORT" bash "$gs"
}

wait_http() {  # wait_http <url> <timeout_s> <label>
  local url="$1" t="$2" label="$3" i
  for ((i = 0; i < t; i += 5)); do
    if ! is_running "$label"; then warn "$label exited — tail ${LOG_DIR}/${label}.log:"; tail -n 20 "${LOG_DIR}/${label}.log" >&2; return 1; fi
    if curl -fsS -m 3 -o /dev/null "$url"; then ok "$label ready (${i}s): $url"; return 0; fi
    sleep 5
  done
  warn "$label not ready after ${t}s"; return 1
}

# ================================================================== copilot
start_copilot() {
  [[ -x "${VENV_DIR}/bin/uvicorn" ]] || die "repo venv missing — run runpod/bootstrap.sh"
  # COSMOS_BACKEND=auto: probe the local vLLM servers, else nim, else replay (worldmodels/config.py)
  COSMOS_BACKEND="${COSMOS_BACKEND:-auto}" DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}" \
  MOSAICO_HOST="${MOSAICO_HOST:-localhost}" MOSAICO_PORT="$MOSAICO_PORT" \
    start_bg copilot "${VENV_DIR}/bin/uvicorn" copilot.app:app --host 0.0.0.0 --port "$COPILOT_PORT" \
      --app-dir "$REPO_DIR" --proxy-headers --forwarded-allow-ips '*'
  wait_port "$COPILOT_PORT" 180 copilot || tail -n 20 "${LOG_DIR}/copilot.log" >&2
}

# ================================================================== main
if want infra; then
  hdr "infra"
  start_postgres
  start_minio
  start_mosaicod
fi
if want models; then
  hdr "Cosmos 3 model servers"
  start_models
  if [[ $WAIT -eq 1 ]]; then
    if is_running reasoner; then wait_http "http://localhost:${REASONER_PORT}/v1/models" "${WAIT_S:-1800}" reasoner || true; fi
    if is_running generator; then wait_http "http://localhost:${GENERATOR_PORT}/health" "${WAIT_S:-1800}" generator || true; fi
  else
    log "model servers load in the background (several minutes); check with runpod/status.sh or --wait"
  fi
fi
if want copilot; then
  hdr "copilot"
  start_copilot
fi
print_urls
