#!/usr/bin/env bash
# Shared helpers for runpod/{bootstrap,start_all,stop_all,status}.sh. Source, don't execute.
# shellcheck disable=SC2034  # variables here are consumed by the scripts that source this file

# ------------------------------------------------------------------ paths
RUNPOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${RUNPOD_DIR}/.." && pwd)}"
WS="${WORKSPACE:-/workspace}"
STATE_DIR="${WS}/.weldworld"            # generated env, secrets, step markers
LOG_DIR="${LOG_DIR:-${WS}/logs}"
RUN_DIR="${WS}/run"                     # pid files
BIN_DIR="${WS}/bin"                     # uv, mosaicod, minio, crane, ...
MODELS_DIR="${MODELS_DIR:-${WS}/models}"
CURATOR_DIR="${CURATOR_DIR:-${WS}/cosmos-curator}"
CURATOR_WS="${WS}/curator_workspace"    # symlinked to /config (Curator looks for /config/models/<id>)
# Postgres never lives on the network volume: depending on the volume the postgres user either cannot enter
# /workspace at all or gets refused the 0700 data-dir permissions it insists on. The Mosaico catalog is rebuilt
# every run (the pipeline re-ingests), so container disk is fine. Its log goes there too, linked into LOG_DIR.
PGDATA_DEFAULT="/var/lib/postgresql/weldworld"
PG_LOG="/var/log/postgresql/weldworld.log"
MINIO_DATA="${WS}/minio-data"
MOSAICO_STORE_DIR="${WS}/mosaico-store"
VENV_DIR="${REPO_DIR}/.venv"

COPILOT_PORT="${COPILOT_PORT:-7860}"
MOSAICO_PORT="${MOSAICO_PORT:-6726}"
MINIO_API_PORT="${MINIO_API_PORT:-9000}"
MINIO_CONSOLE_PORT="${MINIO_CONSOLE_PORT:-9001}"
REASONER_PORT="${REASONER_PORT:-8000}"
GENERATOR_PORT="${GENERATOR_PORT:-8001}"
PG_PORT="${PG_PORT:-5432}"

# ------------------------------------------------------------------ logging
if [[ -t 1 ]]; then
  C_B=$'\033[1m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_C=$'\033[36m'; C_0=$'\033[0m'
else
  C_B=""; C_G=""; C_Y=""; C_R=""; C_C=""; C_0=""
fi
log()  { printf '%s[%s]%s %s\n' "${C_C}" "$(date +%H:%M:%S)" "${C_0}" "$*"; }
ok()   { printf '%s  ok%s  %s\n' "${C_G}" "${C_0}" "$*"; }
warn() { printf '%swarn%s  %s\n' "${C_Y}" "${C_0}" "$*" >&2; }
die()  { printf '%s FAIL%s %s\n' "${C_R}" "${C_0}" "$*" >&2; exit 1; }
hdr()  { printf '\n%s==> %s%s\n' "${C_B}" "$*" "${C_0}"; }

have() { command -v "$1" >/dev/null 2>&1; }

# Postgres log on the container disk (the postgres user may not be able to write to /workspace); a symlink in
# LOG_DIR keeps it next to the other logs for root.
pg_log_prepare() {
  mkdir -p "$(dirname "$PG_LOG")"; touch "$PG_LOG"; chown postgres:postgres "$(dirname "$PG_LOG")" "$PG_LOG" 2>/dev/null || true
  ln -sfn "$PG_LOG" "${LOG_DIR}/postgres.log" 2>/dev/null || true
}

# ------------------------------------------------------------------ env
# Precedence (first non-empty wins): process env (e.g. RunPod pod env vars HF_TOKEN /
# ANTHROPIC_API_KEY) > generated secrets > Curator/Postgres state > $STATE_DIR/demo.env (user-edited
# copy of configs/demo.env.example) > repo .env > generated paths ($STATE_DIR/env.sh).
# An empty `HF_TOKEN=` line in a file therefore never clobbers a token set on the pod.
_env_keys() {
  grep -oE '^[[:space:]]*(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*=' "$1" \
    | sed -E 's/^[[:space:]]*(export[[:space:]]+)?//; s/=$//' | sort -u
}

load_env_file() {
  local f="$1" k missing=()
  [[ -f "$f" ]] || return 0
  for k in $(_env_keys "$f"); do [[ -n "${!k:-}" ]] || missing+=("$k"); done
  [[ ${#missing[@]} -gt 0 ]] || return 0
  eval "$(
    set -a
    # shellcheck disable=SC1090
    . "$f" >/dev/null 2>&1
    for k in "${missing[@]}"; do
      if [[ -n "${!k:-}" ]]; then printf 'export %s=%q\n' "$k" "${!k}"; fi
    done
  )"
}

load_env() {
  local f
  for f in "${STATE_DIR}/secrets.env" "${STATE_DIR}/curator.env" "${STATE_DIR}/pg.env" \
           "${STATE_DIR}/demo.env" "${REPO_DIR}/.env" "${STATE_DIR}/env.sh"; do
    load_env_file "$f"
  done
  export PATH="${BIN_DIR}:${WS}/.pixi/bin:${WS}/go/bin:${WS}/gopath/bin:${WS}/.cargo/bin:${PATH}"
}

# ------------------------------------------------------------------ GPU / driver
gpu_count() {
  if have nvidia-smi; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
  else
    echo "${RUNPOD_GPU_COUNT:-0}"
  fi
}

driver_major() {
  if have nvidia-smi; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1
  else
    echo 0
  fi
}

# CUDA flavour for pip wheels: cu130 needs driver >= 580 (CUDA 13.x minimum driver), else cu128.
cuda_flavor() {
  local d; d="$(driver_major)"
  if [[ "${d:-0}" -ge 580 ]]; then echo cu130; else echo cu128; fi
}

# ------------------------------------------------------------------ processes
pidfile() { echo "${RUN_DIR}/$1.pid"; }

is_running() {
  local pf; pf="$(pidfile "$1")"
  [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

port_open() {  # port_open <port> [host]
  (exec 3<>"/dev/tcp/${2:-127.0.0.1}/$1") >/dev/null 2>&1
}

# start_bg <name> <command...>  — detached, logs to $LOG_DIR/<name>.log, pid in $RUN_DIR/<name>.pid
start_bg() {
  local name="$1"; shift
  mkdir -p "$LOG_DIR" "$RUN_DIR"
  if is_running "$name"; then
    ok "$name already running (pid $(cat "$(pidfile "$name")"))"
    return 0
  fi
  {
    echo
    echo "===== $(date -Is) starting ${name}: $*"
  } >>"${LOG_DIR}/${name}.log"
  nohup setsid "$@" >>"${LOG_DIR}/${name}.log" 2>&1 < /dev/null &
  echo $! >"$(pidfile "$name")"
  ok "$name started (pid $!, log ${LOG_DIR}/${name}.log)"
}

# stop_bg <name> — SIGTERM the whole process group, SIGKILL after 20 s
stop_bg() {
  local name="$1" pf pid i
  pf="$(pidfile "$name")"
  if ! [[ -f "$pf" ]]; then return 0; fi
  pid="$(cat "$pf")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for i in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    ok "$name stopped (pid $pid, ${i}s)"
  fi
  rm -f "$pf"
}

wait_port() {  # wait_port <port> <timeout_s> <label>
  local port="$1" t="$2" label="$3" i
  for ((i = 0; i < t; i++)); do
    if port_open "$port"; then ok "$label listening on :$port"; return 0; fi
    sleep 1
  done
  warn "$label not listening on :$port after ${t}s (see ${LOG_DIR})"
  return 1
}

# ------------------------------------------------------------------ RunPod URLs
proxy_url() {  # proxy_url <port>
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "https://${RUNPOD_POD_ID}-$1.proxy.runpod.net"
  else
    echo "http://localhost:$1"
  fi
}

print_urls() {
  hdr "Endpoints"
  printf '  %-26s %s\n' "Weld Copilot (UI)" "$(proxy_url "$COPILOT_PORT")"
  printf '  %-26s %s\n' "MinIO console" "$(proxy_url "$MINIO_CONSOLE_PORT")"
  if [[ -n "${RUNPOD_PUBLIC_IP:-}" ]]; then
    printf '  %-26s %s\n' "Mosaico (gRPC, TCP)" "${RUNPOD_PUBLIC_IP}:<public port mapped to ${MOSAICO_PORT} — Pod > Connect > Direct TCP Ports>"
  else
    printf '  %-26s %s\n' "Mosaico (gRPC, TCP)" "localhost:${MOSAICO_PORT}  (from a laptop: ssh -L ${MOSAICO_PORT}:localhost:${MOSAICO_PORT} <pod>)"
  fi
  printf '  %-26s %s\n' "Cosmos3 Reasoner (vLLM)" "http://localhost:${REASONER_PORT}/v1   (pod-internal)"
  printf '  %-26s %s\n' "Cosmos3 Generator (Omni)" "http://localhost:${GENERATOR_PORT}      (pod-internal)"
  if [[ -n "${RUNPOD_PUBLIC_IP:-}" && -n "${RUNPOD_TCP_PORT_22:-}" ]]; then
    printf '  %-26s %s\n' "SSH" "ssh root@${RUNPOD_PUBLIC_IP} -p ${RUNPOD_TCP_PORT_22}"
  fi
}
