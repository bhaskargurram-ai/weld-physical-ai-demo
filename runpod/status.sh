#!/usr/bin/env bash
# One-screen health check of the WeldWorld pod: services, health endpoints, GPUs, disk, URLs.
#   bash runpod/status.sh
set -uo pipefail
# shellcheck source=runpod/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env

row() {  # row <name> <port> [health-url]
  local name="$1" port="$2" url="${3:-}" proc="-" listen="-" health="-"
  if is_running "$name"; then proc="${C_G}pid $(cat "$(pidfile "$name")")${C_0}"
  elif [[ -f "$(pidfile "$name")" ]]; then proc="${C_R}dead${C_0}"; fi
  if port_open "$port"; then listen="${C_G}:${port}${C_0}"; else listen="${C_Y}:${port} closed${C_0}"; fi
  if [[ -n "$url" ]]; then
    if curl -fsS -m 3 -o /dev/null "$url"; then health="${C_G}healthy${C_0}"; else health="${C_Y}not ready${C_0}"; fi
  fi
  printf '  %-11s %-22s %-24s %s\n' "$name" "$proc" "$listen" "$health"
}

hdr "Services"
pg="-"
if [[ -f "${STATE_DIR}/pg.env" ]]; then
  # shellcheck disable=SC1091
  . "${STATE_DIR}/pg.env"
  if su postgres -c "${PGBIN}/pg_ctl -D '${PGDATA}' status" >/dev/null 2>&1; then pg="${C_G}running${C_0}"; else pg="${C_R}stopped${C_0}"; fi
fi
printf '  %-11s %-22s %-24s\n' "postgres" "$pg" ":${PG_PORT}"
row minio "$MINIO_API_PORT" "http://localhost:${MINIO_API_PORT}/minio/health/live"
row mosaicod "$MOSAICO_PORT"
row reasoner "$REASONER_PORT" "http://localhost:${REASONER_PORT}/v1/models"
row generator "$GENERATOR_PORT" "http://localhost:${GENERATOR_PORT}/v1/models"
row copilot "$COPILOT_PORT" "http://localhost:${COPILOT_PORT}/api/health"

hdr "Config"
printf '  COSMOS_BACKEND=%s  CURATE_BACKEND=%s  COPILOT_LLM=%s  driver=%s (%s wheels)\n' \
  "${COSMOS_BACKEND:-auto}" "${CURATE_BACKEND:-lite}" "${COPILOT_LLM:-auto}" "$(driver_major)" "$(cuda_flavor)"
[[ -n "${HF_TOKEN:-}" ]] && echo "  HF_TOKEN set" || echo "  HF_TOKEN not set"
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && echo "  ANTHROPIC_API_KEY set" || echo "  ANTHROPIC_API_KEY not set (copilot uses cosmos/scripted chat)"
[[ -n "${NVIDIA_API_KEY:-}" ]] && echo "  NVIDIA_API_KEY set (nim fallback available)"

hdr "Data"
d="${DATA_DIR:-${REPO_DIR}/data}"; [[ "$d" = /* ]] || d="${REPO_DIR}/${d#./}"
printf '  generated passes: %s   exported clips: %s   curated rows: %s   cached world-model calls: %s\n' \
  "$(find "$d/generated" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" \
  "$(find "$d/export" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" \
  "$( (wc -l <"$d/curated/manifest.jsonl") 2>/dev/null || echo 0)" \
  "$(find "$d/cache/worldmodels" -type f -name '*.json' 2>/dev/null | wc -l)"

hdr "GPUs"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version \
    --format=csv,noheader | sed 's/^/  /'
else
  echo "  (no nvidia-smi)"
fi

hdr "Disk"
df -h "$WS" 2>/dev/null | tail -1 | awk '{print "  " $6 ": " $3 " used / " $2 " (" $5 ")"}'
for p in "${HF_HOME:-${WS}/.cache/huggingface}" "${COSMOS_HOME:-${WS}/cosmos}" "$CURATOR_DIR" "$CURATOR_WS"; do
  [[ -d "$p" ]] && printf '  %-40s %s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"
done

print_urls
echo
echo "  logs: ${LOG_DIR}/{postgres,minio,mosaicod,reasoner,generator,copilot,bootstrap}.log"
