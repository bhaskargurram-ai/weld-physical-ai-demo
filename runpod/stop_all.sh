#!/usr/bin/env bash
# Stop WeldWorld services on the pod (reverse start order). Data on /workspace is untouched.
#
#   bash runpod/stop_all.sh            # everything
#   bash runpod/stop_all.sh models     # free the GPUs only (e.g. before a Curator run on 1 GPU)
#   bash runpod/stop_all.sh copilot | infra
set -Eeuo pipefail
# shellcheck source=runpod/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WHAT=("$@"); [[ ${#WHAT[@]} -gt 0 ]] || WHAT=(all)
want() { local w; for w in "${WHAT[@]}"; do [[ "$w" == all || "$w" == "$1" ]] && return 0; done; return 1; }
load_env

if want copilot; then stop_bg copilot; fi
if want models; then
  stop_bg generator
  stop_bg reasoner
  # vLLM spawns engine-core workers in their own groups; make sure the GPUs are really free
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -f 'VLLM::EngineCore' 2>/dev/null || true
fi
if want infra; then
  stop_bg mosaicod
  stop_bg minio
  if [[ -f "${STATE_DIR}/pg.env" ]]; then
    # shellcheck disable=SC1091
    . "${STATE_DIR}/pg.env"
    if su postgres -c "${PGBIN}/pg_ctl -D '${PGDATA}' status" >/dev/null 2>&1; then
      su postgres -c "${PGBIN}/pg_ctl -D '${PGDATA}' -m fast -w stop" >/dev/null && ok "postgres stopped"
    fi
  fi
fi
log "stopped: ${WHAT[*]}"
