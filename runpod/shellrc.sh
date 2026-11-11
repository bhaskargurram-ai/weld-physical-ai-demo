#!/usr/bin/env bash
# Sourced from ~/.bashrc on the pod (added by runpod/bootstrap.sh): puts /workspace tools on PATH
# and loads the demo env (HF caches, Mosaico/MinIO creds) into interactive shells.
# shellcheck source=runpod/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
alias ww-status='bash ${RUNPOD_DIR}/status.sh'
alias ww-logs='tail -n 50 -F ${LOG_DIR}/*.log'
