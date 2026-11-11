# WeldWorld — physical AI demo for pipe-spool / pressure-vessel welding.
#   make help
SHELL        := /bin/bash
.SHELLFLAGS  := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON_VERSION ?= 3.12
VENV       ?= .venv
PY         := $(VENV)/bin/python
DATA_DIR   ?= ./data
N_PASSES   ?= 24
SEED       ?= 7
BACKEND    ?= lite
PORT       ?= 7860
COMPOSE    ?= docker compose -f infra/docker-compose.yml
PIPELINE   := $(PY) scripts/run_pipeline.py --data-dir $(DATA_DIR) --n-passes $(N_PASSES) --seed $(SEED) --backend $(BACKEND)
# load .env (copy of configs/demo.env.example) without clobbering variables already set in the shell
# (an empty `ANTHROPIC_API_KEY=` line must not wipe a key you exported) — same loader as runpod/
LOAD_ENV   := . runpod/lib.sh && load_env_file .env;

.PHONY: help setup-local env cloud-run cloud-serve cloud-status cloud-logs cloud-down infra-up infra-down infra-logs simulate ingest export curate warm-cache \
        pipeline copilot copilot-fixtures demo-local demo-runpod runpod-bootstrap runpod-status runpod-stop \
        test lint clean

help:  ## show targets
	@awk 'BEGIN{FS=":.*## "; printf "\nWeldWorld targets (vars: DATA_DIR=$(DATA_DIR) BACKEND=$(BACKEND) N_PASSES=$(N_PASSES))\n\n"} \
	     /^[a-zA-Z_-]+:.*## /{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------ setup
$(PY):
	@if command -v uv >/dev/null 2>&1; then uv venv --python $(PYTHON_VERSION) $(VENV); \
	 else python3 -m venv $(VENV); fi

setup-local: $(PY) env  ## create .venv and install requirements.txt (uv if available)
	@if command -v uv >/dev/null 2>&1; then uv pip install --python $(PY) -r requirements.txt; \
	 else $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt; fi
	@echo "venv ready: $(VENV)  ($$($(PY) --version))"

env:  ## create .env from .env.example (if missing) — fill in your API keys
	@[ -f .env ] || { cp .env.example .env; echo "created .env (edit for API keys; advanced settings: configs/demo.env.example)"; }

# ------------------------------------------------------------------ local infra (Docker)
infra-up:  ## start Postgres + mosaicod + MinIO (infra/docker-compose.yml)
	$(COMPOSE) up -d
	@for i in $$(seq 1 60); do (exec 3<>/dev/tcp/127.0.0.1/6726) 2>/dev/null && { echo "mosaicod up on :6726"; exit 0; }; sleep 1; done; \
	 echo "mosaicod not reachable yet: $(COMPOSE) logs mosaicod"; exit 1

infra-down:  ## stop local infra (keeps volumes; add V=1 to wipe them)
	$(COMPOSE) down $(if $(V),-v,)

infra-logs:  ## follow local infra logs
	$(COMPOSE) logs -f --tail=50

# ------------------------------------------------------------------ pipeline stages
simulate:  ## 1. generate N_PASSES synthetic weld passes -> data/generated
	$(PIPELINE) --only collect

ingest:  ## 2. ingest data/generated into Mosaico (needs mosaicod: infra-up)
	$(PIPELINE) --only ingest --strict

export:  ## 3. query Mosaico + export 5 s clips with sensor sidecars -> data/export
	$(PIPELINE) --only export

curate:  ## 4. curate clips -> data/curated (BACKEND=lite|curator-next|curator-legacy)
	$(PIPELINE) --only curate

warm-cache:  ## 5. pre-compute Cosmos 3 verdicts/videos into data/cache (replay mode reads these)
	$(PIPELINE) --only warm-cache

pipeline:  ## run all stages (auto-falls back to offline if mosaicod is down)
	$(PIPELINE)

# ------------------------------------------------------------------ app
copilot:  ## start the Weld Copilot on :7860 (uses .env)
	$(LOAD_ENV) DATA_DIR=$(DATA_DIR) $(VENV)/bin/uvicorn copilot.app:app --host 0.0.0.0 --port $(PORT)

copilot-fixtures:  ## start the copilot on checked-in fixtures (no pipeline run, no services)
	$(LOAD_ENV) COPILOT_FIXTURES=1 COSMOS_BACKEND=replay MOSAICO_OFFLINE=1 DATA_DIR=data/fixtures \
	  $(VENV)/bin/uvicorn copilot.app:app --host 0.0.0.0 --port $(PORT)

# ------------------------------------------------------------------ end-to-end
demo-local: setup-local  ## CPU-only end-to-end: simulate -> (Mosaico if up) -> export -> curate-lite -> copilot (replay)
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then $(MAKE) --no-print-directory infra-up || true; \
	 else echo "docker not available -> Mosaico offline mode (data/generated is read directly)"; fi
	$(LOAD_ENV) COSMOS_BACKEND=replay $(PIPELINE) --replay --keep-going
	$(LOAD_ENV) COSMOS_BACKEND=replay $(MAKE) --no-print-directory copilot

demo-runpod:  ## on the pod: infra -> pipeline (Curator if available) -> Cosmos 3 servers -> warm cache -> copilot
	bash runpod/start_all.sh infra
	. runpod/lib.sh && load_env && \
	  $(PY) scripts/run_pipeline.py --data-dir $(DATA_DIR) --n-passes $(N_PASSES) --seed $(SEED) \
	    --backend $${CURATE_BACKEND:-lite} --skip warm-cache --keep-going
	bash runpod/start_all.sh models --wait
	. runpod/lib.sh && load_env && $(PY) scripts/run_pipeline.py --data-dir $(DATA_DIR) --only warm-cache
	bash runpod/start_all.sh copilot

# ------------------------------------------------------------------ Runpod from your laptop (.env keys)
cloud-run:  ## laptop: volume -> GPU pod -> full pipeline -> follow logs -> terminate pod (RUN_MODE from .env)
	python3 runpod/launch.py run

cloud-serve:  ## laptop: same, then keep the Weld Copilot online for SERVE_HOURS, then terminate
	python3 runpod/launch.py run --mode serve

cloud-status:  ## laptop: show the recorded pod / volume
	python3 runpod/launch.py status

cloud-logs:  ## laptop: re-attach to the running pod's logs
	python3 runpod/launch.py logs

cloud-down:  ## laptop: terminate the pod now (DELETE_VOLUME=1 also deletes the storage)
	python3 runpod/launch.py down $(if $(filter 1,$(DELETE_VOLUME)),--delete-volume,)

runpod-bootstrap:  ## on the pod: one-time setup (idempotent)
	bash runpod/bootstrap.sh

runpod-status:  ## on the pod: services, GPUs, URLs
	@bash runpod/status.sh

runpod-stop:  ## on the pod: stop everything (data stays on /workspace)
	bash runpod/stop_all.sh

# ------------------------------------------------------------------ quality
test:  ## run the test suites (pytest)
	@MOSAICO_OFFLINE=1 COSMOS_BACKEND=replay COPILOT_LLM=scripted $(PY) -m pytest -q -p no:cacheprovider \
	   --ignore=data $(PYTEST_ARGS) || { rc=$$?; [ $$rc -eq 5 ] && echo "no tests collected" || exit $$rc; }

lint:  ## shellcheck + bash -n the ops scripts, byte-compile python
	@for f in runpod/*.sh; do bash -n $$f; done
	@if command -v shellcheck >/dev/null 2>&1; then shellcheck -x runpod/*.sh; else echo "shellcheck not installed (bash -n only)"; fi
	$(PY) -m py_compile scripts/run_pipeline.py

clean:  ## remove generated data (keeps data/fixtures and the world-model cache)
	rm -rf $(DATA_DIR)/generated $(DATA_DIR)/export $(DATA_DIR)/curated
