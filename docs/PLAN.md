# WeldWorld — Physical AI for Fabrication Welding

A working prototype for a pipe-spool / pressure-vessel fabrication shop (target audience:
Ameri-Fab Energy Solutions, Weatherford TX — ASME vessels, process piping skids, structural skids).

It demonstrates the full physical-AI loop on real open-source infrastructure:

```
 ┌──────────── 1. COLLECT ────────────┐   ┌──── 2. STORE & QUERY ────┐   ┌──── 3. CURATE ─────┐   ┌──────── 4. WORLD MODELS ────────┐
 │ Robotic weld cell (sim + public    │   │ Mosaico (mosaicod +      │   │ NVIDIA Cosmos      │   │ Cosmos 3 Reasoner: weld QA,     │
 │ data): arc V/I, wire feed, gas,    │──▶│ Python SDK): typed       │──▶│ Curator: cut,      │──▶│   root cause, WPS compliance    │
 │ robot joints/TCP, IR interpass,    │   │ sequences per weld pass, │   │ transcode, caption,│   │ Cosmos 3 Generator: what-if     │
 │ torch camera; WPS + RT labels      │   │ queries on physical vals │   │ embed → Lance      │   │   video, sim→real Transfer      │
 └────────────────────────────────────┘   └──────────────────────────┘   └────────────────────┘   └───────────────┬─────────────────┘
                                                                                                                    ▼
                                         ┌──────────────── 5. WELD COPILOT (chat + dashboard) ────────────────┐
                                         │ "Show me FCAW fill passes on spool SP-1043 that broke the WPS"     │
                                         │ "Why did J07 fail RT?"  "What if travel speed drops 20%?"          │
                                         │ "Generate 10 photoreal porosity examples for training"             │
                                         └────────────────────────────────────────────────────────────────────┘
```

## Data strategy
- **Primary: WeldCell simulator** (`collect/`), domain-matched to the customer: A106 Gr.B carbon-steel pipe,
  GTAW root + FCAW fill/cap per WPS, pipe on a positioner, 6-axis robot. Generates physically consistent
  arc/robot/thermal streams, a procedural torch-camera render, and *control videos (edge/depth)* of the
  same scene, with injected defects (porosity, lack of fusion, undercut, burn-through, arc instability, crack)
  and radiography (RT) accept/reject labels.
- **Sim→Real with Cosmos 3 Transfer**: the procedural render's edge/depth videos are turned into photoreal
  weld footage. This is the "wow" moment and the synthetic-data multiplier story.
- **Real public data adapters** (optional): `Taisanson/welding-seam-following` (LeRobot, Apache-2.0) via the
  Mosaico LeRobot bridge; Intel Robotic Welding Multimodal dataset if access is granted (research-only).
- Any real customer data later: ROS bag / MCAP → `mosaico inject ros`, or CSV/video via our ingest.

## Compute (RunPod)
| Option | GPUs | Use |
|---|---|---|
| **Recommended** | 2× H100 80GB | GPU0: Cosmos3-Nano Reasoner (vLLM, ~23 GiB) + Curator captioning batch. GPU1: Cosmos3-Nano Generator (vLLM-Omni, BF16; measure on pod) |
| Budget | 1× H100 80GB | Reasoner + Generator FP8 w/ layerwise offload (23 + 32 GiB); curation runs first, then serving |
| No GPU / rehearsal | — | `COSMOS_BACKEND=replay` (cached outputs) or `nim` (build.nvidia.com hosted) |

RunPod pods are containers: no docker-in-docker. Everything runs natively (Postgres, MinIO, mosaicod binary,
pixi for Curator, uv venvs for vLLM). Locally (with Docker) use `infra/docker-compose.yml`.

## Repository layout & owners (parallel build)
| Dir | Owner agent | Contents |
|---|---|---|
| `collect/` | A: data | `schema.py` (shared contract, DONE), `simulate.py`, `render.py`, `datasets/` adapters |
| `mosaico_io/`, `infra/` | A: data | custom ontology, `ingest.py`, `query.py`, `export.py`, docker-compose |
| `curate/` | B: curation | Cosmos Curator configs/runner + `lite` fallback, manifest builder |
| `worldmodels/` | C: cosmos | Reasoner / Generator / Transfer clients, backends (local, nim, replay), serve scripts |
| `copilot/` | D: app | FastAPI + web UI, tool-using chat agent, dashboard |
| `runpod/`, `Makefile`, `README.md`, `docs/DEMO_SCRIPT.md` | E: ops | bootstrap, start/stop, end-to-end runner, talk track |

## Interface contracts (all agents code to these)
Paths are relative to repo root; `DATA_DIR` env defaults to `./data`.

**Generated pass directory** — `data/generated/<sequence_name>/`
- `meta.json` = `WeldPass.to_metadata()` + `{"wps": {...limits}}`
- `arc.parquet` (t_ns, current_a, voltage_v, wfs_mpm, gas_lpm) @100 Hz
- `travel.parquet` (t_ns, travel_speed_mmps, positioner_deg, heat_input_kj_mm) @100 Hz
- `robot.parquet` (t_ns, j1..j6, tcp_x, tcp_y, tcp_z, tcp_qx, tcp_qy, tcp_qz, tcp_qw) @50 Hz
- `thermal.parquet` (t_ns, interpass_c) @10 Hz
- `events.jsonl` ({t_ns, type: arc_start|arc_stop|defect_on|defect_off, defect?, severity?})
- `torch_cam.mp4` (24 fps, 1280x720 or 960x540, H.264) + `control/edge.mp4`, `control/depth.mp4` (same timing)

**Mosaico** — one sequence per pass, name = `sequence_name`, user metadata = meta.json (flattened where
needed: process, wps_id, spool_id, joint_id, pass_type, position, material, rt_result, has_defect).
Topics per `collect/schema.py` (TOPIC_*). Arc/travel/thermal use a custom ontology `WeldArc`/`WeldTravel`
(or Floating64 topics if custom ontologies prove impractical — document the choice in mosaico_io/README.md).

**`mosaico_io` Python API**
```python
connect() -> MosaicoClient                       # env MOSAICO_HOST/MOSAICO_PORT (localhost:6726)
ingest_pass_dir(path) -> str                     # returns sequence_name
list_passes(**filters) -> list[dict]             # metadata rows
get_signals(seq, t0_s=None, t1_s=None) -> pandas.DataFrame   # synced @ARC_HZ, columns: t_s + signal names
wps_violations(seq) -> list[dict]                # {param, t_start_s, t_end_s, value, limit_lo, limit_hi}
export_clips(out_dir, sequences=None, clip_s=5) -> list[dict]  # writes <out>/<clip_id>.mp4 + .json sidecar
```
Every function must also work in **offline mode** (`MOSAICO_OFFLINE=1`) by reading `data/generated/` directly,
so the copilot and curation can run without mosaicod.

**Export sidecar** `<clip_id>.json`: {clip_id, sequence_name, t0_s, t1_s, meta, signal_stats{mean,std,min,max per
signal}, wps_violations, defects_in_window, rt_result}

**Curation output** — `data/curated/`
- `lance/` (Curator Next schema: clip_id, clip_uri, start_ns, end_ns, caption_*, embedding_*)
- `clips/<clip_id>.mp4`
- `manifest.jsonl` one row per clip: {clip_id, clip_uri, sequence_name, t0_s, t1_s, caption, caption_model,
  sensor_summary, wps_violations, defects, rt_result, quality{blur, motion}, kept: bool, reject_reason,
  generator_prompt (Cosmos-3 JSON-style prompt for what-if / transfer)}
- CLI: `python -m curate.run --input data/export --out data/curated --backend {curator-next,curator-legacy,lite}`

**World models** — `worldmodels/` (backend via `COSMOS_BACKEND=local|nim|replay`)
```python
reason.assess_clip(video_path, context: dict) -> dict   # {verdict: accept|reject|review, defects:[{type,confidence,t_s}],
                                                         #  root_cause, wps_findings, recommended_action, reasoning}
reason.chat(messages, video_path=None) -> str
generate.what_if(prompt, image_path=None, video_path=None, seconds=5, seed=0) -> str   # mp4 path
transfer.sim2real(control_video, kind: "edge"|"depth", prompt, seed=0) -> str          # mp4 path
```
Env: `COSMOS_REASONER_URL` (default http://localhost:8000/v1), `COSMOS_GENERATOR_URL` (http://localhost:8001),
`NVIDIA_API_KEY` (nim backend). Replay cache: `data/cache/worldmodels/` keyed by sha256 of inputs; local backend
writes to the cache on every call so a rehearsal run populates replay mode.

**Copilot** — `uvicorn copilot.app:app --host 0.0.0.0 --port 7860`. Chat LLM via `COPILOT_LLM=claude|cosmos`
(Claude API tool use with `ANTHROPIC_API_KEY`, or Cosmos3 Reasoner OpenAI endpoint). Tools wrap mosaico_io,
curate manifest, worldmodels. Must run fully in replay/offline mode for a no-GPU rehearsal.

## Milestones
1. M1 (CPU, no GPU): simulate 24 passes → Mosaico → query/export → curate-lite → copilot in replay mode.
2. M2 (RunPod): Cosmos3 Reasoner + Generator served; Curator Next chain on exported clips; cache populated.
3. M3 (polish): dashboard visuals, demo script for Ameri-Fab, recorded fallback video.

## Risks
- Cosmos Curator needs CUDA 13 driver (≥580). If RunPod host driver is older → `--backend lite`.
- Curator Next video-cut accepts only S3 input → local MinIO bucket `raw`.
- Nano 720p generation can take minutes → pre-generate showcase clips, use 480p/fewer frames live, cache.
- Guardrail repo is gated → `--no-guardrails` for the demo (state it in the README).
