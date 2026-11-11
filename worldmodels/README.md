# worldmodels — Cosmos 3 Reasoner / Generator / Transfer clients

Clients for NVIDIA Cosmos 3 (`nvidia/Cosmos3-Nano`, one omnimodal checkpoint) used by the WeldWorld
copilot, plus the RunPod serve scripts and a replay cache so the demo runs without a GPU.

```python
from worldmodels import reason, generate, transfer, anomaly

r = reason.assess_clip("data/curated/clips/<clip_id>.mp4", sidecar)   # sidecar = export <clip_id>.json or manifest row
# {verdict: accept|reject|review, defects: [{type, confidence, t_s}], root_cause,
#  wps_findings: [{param, status: within|violation|marginal, note}], recommended_action,
#  reasoning (the <think> trace), code_basis, source, latency_ms, clip_id, cache_key, ...}
reason.chat([{"role": "user", "content": "Why did J07 fail RT?"}], video_path=clip)  # -> str
reason.explain_signals(sidecar)                                                     # -> str
generate.what_if("Robot travel speed is reduced by 20 percent ...", image_path=frame_png, seconds=5, seed=0)  # -> mp4
transfer.sim2real("data/generated/<seq>/control/edge.mp4", "edge", "photoreal FCAW fill pass")              # -> mp4
transfer.sim2real_multi({"edge": e, "depth": d}, "photoreal FCAW fill pass")                                # -> mp4
anomaly.synth_defect("bead.png", "porosity")                                                                 # -> png
```

Every call has a `*_meta` twin (`what_if_meta`, `sim2real_meta`, `chat_meta`, `explain_signals_meta`,
`synth_defect_meta`) that returns `{path|text, source, latency_ms, cache_key, request, ...}`;
`generate.read_sidecar(mp4)` gives the same for a returned path.
`source` is always one of `cosmos3-local | nim | cache | heuristic-fallback` (cache hits also carry
`origin_source`, `origin_latency_ms` and `cache_match`; placeholders carry `"placeholder": true`).

## Backends (`COSMOS_BACKEND`)

| value | Reasoner | Generator / Transfer |
|---|---|---|
| `auto` (default) | local if `GET $COSMOS_REASONER_URL/models` answers, else `nim` if `NVIDIA_API_KEY`, else `replay` | same probe on `$COSMOS_GENERATOR_URL/v1/models`; transfer also uses a `cosmos-framework` checkout if present |
| `local` | vLLM OpenAI API, video as `file://` (server has `--allowed-local-media-path /`) | vLLM-Omni `POST /v1/videos/sync`, or `cosmos_framework.scripts.inference` for transfer |
| `nim` | `https://integrate.api.nvidia.com/v1`, model `nvidia/cosmos3-nano-reasoner`, video as base64 data URL | `POST https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano` (NIM `/v1/infer` schema) — see caveat below |
| `replay` | cache → same-video cache → rule-based heuristic | cache → same prompt (any conditioning frame) → nearest showcase prompt → labelled placeholder MP4 |

Local/nim calls **read through and write through** the cache (`data/cache/worldmodels`, keyed by
sha256 of function + inputs with file *contents* hashed), so one rehearsal run on the GPU pod makes the
whole demo replayable. On any backend error the client falls back the same way as `replay`
(`COSMOS_FALLBACK=0` to raise instead). Ground-truth labels (defect annotations, RT result) are **not**
put in the Reasoner prompt unless `COSMOS_PROMPT_INCLUDE_LABELS=1`; the heuristic fallback does use them
and says so.

Other env: `COSMOS_REASONER_URL` (http://localhost:8000/v1), `COSMOS_REASONER_MODEL` (nvidia/Cosmos3-Nano,
`auto` = first of `/models`), `COSMOS_GENERATOR_URL` (http://localhost:8001), `NVIDIA_API_KEY`,
`COSMOS_NIM_REASONER_URL/MODEL`, `COSMOS_NIM_GENERATOR_URL` (point at `http://localhost:8000` for a
self-hosted Cosmos3 NIM container), `COSMOS_CACHE_DIR`, `COSMOS_CACHE=readwrite|write|read|off`
(`write` = force fresh calls), `COSMOS_GEN_PRESET=live|showcase|smoke`, `COSMOS_TRANSFER_BACKEND=auto|omni|framework`,
`COSMOS_FRAMEWORK_DIR`, `COSMOS_ANOMALYGEN_DIR`, `COSMOS_GUARDRAILS=0|1`, `COSMOS_REASONER_THINK=1`,
`COSMOS_REASONER_RETRIES=2`, `COSMOS_REASONER_VIDEO_MODE=auto|file|base64`, `COSMOS_GENERATOR_SHARED_FS=1`.

## Run it

```bash
# no GPU: everything works, clearly labelled
python -m worldmodels.cli --backend replay assess clip.mp4 --context clip.json
python -m worldmodels.cli health                     # status of every endpoint, cache, GPUs

# RunPod (runpod/bootstrap.sh + start_all.sh call these with the right GPU layout)
bash worldmodels/serve/install.sh                    # uv venvs: reasoner (vLLM) + generator (vLLM-Omni), weights
INSTALL_FRAMEWORK=1 bash worldmodels/serve/install.sh   # optional: cosmos-framework for spec-based transfer
DAEMON=1 bash worldmodels/serve/reasoner.sh          # :8000, GPU0
DAEMON=1 bash worldmodels/serve/generator.sh         # :8001, GPU1 (1 GPU: GENERATOR_PRECISION=fp8 OFFLOAD=1)
python -m worldmodels.serve.healthcheck --deep

# before the customer meeting: populate the replay cache
python -m worldmodels.cli warm-cache --manifest data/curated/manifest.jsonl          # 720p showcase
python -m worldmodels.cli warm-cache --manifest data/curated/manifest.jsonl --preset live --skip-assess

# one-offs
python -m worldmodels.cli assess data/curated/clips/<id>.mp4 --context data/export/<id>.json
python -m worldmodels.cli whatif --prompt "travel speed drops 20%" --image frame.png --preset live
python -m worldmodels.cli transfer data/generated/<seq>/control/edge.mp4 --kind edge --prompt "photoreal FCAW fill pass"
python -m worldmodels.cli transfer data/generated/<seq>/control/edge.mp4 --kind multi --prompt "..."   # + sibling depth.mp4
bash worldmodels/serve/transfer.sh configs/cosmos/transfer_spec_edge.json outputs/transfer   # framework path
python -m pytest tests/test_worldmodels.py -q
```

`warm-cache` assesses every `kept` clip of the manifest (+ `explain_signals`), renders the what-if jobs
of `configs/cosmos/showcase.json` (ids/prompts mirror the copilot's change templates, conditioned on the
first frame of a matching curated clip) and the sim2real jobs (edge, depth, edge+depth on
`data/generated/<seq>/control/`), and writes `data/cache/worldmodels/warm_report.json`
(`worldmodels.showcase.list_showcase()` / `get_showcase(id)`).

## Verified interface details (sources read 2026-11-11)

Sources: `NVIDIA/cosmos-framework` @ `45fecab` (`packages/vllm-cosmos3/README.md`;
`.agents/skills/cosmos3-inference-reasoning`, `-generate-media`, `-transfer` SKILL.md + references;
`.agents/skills/cookbooks/cosmos3/README.md` and `nim/`), the `nvidia/Cosmos3-Nano` model card,
`vllm-project/vllm-omni` `recipes/cosmos3/Cosmos3-Nano.md`, the NGC endpoint specs
`api.ngc.nvidia.com/v2/endpoints/qc69jvmznzxy/{cosmos3-nano,cosmos3-nano-reasoner}/spec`, and
`NVIDIA/cdf-anomalygen` README + `docs/tutorial/texture/5-generation.md` + the
`nvidia/Cosmos-AnomalyGen-Metal-Nano` model card.

**Reasoner** (verified): `vllm serve nvidia/Cosmos3-Nano --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling --allowed-local-media-path / --media-io-kwargs '{"video": {"num_frames": -1}}' --port 8000`
with `uv pip install --torch-backend=cu130 "vllm==0.21.0" "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"`
(CUDA 12.8 driver: `cu128` + `vllm==0.19.1`); the cookbook additionally documents native support in
`vllm>=0.23.0` without the plugin/`--hf-overrides` (`REASONER_ROUTE=native`). Requests: OpenAI chat
completions, `{"type": "video_url", ...}` **before** the text, `media_io_kwargs={"video": {"num_frames": -1, "fps": 4}}`
(NIM: `{"video": {"fps": 4.0}}`), reasoning via the prompt-guide suffix
(`<think>…</think>` then the answer), sampling with reasoning `temperature 0.6, top_p 0.95, top_k 20, presence_penalty 0, repetition_penalty 1.0`.
NIM Chat Completions supports `response_format` JSON schema (used as the 3rd retry). Answer parsing
handles `<think>` in `content`, or `reasoning_content` / `reasoning` fields.

**Generator** (verified): vLLM-Omni `vllm serve nvidia/Cosmos3-Nano --omni --model-class-name Cosmos3OmniDiffusersPipeline --allowed-local-media-path / --init-timeout 1800 [--no-guardrails] [--enable-layerwise-offload] [--quantization fp8]`,
`VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600`. `POST /v1/videos/sync` form fields `prompt` (compact JSON prompt),
`negative_prompt`, `size=WxH`, `num_frames`, `fps`, `num_inference_steps`, `guidance_scale`, `flow_shift`,
`seed`, `extra_params` = JSON `{"use_resolution_template": false, "use_duration_template": false, "guardrails": false}`;
`input_reference` file = image (image2video) or video (video2video, with `condition_frame_indexes_vision: [0,1]`
and `condition_video_keep: "first"|"last"` in `extra_params`); response body = MP4. Frame counts are
4k+1 (5 s @ 24 fps = 121). Presets: `live` 832x480/35 steps, `showcase` 1280x720/35 steps, `smoke` 320x192/20.

**Transfer** (verified for vLLM-Omni single control, from the skill's `build_transfer_request`):
same endpoint, `extra_params` = `{"edge": {"control_path": "<server-readable path>", "preset_edge_threshold": "medium"}, "resolution": "720", "control_guidance": 1.5, "num_video_frames_per_chunk": 121, "num_conditional_frames": 1, "num_first_chunk_conditional_frames": 0, "share_vision_temporal_positions": true, "max_frames": N, ...}`,
`guidance_scale 3.0`, 50 steps (showcase). "vLLM-Omni accepts multiple control hints in one request, but
does not support the per-hint weight field available in Cosmos Framework" (cookbook; the vllm-omni main
recipe now accepts `control_weight`). Framework path: `python -m cosmos_framework.scripts.inference --parallelism-preset=latency -i spec.json -o out/ --checkpoint-path Cosmos3-Nano --seed N --no-guardrails`
→ `out/<name>/vision.mp4`; spec fields copied from `specs/edge.json` (examples in `configs/cosmos/transfer_spec_*.json`).

**NIM `/v1/infer`** (verified from the NIM cookbook api-reference/transfer and the hosted OpenAPI spec):
strict JSON types; `model_mode` text2video|image2video|video2video, `resolution` keys `256|480|720[_4_3|_1_1|...]`,
`input_reference` data URL, `transfer: {"edge": {"video": data-url}, "control_guidance", ...}`; response
`{"b64_video"}` = VP9-in-MP4 (we transcode to H.264 for browsers).

## GPU memory (H100 80 GB) — what the sources say

| Component | Figure | Source |
|---|---|---|
| Cosmos3-Nano checkpoint | ~33 GB on disk; one download serves Reasoner, Generator, Transfer | vllm-omni recipe |
| Reasoner Nano, vLLM/TRT-LLM/Transformers/NIM | "~16–17 GB" (vs ~34 GB via the full-omni framework) | reasoner cookbook README |
| Reasoner Nano, reasoner-only framework image request | ~19 GiB peak, 1×H100 | reasoning skill eval 2026-11-08 |
| Reasoner vLLM process as we launch it | `--gpu-memory-utilization` × 80 GB: 0.45 → 36 GB (2 GPUs), 0.32 → 25.6 GB (1 GPU) | our setting (vLLM pre-allocates) |
| Generator Nano bf16 weights | 30.56 GiB resident after load | generate-media skill notes §2 |
| Generator Nano vLLM-Omni bf16, 720p video | peak ~46 GiB (1×B300); "~50 GB" | vllm-omni recipe |
| Generator Nano vLLM-Omni `--quantization fp8`, 720p video | ~36 GB | vllm-omni recipe |
| Generator Nano diffusers 1280x720 t2i | 33.8 GiB peak | skill notes §2 |
| Generator Nano TRT-LLM (cosmos3-nano-1gpu.yaml) | 50.6–52.1 GiB | skill notes §2 |
| Transfer | no memory figures published; Nano edge transfer 700–810 s on 1×H100 (vLLM-Omni) | transfer skill note |
| Latency, T2V 1280x720 35 steps, 1×B300 bf16 | 25/49/93/189 frames → ~7/15/33/93 s; first request has a one-time warm-up (~9 min measured for Nano FP8 on B200) | vllm-omni recipe / cookbook |

Layouts: **2×H100** — GPU0 reasoner (util 0.45) + Curator captioning, GPU1 generator bf16.
**1×H100** — reasoner util 0.32 (~26 GB) + generator `GENERATOR_PRECISION=fp8` (~36 GB peak) [+ `OFFLOAD=1`];
start the reasoner first (runpod/start_all.sh does). The planning figures in `docs/research/cosmos3.md`
(58/64/44/50/32/38/23 GiB) were not found in the primary sources above — treat them as estimates and
measure with `python -m worldmodels.serve.healthcheck` (prints per-GPU memory).

## Unverified / deviations

- **Hosted generator**: build.nvidia.com publishes `POST https://ai.api.nvidia.com/v1/cosmos/nvidia/cosmos3-nano`
  (NGC spec, function `d09cd49d-…`, path `/v1/infer`), but NVIDIA forum threads (Jun–Nov 9 2026) report a
  plain-text 404 for most accounts. The client implements it and reports that case explicitly;
  `healthcheck --nim-probe` sends `{}` (422 = route live, plain 404 = not provisioned). Hosted reasoner
  (`integrate.api.nvidia.com/v1`, `nvidia/cosmos3-nano-reasoner`) is taken from the NGC spec; not call-tested (no key here).
- No GPU here: no request was run against a real Cosmos server; tests use a fake OpenAI/vLLM-Omni/NIM server.
- `vllm-omni` native install tracks `main` (docs give no release pin); pin `VLLM_OMNI_REF` to the commit you validate.
  The docs' recommended route is the `vllm/vllm-omni:cosmos3` Docker image, which RunPod cannot run inside a pod
  (use it as the pod image instead if the native install misbehaves).
- `control_reference` upload (`COSMOS_GENERATOR_SHARED_FS=0`) and per-control `control_weight` are documented for
  vllm-omni `main`, not for the released cosmos3 image. Precomputed multi-control (edge+depth `control_path`)
  via the framework follows the spec schema but the upstream multi-control example derives its controls from `vision_path`.
- The `max_frames` key in framework specs is inferred from the transfer README ("capped by max_frames").
- AnomalyGen: usage verified from the repo docs, but the released LoRA is trained on magnetic tiles
  (Blowhole/Break/Crack/Fray/Uneven) — weld mapping porosity→Blowhole, crack→Crack, burn-through→Break,
  LOF/arc instability→Uneven, undercut→Fray is an approximation; checkpoint groups for
  `scripts/download_checkpoints.sh` must be chosen via `--list` (`ANOMALYGEN_CKPT_GROUPS`).
- **Guardrails** are off by default (`--no-guardrails`, `"guardrails": false`) because `nvidia/Cosmos-1.0-Guardrail`
  is gated; with guardrails off you are responsible for license compliance (OpenMDW-1.1 / NVIDIA Open Model License).
  `COSMOS_GUARDRAILS=1` turns them back on (needs `HF_TOKEN` with access and `cosmos-guardrail`).
- The Reasoner prompt guide recommends the system prompt "You are a helpful assistant."; we use a CWI /
  ASME Sec IX / B31.3 system prompt instead (domain framing); revert in `prompts.CWI_SYSTEM_PROMPT` if answers degrade.
- `configs/cosmos/negative_prompt_*.json` are vendored verbatim from cosmos-framework (OpenMDW-1.1).
