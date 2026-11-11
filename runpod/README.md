# Running WeldWorld on RunPod

> **Fastest path:** `cp .env.example .env`, fill in the keys, then `python3 runpod/launch.py` on your laptop.
> It creates the network volume and GPU pod through the Runpod API v2, runs the whole pipeline
> (`runpod/pod_entry.sh`), streams the logs, and terminates the pod when the code ends. The rest of this
> guide explains what happens inside the pod and how to do it by hand.

This guide takes you from an empty RunPod account to the Weld Copilot running on a GPU pod with
Mosaico, Cosmos Curator and Cosmos 3 (Reasoner and Generator), in about **90 minutes**. Most of
that time goes to downloads, and you only pay it once because everything lands on a network volume.

> RunPod pods are containers, so docker-in-docker is not available. Postgres, MinIO, `mosaicod`,
> Cosmos Curator (pixi) and vLLM all run **natively** in the pod. Locally you can use
> `infra/docker-compose.yml` instead.

Facts below were checked against docs.runpod.io and runpod.io/pricing on **2026-11-11**. Prices change, so confirm them before you quote a customer.

---

## 0. What runs where

| Service | Port | Exposed as | Notes |
|---|---|---|---|
| Weld Copilot (FastAPI + UI) | 7860 | HTTP proxy | the demo screen |
| MinIO console | 9001 | HTTP proxy | optional; shows the `raw` bucket Curator reads |
| mosaicod (Arrow Flight / gRPC) | 6726 | TCP (public IP) | optional; for SDK access from a laptop, an SSH tunnel is safer |
| MinIO S3 API | 9000 | pod-internal | Curator Next `video-cut` only accepts `s3://` input |
| Postgres | 5432 | pod-internal | Mosaico catalog |
| Cosmos3 Reasoner (vLLM) | 8000 | pod-internal | OpenAI-compatible `/v1` |
| Cosmos3 Generator (vLLM-Omni) | 8001 | pod-internal | `/v1/videos/sync` |
| SSH | 22 | TCP | |

GPU memory plan (from `docs/research/cosmos3.md`):

| Layout | GPU0 | GPU1 |
|---|---|---|
| **2× H100 80GB (recommended)** | Reasoner, about 23 GiB (vLLM capped at 0.45 of memory) + Curator captioning batches (Qwen3.8-27B-FP8) | Generator BF16, about 58 GiB (Transfer about 64 GiB) |
| **1× H100 80GB (budget)** | Reasoner (0.32 of memory) + Generator FP8 with layer-wise offload (about 32 GiB). Run curation **before** starting the model servers | — |
| 2× A100 80GB | Reasoner | Generator **BF16 only**: A100 has no FP8 |
| no GPU | `COSMOS_BACKEND=replay` (cached) or `nim` (build.nvidia.com) | — |

---

## 1. Pick the GPU (and know the cost)

On-demand **Secure Cloud** prices from [runpod.io/pricing](https://www.runpod.io/pricing), billed per second.
Network volumes only work with Secure Cloud, so Community Cloud prices don't apply to this setup.

| GPU (RunPod GPU ID) | VRAM | $/GPU-hr (Secure) | Demo config | $/hr |
|---|---|---|---|---|
| H100 SXM (`NVIDIA H100 80GB HBM3`) | 80 GB | $3.99 | **2× recommended** | **$7.98** |
| H100 PCIe (`NVIDIA H100 PCIe`) | 80 GB | $2.89 | 2× (also good) / 1× budget | $5.78 / **$2.89** |
| A100 SXM (`NVIDIA A100-SXM4-80GB`) | 80 GB | $1.79 | 2×, no FP8 | $3.58 |
| A100 PCIe (`NVIDIA A100 80GB PCIe`) | 80 GB | $1.79 | 2×, no FP8 | $3.58 |
| L40S (`NVIDIA L40S`) | 48 GB | $1.09 | not recommended: Reasoner + FP8 Generator (about 55 GiB) does not fit on one card | — |

GPU IDs come from [docs.runpod.io/references/gpu-types](https://docs.runpod.io/references/gpu-types).

**Typical demo budget:** 1.5 h first setup + 1 h rehearsal + 1 h meeting comes to about 3.5 h. That is about $28 on
2× H100 SXM or about $10 on 1× H100 PCIe, plus the volume (below). Stop the pod as soon as you are done.
A stopped pod keeps its volume, and you keep paying for storage until you delete it.

---

## 2. Template and CUDA driver

The **host driver** is what matters. The image's CUDA version matters much less, because pixi and uv bring their
own CUDA runtimes.

| Component | Needs | Why |
|---|---|---|
| Cosmos Curator (pixi env `default`) | **CUDA 13 → driver ≥ 580** | `pixi.toml` pins `cuda = "13.0.3"` (ref/cosmos-curator); CUDA 13.x minimum driver is 580 ([NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)) |
| Cosmos 3 Reasoner (vLLM) | cu130 wheels (`vllm==0.21.0`) on driver ≥ 580, or **cu128** wheels (`vllm==0.19.1`) on CUDA 12.8 hosts | `worldmodels/serve/install.sh` is called with `COSMOS_CUDA_TAG=cu130/cu128`, chosen from the driver |
| Cosmos 3 Generator (vLLM-Omni) | same as above | |

**Recommended image:** `runpod/pytorch:2.1.0-cu1300-torch2130-ubuntu2404` (official RunPod PyTorch, CUDA 13.0, see
[runpod/containers](https://github.com/runpod/containers/blob/main/official-templates/pytorch/README.md)).
If no CUDA-13 host is available, `runpod/pytorch:1.2.0-cu1281-torch2130-ubuntu2404` works too. Bootstrap then skips
Curator and the demo curates with `BACKEND=lite`, and vLLM uses cu128 wheels.

To land on a host with a driver of 580 or newer, choose **CUDA version 13.0** (or newer) in the deploy page's **Filters**, or pass
`--min-cuda-version 13.0` to `runpodctl pod create`
([runpodctl pod reference](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)). After the pod starts, confirm
with `nvidia-smi`: the header must show `Driver Version: 580.x` or newer and `CUDA Version: 13.x`.

---

## 3. Create the network volume (≥ 300 GB)

Console → **Storage → Network Volumes → New**. Pick a data center that has the GPUs you want, and give it a name such as
`weldworld` and a size of **300 GB**.

- Price: **$0.07/GB/month** for the first TB, so 300 GB costs about **$21/month**. A volume can grow later but cannot shrink
  ([docs](https://docs.runpod.io/storage/network-volumes)).
- It replaces the pod's volume disk and is **mounted at `/workspace`**. It is **Secure Cloud only** and must be
  **attached at deploy time**: you can't add it to a running pod.
- The container disk is ephemeral and is wiped on restart. That's why everything below lives on `/workspace`.

Space budget (measured from Hugging Face repo sizes):

| Item | Size |
|---|---|
| `nvidia/Cosmos3-Nano` (Reasoner, Generator, Transfer: one checkpoint) | 34 GB |
| `nvidia/Cosmos3-Nano-FP8` (only with `GENERATOR_PRECISION=fp8-ckpt`; the default `fp8` quantises online) | 21 GB |
| Curator models: `Qwen/Qwen3.8-27B-FP8` captioner, TransNetV2, Cosmos-Embed1-336p | about 32 GB |
| Curator pixi env + vLLM venvs (`/workspace/cosmos`) + uv/pixi caches | about 60–80 GB |
| optional `nvidia/Cosmos3-Nano-Sim-Transfer` (`WITH_SIM_TRANSFER=1`) | 31 GB |
| data, logs, Mosaico store, generated videos | 10–30 GB |
| **Total** | **about 180–250 GB → 300 GB leaves headroom** |

---

## 4. Deploy the pod

### Console
1. **Pods → Deploy → Secure Cloud**, select the network volume (this locks the data center), then pick **2× H100 SXM**
   (or 1× H100 PCIe).
2. **Filters → CUDA versions: 13.0** or newer.
3. **Change template / Edit template**:
   - Container image: `runpod/pytorch:2.1.0-cu1300-torch2130-ubuntu2404`
   - Container disk: **60 GB**. apt packages and `/tmp` live here. Weights do **not**.
   - **Expose HTTP ports:** `7860,9001`
   - **Expose TCP ports:** `22,6726`
   - **Environment variables** (store keys as [RunPod secrets](https://docs.runpod.io/pods/templates/secrets)):

     | Key | Value | Required |
     |---|---|---|
     | `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` | recommended: Cosmos 3 repos are ungated, but a token avoids rate limits and is used by Curator's `model-download` |
     | `ANTHROPIC_API_KEY` | `{{ RUNPOD_SECRET_anthropic_api_key }}` | optional: copilot chat with Claude tool use. Without it the copilot uses the Cosmos 3 Reasoner or its scripted router |
     | `NVIDIA_API_KEY` | `{{ RUNPOD_SECRET_nvidia_api_key }}` | optional: `nim` fallback to the hosted Cosmos 3 on build.nvidia.com |

4. Deploy. Wait for the pod to show telemetry in the console. A green "Running" dot alone doesn't mean it's ready.

### CLI (`runpodctl`)
```bash
runpodctl network-volume list                      # note the volume id
runpodctl pod create \
  --name weldworld \
  --image "runpod/pytorch:2.1.0-cu1300-torch2130-ubuntu2404" \
  --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 2 \
  --network-volume-id <VOLUME_ID> \
  --container-disk-in-gb 60 \
  --ports "7860/http,9001/http,22/tcp,6726/tcp" \
  --min-cuda-version 13.0 \
  --env '{"HF_TOKEN":"{{ RUNPOD_SECRET_hf_token }}","ANTHROPIC_API_KEY":"{{ RUNPOD_SECRET_anthropic_api_key }}"}'
runpodctl pod list                                 # wait for runtimeStatus=running
```

### SSH
The pod's **Connect** tab shows two commands: `ssh <podid>-<hash>@ssh.runpod.io` (proxy) and
`ssh root@<ip> -p <port>` ("SSH over exposed TCP", which supports scp/rsync and port forwarding). Inside the pod,
`RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_22` hold the same details
([env vars](https://docs.runpod.io/pods/templates/environment-variables)).

---

## 5. Bootstrap (once per volume, about 60–80 min, mostly downloads)

```bash
cd /workspace
git clone <this-repo-url> weld-physical-ai-demo
cd weld-physical-ai-demo
tmux new -s ww                         # survives SSH drops
bash runpod/bootstrap.sh               # idempotent: re-run any time
```

`runpod/bootstrap.sh` does the following, logging to `/workspace/logs/bootstrap.log`:

1. **apt**: ffmpeg, postgresql, build-essential, pkg-config, libpq-dev, protobuf-compiler, git-lfs, jq, tmux, …
2. **uv** in `/workspace/bin` and the repo venv `.venv` (Python 3.12, `requirements.txt`).
3. **Postgres**: cluster in `/workspace/pgdata`, database `mosaico`, random password in
   `/workspace/.weldworld/secrets.env`. If the volume filesystem refuses `chmod 700`, the cluster moves to the
   container disk instead. The catalog is cheap to rebuild with `make ingest`.
4. **MinIO**: the community edition is now source-only (`dl.min.io` returns HTTP 410). Bootstrap first extracts
   the binary from `cgr.dev/chainguard/minio` with `crane` and keeps it only if it runs. Otherwise it installs Go and runs
   `go install github.com/minio/minio@latest` into `/workspace/bin` (about 2 minutes). It uses random root credentials and
   writes `~/.aws/credentials` (with `endpoint_url`) for Curator.
5. **mosaicod**: tries, in order, the static musl binary from the GitHub release (`mosaicod-linux-amd64.tar.gz`,
   sha256-checked), then `crane export ghcr.io/mosaico-labs/mosaicod:latest`, which pulls `/usr/local/bin/mosaicod`
   out of the distroless image, then `SQLX_OFFLINE=true cargo build --release` from source. `protox` is pure Rust,
   so no `protoc` is needed.
6. **Cosmos Curator**: clones it with submodules to `/workspace/cosmos-curator` and runs `pixi install --frozen -e default`,
   writes `~/.config/cosmos_curator/config.yaml` (HF token), links `/config` to `/workspace/curator_workspace`
   (Curator expects weights at `/config/models/<id>`), and runs `model-download` for
   `transnetv2,qwen3_8_27b_fp8,cosmos_embed1_336p`. **Skipped automatically if the driver is below 580.** The demo then
   uses `CURATE_BACKEND=lite`.
7. **Cosmos 3 weights**: `hf download nvidia/Cosmos3-Nano` into the shared HF cache `/workspace/.cache/huggingface`. vLLM loads it by repo id.
8. **Serving stack**: `worldmodels/serve/install.sh` (vLLM + vLLM-Omni venvs in `/workspace/cosmos`, cu130 or cu128 picked from the driver).
9. Starts the infra and creates the `raw` bucket.

Useful flags: `--skip-curator`, `--skip-models`, `--skip-worldmodels`, `--mosaicod-from=release|image|source`,
`WITH_SIM_TRANSFER=1`, `WITH_ANOMALYGEN=1`, `MODELS="nvidia/Cosmos3-Edge"` (a small 4B model for quick tests).

---

## 6. Run the demo

```bash
make demo-runpod          # infra → simulate → ingest → export → curate → model servers → warm cache → copilot
# or step by step:
bash runpod/start_all.sh infra
make pipeline BACKEND=curator-next          # or BACKEND=lite
bash runpod/start_all.sh models --wait      # first load: 5–15 min (weights from the network volume)
make warm-cache
bash runpod/start_all.sh copilot
bash runpod/status.sh                       # health, GPUs, disk, URLs
```

`start_all.sh` prints the URLs it builds from `$RUNPOD_POD_ID`:

```
Weld Copilot (UI)          https://<podid>-7860.proxy.runpod.net
MinIO console              https://<podid>-9001.proxy.runpod.net
Mosaico (gRPC, TCP)        <RUNPOD_PUBLIC_IP>:<public port mapped to 6726 — Pod > Connect > Direct TCP Ports>
```

GPU assignment is automatic: two or more GPUs split Reasoner and Generator, and one GPU shares it with FP8 plus offload. Override it with
`REASONER_GPU`, `REASONER_GPU_UTIL`, `GENERATOR_GPU`, `GENERATOR_PRECISION` (`bf16|fp8|fp8-ckpt`) and `OFFLOAD`
(see `configs/demo.env.example`). Logs are in `/workspace/logs/<service>.log`. `bash runpod/stop_all.sh models` frees the GPUs,
for example before a Curator run on a single GPU.

**Configuration** lives in `/workspace/.weldworld/demo.env`, a copy of `configs/demo.env.example` that you edit.
Variables set on the pod, such as `HF_TOKEN`, always win over empty lines in that file.

### After a pod restart
The container disk is wiped, so the apt packages are gone, but everything else is still on `/workspace`:
```bash
cd /workspace/weld-physical-ai-demo && bash runpod/start_all.sh   # re-runs the apt step itself if needed
```

---

## 7. Gotchas

- **100-second proxy timeout.** `*.proxy.runpod.net` passes through Cloudflare, which closes requests after 100 s
  with an HTTP 524 ([docs](https://docs.runpod.io/pods/configuration/expose-ports)). Video generation takes longer than that,
  so the copilot runs it as a background job and polls. Showcase clips should be generated ahead of time with `make warm-cache`.
- **Proxy URLs are public.** Anyone with the URL can reach port 7860 and 9001. The MinIO credentials are random, and the data is
  synthetic. Stop the pod after the meeting.
- **mosaicod on TCP 6726** has no TLS or API key in this demo setup. For laptop access, prefer
  `ssh -L 6726:localhost:6726 root@<ip> -p <port>`, or bind it locally with `MOSAICOD_BIND=127.0.0.1`.
- **External TCP ports change** whenever the pod resets. Check **Connect → Direct TCP Ports** each time.
- **FP8 needs Hopper or Ada.** On A100, keep `GENERATOR_PRECISION=bf16` (2 GPUs).
- **Gated guardrail repo.** `nvidia/Cosmos-1.0-Guardrail` is gated, so the demo runs the Generator with guardrails off
  (`COSMOS_GUARDRAILS=0`). Say so if you're asked. See the top-level README's license section.
- **Curator on 1 GPU.** The captioner needs about 30 GB or more. Run `make curate` *before* `start_all.sh models`, or stop the models first.

## 8. Shut down
```bash
bash runpod/stop_all.sh            # services only
runpodctl pod stop <pod-id>        # stop GPU billing (volume still billed)
runpodctl pod delete <pod-id>      # when done; delete the network volume in the console to stop storage billing
```
