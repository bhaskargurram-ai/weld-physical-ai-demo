# NVIDIA Cosmos 3 notes (as of 2026-11)

Cosmos 3 (released 2026-05-31) supersedes Predict2.5 / Transfer2.5 / Reason2. One omnimodal model:
Reasoner (text+vision -> text) and Generator (text/image/video -> video). Code: github.com/NVIDIA/cosmos-framework,
landing/cookbook: github.com/NVIDIA/cosmos. License OpenMDW-1.1 (commercial OK). HF repos ungated;
guardrail repo nvidia/Cosmos-1.0-Guardrail IS gated -> pass --no-guardrails / enable_safety_checker=False for demo.

Models: nvidia/Cosmos3-Edge (4B), nvidia/Cosmos3-Nano (16B, ~31.5GB bf16), nvidia/Cosmos3-Super (64B, 8 GPU),
*-FP8 variants (FP8 needs Hopper/Ada, not A100), Cosmos3-Nano-Sim-Transfer (real-time sim2real),
Cosmos-AnomalyGen-Metal-Nano (LoRA on Cosmos3-Nano that paints defects like cracks/blowholes on metal surfaces; github.com/NVIDIA/cdf-anomalygen).

## Reasoner (vLLM, OpenAI-compatible)
```
uv pip install --torch-backend=cu130 "vllm==0.21.0" "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3" openai
vllm serve nvidia/Cosmos3-Nano --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 --mm-encoder-tp-mode data --allowed-local-media-path / \
  --media-io-kwargs '{"video": {"num_frames": -1}}' --port 8000
```
(CUDA 12.8 hosts: cu128 + vllm==0.19.1). Chat completions, video at 4 fps. Nano Reasoner ~23 GiB; Edge ~12 GiB.

## Generator (vLLM-Omni)
```
docker image vllm/vllm-omni:cosmos3   (on RunPod: use as pod image, or pip-install vllm-omni)
vllm serve nvidia/Cosmos3-Nano --omni --port 8001 --init-timeout 1800 [--enable-layerwise-offload] [--tensor-parallel-size 2]
POST multipart /v1/videos/sync : prompt, size=1280x720, num_frames=189, fps=24, num_inference_steps=35,
     guidance_scale=6.0, flow_shift=10.0, [input_reference=@image]   -> mp4
```
Prompts work best as JSON "upsampled" prompts (examples in model repo assets/: `hf download nvidia/Cosmos3-Nano assets/`).
Diffusers alt: `Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda")`
`pipe(prompt=..., num_frames=189, height=720, width=1280, num_inference_steps=35, guidance_scale=6.0)`.

## Transfer (control -> video: edge / depth / seg / blur), base Cosmos3-Nano
```
python -m cosmos_framework.scripts.inference --parallelism-preset=latency -i specs/edge.json -o outputs/ --checkpoint-path Cosmos3-Nano --seed 2026
```
Spec examples: cosmos-framework/.agents/skills/cosmos3-inference-transfer/ (SKILL.md + assets/). Also via vLLM-Omni / Diffusers Cosmos3OmniModularPipeline.

## Memory (GiB)  Nano gen BF16 58 (transfer 64) | FP8 44 (50) | FP8+offload 32 (38) | Reasoner 23
H100 80GB: everything fits; Edge i2v 480p 189f ~25s on H100; Nano 720p: minutes (measure).
Hosted fallback: build.nvidia.com/nvidia/cosmos3-nano (free endpoint), build.nvidia.com/nvidia/cosmos3-nano-reasoner. NIM: nvcr.io/nim/nvidia/cosmos3:2.1.0.

## Welding data
- IntelLabs/Intel_Robotic_Welding_Multimodal_Dataset (HF, gated manual approval, research-only): video+audio+CURRENT_A,VOLTAGE_V,GAS_BAR,ROBOT_SPEED_CPM, 12 defect classes, 40GB. Optional adapter only.
- Taisanson/welding-seam-following (LeRobot, Apache-2.0, 0.33GB) — seam-following robot video+actions (mock weld).
- Kaggle tig-aluminium-5083 (CC BY-SA, weld pool images), RIAWELC X-ray (24k imgs), rikkarth/welding-defect-object-detection (HF, CC0).
- No published Cosmos-for-welding work -> demo is novel. Closest: Cosmos AnomalyGen-Metal, Cosmos Data Factory defect blueprint.
