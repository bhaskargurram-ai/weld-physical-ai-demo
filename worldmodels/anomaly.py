"""Defect image synthesis with Cosmos AnomalyGen (stretch goal, best effort).

Model: ``nvidia/Cosmos-AnomalyGen-Metal-Nano`` - per-defect-type LoRA adapters on the
frozen Cosmos3-Nano diffusion backbone, trained few-shot on the *magnetic-tile*
defect dataset (classes ``metal_surface+MT_{Blowhole,Break,Crack,Fray,Uneven}``).
It inpaints a defect into a clean image inside a binary mask. Usage (verified from
github.com/NVIDIA/cdf-anomalygen docs/tutorial/texture/5-generation.md)::

    torchrun --nproc_per_node=1 anomalygen/scripts/texture/generate.py \
        --checkpoint <.../model/iter_000014000.pt> --recipe <exp_texture_ft_magnetic_tile.yaml> \
        --input_data_path testcase.jsonl --output_dir out/ --no-guardrail

testcase.jsonl rows: {image_filename, mask_filename, anomaly_type, guidance, num_steps,
crop_ratio, num_generated_images, poisson_blend, seed}; results in
``out/reconstructed_image/{anomaly_type}_{idx:05d}.png``. ``--no-guardrail`` matters:
the text guardrail blocks every ``MT_Crack`` caption (tutorial note).

Domain caveat: the released adapters were trained on magnetic tiles, not weld beads,
so weld-defect -> tile-class mapping below is an approximation (porosity ~ blowhole,
crack ~ crack, burn-through ~ break, lack of fusion ~ uneven, undercut ~ fray).
For production, fine-tune a weld LoRA with the same pipeline (~5 examples/defect).
Without the checkout / GPU a procedural painter produces a clearly-labelled stand-in.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import cache
from .config import SOURCE_CACHE, SOURCE_HEURISTIC, SOURCE_LOCAL, Settings, get_settings
from .http import BackendError

FN_ANOMALY = "anomaly"

WELD_TO_ANOMALYGEN = {
    "porosity": "metal_surface+MT_Blowhole",
    "crack": "metal_surface+MT_Crack",
    "burn_through": "metal_surface+MT_Break",
    "lack_of_fusion": "metal_surface+MT_Uneven",
    "undercut": "metal_surface+MT_Fray",
    "arc_instability": "metal_surface+MT_Uneven",
}


def _paths(s: Settings) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if not s.anomalygen_dir:
        return None, None, None
    root = Path(s.anomalygen_dir)
    ck = Path(s.anomalygen_checkpoint) if s.anomalygen_checkpoint else root / "checkpoints" / "Cosmos-AnomalyGen-Metal-Nano" / "model" / "iter_000014000.pt"
    rc = Path(s.anomalygen_recipe) if s.anomalygen_recipe else root / "checkpoints" / "Cosmos-AnomalyGen-Metal-Nano" / "exp_texture_ft_magnetic_tile.yaml"
    return root, ck, rc


def anomalygen_available(s: Optional[Settings] = None) -> bool:
    s = s or get_settings()
    root, ck, rc = _paths(s)
    return bool(root and (root / ".venv" / "bin" / "torchrun").exists() and ck and ck.exists() and rc and rc.exists())


def auto_mask(image_path: str, defect: str, out_png: Path, seed: int = 0) -> Path:
    """Binary mask (255 = defect) on the bead region: the brightest-texture horizontal band."""
    import cv2
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    h, w = img.shape
    rows = np.abs(np.diff(img, axis=1)).mean(axis=1)  # texture energy per row ~ bead ripples
    cy = int(np.argmax(np.convolve(rows, np.ones(max(3, h // 12)) / max(3, h // 12), mode="same")))
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    if defect == "porosity":
        for _ in range(int(rng.integers(4, 9))):
            x = int(rng.integers(w // 4, 3 * w // 4))
            y = int(np.clip(cy + rng.normal(0, h / 30), 0, h - 1))
            cv2.circle(m, (x, y), int(rng.integers(max(2, w // 120), max(3, w // 60))), 255, -1)
    elif defect == "crack":
        x0 = int(w * 0.3)
        pts = [(x0 + i * w // 40, int(cy + rng.normal(0, h / 120))) for i in range(16)]
        cv2.polylines(m, [np.array(pts, np.int32)], False, 255, max(2, w // 300))
    else:
        cv2.ellipse(m, (w // 2, int(cy + h / 25)), (w // 6, max(3, h // 50)), 0, 0, 360, 255, -1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m).save(out_png)
    return out_png


def paint_procedural(image_path: str, mask_path: str, defect: str, out_png: Path, seed: int = 0) -> Path:
    """Clearly labelled non-AI stand-in: darken/texture the masked region."""
    import cv2
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("RGB")).copy()
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    rng = np.random.default_rng(seed)
    dark = (img.astype(np.float32) * (0.25 if defect in ("porosity", "crack") else 0.55)).astype(np.uint8)
    noise = rng.normal(0, 10, img.shape).astype(np.int16)
    painted = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[m] = painted[m]
    edges = cv2.Canny(m.astype(np.uint8) * 255, 50, 150) > 0
    img[edges] = np.clip(img[edges].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    label = f"PROCEDURAL PLACEHOLDER ({defect}) - run AnomalyGen on GPU"
    cv2.putText(img, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 1800), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 1800), (255, 200, 60), 1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_png)
    return out_png


def _run_anomalygen(image_path: str, mask_path: str, anomaly_type: str, seed: int, s: Settings, work: Path,
                    guidance: float, num_steps: int, crop_ratio: float) -> Path:
    root, ck, rc = _paths(s)
    if not anomalygen_available(s):
        raise BackendError("cdf-anomalygen checkout / checkpoint not found (COSMOS_ANOMALYGEN_DIR)")
    work.mkdir(parents=True, exist_ok=True)
    tc = work / "testcase.jsonl"
    tc.write_text(json.dumps({"image_filename": str(Path(image_path).resolve()),
                              "mask_filename": str(Path(mask_path).resolve()), "anomaly_type": anomaly_type,
                              "guidance": guidance, "num_steps": num_steps, "crop_ratio": crop_ratio,
                              "num_generated_images": 1, "poisson_blend": False, "seed": seed}) + "\n")
    out_dir = work / "out"
    cmd = [str(root / ".venv" / "bin" / "torchrun"), "--nproc_per_node=1", "anomalygen/scripts/texture/generate.py",
           "--checkpoint", str(ck), "--recipe", str(rc), "--input_data_path", str(tc), "--output_dir", str(out_dir),
           "--no-guardrail"]
    env = dict(os.environ)
    if os.environ.get("COSMOS_ANOMALYGEN_GPU"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["COSMOS_ANOMALYGEN_GPU"]
    log = work / "generate.log"
    with open(log, "w") as lf:
        rc_ = subprocess.run(cmd, cwd=root, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
    produced = sorted((out_dir / "reconstructed_image").glob(f"{anomaly_type}_*.png"))
    if rc_ != 0 or not produced:
        raise BackendError(f"anomalygen generate.py failed (rc={rc_}); see {log}")
    return produced[0]


def synth_defect_meta(image_path: str, defect: str, mask_path: Optional[str] = None, seed: int = 0, *,
                      guidance: float = 3.0, num_steps: int = 35, crop_ratio: float = 2.0) -> dict:
    """Paint a ``defect`` (porosity|crack|burn_through|lack_of_fusion|undercut) into a weld still."""
    t0 = time.perf_counter()
    s = get_settings()
    ms = lambda: int(round((time.perf_counter() - t0) * 1000))  # noqa: E731
    anomaly_type = WELD_TO_ANOMALYGEN.get(defect)
    if not anomaly_type:
        raise ValueError(f"unsupported defect {defect!r}; one of {sorted(WELD_TO_ANOMALYGEN)}")
    key = cache.make_key(FN_ANOMALY, {"image": cache.FileRef(image_path),
                                      "mask": cache.FileRef(mask_path) if mask_path else None,
                                      "defect": defect, "seed": seed, "guidance": guidance, "steps": num_steps,
                                      "crop_ratio": crop_ratio})
    if cache.reading_enabled():
        hit = cache.get_media(FN_ANOMALY, key, suffix=".png")
        if hit:
            return {**hit[1], "path": str(hit[0]), "source": SOURCE_CACHE, "origin_source": hit[1].get("source"),
                    "latency_ms": ms()}
    work = s.cache_dir / "anomaly_runs" / key[:16]
    mask = mask_path or str(auto_mask(image_path, defect, work / "mask.png", seed))
    error = None
    if anomalygen_available(s):
        try:
            png = _run_anomalygen(image_path, mask, anomaly_type, seed, s, work, guidance, num_steps, crop_ratio)
            meta = {"source": SOURCE_LOCAL, "model": "nvidia/Cosmos-AnomalyGen-Metal-Nano", "anomaly_type": anomaly_type,
                    "defect": defect, "mask": mask, "latency_ms": ms(),
                    "note": "magnetic-tile LoRA applied to a weld still (domain-shift approximation)"}
            dst = cache.put_media(FN_ANOMALY, key, png, meta, suffix=".png") if cache.writing_enabled() else png
            return {**meta, "path": str(dst), "cache_key": key}
        except (BackendError, OSError, subprocess.TimeoutExpired) as e:
            error = str(e)
            if not s.fallback:
                raise
    ph = cache.placeholder_path(key, ".png")
    if not ph.exists():
        paint_procedural(image_path, mask, defect, ph, seed)
    return {"path": str(ph), "source": SOURCE_HEURISTIC, "placeholder": True, "defect": defect,
            "anomaly_type": anomaly_type, "mask": mask, "latency_ms": ms(), "error": error, "cache_key": key}


def synth_defect(image_path: str, defect: str, mask_path: Optional[str] = None, seed: int = 0, **kw) -> str:
    """Returns the PNG path of the defect-painted image (AnomalyGen, cache, or labelled placeholder)."""
    return synth_defect_meta(image_path, defect, mask_path, seed, **kw)["path"]
