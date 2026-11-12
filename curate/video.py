"""Video I/O (probe, cut+transcode to H.264) and CPU quality metrics / embeddings."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from curate.common import log, normalize_signal_stats

ANALYSIS_WIDTH = 320
EMBED_FRAMES = 4
EMBED_HW = (9, 16)
EMBED_SIGNALS = ("current_a", "voltage_v", "wfs_mpm", "gas_lpm", "travel_speed_mmps",
                 "heat_input_kj_mm", "interpass_c")
# rough full-scale for each signal, so the signal part of the embedding is ~unit range
_SIGNAL_SCALE = {"current_a": 300.0, "voltage_v": 35.0, "wfs_mpm": 12.0, "gas_lpm": 25.0,
                 "travel_speed_mmps": 12.0, "heat_input_kj_mm": 2.5, "interpass_c": 300.0}
EMBED_DIM = EMBED_FRAMES * EMBED_HW[0] * EMBED_HW[1] + 16 + 2 * len(EMBED_SIGNALS)
EMBED_NAME = "weldlite"


@dataclass
class Probe:
    duration_s: float
    fps: float
    frames: int
    width: int
    height: int


def probe(path: str | Path) -> Probe:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or n <= 0:
        # fall back to PyAV for containers OpenCV cannot size
        import av
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate or 24)
            dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
            n = s.frames or int(round(dur * fps))
            w, h = s.codec_context.width, s.codec_context.height
    return Probe(duration_s=n / fps if fps else 0.0, fps=float(fps), frames=n, width=w, height=h)


def plan_spans(duration_s: float, clip_s: float, stride_s: float, min_s: float,
               whole_tolerance_s: float = 0.6) -> list[tuple[float, float]]:
    """Fixed-stride spans (Curator semantics); whole clip if it is ~one clip long."""
    if duration_s <= 0:
        return []
    if duration_s <= clip_s + whole_tolerance_s:
        return [(0.0, duration_s)] if duration_s >= min_s else []
    spans, start = [], 0.0
    while start < duration_s - 1e-6:
        end = min(start + clip_s, duration_s)
        if end - start >= min_s:
            spans.append((round(start, 6), round(end, 6)))
        start += stride_s
    return spans


def transcode(src: str | Path, dst: str | Path, start_s: float | None = None,
              dur_s: float | None = None, crf: int = 20) -> None:
    """Cut [start, start+dur) and encode H.264 yuv420p mp4 (ffmpeg CLI, PyAV fallback)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if start_s:
            cmd += ["-ss", f"{start_s:.3f}"]
        cmd += ["-i", str(src)]
        if dur_s:
            cmd += ["-t", f"{dur_s:.3f}"]
        cmd += ["-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags", "+faststart", "-an", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return
        log.warning("ffmpeg failed for %s (%s); falling back to PyAV", src, r.stderr.strip()[-300:])
    _transcode_pyav(src, dst, start_s or 0.0, dur_s, crf)


def _transcode_pyav(src, dst, start_s: float, dur_s: float | None, crf: int) -> None:
    import av
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        ist = inp.streams.video[0]
        rate = ist.average_rate or Fraction(24, 1)
        ost = out.add_stream("libx264", rate=rate)
        ost.width = ist.codec_context.width // 2 * 2
        ost.height = ist.codec_context.height // 2 * 2
        ost.pix_fmt = "yuv420p"
        ost.options = {"crf": str(crf), "preset": "veryfast"}
        end = start_s + dur_s if dur_s else float("inf")
        for frame in inp.decode(ist):
            t = float(frame.pts * ist.time_base) if frame.pts is not None else 0.0
            if t < start_s - 1e-6:
                continue
            if t >= end - 1e-6:
                break
            img = frame.to_ndarray(format="rgb24")[: ost.height, : ost.width]
            for pkt in ost.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in ost.encode():
            out.mux(pkt)


def _read_gray_frames(path: str | Path, max_frames: int = 48) -> tuple[list[np.ndarray], float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, n // max_frames) if n > 0 else 1
    frames, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok and img is not None:
                h, w = img.shape[:2]
                if w > ANALYSIS_WIDTH:
                    img = cv2.resize(img, (ANALYSIS_WIDTH, max(2, round(h * ANALYSIS_WIDTH / w))),
                                     interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return frames, float(fps), step


def analyze(path: str | Path) -> tuple[dict, np.ndarray]:
    """Quality metrics + visual embedding part for one clip.

    blur        median variance of the Laplacian (higher = sharper)
    motion      mean |frame difference| between sampled frames, per source frame, 0-255 scale
    glare_frac  mean fraction of saturated pixels (>=250) - arc glare / blown highlights
    dark_frac   mean fraction of near-black pixels (<16)
    mean_luma   mean grey level 0-255
    arc_on_frac fraction of sampled frames with a visible arc (>=0.05 % of pixels at grey >=220; the arc core
                is a small spot in a dark torch-cam frame, measured 0.16-0.74 % on arc-on frames, ~0 when off)
    """
    frames, fps, step = _read_gray_frames(path)
    if len(frames) < 2:
        raise IOError(f"{path}: fewer than 2 decodable frames")
    lap = [float(cv2.Laplacian(f, cv2.CV_64F).var()) for f in frames]
    diffs = [float(np.mean(cv2.absdiff(a, b))) / step for a, b in zip(frames, frames[1:])]
    sat = [float(np.mean(f >= 250)) for f in frames]
    bright = [float(np.mean(f >= 220)) for f in frames]
    dark = [float(np.mean(f < 16)) for f in frames]
    q = {
        "blur": round(float(np.median(lap)), 3),
        "motion": round(float(np.mean(diffs)), 4),
        "glare_frac": round(float(np.mean(sat)), 4),
        "dark_frac": round(float(np.mean(dark)), 4),
        "mean_luma": round(float(np.mean([f.mean() for f in frames])), 2),
        "arc_on_frac": round(float(np.mean([b >= 0.0005 for b in bright])), 3),
    }
    # visual embedding: EMBED_FRAMES evenly spaced frames at 16x9, per-frame standardised, + luma hist
    idx = np.linspace(0, len(frames) - 1, EMBED_FRAMES).round().astype(int)
    parts = []
    for i in idx:
        small = cv2.resize(frames[i], (EMBED_HW[1], EMBED_HW[0]), interpolation=cv2.INTER_AREA)
        v = small.astype(np.float32).ravel()
        parts.append((v - v.mean()) / (v.std() + 1e-3))
    hist = np.histogram(np.concatenate([f.ravel() for f in frames[:: max(1, len(frames) // 8)]]),
                        bins=16, range=(0, 256))[0].astype(np.float32)
    hist = hist / (hist.sum() + 1e-6) * 8.0
    visual = np.concatenate(parts + [hist])
    return q, visual


def embed(visual: np.ndarray, signal_stats: dict | None, w_signal: float = 0.35) -> np.ndarray:
    """L2-normalised clip embedding = visual part (+) weighted sensor-stat part."""
    v = visual / (np.linalg.norm(visual) + 1e-9)
    stats = normalize_signal_stats(signal_stats or {})
    sig = []
    for s in EMBED_SIGNALS:
        d = stats.get(s, {})
        sc = _SIGNAL_SCALE[s]
        m, sd = d.get("mean", 0.0), d.get("std", 0.0)
        sig += [0.0 if m != m else m / sc, 0.0 if sd != sd else 5.0 * sd / sc]
    sv = np.asarray(sig, dtype=np.float32)
    n = np.linalg.norm(sv)
    sv = sv / n if n > 0 else sv
    out = np.concatenate([(1 - w_signal) * v, w_signal * sv]).astype(np.float32)
    return out / (np.linalg.norm(out) + 1e-9)
