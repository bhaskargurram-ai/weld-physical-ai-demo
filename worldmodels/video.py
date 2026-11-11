"""Small video utilities: probing, frame extraction, placeholder rendering, H.264.

Only depends on PyAV / OpenCV / numpy (present in the project venv); never on a GPU.
"""
from __future__ import annotations

import base64
import mimetypes
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    codec: str

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.base_rate or 24.0)
        frames = s.frames or 0
        if not frames:
            frames = sum(1 for _ in c.decode(s))
        return VideoInfo(s.codec_context.width, s.codec_context.height, fps, int(frames), s.codec_context.name)


def frames_4k1(n: int, minimum: int = 5) -> int:
    """Largest 4k+1 frame count <= n (Cosmos3 VAE temporal stride is 4)."""
    n = max(int(n), minimum)
    return max(minimum, ((n - 1) // 4) * 4 + 1)


def frames_for_seconds(seconds: float, fps: float) -> int:
    """4k+1 frame count closest to ``seconds * fps`` (e.g. 5 s @ 24 fps -> 121)."""
    target = max(1.0, float(seconds) * float(fps))
    k = max(1, round((target - 1) / 4))
    return 4 * k + 1


def extract_frame(video: str | Path, out_png: str | Path, which: str = "first") -> Path:
    """Write the first / middle / last frame of ``video`` to ``out_png``."""
    import av
    from PIL import Image

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as c:
        if which == "first":
            frame = next(c.decode(video=0))
        else:
            frames = list(c.decode(video=0))
            frame = frames[-1] if which == "last" else frames[len(frames) // 2]
        Image.fromarray(frame.to_ndarray(format="rgb24")).save(out)
    return out


def read_frames(video: str | Path, max_frames: Optional[int] = None) -> tuple[list[np.ndarray], float]:
    import av

    out: list[np.ndarray] = []
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 24.0)
        for f in c.decode(s):
            out.append(f.to_ndarray(format="rgb24"))
            if max_frames and len(out) >= max_frames:
                break
    return out, fps


def write_mp4(frames: Sequence[np.ndarray], out: str | Path, fps: float = 24.0) -> Path:
    """Encode RGB frames to H.264/yuv420p MP4 (browser-playable)."""
    import av

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    tmp = out.with_name(f".tmp-{out.name}")
    with av.open(str(tmp), "w") as c:
        codec = "libx264" if "libx264" in av.codecs_available else "h264"
        s = c.add_stream(codec, rate=int(round(fps)))
        s.width, s.height, s.pix_fmt = w2, h2, "yuv420p"
        for fr in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr[:h2, :w2, :3]), format="rgb24")
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    tmp.replace(out)
    return out


def _put_lines(img: np.ndarray, lines: list[str], y0: int, scale: float, color=(255, 255, 255)) -> int:
    import cv2

    y = y0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x = max(8, (img.shape[1] - tw) // 2)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(th * 1.9)
    return y


def render_placeholder(out: str | Path, title: str, subtitle: str = "", *, seconds: float = 3.0,
                       fps: float = 24.0, size: tuple[int, int] = (832, 480),
                       background: Optional[str | Path] = None, seed: int = 0) -> Path:
    """Deterministic, clearly-labelled stand-in clip for replay-mode cache misses.

    If ``background`` is a video (e.g. the control video of a transfer request)
    its frames are shown dimmed underneath the label, so the placeholder still
    relates to the request.
    """
    import cv2

    w, h = size
    n = max(2, int(round(seconds * fps)))
    bg_frames: list[np.ndarray] = []
    if background and Path(background).exists():
        try:
            if Path(background).suffix.lower() in (".png", ".jpg", ".jpeg"):
                from PIL import Image

                bg_frames = [np.asarray(Image.open(background).convert("RGB"))]
            else:
                bg_frames, src_fps = read_frames(background, max_frames=n)
        except Exception:
            bg_frames = []
    rng = np.random.default_rng(seed)
    base_color = rng.integers(20, 60, size=3)
    sub_lines = textwrap.wrap(subtitle, width=max(20, w // 16))[:6] if subtitle else []
    frames = []
    for i in range(n):
        if bg_frames:
            fr = cv2.resize(bg_frames[min(i, len(bg_frames) - 1)], (w, h))
            fr = (fr.astype(np.float32) * 0.45).astype(np.uint8)
        else:
            fr = np.zeros((h, w, 3), np.uint8)
            fr[:] = base_color
            # slow diagonal stripes so it is obviously synthetic and moving
            xs = (np.arange(w)[None, :] + np.arange(h)[:, None] + i * 3) % 80 < 40
            fr[xs] = np.clip(base_color + 18, 0, 255)
        cv2.rectangle(fr, (6, 6), (w - 7, h - 7), (255, 170, 0), 3)
        y = _put_lines(fr, ["REPLAY PLACEHOLDER", "run on GPU to generate"], int(h * 0.22), w / 900, (255, 200, 60))
        y = _put_lines(fr, textwrap.wrap(title, width=max(20, w // 20))[:2], y + 10, w / 1300)
        _put_lines(fr, sub_lines, y + 6, w / 1700, (210, 210, 210))
        frames.append(fr)
    return write_mp4(frames, out, fps)


def ensure_h264(path: str | Path) -> Path:
    """Transcode in place to H.264 if the codec is not already H.264 (best effort)."""
    p = Path(path)
    try:
        if probe(p).codec in ("h264",):
            return p
    except Exception:
        return p
    ffmpeg = shutil.which("ffmpeg")
    tmp = p.with_name(f".h264-{p.name}")
    try:
        if ffmpeg:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(p), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", "-preset", "veryfast", "-c:a", "aac", str(tmp)], check=True)
        else:
            frames, fps = read_frames(p)
            write_mp4(frames, tmp, fps)
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)
    return p


def data_url(path: str | Path, default_mime: str = "video/mp4") -> str:
    mime = mimetypes.guess_type(str(path))[0] or default_mime
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
