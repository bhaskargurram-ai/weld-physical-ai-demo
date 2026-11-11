"""Export curation-ready clips (+ sidecar JSON) from weld passes.

    python -m mosaico_io.export --out data/export [--clip-s 5] [--sequences A B ...]

For each pass, windows are chosen around every defect (centred on it) plus clean
windows away from any defect. Each clip gets:

    <out>/<clip_id>.mp4                      torch camera, 24 fps H.264
    <out>/<clip_id>.json                     sidecar (contract in docs/PLAN.md)
    <out>/control/<clip_id>_edge.mp4         Cosmos Transfer controls, same window
    <out>/control/<clip_id>_depth.mp4

Signals, stats and WPS findings come from Mosaico (or local files offline). Video
comes from the pass's torch_cam.mp4 when present (full 24 fps), otherwise it is
rebuilt from the JPEG frames of the Mosaico /camera/torch/image topic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from collect.schema import CAMERA_FPS, CLIP_SECONDS, TOPIC_CAMERA
from mosaico_io.common import SIGNALS, generated_dir, is_offline, json_default, meta_row
from mosaico_io.query import get_meta, get_signals, list_passes, violations_from_signals


def choose_windows(meta: dict, clip_s: float = CLIP_SECONDS, n_clean: Optional[int] = None) -> list[dict]:
    D = float(meta["duration_s"])
    lo, hi = 0.5, max(0.5, D - clip_s - 0.5)
    defects = meta.get("defects", [])
    wins = []
    for d in defects:
        c = 0.5 * (d["t_start"] + d["t_end"])
        t0 = float(np.clip(c - clip_s / 2, lo, hi))
        wins.append({"t0_s": round(t0, 2), "kind": "defect", "defect": d["defect"]})
    if n_clean is None:
        n_clean = 1 if defects else 2
    cands = np.arange(lo, hi + 1e-9, 0.5)
    ok = [t for t in cands if all(t + clip_s < d["t_start"] - 0.5 or t > d["t_end"] + 0.5 for d in defects)]
    if ok and n_clean:
        picks = np.linspace(0, len(ok) - 1, n_clean + 2)[1:-1] if n_clean > 1 else [len(ok) // 3]
        for p in picks:
            t0 = float(ok[int(round(p))])
            if all(abs(t0 - w["t0_s"]) >= clip_s for w in wins):
                wins.append({"t0_s": round(t0, 2), "kind": "clean", "defect": None})
    return sorted(wins, key=lambda w: w["t0_s"])


def clip_id_for(seq: str, t0_s: float) -> str:
    return f"{seq}_t{int(round(t0_s * 10)):04d}"


class _Enc:
    def __init__(self, path: Path, fps: float, w: int, h: int, crf: int = 20):
        import av
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(path), "w")
        self.s = self.c.add_stream("libx264", rate=int(round(fps)))
        self.s.width, self.s.height, self.s.pix_fmt = w, h, "yuv420p"
        self.s.options = {"crf": str(crf), "preset": "veryfast", "movflags": "+faststart"}
        self.n = 0

    def write(self, frame):
        import av
        if not isinstance(frame, av.VideoFrame):
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        frame.pts = None
        for p in self.s.encode(frame.reformat(format="yuv420p")):
            self.c.mux(p)
        self.n += 1

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()


def _cut_local(src: Path, windows: list[tuple[float, float, Path]], fps: int = CAMERA_FPS) -> list[int]:
    """Single decode pass over src, frames [t0, t1) go to each window's output."""
    import av
    if not windows:
        return []
    encs: dict[int, _Enc] = {}
    counts = [0] * len(windows)
    t_end = max(w[1] for w in windows)
    with av.open(str(src)) as c:
        st = c.streams.video[0]
        for k, fr in enumerate(c.decode(st)):
            t = k / fps
            if t >= t_end:
                break
            for i, (a, b, out) in enumerate(windows):
                if a - 1e-6 <= t < b - 1e-6:
                    if i not in encs:
                        encs[i] = _Enc(out, fps, fr.width, fr.height)
                    encs[i].write(fr)
                    counts[i] += 1
    for e in encs.values():
        e.close()
    return counts


def _cut_mosaico(seq: str, t0_ns: int, a: float, b: float, out: Path) -> int:
    """Rebuild a clip from the JPEG frames in the Mosaico camera topic."""
    import cv2
    from mosaicolabs import CompressedImage
    from mosaico_io.common import connect
    th = connect().topic_handler(seq, TOPIC_CAMERA)
    if th is None:
        return 0
    fps = float(th.user_metadata.get("fps", 12))
    enc = None
    n = 0
    for msg in th.get_data_streamer(start_timestamp_ns=int(t0_ns + a * 1e9), end_timestamp_ns=int(t0_ns + b * 1e9)):
        img = msg.get_data(CompressedImage) if hasattr(msg, "get_data") else msg.data
        arr = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        if enc is None:
            enc = _Enc(out, fps, arr.shape[1], arr.shape[0])
        enc.write(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        n += 1
    if enc:
        enc.close()
    return n


def _stats(df) -> dict:
    out = {}
    for c in SIGNALS:
        if c in df and len(df):
            x = df[c].to_numpy(dtype=float)
            out[c] = {"mean": round(float(np.nanmean(x)), 4), "std": round(float(np.nanstd(x)), 4),
                      "min": round(float(np.nanmin(x)), 4), "max": round(float(np.nanmax(x)), 4)}
    return out


def export_clips(out_dir, sequences: Optional[list[str]] = None, clip_s: float = CLIP_SECONDS,
                 include_control: bool = True, quiet: bool = True) -> list[dict]:
    """Write <out>/<clip_id>.mp4 + .json sidecars for defect and clean windows. Returns sidecars."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if sequences is None:
        sequences = [r["sequence_name"] for r in list_passes()]
    results = []
    for seq in sequences:
        meta = get_meta(seq)
        df_all = get_signals(seq)
        viol = violations_from_signals(df_all, meta)
        wins = choose_windows(meta, clip_s)
        local = generated_dir() / seq
        jobs = {"rgb": [], "edge": [], "depth": []}
        sidecars = []
        for w in wins:
            t0, t1 = w["t0_s"], round(w["t0_s"] + clip_s, 2)
            cid = clip_id_for(seq, t0)
            df = df_all[(df_all["t_s"] >= t0) & (df_all["t_s"] < t1)]
            dwin = [{"defect": d["defect"], "severity": d["severity"],
                     "t_start_s": d["t_start"], "t_end_s": d["t_end"],
                     "overlap_s": round(max(0.0, min(t1, d["t_end"]) - max(t0, d["t_start"])), 2)}
                    for d in meta.get("defects", []) if d["t_start"] < t1 and d["t_end"] > t0]
            vwin = [v for v in viol if v["t_start_s"] < t1 and v["t_end_s"] > t0]
            sc = {"clip_id": cid, "sequence_name": seq, "t0_s": t0, "t1_s": t1,
                  "meta": {**meta_row(meta), "wps": meta.get("wps", {})},
                  "signal_stats": _stats(df), "wps_violations": vwin, "defects_in_window": dwin,
                  "rt_result": meta.get("rt_result"), "joint_rt_result": meta.get("joint_rt_result"),
                  "label": "defect" if dwin else "clean", "window_kind": w["kind"],
                  "video": f"{cid}.mp4", "fps": CAMERA_FPS,
                  "source": "local" if is_offline() else "mosaico"}
            if include_control:
                sc["control"] = {"edge": f"control/{cid}_edge.mp4", "depth": f"control/{cid}_depth.mp4"}
            jobs["rgb"].append((t0, t1, out / f"{cid}.mp4"))
            jobs["edge"].append((t0, t1, out / "control" / f"{cid}_edge.mp4"))
            jobs["depth"].append((t0, t1, out / "control" / f"{cid}_depth.mp4"))
            sidecars.append(sc)
        # video
        if (local / "torch_cam.mp4").exists():
            _cut_local(local / "torch_cam.mp4", jobs["rgb"])
            src = "torch_cam.mp4"
        elif not is_offline():
            for t0, t1, p in jobs["rgb"]:
                _cut_mosaico(seq, int(meta["t0_ns"]), t0, t1, p)
            src = f"mosaico:{TOPIC_CAMERA}"
        else:
            src = None
        if include_control:
            for kind in ("edge", "depth"):
                f = local / "control" / f"{kind}.mp4"
                if f.exists():
                    _cut_local(f, jobs[kind])
                else:
                    for s in sidecars:
                        s["control"].pop(kind, None)
        for s in sidecars:
            s["video_source"] = src
            if src is None or not (out / s["video"]).exists():
                s["video"] = None
            with open(out / f"{s['clip_id']}.json", "w") as f:
                json.dump(s, f, indent=2, default=json_default)
            results.append(s)
            if not quiet:
                print(f"  {s['clip_id']:<28} {s['label']:<6} {','.join(d['defect'] for d in s['defects_in_window']) or '-':<16}"
                      f" viol={len(s['wps_violations'])} RT={s['rt_result']}", flush=True)
    with open(out / "clips.jsonl", "w") as f:
        for s in results:
            f.write(json.dumps({k: s[k] for k in ("clip_id", "sequence_name", "t0_s", "t1_s", "label",
                                                  "rt_result", "video")}, default=json_default) + "\n")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export 5 s clips + sidecars for curation")
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "export"))
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--clip-s", type=float, default=CLIP_SECONDS)
    ap.add_argument("--no-control", action="store_true")
    a = ap.parse_args(argv)
    t = time.time()
    res = export_clips(a.out, a.sequences, a.clip_s, include_control=not a.no_control, quiet=False)
    n_def = sum(r["label"] == "defect" for r in res)
    print(f"done: {len(res)} clips ({n_def} defect, {len(res) - n_def} clean) -> {a.out} "
          f"[{'offline' if is_offline() else 'mosaicod'}] in {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
