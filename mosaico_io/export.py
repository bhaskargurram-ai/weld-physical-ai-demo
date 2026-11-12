"""Export curation-ready clips (+ sidecar JSON) from weld passes.

    python -m mosaico_io.export --out data/export [--clip-s 5] [--sequences A B ...]

For each pass, windows are chosen around every defect / NDT indication (centred on it) plus
clean windows away from any defect. Each window gets:

    <out>/<clip_id>.json                     sidecar (contract in docs/PLAN.md + media_type, photos, source, ndt)
    <out>/<clip_id>.mp4                      camera clip at the source frame rate (passes with video only)
    <out>/control/<clip_id>_edge.mp4         Cosmos Transfer controls, same window (when the pass has them)
    <out>/control/<clip_id>_depth.mp4
    <out>/photos/<seq>/*.jpg                 post-weld bead photos (passes with photos, e.g. AIMEN SAW)

Passes without video (AIMEN SAW) produce photo-only sidecars (``media_type: "image"``, ``video: null``).

Signals, stats and WPS findings come from Mosaico (or local files offline). Video
comes from the pass's torch_cam.mp4 when present (full 24 fps), otherwise it is
rebuilt from the JPEG frames of the Mosaico /camera/torch/image topic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

from collect.schema import CAMERA_FPS, CLIP_SECONDS, TOPIC_CAMERA
from mosaico_io.common import (EXTRA_SIGNALS, SIGNALS, TOPIC_BEAD_PHOTO, clean_json, generated_dir, is_offline,
                               json_default, meta_row, ndt_indications, source_info)
from mosaico_io.query import get_meta, get_signals, list_passes, violations_from_signals


def _clean_windows(lo: float, hi: float, clip_s: float, defects: list[dict], wins: list[dict], n_clean: int) -> list[dict]:
    cands = np.arange(lo, hi + 1e-9, 0.5)
    ok = [t for t in cands if all(t + clip_s < d["t_start"] - 0.5 or t > d["t_end"] + 0.5 for d in defects)]
    out: list[dict] = []
    if ok and n_clean:
        picks = np.linspace(0, len(ok) - 1, n_clean + 2)[1:-1] if n_clean > 1 else [len(ok) // 3]
        for p in picks:
            # nearest allowed candidate to the preferred position that does not overlap another window
            for t0 in sorted(ok, key=lambda t: abs(t - ok[int(round(p))])):
                if all(abs(t0 - w["t0_s"]) >= clip_s for w in wins + out):
                    out.append({"t0_s": round(float(t0), 2), "kind": "clean", "defect": None})
                    break
    return out


def choose_windows(meta: dict, clip_s: float = CLIP_SECONDS, n_clean: Optional[int] = None,
                   media: str = "video") -> list[dict]:
    """Windows centred on each defect / NDT indication (indications already covered by an earlier window are
    merged into it) plus clean windows away from any defect.

    ``media="image"`` (pass without video, e.g. AIMEN SAW: the evidence is the post-weld bead photo of the
    whole bead) -> indication windows only, or one clean window when the pass has no indication: a "clean"
    sample from a bead that has indications elsewhere would carry a photo that shows them."""
    D = float(meta.get("duration_s") or 0.0)
    lo, hi = 0.5, max(0.5, D - clip_s - 0.5)
    defects = sorted((d for d in meta.get("defects", []) if d.get("t_start") is not None),
                     key=lambda d: d["t_start"])
    wins: list[dict] = []
    for d in defects:
        last = wins[-1] if wins else None
        if last and last["kind"] == "defect" and d["t_start"] >= last["t0_s"] and d["t_end"] <= last["t0_s"] + clip_s:
            if d["defect"] not in last["defects"]:
                last["defects"].append(d["defect"])
            continue
        c = 0.5 * (d["t_start"] + d["t_end"])
        t0 = float(np.clip(c - clip_s / 2, lo, hi))
        wins.append({"t0_s": round(t0, 2), "kind": "defect", "defect": d["defect"], "defects": [d["defect"]]})
    if media == "image":
        n_clean = 0 if defects else 1
    elif n_clean is None:
        n_clean = 1 if defects else 2
    wins += _clean_windows(lo, hi, clip_s, defects, wins, n_clean)
    for w in wins:
        w.pop("defects", None)
    return sorted(wins, key=lambda w: w["t0_s"])


def clip_id_for(seq: str, t0_s: float) -> str:
    return f"{seq}_t{int(round(t0_s * 10)):04d}"


def _rate(fps: float) -> Fraction:
    return Fraction(float(fps)).limit_denominator(1001)


class _Enc:
    def __init__(self, path: Path, fps: float, w: int, h: int, crf: int = 20):
        import av
        path.parent.mkdir(parents=True, exist_ok=True)
        rate = _rate(fps)
        self.w, self.h = w // 2 * 2, h // 2 * 2
        self.c = av.open(str(path), "w")
        self.s = self.c.add_stream("libx264", rate=rate)
        self.s.width, self.s.height, self.s.pix_fmt = self.w, self.h, "yuv420p"
        self.s.options = {"crf": str(crf), "preset": "veryfast", "movflags": "+faststart"}
        self.s.codec_context.time_base = 1 / rate
        self.n = 0

    def write(self, frame):
        import av
        if isinstance(frame, av.VideoFrame):
            frame = frame.reformat(width=self.w, height=self.h, format="yuv420p")
        else:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[:self.h, :self.w]), format="rgb24") \
                .reformat(format="yuv420p")
        frame.pts = self.n
        frame.time_base = self.s.codec_context.time_base
        for p in self.s.encode(frame):
            self.c.mux(p)
        self.n += 1

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()


def _cut_local(src: Path, windows: list[tuple[float, float, Path]], fps: Optional[float] = None,
               t_offset_s: float = 0.0) -> list[int]:
    """Single decode pass over src; frames whose pass time (frame time + t_offset_s) is in [t0, t1) go to each
    window's output, encoded at the source frame rate."""
    import av
    if not windows:
        return []
    encs: dict[int, _Enc] = {}
    counts = [0] * len(windows)
    t_end = max(w[1] for w in windows)
    with av.open(str(src)) as c:
        st = c.streams.video[0]
        rate = float(fps or st.average_rate or CAMERA_FPS)
        for k, fr in enumerate(c.decode(st)):
            t = (float(fr.time) if fr.time is not None else k / rate) + t_offset_s
            if t >= t_end:
                break
            for i, (a, b, out) in enumerate(windows):
                if a - 1e-6 <= t < b - 1e-6:
                    if i not in encs:
                        encs[i] = _Enc(out, rate, fr.width, fr.height)
                    encs[i].write(fr)
                    counts[i] += 1
    for e in encs.values():
        e.close()
    for (_, _, out), n in zip(windows, counts):
        if 0 < n < 2 and out.exists():     # a single frame is not a clip
            out.unlink()
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
    for c in (*SIGNALS, *EXTRA_SIGNALS):
        if c in df and len(df):
            x = df[c].to_numpy(dtype=float)
            if not np.isfinite(x).any():
                continue                      # not recorded (gas on SAW, trail wire on a single-wire pass)
            out[c] = {"mean": round(float(np.nanmean(x)), 4), "std": round(float(np.nanstd(x)), 4),
                      "min": round(float(np.nanmin(x)), 4), "max": round(float(np.nanmax(x)), 4)}
    return out


def _video_info(meta: dict, local: Path) -> dict:
    v = meta.get("video") if isinstance(meta.get("video"), dict) else {}
    out = {"file": v.get("file") or "torch_cam.mp4", "kind": v.get("kind"), "t_offset_s": float(v.get("t_offset_s") or 0.0),
           "fps": v.get("fps")}
    if not out["kind"]:
        out["kind"] = "simulated_torch_camera" if source_info(meta)["synthetic"] else "process_camera"
    if out["fps"] is None:
        out["fps"] = float(CAMERA_FPS) if source_info(meta)["synthetic"] else None
    return out


def _probe_fps(path: Path) -> Optional[float]:
    try:
        import av
        with av.open(str(path)) as c:
            return float(c.streams.video[0].average_rate)
    except Exception:  # noqa: BLE001
        return None


def _export_photos(seq: str, meta: dict, local: Path, out: Path) -> list[str]:
    """Copy the pass's bead photos to <out>/photos/<seq>/ (from the pass dir, or from the Mosaico
    /camera/bead/photo topic when the pass dir is not local). Returns paths relative to <out>."""
    dst_dir = out / "photos" / seq
    rels: list[str] = []
    srcs = [local / p for p in meta.get("photos") or [] if (local / p).exists()]
    if srcs:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in srcs:
            dst = dst_dir / p.name
            if not dst.exists() or dst.stat().st_size != p.stat().st_size:
                shutil.copyfile(p, dst)
            rels.append(dst.relative_to(out).as_posix())
        return rels
    if meta.get("photos") and not is_offline():
        try:
            from mosaicolabs import CompressedImage
            from mosaico_io.common import connect
            th = connect().topic_handler(seq, TOPIC_BEAD_PHOTO)
            if th is None:
                return []
            names = list(th.user_metadata.get("files") or [])
            dst_dir.mkdir(parents=True, exist_ok=True)
            for k, msg in enumerate(th.get_data_streamer()):
                img = msg.get_data(CompressedImage) if hasattr(msg, "get_data") else msg.data
                name = Path(names[k]).name if k < len(names) else f"{seq}_photo_{k}.jpg"
                (dst_dir / name).write_bytes(img.data)
                rels.append((dst_dir / name).relative_to(out).as_posix())
        except Exception as e:  # noqa: BLE001 - photos are optional evidence
            print(f"  ! {seq}: bead photos not exported from Mosaico ({e})", file=sys.stderr)
    return rels


def _photo_note(meta: dict, dwin: list[dict]) -> Optional[str]:
    if not meta.get("photos"):
        return None
    xs = [f"x={d['x_start_mm']:.1f}-{d['x_end_mm']:.1f} mm" for d in dwin
          if isinstance(d.get("x_start_mm"), (int, float)) and isinstance(d.get("x_end_mm"), (int, float))]
    note = "Post-weld photo(s) of the complete bead from the source dataset; not time-aligned to this window."
    if xs:
        note += " NDT indication position along the weld: " + ", ".join(xs) + "."
    return note


def select_sequences(data_source: Optional[str] = None) -> list[str]:
    """Pass names for DATA_SOURCE: ``real`` (non-synthetic sources), ``sim`` (simulator), ``all``/None."""
    rows = list_passes()
    if data_source in ("real", "sim"):
        want_synth = data_source == "sim"
        rows = [r for r in rows if bool((r.get("source") or {}).get("synthetic", False)) == want_synth]
    return [r["sequence_name"] for r in rows]


def export_clips(out_dir, sequences: Optional[list[str]] = None, clip_s: float = CLIP_SECONDS,
                 include_control: bool = True, quiet: bool = True, data_source: Optional[str] = None) -> list[dict]:
    """Write <out>/<clip_id>.mp4 + .json sidecars for defect / NDT-indication and clean windows.

    Passes with video (simulator, BAM) -> 5 s clips cut from the real video at its own frame rate.
    Passes without video (AIMEN SAW) -> no mp4; a sidecar per indication window (``media_type: "image"``)
    referencing the bead photo(s) copied to <out>/photos/<seq>/. Returns the sidecars."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if sequences is None:
        sequences = select_sequences(data_source)
    results = []
    for seq in sequences:
        meta = get_meta(seq)
        df_all = get_signals(seq)
        viol = violations_from_signals(df_all, meta)
        local = generated_dir() / seq
        vinfo = _video_info(meta, local)
        vfile = local / vinfo["file"]
        has_local_video = vfile.exists()
        topics = {t.get("topic") for t in meta.get("topics") or [] if isinstance(t, dict)}
        has_video = has_local_video or (not is_offline() and (TOPIC_CAMERA in topics or (
            not topics and source_info(meta)["synthetic"])))
        if has_local_video and not vinfo["fps"]:
            vinfo["fps"] = _probe_fps(vfile)
        photos = _export_photos(seq, meta, local, out)
        media = "video" if has_video else ("image" if photos else "none")
        wins = choose_windows(meta, clip_s, media=media)
        src_info = source_info(meta)
        inds = ndt_indications(meta)
        ndt = meta.get("ndt") if isinstance(meta.get("ndt"), dict) else {}
        jobs = {"rgb": [], "edge": [], "depth": []}
        sidecars = []
        for w in wins:
            t0, t1 = w["t0_s"], round(w["t0_s"] + clip_s, 2)
            cid = clip_id_for(seq, t0)
            df = df_all[(df_all["t_s"] >= t0) & (df_all["t_s"] < t1)]
            dwin = []
            for d in meta.get("defects", []):
                if not (d["t_start"] < t1 and d["t_end"] > t0):
                    continue
                e = {"defect": d["defect"], "severity": d.get("severity"), "t_start_s": d["t_start"],
                     "t_end_s": d["t_end"],
                     "overlap_s": round(max(0.0, min(t1, d["t_end"]) - max(t0, d["t_start"])), 2)}
                ind = next((i for i in inds if abs(float(i.get("t_start", -1e9)) - float(d["t_start"])) < 1e-3
                            and (i.get("type") or i.get("defect")) == d["defect"]), None)
                if ind:
                    e.update({k: ind[k] for k in ("x_start_mm", "x_end_mm") if k in ind})
                    e["source"] = "NDT"
                dwin.append(e)
            vwin = [v for v in viol if v["t_start_s"] < t1 and v["t_end_s"] > t0]
            sc = {"clip_id": cid, "sequence_name": seq, "t0_s": t0, "t1_s": t1,
                  "meta": {**meta_row(meta), "wps": meta.get("wps", {})},
                  "signal_stats": _stats(df), "wps_violations": vwin, "defects_in_window": dwin,
                  "rt_result": meta.get("rt_result"), "joint_rt_result": meta.get("joint_rt_result"),
                  "label": "defect" if dwin else "clean", "window_kind": w["kind"],
                  "media_type": media, "video": f"{cid}.mp4" if has_video else None,
                  "fps": vinfo["fps"] if has_video else None,
                  "video_kind": vinfo["kind"] if has_video else None,
                  "photos": photos, "photo_note": _photo_note(meta, dwin) if photos else None,
                  "electrodes": int(meta.get("electrodes") or 1),
                  "ndt": {"methods": ndt.get("methods") or [], "inspected": bool(ndt.get("inspected", bool(ndt))),
                          "report_files": ndt.get("report_files") or [],
                          "indications_in_window": [d for d in dwin if d.get("source") == "NDT"]},
                  "source": src_info,
                  "read_from": "local" if is_offline() else "mosaico"}
            if include_control and has_video:
                sc["control"] = {"edge": f"control/{cid}_edge.mp4", "depth": f"control/{cid}_depth.mp4"}
            jobs["rgb"].append((t0, t1, out / f"{cid}.mp4"))
            jobs["edge"].append((t0, t1, out / "control" / f"{cid}_edge.mp4"))
            jobs["depth"].append((t0, t1, out / "control" / f"{cid}_depth.mp4"))
            sidecars.append(sc)
        # video
        src = None
        if has_local_video:
            _cut_local(vfile, jobs["rgb"], vinfo["fps"], vinfo["t_offset_s"])
            src = vinfo["file"]
        elif has_video:
            for t0, t1, p in jobs["rgb"]:
                _cut_mosaico(seq, int(meta["t0_ns"]), t0, t1, p)
            src = f"mosaico:{TOPIC_CAMERA}"
        if include_control and has_video:
            for kind in ("edge", "depth"):
                f = local / "control" / f"{kind}.mp4"
                if f.exists():
                    _cut_local(f, jobs[kind], vinfo["fps"], vinfo["t_offset_s"])
                for s in sidecars:
                    if not (out / s["control"].get(kind, "")).is_file():
                        s["control"].pop(kind, None)
        for s in sidecars:
            s["video_source"] = src
            if src is None or not s["video"] or not (out / s["video"]).exists():
                s["video"] = None
                if s["media_type"] == "video":
                    s["media_type"] = "image" if s["photos"] else "none"
            with open(out / f"{s['clip_id']}.json", "w") as f:
                json.dump(clean_json(s), f, indent=2, default=json_default)
            results.append(s)
            if not quiet:
                print(f"  {s['clip_id']:<28} {s['media_type']:<5} {s['label']:<6} "
                      f"{','.join(d['defect'] for d in s['defects_in_window']) or '-':<16}"
                      f" viol={len(s['wps_violations'])} RT={s['rt_result']}", flush=True)
    with open(out / "clips.jsonl", "w") as f:
        for s in results:
            f.write(json.dumps({k: s.get(k) for k in ("clip_id", "sequence_name", "t0_s", "t1_s", "label",
                                                      "rt_result", "media_type", "video", "photos")},
                               default=json_default) + "\n")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export 5 s clips + sidecars for curation")
    ap.add_argument("--out", default=os.path.join(os.environ.get("DATA_DIR", "data"), "export"))
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--clip-s", type=float, default=CLIP_SECONDS)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--data-source", default=os.environ.get("DATA_SOURCE", "all"), choices=["real", "sim", "all"],
                    help="which passes to export when --sequences is not given (default DATA_SOURCE or all)")
    a = ap.parse_args(argv)
    t = time.time()
    res = export_clips(a.out, a.sequences, a.clip_s, include_control=not a.no_control, quiet=False,
                       data_source=a.data_source)
    n_def = sum(r["label"] == "defect" for r in res)
    n_img = sum(r["media_type"] == "image" for r in res)
    print(f"done: {len(res)} windows ({n_def} defect, {len(res) - n_def} clean; {len(res) - n_img} video clips, "
          f"{n_img} photo-only) -> {a.out} "
          f"[{'offline' if is_offline() else 'mosaicod'}] in {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
