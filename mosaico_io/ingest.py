"""Ingest weld pass directories (simulator or real data, docs/REAL_DATA.md) into Mosaico: one sequence per pass.

    python -m mosaico_io.ingest data/generated [--no-replace] [--camera-fps 12]

Topics (collect/schema.py); only arc + events are always present, the rest when the pass has the data:
    /weld/arc            WeldArc     (custom ontology) @100 Hz
    /weld/travel         WeldTravel  (custom ontology) @100 Hz
    /weld/thermal        WeldThermal (custom ontology) @10 Hz
    /robot/joint_states  RobotJoint  @50 Hz
    /robot/tcp_pose      Pose        @50 Hz
    /camera/torch/image  CompressedImage (JPEG) @MOSAICO_CAMERA_FPS (default 12; real video at its own fps)
    /camera/bead/photo   CompressedImage (JPEG) post-weld bead photos (AIMEN)
    /weld/events         String (JSON event payload)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from collect.schema import (ARC_HZ, CAMERA_FPS, ROBOT_HZ, THERMAL_HZ, TOPIC_ARC, TOPIC_CAMERA,
                            TOPIC_EVENTS, TOPIC_ROBOT_JOINTS, TOPIC_TCP_POSE, TOPIC_THERMAL, TOPIC_TRAVEL)
from mosaico_io.common import (TOPIC_BEAD_PHOTO, metadata_safe, connect, flatten_meta, load_meta, movavg, movmedian,
                               read_optional_parquet, source_info)
from mosaico_io.ontology import WeldArc, WeldThermal, WeldTravel


def _iter_jpeg_frames(video: Path, every: int, quality: int, max_width: int = 1280):
    """(frame_index, t_video_s, jpeg_bytes) for every ``every``-th frame."""
    import av
    import cv2
    with av.open(str(video)) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or CAMERA_FPS)
        for k, fr in enumerate(c.decode(st)):
            if k % every:
                continue
            bgr = fr.to_ndarray(format="bgr24")
            if bgr.shape[1] > max_width:
                bgr = cv2.resize(bgr, (max_width, round(bgr.shape[0] * max_width / bgr.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            t = float(fr.time) if fr.time is not None else k / fps
            if ok:
                yield k, t, buf.tobytes()


def video_fps(video: Path, meta: dict) -> float:
    v = meta.get("video") if isinstance(meta.get("video"), dict) else {}
    if v.get("fps"):
        return float(v["fps"])
    try:
        import av
        with av.open(str(video)) as c:
            return float(c.streams.video[0].average_rate or CAMERA_FPS)
    except Exception:  # noqa: BLE001
        return float(CAMERA_FPS)


def _rate_hz(ts: np.ndarray, default: float) -> float:
    if len(ts) < 3:
        return default
    dt = float(np.median(np.diff(ts))) / 1e9
    return round(1.0 / dt, 3) if dt > 0 else default


def _decimate(df: pd.DataFrame, max_hz: float) -> pd.DataFrame:
    hz = _rate_hz(df["t_ns"].to_numpy(), max_hz)
    step = max(1, int(round(hz / max_hz)))
    return df.iloc[::step].reset_index(drop=True) if step > 1 else df


def _f(x) -> Optional[float]:
    """float, NaN kept (Arrow double NaN is a value, not a null)."""
    return float(x)


def _opt_cols(df: Optional[pd.DataFrame], cols) -> list[str]:
    return [c for c in cols if df is not None and c in df.columns and df[c].notna().any()]


def ingest_pass_dir(path, client=None, replace: bool = True, camera_fps: Optional[int] = None,
                    jpeg_quality: int = 80, with_camera: bool = True, with_photos: bool = True) -> str:
    """Upload one pass directory (simulator or real-data contract) as a Mosaico sequence.

    Required: meta.json, arc.parquet, events.jsonl. Optional (skipped when absent): travel/thermal/robot
    parquet, torch_cam.mp4 (-> /camera/torch/image), photos/*.jpg (-> /camera/bead/photo).
    Returns the sequence name."""
    from mosaicolabs import (CompressedImage, Message, Point3d, Pose, Quaternion, RobotJoint,
                             SessionLevelErrorPolicy, String)
    from mosaicolabs.models.sensors.image import ImageFormat

    d = Path(path)
    meta = load_meta(d)
    seq = meta["sequence_name"]
    client = client or connect()
    if client.sequence_exists(seq):
        if not replace:
            return seq
        client.sequence_delete(seq)

    arc = pd.read_parquet(d / "arc.parquet")
    trv = read_optional_parquet(d / "travel.parquet")
    thm = read_optional_parquet(d / "thermal.parquet")
    rob = read_optional_parquet(d / "robot.parquet")
    ev_path = d / "events.jsonl"
    events = [json.loads(line) for line in ev_path.read_text().splitlines() if line.strip()] if ev_path.exists() else []
    t0_ns = int(meta.get("t0_ns", arc["t_ns"].iloc[0]))
    synthetic = source_info(meta)["synthetic"]
    video = d / "torch_cam.mp4"
    photos = [d / p for p in meta.get("photos") or [] if (d / p).exists()]
    if not photos and (d / "photos").is_dir():
        photos = sorted((d / "photos").glob("*.jpg"))
    has_cam = with_camera and video.exists()
    has_photo = with_photos and bool(photos)

    n_avg = ARC_HZ // 2
    arc_cols = ["current_a", "voltage_v", "wfs_mpm", "gas_lpm"]
    for c in arc_cols:
        if c not in arc.columns:
            arc[c] = np.nan
    arc_extra = _opt_cols(arc, ("current2_a", "voltage2_v", "wfs2_mpm"))
    arc_avg = {c: movavg(arc[c].to_numpy(dtype=float), n_avg) for c in arc_cols + arc_extra}
    avg_name = {"current_a": "current_avg_a", "voltage_v": "voltage_avg_v", "wfs_mpm": "wfs_avg_mpm",
                "gas_lpm": "gas_avg_lpm", "current2_a": "current2_avg_a", "voltage2_v": "voltage2_avg_v",
                "wfs2_mpm": "wfs2_avg_mpm"}

    seq_meta = flatten_meta(meta)
    topics = [(TOPIC_ARC, "WeldArc")]
    if trv is not None:
        topics.append((TOPIC_TRAVEL, "WeldTravel"))
    if thm is not None and "interpass_c" in thm.columns:
        topics.append((TOPIC_THERMAL, "WeldThermal"))
    if rob is not None and all(f"j{k}" in rob.columns for k in range(1, 7)):
        topics.append((TOPIC_ROBOT_JOINTS, "RobotJoint"))
    if rob is not None and {"tcp_x", "tcp_y", "tcp_z"} <= set(rob.columns):
        topics.append((TOPIC_TCP_POSE, "Pose"))
    if has_cam:
        topics.append((TOPIC_CAMERA, "CompressedImage/JPEG"))
    if has_photo:
        topics.append((TOPIC_BEAD_PHOTO, "CompressedImage/JPEG"))
    topics.append((TOPIC_EVENTS, "String/JSON"))
    # (metadata keys may not contain '/', so topics are listed as records)
    seq_meta["topics"] = [{"topic": t, "ontology": o} for t, o in topics]
    names = {t for t, _ in topics}

    with client.sequence_create(sequence_name=seq, metadata=seq_meta,
                                on_error=SessionLevelErrorPolicy.Delete) as sw:
        # --- arc
        units = {"current": "A", "voltage": "V", "wfs": "m/min", "gas": "L/min"}
        tw = sw.topic_create(TOPIC_ARC, metadata_safe({
            "rate_hz": _rate_hz(arc["t_ns"].to_numpy(), ARC_HZ), "units": units, "avg_window_s": 0.5,
            "electrodes": int(meta.get("electrodes") or 1), "trail_fields": arc_extra,
            "gas": "none (flux-shielded SAW): gas_lpm is NaN" if not arc["gas_lpm"].notna().any() else "measured"}),
            WeldArc)
        ts = arc["t_ns"].to_numpy()
        cols = {c: arc[c].to_numpy(dtype=float) for c in arc_cols + arc_extra}
        for i in range(len(ts)):
            kw = {c: _f(cols[c][i]) for c in cols}
            kw.update({avg_name[c]: _f(arc_avg[c][i]) for c in arc_avg})
            tw.push(Message(timestamp_ns=int(ts[i]), data=WeldArc(**kw)))
        # --- travel
        if TOPIC_TRAVEL in names:
            tw = sw.topic_create(TOPIC_TRAVEL, {"rate_hz": _rate_hz(trv["t_ns"].to_numpy(), ARC_HZ)}, WeldTravel)
            ts = trv["t_ns"].to_numpy()
            req = {c: (trv[c].to_numpy(dtype=float) if c in trv.columns else np.full(len(trv), np.nan))
                   for c in ("travel_speed_mmps", "heat_input_kj_mm")}
            opt = {c: trv[c].to_numpy(dtype=float) for c in _opt_cols(trv, ("positioner_deg", "position_mm"))}
            for i in range(len(ts)):
                kw = {c: _f(v[i]) for c, v in {**req, **opt}.items()}
                tw.push(Message(timestamp_ns=int(ts[i]), data=WeldTravel(**kw)))
        # --- thermal
        if TOPIC_THERMAL in names:
            thm = thm[thm["interpass_c"].notna()].reset_index(drop=True)
            hz = _rate_hz(thm["t_ns"].to_numpy(), THERMAL_HZ)
            thm_avg = movmedian(thm["interpass_c"].to_numpy(dtype=float), max(1, int(round(hz))))
            sensor = ("IR pyrometer, 25 mm from toe" if synthetic else
                      meta.get("thermal_sensor") or "see source dataset (interpass_c = interpass proxy)")
            tw = sw.topic_create(TOPIC_THERMAL, {"rate_hz": hz, "sensor": sensor}, WeldThermal)
            ts = thm["t_ns"].to_numpy()
            T = thm["interpass_c"].to_numpy(dtype=float)
            opt = {c: thm[c].to_numpy(dtype=float) for c in _opt_cols(thm, ("temp_left_c", "temp_right_c",
                                                                             "temp_back_c", "thermo2_c"))}
            for i in range(len(ts)):
                tw.push(Message(timestamp_ns=int(ts[i]), data=WeldThermal(
                    interpass_c=float(T[i]), interpass_avg_c=float(thm_avg[i]),
                    **{c: _f(v[i]) for c, v in opt.items()})))
        # --- robot joints + TCP pose (decimated to <= ROBOT_HZ)
        if rob is not None and (TOPIC_ROBOT_JOINTS in names or TOPIC_TCP_POSE in names):
            rob = _decimate(rob, ROBOT_HZ)
            ts = rob["t_ns"].to_numpy()
            hz = _rate_hz(ts, ROBOT_HZ)
            if TOPIC_ROBOT_JOINTS in names:
                jn = [f"j{k}" for k in range(1, 7)]
                Q = rob[jn].to_numpy(dtype=float)
                Qd = np.gradient(Q, 1.0 / hz, axis=0) if len(Q) > 1 else np.zeros_like(Q)
                tj = sw.topic_create(TOPIC_ROBOT_JOINTS, {"rate_hz": hz, "robot": "6-axis arc welding robot",
                                                          "units": "rad"}, RobotJoint)
                for i in range(len(ts)):
                    tj.push(Message(timestamp_ns=int(ts[i]), data=RobotJoint(
                        names=jn, positions=[float(x) for x in Q[i]], velocities=[float(x) for x in Qd[i]],
                        efforts=[0.0] * 6)))
            if TOPIC_TCP_POSE in names:
                qcols = ["tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]
                has_q = set(qcols) <= set(rob.columns)
                tp = sw.topic_create(TOPIC_TCP_POSE, {"rate_hz": hz, "frame": "robot_base" if synthetic else "workpiece/robot",
                                                      "units": "m" if synthetic else "mm",
                                                      "orientation": "logged" if has_q else "not logged (identity)"},
                                     Pose)
                X = rob[["tcp_x", "tcp_y", "tcp_z"]].to_numpy(dtype=float)
                Qr = rob[qcols].to_numpy(dtype=float) if has_q else np.tile([0.0, 0.0, 0.0, 1.0], (len(rob), 1))
                for i in range(len(ts)):
                    x, q = X[i], Qr[i]
                    tp.push(Message(timestamp_ns=int(ts[i]), data=Pose(
                        position=Point3d(x=float(x[0]), y=float(x[1]), z=float(x[2])),
                        orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))))
        # --- events
        te = sw.topic_create(TOPIC_EVENTS, {"schema": "{t_ns,type,defect?,severity?,source?}"}, String)
        for e in events:
            te.push(Message(timestamp_ns=int(e["t_ns"]), data=String(data=json.dumps(metadata_safe(e)))))
        # --- camera (JPEG frames of the torch / process camera)
        if has_cam:
            src_fps = video_fps(video, meta)
            fps = float(camera_fps or os.environ.get("MOSAICO_CAMERA_FPS", 12))
            every = max(1, round(src_fps / fps))
            vmeta = meta.get("video") if isinstance(meta.get("video"), dict) else {}
            t_off = float(vmeta.get("t_offset_s") or 0.0)
            tc = sw.topic_create(TOPIC_CAMERA, metadata_safe({
                "fps": src_fps / every, "source_fps": src_fps, "format": "jpeg", "t_offset_s": t_off,
                "kind": vmeta.get("kind") or ("simulated torch camera render" if synthetic else "process camera"),
                "source": "torch-mounted HDR weld camera (simulated)" if synthetic else
                          f"{source_info(meta).get('id')} process camera video"}), CompressedImage)
            for k, t_v, jpg in _iter_jpeg_frames(video, every, jpeg_quality):
                tc.push(Message(timestamp_ns=int(t0_ns + round((t_v + t_off) * 1e9)),
                                data=CompressedImage(data=jpg, format=ImageFormat.JPEG)))
        # --- post-weld bead photos (one message each, stamped at arc stop)
        if has_photo:
            tph = sw.topic_create(TOPIC_BEAD_PHOTO, {"format": "jpeg", "kind": "post-weld bead photo",
                                                     "files": [p.relative_to(d).as_posix() for p in photos],
                                                     "time": "stamped after arc stop (not time-aligned to the weld)"},
                                  CompressedImage)
            t_stop = t0_ns + int(float(meta.get("duration_s") or 0) * 1e9)
            for k, p in enumerate(photos):
                tph.push(Message(timestamp_ns=int(t_stop + k * 1_000_000),
                                 data=CompressedImage(data=p.read_bytes(), format=ImageFormat.JPEG)))
    return seq


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest data/generated/<pass>/ dirs into mosaicod")
    ap.add_argument("paths", nargs="*", default=[os.path.join(os.environ.get("DATA_DIR", "data"), "generated")])
    ap.add_argument("--no-replace", action="store_true", help="skip sequences that already exist")
    ap.add_argument("--camera-fps", type=int, default=None)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-photos", action="store_true")
    args = ap.parse_args(argv)
    dirs = []
    for p in map(Path, args.paths):
        if (p / "meta.json").exists():
            dirs.append(p)
        elif p.is_dir():
            dirs += sorted(x for x in p.iterdir() if (x / "meta.json").exists())
    if not dirs:
        print("no pass directories found", file=sys.stderr)
        return 1
    client = connect()
    t = time.time()
    failed = []
    for d in dirs:
        t1 = time.time()
        try:
            seq = ingest_pass_dir(d, client, replace=not args.no_replace, camera_fps=args.camera_fps,
                                  with_camera=not args.no_camera, with_photos=not args.no_photos)
        except Exception as e:  # noqa: BLE001 - one bad pass must not stop the others
            failed.append(d.name)
            print(f"  FAILED {d.name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        print(f"  ingested {seq} ({time.time() - t1:.1f}s)", flush=True)
    print(f"done: {len(dirs) - len(failed)}/{len(dirs)} sequences in {time.time() - t:.1f}s -> mosaicod "
          f"{os.environ.get('MOSAICO_HOST', 'localhost')}:{os.environ.get('MOSAICO_PORT', '6726')}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
