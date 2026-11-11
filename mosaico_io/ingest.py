"""Ingest generated weld passes into Mosaico: one sequence per pass.

    python -m mosaico_io.ingest data/generated [--replace] [--camera-fps 12]

Topics (collect/schema.py):
    /weld/arc            WeldArc     (custom ontology) @100 Hz
    /weld/travel         WeldTravel  (custom ontology) @100 Hz
    /weld/thermal        WeldThermal (custom ontology) @10 Hz
    /robot/joint_states  RobotJoint  @50 Hz
    /robot/tcp_pose      Pose        @50 Hz
    /camera/torch/image  CompressedImage (JPEG) @MOSAICO_CAMERA_FPS (default 12)
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
from mosaico_io.common import connect, flatten_meta, load_meta, movavg, movmedian
from mosaico_io.ontology import WeldArc, WeldThermal, WeldTravel


def _iter_jpeg_frames(video: Path, every: int, quality: int):
    import av
    import cv2
    with av.open(str(video)) as c:
        for k, fr in enumerate(c.decode(video=0)):
            if k % every:
                continue
            bgr = fr.to_ndarray(format="bgr24")
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                yield k, buf.tobytes()


def ingest_pass_dir(path, client=None, replace: bool = True, camera_fps: Optional[int] = None,
                    jpeg_quality: int = 80, with_camera: bool = True) -> str:
    """Upload one generated pass directory as a Mosaico sequence. Returns the sequence name."""
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
    trv = pd.read_parquet(d / "travel.parquet")
    thm = pd.read_parquet(d / "thermal.parquet")
    rob = pd.read_parquet(d / "robot.parquet")
    events = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines() if line.strip()]
    t0_ns = int(meta.get("t0_ns", arc["t_ns"].iloc[0]))

    n_avg = ARC_HZ // 2
    arc_avg = {c: movavg(arc[c].to_numpy(), n_avg) for c in ("current_a", "voltage_v", "wfs_mpm", "gas_lpm")}
    thm_avg = movmedian(thm["interpass_c"].to_numpy(), THERMAL_HZ)

    seq_meta = flatten_meta(meta)
    # (metadata keys may not contain '/', so topics are listed as records)
    seq_meta["topics"] = [{"topic": t, "ontology": o} for t, o in (
        (TOPIC_ARC, "WeldArc"), (TOPIC_TRAVEL, "WeldTravel"), (TOPIC_THERMAL, "WeldThermal"),
        (TOPIC_ROBOT_JOINTS, "RobotJoint"), (TOPIC_TCP_POSE, "Pose"),
        (TOPIC_CAMERA, "CompressedImage/JPEG"), (TOPIC_EVENTS, "String/JSON"))]

    with client.sequence_create(sequence_name=seq, metadata=seq_meta,
                                on_error=SessionLevelErrorPolicy.Delete) as sw:
        # --- arc
        tw = sw.topic_create(TOPIC_ARC, {"rate_hz": ARC_HZ, "units": {"current": "A", "voltage": "V",
                                         "wfs": "m/min", "gas": "L/min"}, "avg_window_s": 0.5}, WeldArc)
        ts = arc["t_ns"].to_numpy()
        I, V, F, G = (arc[c].to_numpy() for c in ("current_a", "voltage_v", "wfs_mpm", "gas_lpm"))
        for i in range(len(ts)):
            tw.push(Message(timestamp_ns=int(ts[i]), data=WeldArc(
                current_a=float(I[i]), voltage_v=float(V[i]), wfs_mpm=float(F[i]), gas_lpm=float(G[i]),
                current_avg_a=float(arc_avg["current_a"][i]), voltage_avg_v=float(arc_avg["voltage_v"][i]),
                wfs_avg_mpm=float(arc_avg["wfs_mpm"][i]), gas_avg_lpm=float(arc_avg["gas_lpm"][i]))))
        # --- travel
        tw = sw.topic_create(TOPIC_TRAVEL, {"rate_hz": ARC_HZ}, WeldTravel)
        ts = trv["t_ns"].to_numpy()
        S, P, HI = (trv[c].to_numpy() for c in ("travel_speed_mmps", "positioner_deg", "heat_input_kj_mm"))
        for i in range(len(ts)):
            tw.push(Message(timestamp_ns=int(ts[i]), data=WeldTravel(
                travel_speed_mmps=float(S[i]), positioner_deg=float(P[i]), heat_input_kj_mm=float(HI[i]))))
        # --- thermal
        tw = sw.topic_create(TOPIC_THERMAL, {"rate_hz": THERMAL_HZ, "sensor": "IR pyrometer, 25 mm from toe"},
                             WeldThermal)
        ts = thm["t_ns"].to_numpy()
        T = thm["interpass_c"].to_numpy()
        for i in range(len(ts)):
            tw.push(Message(timestamp_ns=int(ts[i]), data=WeldThermal(
                interpass_c=float(T[i]), interpass_avg_c=float(thm_avg[i]))))
        # --- robot joints + TCP pose
        names = [f"j{k}" for k in range(1, 7)]
        Q = rob[names].to_numpy()
        dt = 1.0 / ROBOT_HZ
        Qd = np.gradient(Q, dt, axis=0) if len(Q) > 1 else np.zeros_like(Q)
        ts = rob["t_ns"].to_numpy()
        tj = sw.topic_create(TOPIC_ROBOT_JOINTS, {"rate_hz": ROBOT_HZ, "robot": "6-axis arc welding robot",
                                                  "units": "rad"}, RobotJoint)
        for i in range(len(ts)):
            tj.push(Message(timestamp_ns=int(ts[i]), data=RobotJoint(
                names=names, positions=[float(x) for x in Q[i]], velocities=[float(x) for x in Qd[i]],
                efforts=[0.0] * 6)))
        tp = sw.topic_create(TOPIC_TCP_POSE, {"rate_hz": ROBOT_HZ, "frame": "robot_base", "units": "m"}, Pose)
        X = rob[["tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]].to_numpy()
        for i in range(len(ts)):
            x = X[i]
            tp.push(Message(timestamp_ns=int(ts[i]), data=Pose(
                position=Point3d(x=float(x[0]), y=float(x[1]), z=float(x[2])),
                orientation=Quaternion(x=float(x[3]), y=float(x[4]), z=float(x[5]), w=float(x[6])))))
        # --- events
        te = sw.topic_create(TOPIC_EVENTS, {"schema": "{t_ns,type,defect?,severity?}"}, String)
        for e in events:
            te.push(Message(timestamp_ns=int(e["t_ns"]), data=String(data=json.dumps(e))))
        # --- camera (JPEG frames)
        video = d / "torch_cam.mp4"
        if with_camera and video.exists():
            fps = int(camera_fps or os.environ.get("MOSAICO_CAMERA_FPS", 12))
            every = max(1, round(CAMERA_FPS / fps))
            tc = sw.topic_create(TOPIC_CAMERA, {"fps": CAMERA_FPS / every, "format": "jpeg",
                                                "source": "torch-mounted HDR weld camera"}, CompressedImage)
            frame_ns = 1_000_000_000 / CAMERA_FPS
            for k, jpg in _iter_jpeg_frames(video, every, jpeg_quality):
                tc.push(Message(timestamp_ns=int(t0_ns + round(k * frame_ns)),
                                data=CompressedImage(data=jpg, format=ImageFormat.JPEG)))
    return seq


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest data/generated/<pass>/ dirs into mosaicod")
    ap.add_argument("paths", nargs="*", default=[os.path.join(os.environ.get("DATA_DIR", "data"), "generated")])
    ap.add_argument("--no-replace", action="store_true", help="skip sequences that already exist")
    ap.add_argument("--camera-fps", type=int, default=None)
    ap.add_argument("--no-camera", action="store_true")
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
    for d in dirs:
        t1 = time.time()
        seq = ingest_pass_dir(d, client, replace=not args.no_replace, camera_fps=args.camera_fps,
                              with_camera=not args.no_camera)
        print(f"  ingested {seq} ({time.time() - t1:.1f}s)", flush=True)
    print(f"done: {len(dirs)} sequences in {time.time() - t:.1f}s -> mosaicod "
          f"{os.environ.get('MOSAICO_HOST', 'localhost')}:{os.environ.get('MOSAICO_PORT', '6726')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
