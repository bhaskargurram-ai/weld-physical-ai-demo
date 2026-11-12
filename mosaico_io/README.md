# mosaico_io — WeldWorld data layer on Mosaico

One **Mosaico sequence per weld pass** (name = `sequence_name`, e.g. `SP-1043_J07_P3_FCAW`),
written and read with the real `mosaicolabs` SDK (0.7.x, matches `mosaicod 0.7`).

## Quick start

```bash
docker compose -f infra/docker-compose.yml up -d         # postgres + mosaicod (:6726) + minio (:9000, bucket raw)
python -m collect.simulate --n-passes 24 --seed 7 --out data/generated
python -m mosaico_io.ingest data/generated               # 24 sequences, ~10 s each incl. camera JPEGs
python -m mosaico_io.export --out data/export            # 5 s clips + sidecars + control clips
python -m mosaico_io.query list --process FCAW --has_defect true
python -m mosaico_io.query anomalies gas_lpm --lt 12     # query by physical value
python -m mosaico_io.query violations SP-1043_J07_P3_FCAW
```

Offline (no mosaicod): `MOSAICO_OFFLINE=1` reads `data/generated/` directly with the same
return shapes. If `MOSAICO_OFFLINE` is unset and mosaicod is unreachable, the package
falls back to offline automatically (with a warning); `MOSAICO_OFFLINE=0` forces online.
Env: `MOSAICO_HOST`/`MOSAICO_PORT` (localhost:6726), `DATA_DIR` (./data),
`MOSAICO_CAMERA_FPS` (camera topic rate at ingest, default 12).

## Python API

```python
from mosaico_io import connect, ingest_pass_dir, list_passes, get_signals, wps_violations, export_clips, find_anomalies

list_passes(process="FCAW", spool_id="SP-1043", pass_type=["fill", "cap"], has_defect=True)
list_passes(heat_input_mean_kj_mm__gt=1.3)              # op suffixes: __gt __lt __geq __leq __neq __in __match
get_signals("SP-1043_J07_P3_FCAW", t0_s=18, t1_s=28)    # DataFrame @100 Hz: t_s + current_a voltage_v wfs_mpm gas_lpm
                                                        #   travel_speed_mmps positioner_deg heat_input_kj_mm interpass_c
wps_violations("SP-1043_J07_P3_FCAW")                   # [{param, t_start_s, t_end_s, value, limit_lo, limit_hi, kind}]
find_anomalies("gas_lpm", lt=12)                        # [{sequence_name, t_start_s, t_end_s, duration_s, peak, ...}]
find_anomalies("interpass_c", gt=250)
find_anomalies("voltage_v", gt=27.5, process="FCAW")
export_clips("data/export", sequences=None, clip_s=5)   # -> list of sidecar dicts
```

`t_s` is seconds from arc start (`meta["t0_ns"]`). `list_passes` rows carry the flat
metadata keys plus `defects`, `wps` and `setpoints`.

* **list_passes** → server-side `QuerySequence().with_user_metadata(key, op=value)` (dot keys
  reach nested metadata, e.g. `**{"wps.max_interpass_c__lt": 255}`).
* **get_signals** → `mosaicolabs.ml.DataFrameExtractor` over `/weld/arc`, `/weld/travel`,
  `/weld/thermal`, re-gridded onto the 100 Hz arc clock (thermal interpolated).
* **wps_violations** → 0.5 s monitor averages vs. the pass WPS window, arc-on only
  (hot-start ramp and programmed crater fill excluded), runs ≥ 0.2 s; interpass vs. `max_interpass_c`.
* **find_anomalies** — the "query by physical values" showcase. Online it is one
  server-side query: `Query(QuerySequence(...filters), QueryTopic().with_name("/weld/arc"),
  QueryOntologyCatalog(WeldArc.Q.gas_avg_lpm.lt(12)))`; time intervals come from
  `topic.clusterize(250 ms)` and are converted to pass-relative seconds; the peak value is read
  back for each interval. `raw=True` queries the raw 100 Hz field instead of the monitor
  average. Passes whose WPS window already satisfies the condition (e.g. `gas < 12` on
  gasless SAW, `current > 245` on SAW) are skipped unless `relative_to_wps=False`.
* **export_clips** → for each pass, one window centred on every defect plus 1–2 clean windows
  (away from defects). Writes `<out>/<clip_id>.mp4` (24 fps H.264), `<clip_id>.json` sidecar
  (`clip_id, sequence_name, t0_s, t1_s, meta, signal_stats{mean,std,min,max}, wps_violations,
  defects_in_window, rt_result` + `label`, `window_kind`, `joint_rt_result`, `control`),
  `control/<clip_id>_{edge,depth}.mp4` (Cosmos Transfer inputs) and `clips.jsonl`.
  `clip_id = <sequence>_t<deciseconds>` (e.g. `SP-1043_J07_P3_FCAW_t0210`).
  Signals/stats/violations come from Mosaico (or local files offline); the video is cut from
  the pass's `torch_cam.mp4` when present, otherwise rebuilt from the JPEG frames stored in
  `/camera/torch/image`.

## Real data (docs/REAL_DATA.md)

Only `meta.json`, `arc.parquet` and `events.jsonl` are required; travel / thermal / robot parquet, `torch_cam.mp4`,
`control/*` and `photos/*.jpg` are ingested when present and skipped otherwise. Tandem passes add the trail electrode
as `current2_a, voltage2_v, wfs2_mpm` (+ `*2_avg_*`) on `WeldArc`; `gas_lpm` is NaN on SAW; `WeldThermal` carries
`temp_left_c / temp_right_c / temp_back_c` (AIMEN pyrometers) or `thermo2_c` (BAM) as optional fields.
`get_signals` always returns the core columns (NaN when a dataset lacks them; `positioner_deg` only for the
simulator) followed by the extras present. `wps_violations` skips parameters whose range or signal is missing.
Export: passes with video -> clips around NDT indications + clean windows at the source frame rate; passes without
video -> one sidecar per indication window with `media_type: "image"`, `video: null`, `photos: [photos/<seq>/*.jpg]`,
`photo_note` (indication x position) — curation and Cosmos use the photo. Sidecars also carry `source`
(id, license, doi, attribution), `ndt`, `electrodes`, `video_kind`; the old `source: local|mosaico` field is now
`read_from`. `python -m mosaico_io.export --data-source real|sim|all` limits the default sequence set.
`python -m mosaico_io.testdata <dir>` writes a contract-compliant two-pass fixture.

## Topics and ontology choice

| Topic | Ontology | Rate | Notes |
|---|---|---|---|
| `/weld/arc` | **`WeldArc`** (custom) | 100 Hz | `current_a, voltage_v, wfs_mpm, gas_lpm` raw + `*_avg_*` 0.5 s averages |
| `/weld/travel` | **`WeldTravel`** (custom) | 100 Hz | `travel_speed_mmps, positioner_deg, heat_input_kj_mm` |
| `/weld/thermal` | **`WeldThermal`** (custom) | 10 Hz | `interpass_c` raw + `interpass_avg_c` (1 s median) |
| `/robot/joint_states` | `RobotJoint` | 50 Hz | names j1..j6, positions [rad], velocities (finite diff), efforts 0 |
| `/robot/tcp_pose` | `Pose` | 50 Hz | torch TCP in robot base frame [m] + quaternion |
| `/camera/torch/image` | `CompressedImage` (JPEG q80) | 12 fps | `MOSAICO_CAMERA_FPS` / `--camera-fps 24` for full rate; real video (BAM, 17.96 fps) at its own rate, stamped `t0 + frame time + video.t_offset_s` |
| `/camera/bead/photo` | `CompressedImage` (JPEG) | per pass | post-weld bead photos (AIMEN), stamped after arc stop; topic metadata `files` keeps the names |
| `/weld/events` | `String` | events | JSON `{t_ns, type: arc_start/arc_stop/defect_on/defect_off, defect?, severity?}` |

**Why a custom ontology (not `Floating64` topics):** the custom-ontology path in the SDK is
simple (`mosaico_io/ontology.py`, ~40 lines, same pattern as the SDK's Isaac `EncoderTicks`
example) and it works end-to-end with mosaicod 0.7: typed Arrow columns, one topic per stream
instead of 8 scalar topics, and `.Q` query proxies on every field so the server can answer
"where did gas flow drop below 12 L/min" directly. The `*_avg_*` fields exist because FCAW
short-circuit transfer produces real single-sample current/voltage spikes; WPS compliance and
anomaly queries use the monitor average so a 10 ms short circuit is not reported as a WPS
excursion (use `raw=True` to query raw samples). Readers must `import mosaico_io.ontology`
(done by `mosaico_io.connect()`) so the classes are registered before decoding.

**Sequence metadata** = full `meta.json` (nested `wps`, `setpoints` kept; `defects` as a list)
plus flat queryable keys: `process, wps_id, spool_id, joint_id, pass_no, pass_type, position,
material, rt_result, joint_rt_result, has_defect, defect_types, max_severity, t0_ns,
heat_input_mean_kj_mm, interpass_start_c, interpass_max_c, electrodes, thickness_mm, source_id, source_license,
source_doi, synthetic, ndt_inspected, n_indications, has_video, n_photos, ...` (nested `source`, `ndt`, `photos`,
`video` kept). Metadata keys may not contain `/`, `.` or `:` (nested keys are sanitised to `_`), so the topic list
is stored as records under `topics`.

## Tests

```bash
pytest tests/test_mosaico_io.py tests/test_mosaico_io_real.py   # offline + live round-trip if mosaicod is up
pytest tests/test_mosaico_io.py -m mosaico   # only the live round-trip (skipped if :6726 is closed)
```
