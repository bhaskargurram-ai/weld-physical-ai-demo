# Real-data contract (replaces the simulator as the default source)

The demo now runs on **real, public, commercially usable** welding data. Everything downstream
(Mosaico → Curator → Cosmos 3 → Copilot) keeps working on the same *pass directory* layout as the
simulator, extended so that streams a real dataset does not have are **optional**.

## Sources (both CC BY 4.0 — attribution required, shown in the app)

| id | dataset | what we use | pass naming |
|---|---|---|---|
| `aimen_saw` | Tapia Suárez E. A., Pérez Couñago I., Eike Precker C., Fernández Montenegro J. M., Ruiz C. — *Submerged Arc Welding Open Repository* (PENELOPE, H2020 GA 958303), AIMEN. Zenodo, DOI 10.5281/zenodo.15083865 | coupons **PP3–PP7** (SA-516 Gr.70N, 52 mm butt weld, 24 SAW beads each): `1. process_data/<PPx_NN>/<PPx_NN>.hdf5` (data0 ≈120 Hz: process_status, pyrometer_L/R/back °C, wire_speed_a/b m/min, x/y/z mm, timestamp, idx_data1; data1 5 kHz: current_a/b A, voltage_a/b V, timestamp — `a` = DC lead electrode, `b` = AC trail electrode), per-bead photos `PPx_NN_*.jpg`, `2. ndt_data/defects_xlocation.xlsx` (sheets `defect_code`, `welding_settings_code`, `PPx`: defect, welding_setting, bead, x_init/x_end mm, x_init/x_end # sample = **data1 index**), `RX.pdf`, `UT.pdf`, `Design_Materials_Settings.pdf` | `PP5_B02_SAW` (spool_id `PP5`, joint_id `BW`, pass_no = bead) |
| `bam_tgmaw` | Chaurasia P. K., Fabry Ç., Pittner A., Rethmeier M. (BAM) — *Tandem-GMAW: advanced pulsed and short-circuiting process mode experimental data*. Zenodo, DOI 10.5281/zenodo.21824586 | workpieces WID615/617/630/631/632 × layers L1–L6 (S355J2+N, 30 mm groove; L1 root single-wire, L2–L6 tandem): `*_process.nc` (10 kHz current/voltage leading/trailing, `video_frame` sync pulse), `*_robot.nc` (1 kHz arc_on, gas_flow_leading/trailing L/min, thermo_1/2 °C, tcp_coords mm), `*_power_source_{leading,trailing}.nc` (ACTUAL_WFS, WFS_COMMANDVALUE, ACTUAL_GASFLOW, GASVALUE, JOBNUMBER…), `*_video.avi` (1280×1024, ~18 fps, HDR grayscale, BAM overlay), `experiment_list.csv` (row 2 = units; wts mm/s, mode, wfs commands, weave, t85) | `WID617_L3_TGMAW` (spool_id `WID617`, joint_id `V30`, pass_no = layer) |

Downloads are cached under `DATA_DIR/raw/<source>/` (gitignored). AIMEN is one 12.7 GB zip: read it
**remotely with HTTP range requests** and extract only the needed members (see `/home/user/realdata/remotezip.py`
for a working reader) — never download the whole zip. Skip `*.txt`/`*profilo*` (profilometer) and `*.nde`.

Facts verified on the real files (2026-11-12): AIMEN PP5 bead 2 — data1 rate 4999 Hz, data0 ≈120 Hz (README says
66.7, wrong); robot x = spreadsheet x + ≈536.7 mm on PP5 (derive the offset per coupon from
`x(idx_data1 = x_init_sample) − x_init_mm`, do not hard-code); mean current/voltage inside the 14 porosity windows equal
the clean weld (535 vs 531 A) — **real defects do not necessarily show in the electrical signals; never fake a signature.**

## Pass directory (`DATA_DIR/generated/<sequence_name>/`)

Required: `meta.json`, `arc.parquet`, `events.jsonl`. Everything else optional — readers must tolerate absence.

`meta.json` — all simulator keys (sequence_name, spool_id, joint_id, pass_no, pass_type, wps_id, process, position,
material, cell_id, duration_s, defects, rt_result, has_defect, defect_types, t0_ns, wps, setpoints) **plus**:
```json
"source": {"id": "aimen_saw", "title": "...", "authors": "...", "doi": "10.5281/zenodo.15083865",
           "url": "https://zenodo.org/records/15083865", "license": "CC BY 4.0", "synthetic": false,
           "files": ["PP5/1. process_data/PP5_02/PP5_02.hdf5", "..."]},
"electrodes": 2,                       # 1 single wire, 2 tandem (lead + trail)
"wps_basis": "published nominal setting ±10 % (demo assumption, not a qualified WPS)",
"ndt": {"methods": ["RT", "UT"], "inspected": true, "report_files": ["RX.pdf", "UT.pdf"],
        "indications": [{"type": "porosity", "t_start": 17.0, "t_end": 17.1, "x_start_mm": 136, "x_end_mm": 136.7}]},
"thickness_mm": 52, "pass_type": "root|fill|cap",
"photos": ["photos/PP5_02_1.jpg"],     # relative to the pass dir
"video": {"file": "torch_cam.mp4", "kind": "real_process_camera", "fps": 17.96, "t_offset_s": 0.0}
```
- `rt_result`: `"reject"`/`"accept"` (simulator), `"indications"` (NDT found defects in this pass), `"no_indications"`
  (inspected, nothing in this pass), `"not_inspected"` (BAM). `has_defect` = any indication. `defects` = the indications
  with `defect`, `t_start`, `t_end`, `severity: null`.
- `wps` ranges: nominal ±10 % (current, voltage, wfs, travel speed) from the published settings, gas `[GASVALUE·0.8, ·1.2]`
  for BAM, no gas for SAW (flux). Keep `wps_basis` honest.

`arc.parquet` @ 100 Hz (10 ms means of the raw signal; also `current_rms_a` not required):
`t_ns, current_a, voltage_v, wfs_mpm, gas_lpm` (lead electrode; `gas_lpm` NaN for SAW) + optional
`current2_a, voltage2_v, wfs2_mpm` (trail electrode). t = 0 at arc start.
`travel.parquet` @100 Hz: `t_ns, travel_speed_mmps, position_mm, heat_input_kj_mm` (`positioner_deg` optional).
`thermal.parquet`: `t_ns, interpass_c` (+ optional `temp_left_c`, `temp_right_c` …). AIMEN: pyrometer_back as
interpass proxy is wrong (it reads the pool region) → use max(pyrometer_L, pyrometer_R) as `interpass_c` and keep the others.
BAM: thermo_1 as `interpass_c` (document it), thermo_2 raw.
`robot.parquet` optional: `t_ns, tcp_x, tcp_y, tcp_z` (+ j1..j6/quaternions if available).
`events.jsonl`: arc_start, arc_stop, defect_on/defect_off per NDT indication (`source: "NDT"`).
`torch_cam.mp4` optional: H.264 yuv420p, frame 0 = arc start (trim using sync), ≤1280 px wide; keep the BAM overlay
(attribution). `control/edge.mp4` optional (BAM: Canny edges of the real video, same timing, for Cosmos Transfer).
`photos/*.jpg` optional (AIMEN bead photos, long side ≤1600 px, EXIF stripped).

## Pipeline switch
`DATA_SOURCE=real|sim|all` (default **real**). `collect` stage: `real` → `python -m collect.datasets.fetch --sources
aimen_saw,bam_tgmaw [--limit ...]` (download + convert); `sim` → the simulator (sequences tagged `source.synthetic: true`).
Defaults for size: AIMEN PP3–PP7 all beads; BAM WID617 + WID630 (`BAM_WIDS=all` on the pod).
