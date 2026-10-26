# Plan: kinematics — from tracker output to tidy tables

> **Status: revision 1 (2026-10-26), implemented on branch `feat/kinematics-core`, not released.**
> Replaces `kinematics/tongue_analysis.py` and `kinematics/tongue_kinematics_utils.py` with a
> general core (`kinematics/`) and one task module (`kinematics/tasks/dynamic_foraging_tongue.py`).
> A clean break: output files and column names change (map below); `kinematics_analysis` updates
> its loaders when it re-runs the batch. Written so a new contributor or agent can pick it up
> without the conversation that produced it.

## Use cases

1. **Any keypoint, processed its own way.** A tongue that is tracked only while it is out, a jaw
   or nose that is always visible, a paw: each needs its own masking, smoothing, segmentation and
   measurements, from the same code.
2. **Tidy tables for analysis.** Per frame, per segment (movement, submovement, bout), per trial
   or time window, and one record per session, all keyed by `session` and safe to pool.
3. **Questions downstream asks repeatedly**, answered once: end points relative to a landmark,
   movements per trial or per second, first movement after a cue, bouts and their rate, rhythm.
4. **Other tasks and trackers.** Nothing in the core knows about dynamic foraging, Harp or NWB, so
   the core can move to its own package once a second project uses it.

## What was wrong before

| Problem | Effect |
|---|---|
| One keypoint hard-coded (`tongue_tip_center`, `jaw` as literals) | No second keypoint, no different processing |
| Three time columns (`time`, `time_raw`, `time_in_session`) | Downstream overwrote `kps_raw_*.time` by row position |
| Segmentation by tracking gaps only | Useless for body parts that never disappear |
| Cubic interpolation across every gap, then one filter pass | Neighbouring movements leak into each other's edges |
| Features in absolute pixels; duplicates (`max_x`, `max_x_from_jaw`, `max_x_distance` ρ = 1.000; `time_to_endpoint` = `out_duration`) | Not poolable across sessions; every notebook rebuilt the jaw-centred frame |
| NWB, go cues, `/root/capsule` paths inside the processing | Not reusable |
| No settings object, no record of how outputs were made | Thresholds drifted between callers |
| Downstream rebuilt bouts, per-trial counts, first-move tables, quality filters | Several definitions of "first movement" |

## Design

### Principles

1. **Four kinds of table, and functions that take one and return another**: frames → segments →
   summaries per window, plus one record per session.
2. **One clock per table.** Frame tables have `frame` (= video frame = CSV row = pose row) and
   `time` on the caller's clock. The core never shifts clocks; the task module picks the origin
   and records it (`time_zero`).
3. **Segments are intervals** (`start`, `end` rows). Trials and time bins are intervals too, so
   "which movement was this lick in", "movements per trial" and "first movement after the cue"
   are the same few helpers.
4. **One settings object per keypoint**, one mechanism. Different keypoints, different settings.
5. **Relative positions in features.** Segment geometry is measured from a reference landmark,
   never in absolute pixels. Raw pixels stay in `pose`; smoothed pixels in `kinematics`.
6. **A record with every output**: settings, library versions, timing verdict, frame counts.

### Separation

| Job | Where |
|---|---|
| Tracker output → `pose` (`frame, keypoint, x, y, likelihood`) | `kinematics.pose_io` (LP/DLC CSV, DLC HDF5, SLEAP HDF5, ndx-pose, arrays) |
| Frame times → `time`; mask; smooth; velocity; radial velocity | `kinematics.signal` |
| One position per landmark per session | `kinematics.landmarks` |
| Frames → segments | `kinematics.segment` |
| Segments → measurements | `kinematics.features` (groups; others add their own) |
| Which interval things fall in; summaries per window | `kinematics.intervals` |
| All of the above from `Settings`; save, read | `kinematics.pipeline` |
| Dynamic foraging files, Harp clock, licks, trials, quality, batch | `kinematics.tasks.dynamic_foraging_tongue` |
| Which cameras to use | `video_screen` (unchanged) |
| Corrected frame times | `video_timing_qc` (unchanged) |

The core imports numpy, pandas and scipy only. `kinematics/__init__.py` imports nothing, so
`tongue_lickometer_utils` still works with the core install.

### Segmentation methods

| Method | Use when | Parameters |
|---|---|---|
| `tracked_runs` | The part is tracked only while it moves (tongue) | `max_gap_frames` |
| `above_threshold` | The part is always visible (jaw, nose, paw) | `column`, `on`, `off`, `min_frames`, `max_gap_frames` |
| `zero_crossings` | Split at changes of direction (protrusion / retraction) | `column` (default radial velocity), `deadband`, `max_gap_frames`, `min_samples`, `short` |
| `speed_minima` | Split at dips in speed | `min_prominence`, `max_minimum_speed`, ... |
| `bouts` | Group segments that follow each other closely | `of`, `max_gap_s`, `min_count` |

`within` nests one kind inside another (submovements within movements). `zero_crossings` and
`speed_minima` are the uncommitted `feat/submovements` work, ported with behavior unchanged and
its tests carried over.

### Feature groups

| Group | Columns |
|---|---|
| `path` | `path_length, displacement, peak_speed, mean_speed` |
| `excursion` | `amplitude, end_x, end_y, start_x, start_y, angle_deg, extent_x, extent_y` (relative to the reference) |
| `outbound` | `out_duration, out_path_length, out_peak_speed, out_mean_speed` (first frame to farthest frame) |
| `spectrum` | `dominant_hz, band_power, band_fraction` (Welch; for continuous signals) |

Every segment table also has `<kind>_id, keypoint, start_frame, end_frame, start_time, end_time,
duration, n_frames, n_valid`. Bouts add `n_<of>s` and `rate_hz`; nested kinds add `<within>_id`
and `number_in_<within>`; `zero_crossings` adds `direction`.

### Settings

```python
from aind_dynamic_foraging_behavior_video_analysis.kinematics import pipeline as pl

settings = pl.Settings(
    fs_hz=500.0,
    landmarks=("jaw", "spout_l", "spout_r"),
    keypoints=(
        pl.KeypointSettings(
            "tongue_tip_center", min_likelihood=0.9, reference="jaw",
            smooth=pl.Smooth(cutoff_hz=50.0, order=4, max_gap_s=0.02),
            segments={
                "movement": pl.Segmentation("tracked_runs", {"max_gap_frames": 10},
                                            features=("path", "excursion", "outbound")),
                "submovement": pl.Segmentation("zero_crossings", within="movement",
                                               features=("path",)),
                "bout": pl.Segmentation("bouts", {"of": "movement", "max_gap_s": 0.5,
                                                  "min_count": 2}),
            }),
        pl.KeypointSettings(
            "jaw", smooth=pl.Smooth(cutoff_hz=30.0),
            segments={"jaw_movement": pl.Segmentation(
                "above_threshold", {"on": 20.0, "off": 1.0, "max_gap_frames": 5},
                features=("path", "spectrum"))},
            feature_options={"spectrum": {"column": "x", "band": (2, 20)}}),
    ),
)
result = pl.process(pose, settings, session="...")   # dict of tables + record
pl.write(result, out_dir); pl.read(out_dir, tables=["movements"])
```

`Settings.to_dict()` / `from_dict()` round-trip through JSON; the record stores it.

### Smoothing

Per run of valid frames (missing stretches up to `max_gap_s` bridged): add samples where the
camera dropped frames, fill gaps by interpolation, Butterworth low-pass forward and backward,
differentiate on time. Runs are filtered separately, so nothing is interpolated across a long gap
and one run's edges do not leak into the next. Positions and velocities are NaN on frames that
were not valid; filled values only feed the filter. This deliberately differs from before 0.5.0
near movement edges.

### Dynamic foraging task module

- Frame times: `video_timing_qc.correct_video_timing` (with the trigger log when present);
  `ValueError` when the verdict is not `use`.
- Clock: session time, seconds from the first go cue; `time_zero` from the trials table
  (`goCue_start_time - goCue_start_time_in_session`).
- Licks belong to the movement they fall in, within `lick_tolerance_s` (0.01 s).
- Movements get `trial` (last go cue at or before the start), `go_cue_time`,
  `latency_from_go`, `spans_go_cue`, `number_in_trial`, `n_licks`, and from the first lick
  `lick_time, cue_response, rewarded, lick_side`. `cue_response`/`rewarded` are real booleans.
- Trials table = NWB trials plus `n_movements, n_lick_movements, first_movement_id,
  first_movement_latency, cue_response_movement_id, cue_response_movement_number, lick_latency,
  n_licks, lick_coverage`.
- Record adds `clock`, `frames`, `timing`, `quality` (`n_licks, n_licks_in_movement,
  lick_coverage, movement_duration_p50`). Read with `quality_summary(session_dir)`.
- Output folder `<out_root>/<session>/`: `pose, kinematics, landmarks, movements, submovements,
  bouts, trials, licks, events` (parquet) and `record.json` (written last; its presence means
  done).

## Old → new

### Files

| Before 0.5.0 | 0.5.0 |
|---|---|
| `<session>/intermediate_data/tongue_kins.parquet` | `<session>/kinematics.parquet` (all keypoints, long) |
| `tongue_movs.parquet` | `movements.parquet` (+ `submovements`, `bouts`) |
| `kps_raw_<kp>.parquet` | `pose.parquet` (raw, long) and `landmarks.parquet` |
| `tongue_trials.parquet`, `nwb_df_trials.parquet` | `trials.parquet` |
| `nwb_df_licks.parquet` | `licks.parquet` (`movement_id` replaces `nearest_movement_id`) |
| `nwb_df_events.parquet` | `events.parquet` |
| `tongue_quality_stats.json` | `record.json` → `quality` |
| `batch_error_log.txt` | `errors.txt` |

### Functions

| Before | Now |
|---|---|
| `run_batch_analysis` | `tasks.dynamic_foraging_tongue.run_batch` (no QC figures or example clips) |
| `generate_tongue_dfs` | `tasks.dynamic_foraging_tongue.run_session` |
| `load_keypoints_from_csv` | `pose_io.read_pose` |
| `integrate_keypoints_with_video_time` | `dynamic_foraging_tongue.frame_times` + `signal.add_time` |
| `mask_keypoint_data`, `kinematics_filter` | `signal.mask`, `signal.smooth` |
| `segment_movements_trimnans` | `segment.tracked_runs` |
| `segment_submovements`, `add_radial_velocity` | `segment.zero_crossings` / `speed_minima`, `signal.radial_velocity` |
| `aggregate_tongue_movements`, `compute_outbound_metrics` | `features.describe` with `path`, `excursion`, `outbound` |
| `annotate_*`, `assign_movements_to_licks`, `add_lick_metadata_to_movements`, `annotate_movement_timing`, `get_trial_level_df` | `dynamic_foraging_tongue.annotate_movements`, `summarize_trials` |
| `load_tongue_quality_stats`, `get_quality_summary`, `session_already_done` | `pipeline.read_record`, `dynamic_foraging_tongue.quality_summary`, `pipeline.is_done` |
| `get_session_name_from_path`, `find_video_csv_path`, `find_behavior_videos_folder` | `dynamic_foraging_tongue.session_name_from_path`, `find_inputs` |
| QC plots (`plot_*`), `select_percentile_movements`, `extract_example_clips_for_session` | removed (follow-up) |
| `ephys.tongue_ephys.load_intermediate_data`, `get_events_dict` | same names, read the new files; `firstMove` from `movements.start_time` |
| `video_clip_utils.extract_trial_clip(..., tongue_kins, ...)` | takes the `kinematics` table (`frame`, `time` in session time) |

### Columns

| Before (`tongue_kins` / `tongue_movs`) | Now |
|---|---|
| `time_in_session` | `time` (session time) |
| `time`, `time_raw` | dropped: `frame` + `time`; Harp time = `time + record.clock.time_zero` |
| `confidence` | `likelihood` (`pose`) |
| `xv, yv, v` | `vx, vy, speed` |
| `start_time, end_time, duration` | same |
| `n_datapoints`, `dropped_frames_n`, `dropped_frames_pct` | `n_valid`, `n_frames - n_valid`, derived |
| `total_distance, peak_velocity, mean_velocity` | `path_length, peak_speed, mean_speed` |
| `min_x … max_yv` | dropped (absolute pixels / redundant) |
| `startpoint_x/y`, `endpoint_x/y` (absolute) | `start_x/y`, `end_x/y` relative to the median jaw |
| `max_x_distance`, `max_y_distance` | `abs(extent_x)`, `abs(extent_y)` (signed, relative) |
| `max_x_from_jaw`, `max_y_from_jaw`, `max_*_from_jaw_*` | dropped (absolute pixels; `extent_*` + landmark if needed) |
| `time_to_endpoint` | `out_duration` (they were equal) |
| `excursion_angle_deg` | `angle_deg` |
| `out_total_distance, out_peak_velocity, out_mean_velocity` | `out_path_length, out_peak_speed, out_mean_speed` |
| `has_lick, lick_count, first_lick_index` | `n_licks > 0`, `n_licks`, licks' `movement_id` |
| `lick_time, cue_response, rewarded, event` | same, `lick_side` ('left'/'right') replaces `event` |
| `trial` (NA when spanning a go cue) | `trial` + `spans_go_cue` |
| `goCue_start_time_in_session` | `go_cue_time` |
| `movement_number_in_trial, movement_latency_from_go` | `number_in_trial, latency_from_go` |
| `cue_response_movement_number, lick_latency`, `movement_before_cue_response` | on `trials`; `number_in_trial < cue_response_movement_number` |
| `tongue_quality_stats.coverage_pct` (0-100), `duration_p50` | `quality.lick_coverage` (0-1), `movement_duration_p50` |

The reference also changes: the jaw is the **median** of confident jaw points (it was the mean
of all points, even low-confidence ones). The parity check below uses the old jaw position, so
values agree exactly there. On re-run, relative positions shift by the difference between the
two jaw estimates.

## Verification

Run with `kinematics_analysis/.venv-fip` (Python 3.12, pandas 2.3, numpy 2.0, scipy 1.13).

- **Tests**: `python -m unittest discover` — 183 tests pass (2026-10-26), 57 of them new for
  the kinematics core and task module. New modules are at 97–100% line coverage; flake8 and
  interrogate (100%) pass on them. `test_video_clips.ExtractTrialClipTest` and the import test
  need `cv2`/`moviepy`, absent in that environment (as on `main`); they pass with stand-in
  modules for those two imports, which `extract_trial_clip` does not use.
- **Parity against the old pipeline** (`scripts/compare_kinematics_parity.py`), on session
  `behavior_716325_2024-05-31_10-31-14` (2,689,718 frames, 7,393 movements), from its old frame
  table and its rows in `all_tongue_movements_04022026.parquet`:
  - `segment.tracked_runs(valid, 10)` reproduces every old movement exactly.
  - `features.describe` reproduces `duration, n_datapoints, dropped_frames_n, total_distance,
    peak_velocity, mean_velocity, startpoint_*, endpoint_*, max_y_distance, time_to_endpoint,
    out_*` to float precision (≤ 2e-12), `excursion_angle_deg` to 4e-6° and `max_x_distance` to
    3e-7 px (both limited by recovering the jaw from the old columns to 1e-6 px; the old
    `max_x_distance` equals `abs(extent_x)` for the 97.8% of movements farthest on +x).
  - Smoothing is not compared: the old frame table holds only filtered positions.
- **Speed**: `process` with `TONGUE_SETTINGS` on that session takes 36 s (7,393 movements,
  22,526 submovements, 971 bouts).
- **Not run**: end to end on raw inputs (LP CSV + video CSV + trigger log + NWB). Those are on
  Code Ocean only. `run_session` and `run_batch` are tested on simulated files with the NWB calls
  mocked.

## Follow-ups

- **kinematics_analysis migration** (with the batch re-run its TODO already stages): update
  `data_loading.py`, `build_all_tongue_movements.py` (quality filter → `quality_summary`, coverage
  now 0–1), `ephys_utils`, `lickometer_qc.load_session` (`mask_keypoint_data` → `signal.mask`),
  `kin_utils.load_kps_raw` / `jaw_position` (→ `landmarks.parquet`), `val_02`/`val_03`,
  `model_quality`, `tongue_movements_all`, and notebooks using the renamed columns.
- QC figures (confidence, lick coverage, example clips) as a small task-side module.
- `run_batch(screen=...)` gate from `video_screen` (`VIDEO_SCREEN_PLAN.md`).
- Spout distance per frame and pose-derived lick detection (`tongue_lickometer_utils`) as
  `above_threshold` on distance to a landmark.
- A `flip_x`/`flip_y` choice for the bottom camera so +x is the same side in every rig (needs
  the rig geometry; left as False).
- Move `kinematics/` to its own package once a second project uses it.

## Changes

- Revision 1 (2026-10-26): first design and implementation.
