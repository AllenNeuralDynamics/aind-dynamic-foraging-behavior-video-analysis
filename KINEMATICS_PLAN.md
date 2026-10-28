# Plan: kinematics — from tracker output to tidy tables

> **Status: revision 4 (2026-10-27), released in 0.5.0 (PR #16, tag `v0.5.0`).**
> Replaces `kinematics/tongue_analysis.py` and `kinematics/tongue_kinematics_utils.py` with a
> general core (`kinematics/`) and one task module (`kinematics/tasks/dynamic_foraging_tongue.py`).
> A clean break: output files and column names change (map below); `kinematics_analysis` updates
> its loaders when it re-runs the batch. Written so a new contributor or agent can pick it up
> without the conversation that produced it.
>
> **How to use it:** `KINEMATICS_GUIDE.md`.

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
`speed_minima` are the uncommitted `feat/submovements` work, ported with its tests carried over.
Compared with that code on 14,400 random cases (both methods, nested and standalone, gap,
deadband, minimum length, merge or drop), every frame's id, number and direction match, except
in one place, changed on purpose (revision 2):

- **Direction of a piece that absorbed a short one.** The original took the sign of the piece's
  largest sample, so a 1-frame blip merged into a retraction could flip it to +1 and fuse it
  with the protrusion. This is common at the end of a lick, where the tongue is back at the jaw
  and radial velocity is unreliable. Direction is now the sign of the sum of the piece's values
  outside the deadband. Unmerged pieces are unchanged; the 1,240 differing cases are all
  `zero_crossings` with `min_samples > 1` and `short="merge"`.

**Submovement defaults** (`TONGUE_SETTINGS`): every reversal of radial velocity is kept, with no
minimum length and no deadband, so nothing is merged or fused and no decision is made about
which pieces are real. Each piece carries `radial_velocity_sum` (its net motion along the
axis) and `radial_velocity_peak` (signed value farthest from 0) next to `n_frames` and
`direction`, so pieces can be filtered or merged later without rereading frames. Pieces of 1
frame have NaN path measurements.

What a minimum length would do, on session `behavior_716325_2024-05-31_10-31-14` (7,393
movements, radial velocity from the old filtered positions, new direction rule):

| `min_samples` | Pieces per movement | Out then back | Out then back, lick movements | One piece | 3+ pieces |
|---|---|---|---|---|---|
| **1 (default)** | **3.57** | **32.2%** | **34.6%** | **1.8%** | **64.9%** |
| 3 | 2.88 | 54.7% | 56.9% | 3.1% | 41.6% |
| 5 | 2.43 | 71.5% | 74.5% | 4.1% | 23.9% |
| 8 | 2.08 | 85.1% | 90.6% | 6.3% | 8.1% |

In movements with 3+ pieces, extra pieces sit at the start (median 6 frames), the end (median 10)
or the middle (median 9, with a peak at a median 22% of the movement's largest radial
velocity). The most common pattern after a plain out-and-back (2,384 movements) is a second
out-and-back within one tracked movement (`(1, -1, 1, -1)`, 1,319). At the ends the tongue is
near the jaw, where radial velocity is least reliable.
5 frames (10 ms) is the shortest reversal the 50 Hz smoothing resolves (half its period). A
deadband barely matters (0 to 200 px/s changes the out-then-back share by 3.3 points at 5
frames; median |radial velocity| is 602 px/s). Submovements get the same feature groups as
movements and their own lick matches.

**Merging, when it is decided** (follow-up): a separate step after `zero_crossings`, like
`bouts`, that groups raw pieces into merged ones and gives each raw piece the merged piece's id.
Raw pieces stay the saved primary table; the rule is explicit in the settings; the link from raw
to merged is kept. Its inputs are already saved (`n_frames`, `direction`,
`radial_velocity_sum`). `min_samples` / `short` in `zero_crossings` and `speed_minima` would
then be removed, so there is one way to merge.

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
were not valid; filled values only feed the filter. The filter is a Butterworth low-pass of
order `order / 2` (default 2, 50 Hz) applied forward and backward (`sosfiltfilt`), so 4th order
in effect with no phase shift; at each end of a run the signal is extended by its point
reflection (`padtype="odd"`, 9 samples), which continues it at its current slope. Velocity is
`np.gradient` on time: central differences inside, one-sided at the first and last frame.

Before 0.5.0 the whole session was filled by one cubic spline, including the 40-120 ms between
licks when the tongue is in the mouth and unobserved, then filtered once. A movement's edge
frames were therefore averaged (over the filter's ±5 frames) with that invented path. The new
method assumes instead that the tongue keeps its current velocity past the edge, which is closer
to a tongue that becomes visible already moving. Simulated licks with known velocity
(`scripts/simulate_smoothing_edges.py`; speed error in px/s, true speeds about 1,500-4,000
px/s):

| Noise | Method | Edge frames (first/last 3), median | Edge, 90th pct | Interior, median | Peak speed bias |
|---|---|---|---|---|---|
| none | before 0.5.0 | 45 | 147 | 1.6 | −4% |
| none | 0.5.0 | 11 | 56 | 1.3 | −1% |
| 1 px | before 0.5.0 | 209 | 640 | 50 | +9% |
| 1 px | 0.5.0 | 124 | 322 | 49 | +4% |
| 2 px | before 0.5.0 | 407 | 1213 | 100 | +29% |
| 2 px | 0.5.0 | 248 | 638 | 98 | +10% |

On five real sessions (Verification) positions are unchanged (median 0.01 px), interior speeds
are identical (median ratio 1.000) and edge speeds move by −4.5% to +5.6% per session, in
neither direction consistently. In both versions edge speeds are 2-3 times less accurate than
interior ones, and peak speed falls on an edge frame in 49-96% of movements, so `peak_speed`
largely measures the tongue as tracking begins or ends; it is also biased up by noise (the
largest of noisy values).

### Dynamic foraging task module

- Frame times: `video_timing_qc.correct_video_timing` (with the trigger log when present);
  `ValueError` when the verdict is not `use`.
- Clock: session time, seconds from the first go cue; `time_zero` is the first go cue's Harp
  time from the NWB trials (`load_nwb_tables`, as before 0.5.0). The data-utils trials table
  has only `goCue_start_time_in_session`.
- Licks belong to the movement (and submovement) they fall inside, within half a frame
  (`lick_tolerance_s`, default `0.5 / fs_hz` = 1 ms): `licks` gets `movement_id` and
  `submovement_id`; both segment tables get `n_licks` and the first matched lick's details
  (`match_licks`). This is the rule before 0.5.0: the old code matched a lick to its nearest
  frame (within 1 ms at 500 Hz; its 10 ms tolerance only mattered across dropped frames) and
  took that frame's movement. A lick is the tongue on the spout, so it must fall inside a
  tracked movement; the only allowed slack is timing between lickometer and camera. Licks just
  outside are misses for coverage, but are kept: `closest_movement_id` and
  `closest_movement_offset_s` (0 inside, negative before the movement's start, positive after
  its end), the same for submovements, so a looser rule can be applied later.
  Revision 3 briefly matched within 10 ms of either end; on Code Ocean that raised one session's
  coverage from 0.83 to 0.95 (below).
- Trial checks (0.5.1): data-utils' `create_df_trials` raises "Reward before choice time" for
  2025 sessions where an earned reward precedes the choice. On Code Ocean that stopped 19 of 53
  sessions. In the five checked, 1-5 trials each (of 81-200 earned rewards) were early by a
  median of 0.2-1.1 s per session (up to 1.2 s): mislabeled rewards rather than clock jitter. Kinematics does not use reward times.
  `run_batch(ignore_trial_errors=True)` turns the check into a warning; `trials` gets
  `reward_before_choice` for exactly those trials and `quality` counts them, so reward analyses
  drop trials, not sessions. Off by default; saved in the record (`nwb.ignore_trial_errors`).
- `run_batch(screen=...)` takes the `video_screen` table and skips sessions whose camera is not
  in use (`excluded: <reason>`) or was not screened.
- Movements get `trial` (last go cue at or before the start), `go_cue_time`,
  `latency_from_go`, `spans_go_cue`, `number_in_trial`, `n_licks`, and from the first lick
  `lick_time, cue_response, rewarded, lick_side`. `cue_response`/`rewarded` are real booleans.
- Trials table = NWB trials plus `n_movements, n_lick_movements, first_movement_id,
  first_movement_latency, cue_response_movement_id, cue_response_movement_number, lick_latency,
  n_licks, lick_coverage`.
- Record adds `clock`, `frames`, `timing`, `quality` (`n_licks, n_licks_in_movement,
  lick_coverage, lick_coverage_within_10ms, movement_duration_p50`). A gap between the two
  coverages means licks sit just outside tracked movements. Read with `quality_summary(session_dir)`.
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
| `nwb_df_licks.parquet` | `licks.parquet`: `movement_id` replaces `nearest_movement_id` (same rule); new `closest_movement_id`, `closest_movement_offset_s` (and for submovements) |
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
| `video_clip_utils.extract_trial_clip(..., tongue_kins, ...)` | `tasks.dynamic_foraging_tongue.cut_trial_clips(trials, kinematics, ...)` (`video_clip_utils` removed; clips through `video_clips.cut_event_clips`) |
| `video_clip_utils.find_labeled_video` | `tasks.dynamic_foraging_tongue.find_labeled_video` |

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

- **Tests**: `python -m unittest discover` — 189 tests pass (2026-10-27), 66 of them in the new kinematics test files, for
  the kinematics core and task module. New modules are at 97–100% line coverage; flake8 and
  interrogate (100%) pass on them. With `video_clip_utils` removed, nothing in the
  package imports `cv2` or `moviepy`, so the import test and
  `test_video_clips.CutTrialClipsTest` need no stand-ins.
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
- **Old vs new batch**: `scripts/compare_old_new_kinematics.py <old_root> <new_root>` compares
  each session's movements (matched by start and end time), mapped features (median and 99th
  percentile relative difference), lick matches and coverage. Tested on outputs built to match;
  to be run on Code Ocean.
- **Code Ocean, end to end on raw inputs** (2026-10-27, revision 3 code, commit `3eaa746`):
  five sessions from `pred_csv_list_20250113.json` run through `run_batch_analysis` (v0.4.0, the
  capsule image) and `run_batch` (this branch, through `PYTHONPATH`), then compared:
  - Movements identical in every session (7,393 / 6,914 / 6,650 / 6,472 / 6,371: same count,
    start and end times, duration, valid frames). Matched lick times identical.
  - Smoothing: see Smoothing above (positions 0.01 px, interior speeds equal, edges ±5%).
  - Jaw reference moved 2.9-5.7 px in every session (always +x, +y: the old mean included
    low-confidence points); angles changed by a median of 2.6-5.0°.
  - Licks: with the 10 ms rule of revision 3, coverage rose in two sessions. Where licks fall
    relative to movements:

    | Session | Licks | Inside | 0-10 ms before a start | 0-10 ms after an end | Farther |
    |---|---|---|---|---|---|
    | 716325 2024-05-31 | 5,846 | 5,833 | 2 | 1 | 10 |
    | 717259 2024-06-28 | 6,166 | 5,275 | 51 | 15 | 825 |
    | 717263 2024-07-24 | 5,092 | 4,150 | 614 | 76 | 252 |
    | 751004 2024-12-20 | 5,269 | 5,237 | 0 | 0 | 32 |
    | 751004 2024-12-21 | 5,782 | 5,715 | 2 | 1 | 64 |

    In 717263, 614 licks land up to 10 ms *before* a tracked movement starts: the tracker picks
    the tongue up late, or the lickometer and camera clocks are offset by a few ms. Either way
    movement onsets there may be late. The strict rule (revision 4) counts them as missed and
    keeps their offsets. Old coverage 0.827 equals "inside" plus about half of the licks within
    2 ms of an edge, confirming the old rule was inside-within-a-frame.
  - Rerun with revision 4 (commit `0a5ea95`): lick coverage equals the old value in all five
    sessions, and every lick is matched to the same movement as before except one in 717263 (at
    the half-frame boundary). 7-68 movements per session whose first frame was farthest from the
    old jaw now reach farthest later (old `out_duration` 0, new not).
  - **717263 is late tracking, not a clock offset.** In the 5 frames before movements with an
    early lick, confidence is 0.5-0.9 in 52% of frames (median 0.52) with the tongue 49 px from
    the jaw: visible, just under the 0.9 cutoff. In 717263 and 717259 tracked movements start
    ~40 px from the jaw, against 24 px in 716325 (amplitudes 56-63 px). A clock offset would
    move licks earlier relative to the farthest point, which tracking onset does not affect;
    licks there are 27.5 ms before it in 717263 against 36-41 ms in the other two. So where
    the tracker is less confident, movement onsets (and latencies) are late. This predates
    0.5.0.

## Follow-ups

- **kinematics_analysis migration** (with the batch re-run its TODO already stages): update
  `data_loading.py`, `build_all_tongue_movements.py` (quality filter → `quality_summary`, coverage
  now 0–1), `ephys_utils`, `lickometer_qc.load_session` (`mask_keypoint_data` → `signal.mask`),
  `kin_utils.load_kps_raw` / `jaw_position` (→ `landmarks.parquet`), `val_02`/`val_03`,
  `model_quality`, `tongue_movements_all`, and notebooks using the renamed columns.
- QC figures (confidence, lick coverage, example clips) as a small task-side module.
- Spout distance per frame and pose-derived lick detection (`tongue_lickometer_utils`) as
  `above_threshold` on distance to a landmark.
- A `flip_x`/`flip_y` choice for the bottom camera so +x is the same side in every rig (needs
  the rig geometry; left as False).
- A separate merge step for submovements, once the rule is decided (see Segmentation methods).
- Late movement onsets where tracker confidence is low (717263 above): consider two confidence
  levels, like `above_threshold` (a movement starts where confidence first passes ~0.5 and must
  reach 0.9 somewhere), checked against lick timing and against wrong-body-part detections,
  since confidence alone is not a valid noise filter.
- Move `kinematics/` to its own package once a second project uses it.

## Changes

- 0.5.1 (2026-10-27): `ignore_trial_errors` and `reward_before_choice` (Dynamic foraging task
  module).
- Revision 4 (2026-10-27): strict lick matching restored (half a frame) with
  `closest_<kind>_id` / `closest_<kind>_offset_s` kept; `lick_coverage_within_10ms`; clock from
  the NWB trials' first go cue; Code Ocean results and the smoothing simulation recorded;
  comparison script reports angles in degrees, jaw shift and per-lick agreement.
- Revision 3 (2026-10-27): submovements keep every reversal (no merging by default); pieces
  carry `<column>_sum` and `<column>_peak`; merging planned as its own step.
- Revision 2 (2026-10-26): submovements get movement features and lick matches; `min_samples=5`
  default from a real session; direction from the summed signal; `run_batch(screen=...)`;
  `scripts/compare_old_new_kinematics.py`.
- Revision 1 (2026-10-26): first design and implementation.
