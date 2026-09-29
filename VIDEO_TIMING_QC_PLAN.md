# Plan: `video_timing_qc` — QC and correction of behavior-video timestamps

> Status: pre-implementation, revision 1 (2026-09-29). Nothing in this plan is implemented yet.
> Written so a new contributor or agent can pick it up without the conversation that produced it;
> the evidence behind each decision is in "Background" and "Findings".

## Summary

Add one new subpackage, `aind_dynamic_foraging_behavior_video_analysis.video_timing_qc`, that takes
either **a session folder** or **a single behavior-video CSV** and:

1. **Checks** the per-frame timestamps: frame-number continuity, agreement between the Harp clock
   and the camera clock, and the Harp "slip" caused by dropped frames.
2. **Reports** a structured result (class, counts, row indices) instead of printing warnings.
3. **Optionally corrects** the Harp time per frame, for two well-understood failure modes only,
   marking every changed row with where its value came from.

It must use only the core dependencies (`numpy`, `pandas`), and add only new code at first: no
existing function changes behavior until a consumer opts in (see "Rollout without breaking
anything").

Why: the Bonsai workflow that writes these CSVs pairs video frames with Harp trigger times in
**arrival order**. When the host drops a frame, every later frame gets the Harp time of an earlier
trigger. In 18 of 97 recent FIP sessions, Harp times are wrong for essentially the whole session,
ending 280–670 s early. The existing QC (`integrate_keypoints_with_video_time`) does not detect
this at its current threshold, and at a tighter threshold its fix would overwrite the correct
clock. Every pipeline that puts video-derived signals on the behavior clock (Lightning Pose
kinematics, motion energy, video clips, BEAST latents) is affected.

## Background

### The video CSV

One CSV per camera, one row per **saved** video frame. Two layouts exist; `video_alignment.read_video_csv`
already tells them apart by content:

| Layout | Path under `behavior-videos/` | Header | Columns |
|---|---|---|---|
| Old / flat | `bottom_camera.csv`, `side_camera_right.csv` | none | `Behav_Time, Frame, Camera_Time` (the existing code names two more, `Gain, Exposure`, which are absent in the files checked) |
| New / AIND | `BottomCamera/metadata.csv`, `SideCameraRight/metadata.csv` | yes | `ReferenceTime, CameraFrameNumber, CameraFrameTime` |

- **Harp time** (`Behav_Time` / `ReferenceTime`): seconds on the Harp clock, the same clock as NWB
  event times, FIP, and spikes. This is the column analyses use.
- **Frame number** (`Frame` / `CameraFrameNumber`): the camera's own exposure counter. Increments by
  exactly 1 per exposure, including exposures whose frames were later lost.
- **Camera time** (`Camera_Time` / `CameraFrameTime`): the camera's hardware timestamp in
  **nanoseconds**. Also recorded per exposure, so it also records drops.

The video file holds exactly the saved frames: **row *i* of the CSV is frame *i* of the video**.
Verified for all 194 cameras below (MP4 decoded frame count == CSV row count, both layouts).
Anything computed per video frame (Lightning Pose keypoints, motion energy) is therefore row-aligned
with the CSV even when frames were dropped. Only the Harp time on each row can be wrong.

Cameras run at 500 Hz in these sessions (median camera interval 1.9996 ms, i.e. 500.09–500.13 fps).
Do **not** hard-code 500 Hz: derive the frame interval (IFI) from the median camera-time step.

### How Harp times go wrong: arrival-order pairing

`dynamic-foraging-task`'s `foraging.bonsai` joins camera frames to Harp trigger events with
`rx:Zip`, which pairs the *n*-th frame to arrive with the *n*-th Harp event. The camera exposes and
numbers every frame; the host then loses some. After one lost frame, every later row carries the
Harp time meant for the frame before it; each further loss shifts it one more. The error never
resets. `Aind.Behavior.JustFrames` uses the same pairing, so data from either workflow is exposed.
(Source: review by a colleague who analysed `ecephys_786867_2025-09-25_12-43-56`, three cameras at
500 Hz for 85 min, 40,000–47,000 frames lost per camera, Harp times 80–93 s early by the end.
That session is a good extreme test case.)

Consequences, all verified on `behavior_816212_2025-12-05_13-47-41` (bottom camera):

- 2,935,305 frames exposed (last frame number − first + 1); 2,748,281 saved (CSV rows = MP4 frames);
  187,024 dropped, all single frames, about one every 7–11 frames, starting within the first 5 s.
- CSV row *n* holds exactly the *n*-th Harp trigger time. Every Harp value is a real trigger time,
  on the wrong row.
- The last 187,024 trigger times are never paired with a row, so the CSV's Harp column covers only
  ~91.6 min of a 97.8 min recording, and ends 374 s early.
- Each camera drops independently (the side camera saved 2,749,302 rows), so the two cameras'
  errors differ.

### A second, unrelated failure: single-row Harp glitches

In some sessions one Harp value is ~983 ms too early (e.g. `behavior_809491_2025-10-02_09-23-46`,
row 183624: Harp steps −981.024 ms then +985.024 ms while the camera steps 2 ms). Frame numbers are
continuous. The same row is bad in both cameras, and the Harp trigger log (below) contains the same
bad value, so it originates on the Harp side, not in the pairing. A few sessions show smaller
single-row blips of 2–6 ms.

### The Harp trigger log (optional ground truth)

Each raw session has `behavior/raw.harp/BehaviorEvents/Event_94.bin`: one Harp event per camera
trigger. **The register's identity is inferred, not documented**: its event count equals the number
of exposed frames exactly, its values are evenly spaced at the frame interval, and the CSV's Harp
column equals its first *N* events to the microsecond. Confirm against the Harp Behavior register
map before depending on it.

Format: fixed 13-byte messages: `[type=3][length][address=94][port][payload type=17]`
`[seconds: uint32 LE][ticks: uint16 LE][payload: uint8 = 1][checksum]`;
time = seconds + ticks × 32 µs. Both cameras share the one log.

Checked on 4 sessions (clean, glitch, drops, new layout): event count == exposures for both
cameras, CSV row *n* == event *n* for every row.

### Other known issue (detection only)

The acquisition AVIs have bad keyframe timestamps; a default ffmpeg transcode drops six frames at
the second keyframe (~0.5 s in). `aind-video-utils` ≥ 0.7.0 transcodes correctly with
`--normalize_cfr true`. Detect by comparing the video's frame count with the CSV row count. None of
the 194 cameras below had this.

## Findings: 97 FIP sessions × 2 cameras

Sessions: `kinematics_analysis/metadata/me_sessions_fip_curated.csv` rows with
`used_in_fip05_07`. Per-camera results: `kinematics_analysis/metadata/video_csv_qc_fip.csv`
(branch `fip-motion-energy`, column `qc_class`). CSVs read from `s3://aind-open-data/<session>/behavior-videos/`.

| Class | Sessions | Description |
|---|---|---|
| `ok` | 64 | No frame gaps, no backward steps, no clock flags at 0.5×. |
| `harp_glitch` | 15 | No frame drops; 1–3 isolated bad Harp rows (mostly ~983 ms), same rows in both cameras. |
| `frame_drops` | 18 | 140,000–333,000 single-frame drops per camera (5–12%); Harp 280–670 s early by the end. |

Affected sessions: 816212 (all 8), 816214 (all 5), 818586 (2025-12-22, 2025-12-24, 2026-01-02),
818585 (2025-12-22), 808054 (2025-09-26). All between Sept 2025 and Jan 2026.

Threshold comparison on `|ΔHarp − ΔCamera|`:

- **2× IFI** (current default): flagged **0** of 7.9 million drops, because a *k*-frame drop makes
  the clocks disagree by *k* intervals and every drop here is *k* = 1. It caught only the ~983 ms
  glitches.
- **0.5× IFI**: flagged every drop and every glitch, and nothing else. Normal jitter in `ok`
  cameras: p99 ≤ 0.048 ms, max ≤ 0.114 ms, 10× below 1 ms.

Normal clock drift: in `ok` cameras, (camera span − Harp span) is −8.9 to −36.5 frames (18–73 ms)
over a session (the clocks differ by ~13–15 ppm). A slip metric needs a tolerance around that.

Correction test on `behavior_816212_2025-12-05_13-47-41`, both cameras, checked against the trigger
log:

| | Bottom | Side |
|---|---|---|
| Rows re-indexed exactly from the CSV | 93.6% (identical to the log, 0 µs) | 93.7% (0 µs) |
| Tail rows needing an estimate | 175,161 (last 6.2 min) | 174,194 (last 6.2 min) |
| Tail error, linear fit on the last 10 min of exact rows | max 0.062 ms, mean −0.002 ms | max 0.065 ms, mean +0.017 ms |
| Tail error, linear fit on all exact rows | max 0.077 ms | max 0.123 ms |
| Rows over 0.5× after correction (log-based) | 0 (was 187,024) | 0 (was 186,003) |

## What the existing QC does

`kinematics/tongue_kinematics_utils.py::integrate_keypoints_with_video_time(video_csv_path, keypoint_dfs)`:

1. Reads the CSV with `pd.read_csv(path, names=[Behav_Time, Frame, Camera_Time, Gain, Exposure])`
   and divides `Camera_Time` by 1e9.
2. `check_frame_monotonicity`: prints a warning and the rows where the frame step ≠ 1. Detects every
   drop. **Print only**; nothing downstream sees it.
3. `qc_and_fix_timing(tol_multiplier=2, bracket_tol=0.1, auto_fix=True)`: flags rows where
   `|ΔHarp − ΔCamera| > 2 × IFI` or either clock steps backward; for each flagged row, if exactly
   one column's step is off from IFI by more than the threshold, and that column 2 rows apart is
   within `bracket_tol` of 2 × IFI, replaces the row's value with the midpoint of its neighbours.
   Mutates the frame in place.
4. Trims keypoints / CSV to the shorter length and adds `time` / `time_raw` from `Behav_Time`.

Tested against the real function:

- **Glitch session** (`809491_2025-10-02`, full CSV): detects and fixes row 183624 correctly (to
  the neighbours' midpoint; afterwards max |ΔHarp − ΔCamera| 0.09 ms, Harp strictly increasing).
  Side effect: row 183625, which was correct, is also flagged (its step *out of* the bad row is
  +985 ms) and moved by 8 µs. Harmless only because flags are computed before any fix and the bad
  row was fixed first.
- **Frame drops, synthetic CSV built with arrival-order pairing** (one drop each of 1, 2, 3, 5
  frames): the 1-frame drop is not flagged; the 2-, 3- and 5-frame drops are flagged and **the
  fix overwrites the camera time** (by −2, −3, −5 ms), because Harp's step looks normal and the
  camera's looks wrong. It prints `Fixed idx=… in 'Camera_Time' by interpolation`. Harp is left
  shifted.
- **Frame drops at 0.5×** (first 50,000 rows of `816212_2025-12-05`, 3,299 drops): with only
  `tol_multiplier` changed to 0.5, it overwrites **all 3,299 camera times** and still leaves Harp
  shifted.
- **From code reading, not tested:** a header-bearing new-layout CSV read with `names=[...]` makes
  the header row a data row, so `Behav_Time` becomes text. `find_video_csv_path` prefers the new
  layout, so `tongue_analysis.generate_tongue_dfs` may fail or misbehave on new-layout sessions.
  Check before relying on either path.

Rule that follows: **never interpolate at or next to a frame gap, and treat the camera clock as the
trusted one**.

## Goals and non-goals

Goals:

- One place that decides whether a camera's timestamps are trustworthy, used by every pipeline.
- Accept a session folder (both layouts, all cameras) or a single CSV path / DataFrame.
- Structured, serialisable results (JSON-able dict, one row per camera for tables).
- Corrections that are opt-in, exact where possible, marked per row, and verified after applying.
- Work from the CSV alone; use the trigger log only for optional validation.
- Core dependencies only.

Non-goals:

- Changing any existing function's default behavior in the first release.
- Correcting anything other than the two failure modes above. Everything else is reported.
- Reading video files. The transcode check takes a frame count supplied by the caller (from
  `aind-video-utils`, `ffprobe`, or motion-energy metadata `n_frames_decoded`).

## Design

### Module layout

```
src/aind_dynamic_foraging_behavior_video_analysis/video_timing_qc/
    __init__.py      # public API re-exports
    io.py            # CSV -> canonical frame; session-folder discovery
    checks.py        # pure detection functions on arrays
    correct.py       # glitch fix, drop re-indexing, tail estimate
    report.py        # QCReport / CameraTiming dataclasses, to_dict, to_row
    triggers.py      # optional: parse Event_94.bin, validate against CSV
```

`video_alignment.py` stays where it is; `io.py` reuses `read_video_csv` and
`TIME_COLUMN_ALIASES` rather than duplicating layout detection.

### Canonical columns

`io.load_video_timing(csv_path_or_df) -> pandas.DataFrame` with one row per saved frame:

| Column | Type | Meaning |
|---|---|---|
| `row` | int | 0-based row = video frame index |
| `frame_number` | int64 | camera exposure counter |
| `camera_time` | float64 s | camera clock (ns / 1e9) |
| `harp_time_raw` | float64 s | Harp time as written in the CSV, never modified |

Layout-specific column names are mapped by position after `read_video_csv`, with a check that
headers, when present, are the known ones.

### Detection (`checks.py`)

All vectorised; each returns counts and row indices.

- **IFI**: `median(diff(camera_time))`.
- **Frame continuity**: `d = diff(frame_number)`; gaps `d > 1` (frames dropped = `d − 1`),
  repeats/backward `d ≤ 0`.
- **Backward steps** in Harp and camera time.
- **Clock disagreement**: `|ΔHarp − ΔCamera| > threshold × IFI`, default threshold **0.5**
  (parameter, so 2 can be reproduced). Split into flags on a frame gap and flags without one.
- **Harp glitch rows**, found from the **Harp column alone**: row *r* with
  `|harp[r] − (harp[r−1] + harp[r+1]) / 2| > 0.5 × IFI` and
  `|harp[r+1] − harp[r−1] − 2 × IFI| ≤ 0.5 × IFI` (the neighbours agree with each other). This
  picks the one bad row, not its successor. Runs of ≥ 2 consecutive bad rows are reported, not
  classed as glitches.
  Frame numbers and camera time are deliberately **not** part of the rule. Under arrival-order
  pairing the Harp column is the trigger sequence in order, evenly spaced whether or not frames
  were dropped, so a glitch shows up the same way in drop sessions. An earlier version of this
  plan also required frame steps of exactly 1 around the row; with a drop every 7–11 frames that
  misses about a quarter of glitches in drop sessions (it missed 1 of 4 in
  `behavior_816214_2025-12-02_08-28-39`, where a glitch row follows a 2-frame drop on the side
  camera).
  Caveat: this relies on the pairing. In a workflow that leaves real gaps in Harp at drops, the
  Harp steps next to a drop are > 1 IFI and the neighbour test fails, so the row is reported, not
  fixed.
- **Clock slip**: `((camera_time[-1] − camera_time[0]) − (harp[-1] − harp[0])) / IFI`, in frames;
  reported, with the expected range from `ok` sessions (about −40…0 frames for ~90 min) as context.
- **Transcode mismatch** (only if the caller passes `video_frame_count`): `video_frame_count − rows`.

### Classes

Per camera, first match wins:

| Class | Condition | Correctable |
|---|---|---|
| `unreadable` | CSV missing, empty, unknown columns, NaNs in required columns | no |
| `frame_order_error` | any frame step ≤ 0 | no |
| `transcode_mismatch` | video frame count given and ≠ rows | no (row alignment itself is broken) |
| `frame_drops` | any frame step > 1 | yes (drop correction) |
| `harp_glitch` | glitch rows only, no other clock flags | yes (glitch fix) |
| `clock_disagreement` | clock flags that are neither drops nor clean glitches | no |
| `ok` | none of the above | nothing to do |

A camera can carry both drops and glitches (2 of the 97 sessions do:
`behavior_816214_2025-12-02_08-28-39`, `behavior_816212_2025-12-10_13-27-38`). Glitches are fixed
first, then drops. The order matters: re-indexing copies Harp values to other rows, so an unfixed
glitch would be moved to a different row, and the post-checks would then refuse the session.

### Correction (`correct.py`)

`correct_video_timing(timing, report, *, fix_glitches=True, fix_drops=True, tail_fit_window_s=600)`
returns a copy of the timing frame with two added columns and an updated report:

- `harp_time`: corrected Harp time.
- `harp_source`: one of `original`, `glitch_interpolated`, `reindexed`, `estimated_camera_fit`.

Steps:

1. **Glitch fix.** For each glitch row: `harp[r] = (harp[r−1] + harp[r+1]) / 2`. Only Harp is ever
   changed. Camera time is never modified by this package.
2. **Drop re-indexing** (only if frame numbers are strictly increasing):
   `k = frame_number − frame_number[0]` (true trigger index of each row); `N = rows`.
   For rows with `k < N`: `harp_time[row] = harp_after_step1[k]`, source `reindexed` (or `original`
   where `k == row`, i.e. before the first drop).
   This uses only values already in the CSV, moved to the right row.
3. **Tail estimate.** Rows with `k ≥ N` need trigger times the CSV never recorded. Fit
   `harp = a + b × camera_time` on exact rows within the last `tail_fit_window_s` of camera time,
   predict for tail rows, source `estimated_camera_fit`. (Verified error ≤ 0.07 ms; fitting on the
   whole session gave ≤ 0.12 ms.)
4. **Post-checks; refuse on failure.** Corrected Harp strictly increasing; no row over the 0.5×
   threshold; linear-fit residual of `harp_time` against `camera_time` ≤ 1 ms over the session. If
   any fails, return the report with class `correction_failed` and no corrected column, never a
   partially corrected one.

Assumptions (stated in the report and docstrings):

- One Harp trigger per exposure, and no triggers lost. Verified via the trigger log on 4 sessions;
  checked indirectly on every session by the post-checks (a missing trigger leaves a 1-frame step
  error the post-check catches).
- The first saved row is the first exposure. If frames were lost before the first saved row, every
  row is off by that constant. **The CSV cannot detect this**; the trigger log can (event count >
  exposures). Record it as a known limitation.

### Optional trigger-log validation (`triggers.py`)

`read_trigger_log(path) -> ndarray` and `validate_against_triggers(timing, trigger_times) -> dict`:
event count vs exposures; CSV row *n* == event *n*; after correction, max |corrected − log[k]|
(expected: 0 for exact rows, ≤ ~0.1 ms for the tail). Used in tests and for one-off validation;
never required by `correct_video_timing`.

### Report (`report.py`)

`QCReport` dataclass, one per camera, with `to_dict()` (JSON) and `to_row()` (flat, for a table):
source path, layout, camera name (normalised `bottom_camera` / `side_camera_right` from either
layout), rows, IFI, fps, class, frame gaps, frames dropped, first drop row and time, backward
counts, clock flags on / off gaps, glitch rows (list), clock slip (frames, s), p99 and max
|ΔHarp − ΔCamera|, transcode mismatch, correction applied / sources counts / post-check results,
package version, threshold used.

### Public API

```python
from aind_dynamic_foraging_behavior_video_analysis import video_timing_qc as vtq

timing = vtq.load_video_timing("…/behavior-videos/bottom_camera.csv")
report = vtq.check_video_timing(timing, threshold=0.5, video_frame_count=None)
fixed, report = vtq.correct_video_timing(timing, report)          # opt-in

reports = vtq.check_session("…/behavior_816212_2025-12-05_13-47-41")  # all cameras, both layouts
vtq.frame_index_for_harp_time(event_times, fixed)                  # event -> video frame row
```

`frame_index_for_harp_time` (searchsorted on corrected `harp_time`) is how consumers should map
behavior events to video frames. See "Consequence for `video_alignment`".

## Consequence for `video_alignment`

`behavior_time_to_video_time(t, first_frame_behavior_time)` computes video position by subtraction.
That assumes Harp time advances one IFI per saved row. In a drop session with **uncorrected** Harp,
that assumption happens to hold (the pairing makes row *n*'s Harp ≈ first + *n* × IFI), so today's
clip positions are consistent with the file even though the events are mapped to the wrong frames.
Once Harp is corrected, subtraction is wrong in both senses: corrected Harp advances by more than
one IFI across each drop, but the file has no frame there.

So:

- Don't change `video_alignment`'s functions. Document the caveat in their docstrings.
- New code that maps behavior events to video frames (the `video_clips.py` plan, which already
  chooses the frame index first) should use `frame_index_for_harp_time` on corrected timing, then
  convert frame index to file position (`index / fps` for a CFR file).

## Consumers

| Consumer | Where | Today | With this package |
|---|---|---|---|
| Lightning Pose tongue kinematics | `kinematics/tongue_analysis.py::generate_tongue_dfs` → `integrate_keypoints_with_video_time` | Uses `Behav_Time` after the 2× fix as keypoint `time_raw`. Drop sessions pass silently with shifted times; drops of ≥ 2 frames get camera times overwritten. | Opt-in path: check + correct, keypoint time from corrected `harp_time`, save `QCReport` next to the session outputs and into the batch summary; skip or mark sessions that fail. |
| Motion-energy table | `kinematics_analysis/code/fip_me_aligned_table_plan.md` (branch `fip-motion-energy`) | Planned | Store corrected `harp_time`, `harp_source`, QC class per row / camera; ME is row-aligned with the CSV, so no other change. |
| FIP motion energy in analysis | `kinematics_analysis/code/fip_utils.py::motion_energy_to_session` | Reads CSV via `read_video_csv`, Harp time via `TIME_COLUMN_ALIASES` | Read the ME table's corrected time instead. |
| Video clips | `VIDEO_CLIPS_MIGRATION_PLAN.md` | Planned | Event → frame via `frame_index_for_harp_time`. |
| BEAST latents | `aind-BEAST-train-test/code/analyze_latents.ipynb` | `compute_video_session_offset`, `session_time_to_video_time` | Same as clips for drop sessions. |
| Notebooks | `kinematics_analysis/code/{model_quality,test_session_wrapper}.ipynb` call `integrate_keypoints_with_video_time`; `val_03`, `val_04` use `session_time_to_video_time` | Unchanged | Unchanged until migrated. |

## Rollout without breaking anything

1. **Phase 1: new subpackage only.** No edits to existing modules. Ship with tests. Safe by
   construction: nothing imports it yet.
2. **Phase 2: opt-in in the LP path.** Add a keyword to `integrate_keypoints_with_video_time`,
   e.g. `timing_qc="legacy"` (default, today's exact behavior) or `"strict"` (new package; raises
   or returns the report for sessions it can't trust). Same for `generate_tongue_dfs` /
   `run_batch_analysis`. Legacy output must stay identical; add a regression test that pins
   today's output on a fixture.
3. **Phase 3: reference comparison, then flip the default.** Run the LP batch in both modes on a
   set covering `ok`, `harp_glitch`, `frame_drops`, and new-layout sessions. Expect identical
   times for `ok`; changes only on glitch rows (µs to ~1 s) and drop sessions (up to minutes).
   Record the comparison in this plan, then make `"strict"` the default in a minor release with a
   changelog note. Keep `"legacy"` for one release.
4. **Phase 4: migrate other consumers** (ME table, clips, BEAST) to `frame_index_for_harp_time`
   and the corrected time.

Consumers pin this library by SHA or tag (see `PYTHON_311_UPGRADE_PLAN.md`), so none of them
changes until it moves its pin.

### Glitch plus drops: verified

`behavior_816214_2025-12-02_08-28-39` (2,665,562 triggers; bottom 172,631 drops, side 166,788; two
~983 ms glitches at trigger events 554395 and 1357400, present in the trigger log itself and on the
same CSV rows in both cameras):

- The Harp-only rule found both glitches in both cameras.
- Glitch fix, then re-indexing, then tail estimate: 0 rows over 0.5×, Harp strictly increasing,
  both cameras.
- Against the trigger log with its own two glitches repaired: re-indexed rows exact (0 µs), tail
  within 0.06 ms. Against the raw log, the only differences are at those two triggers, where the
  log is wrong. (On the side camera, trigger 554395's exposure was itself dropped, so no row
  needs it after re-indexing.)
- Skipping the glitch fix: 4 (bottom) / 2 (side) rows over 0.5× after re-indexing, so the
  post-check refuses the session.

## Tests

`unittest`, as the rest of the repo (CI runs `coverage run -m unittest discover` and
`flake8 --select=E9,F63,F7,F82`; black line length 79).

Synthetic (fast, in CI): build CSVs by simulating arrival-order pairing (camera exposes every
frame; drop a set of exposures; row *n* gets trigger *n*):

- clean; glitch (one row −983 ms; one row +3 ms); consecutive bad rows (not a glitch);
- glitch next to a drop (row after a 2-frame drop), and glitch on a trigger whose frame was dropped;
- drops of 1, 2, 3, 5 frames; drops starting at row 0 vs later; many drops (tail estimate);
- both layouts (header / headerless), extra columns, nanosecond camera time;
- frame repeats / backward steps; video frame count mismatch;
- post-check failure (inject a lost trigger) → `correction_failed`, no corrected column;
- threshold 2 reproduces the legacy flags on the same inputs.

Real-data fixtures (small, committed): 50,000-row slices of the three cases, with expected counts
from the findings above. Full-session checks against S3 are optional and skipped without network:

| Session | Case | Expected (bottom camera) |
|---|---|---|
| `behavior_800886_2025-09-03_13-03-45` | ok, flat | 2,672,569 rows, 0 gaps, 0 flags |
| `behavior_809491_2025-10-02_09-23-46` | harp_glitch | glitch row 183624 (−981.024 ms step), 0 gaps |
| `behavior_816212_2025-12-05_13-47-41` | frame_drops | 2,748,281 rows, 187,024 gaps, 2,935,305 exposures, tail 175,161 rows |
| `behavior_818586_2026-01-21_09-43-54` | harp_glitch, new layout | 2,517,841 rows, 2 flags |
| `behavior_816214_2025-12-02_08-28-39` | frame_drops + harp_glitch | 2,492,931 rows, 172,631 gaps, glitch rows 554395 and 1357400 |
| `ecephys_786867_2025-09-25_12-43-56` | extreme drops, 3 cameras | 40–47k drops per camera (colleague's numbers; not yet checked here) |

## Phases

| Phase | Deliverable |
|---|---|
| 1 | `video_timing_qc` (io, checks, report, correct, triggers) + synthetic and fixture tests |
| 2 | Opt-in in `integrate_keypoints_with_video_time` / `tongue_analysis`, legacy regression test |
| 3 | LP batch comparison, flip default, release |
| 4 | ME table, clips, BEAST migrated |

## Open questions

1. Confirm register 94 is the camera-trigger event (Harp Behavior register map / rig owners).
2. Whether any workflow version logs triggers differently (older rigs, `Aind.Behavior.JustFrames`).
3. Can frames be lost before the first saved row? Check the trigger log (event count vs
   exposures) across all 97 sessions once, and add it as an optional check.
4. Default action for `frame_drops` in consumers: correct, or exclude? This plan corrects only on
   request; consumers decide.
5. Whether to also offer a trigger-log-based correction (exact tail) as an alternative when the log
   is available.
6. Whether this should become its own repository later. Keeping it dependency-free and independent
   of the kinematics code makes that a move, not a rewrite.

## Out of scope

- Fixing the Bonsai workflow (`rx:Zip` pairing): report upstream to `dynamic-foraging-task` and
  `Aind.Behavior.JustFrames` owners.
- Re-transcoding videos.
- Changing NWB or FIP timestamps.
