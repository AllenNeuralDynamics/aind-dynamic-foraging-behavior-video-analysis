# Plan: `video_timing_qc` — QC and correction of behavior-video timestamps

> Status: revision 5 (2026-09-30; decision reordered, checks added after a top-down review). Phases 1 and 2 implemented on branch `plan/video-timing-qc`;
> Phase 3 in progress (first pipeline comparison done; the no-frames-lost rule was added after
> it). Written so a new contributor or agent can pick it up without the
> conversation that produced it; the evidence behind each decision is in "Background" and
> "Findings".

## Summary

One new module, `aind_dynamic_foraging_behavior_video_analysis.video_timing_qc`, that takes a
behavior-video CSV (or a whole `behavior-videos` folder) and:

1. **Checks** the per-frame timestamps: frame-number continuity, agreement between the Harp clock
   and the camera clock, isolated bad Harp rows, and the Harp "slip" caused by dropped frames.
2. **Reports** the result as a plain dict (class, counts, row indices), or a DataFrame with one
   row per camera.
3. **Corrects** the Harp time per frame for two well-understood failure modes only, marking every
   row with where its value came from. Rows the CSV cannot supply are estimated from camera time,
   or read from the Harp camera trigger log if the caller passes it.

It uses only the core dependencies (`numpy`, `pandas`). `integrate_keypoints_with_video_time`
now uses it in place of its old QC (Phase 2).

Why: the Bonsai workflow that writes these CSVs pairs video frames with Harp trigger times in
**arrival order**. When the host drops a frame, every later frame gets the Harp time of an earlier
trigger. In 18 of 97 recent FIP sessions, Harp times are wrong for essentially the whole session,
ending 280–670 s early. The old QC in `integrate_keypoints_with_video_time` did not detect
this at its threshold, and at a tighter threshold its fix would have overwritten the correct
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
trigger. Register 94 is `Camera1Frame` in the Harp Behavior `device.yml`
(`harp-tech/device.behavior`). Its event count equals the number of exposed frames, and the CSV's
Harp column equals its first *N* events.

Format: fixed 13-byte messages: `[type=3][length=11][address=94][port][payload type=17]`
`[seconds: uint32 LE][ticks: uint16 LE][payload: uint8 = 1][checksum]`;
time = seconds + ticks × 32 µs. All cameras share the one log. `read_harp_trigger_log` parses it
with numpy and gives the same times as `harp.io.read(path)` from `harp-python` (checked on the 6
sessions below, difference 0). `harp.data.open_dataset` does not apply here: these sessions have
no `device.yml` and do not use `<Device>_<address>.bin` names.

Checked on 6 sessions (clean, glitch, drops, glitch + drops, new layout, 3-camera ecephys):
event count == exposures for every camera, and CSV row *n* == event *n* for every row.

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

## Findings: attached LP sessions (Phase 3 run, Jan–Jul 2025)

The first pipeline comparison (`kinematics_analysis/code/verify_video_timing_qc.ipynb`) scanned the
57 attached sessions with LP predictions and ran 9. None had dropped frames. Two failure modes
not seen in the FIP sessions turned up, each in 2 of the 9:

**Harp clock step** (`behavior_751181_2025-02-26_11-51-15`, `behavior_754897_2025-03-14_11-28-48`).
Twice per session, exactly 1,280 s apart and on a whole Harp second, Harp − camera time drops by
1.8–2.3 ms and stays there. At the step row Harp goes *backward* by 0.06–0.58 ms while camera time
steps 2 ms; no other row disagrees by more than 0.2 ms. The step is in the trigger log too, and
the same board's 1 kHz analog stream (`Event_44`) jumps back ~2 ms at the same instants, so the
Harp Behavior board's own clock stepped. The frames were still captured every 2 ms (camera time is
smooth); only the board's timestamps moved. Hypothesis (unconfirmed): the board runs ~1.7 ppm fast
and is pulled back into sync. Not seen in the 97 FIP sessions (Sep 2025+), where a 1 ms step would
have been flagged. Whether the lickometers and sound card (separate Harp devices) step with it is
unknown. The old QC interpolated the one backward row (+1.0–1.3 ms) and kept the rest. **Decision:
refuse** (`harp_evenly_spaced` fails). The behavior clock itself is off, and a value between the two
clocks would be neither; the pipeline also cannot take backward time as written (`kinematics_filter`
asserts its time base, `merge_asof` needs sorted keys).

**Corrupted camera metadata** (`behavior_763590_2025-05-02_11-07-07`,
`behavior_784803_2025-07-02_13-41-41`). Frame number and camera time jump together by +1,049 /
−1,047 frames (±2.1 s), 14 times within ~15 s; the offset stacks to 5 × 1,048 and returns to zero.
Harp stays evenly spaced, and exposures (first/last frame number) = rows = trigger-log events
(763590: 1,932,897 each; CSV Harp == log, 0 µs). The video is continuous across all 14 jump rows
(consecutive-frame difference 0.1–2.3 grey levels, same as elsewhere; a real 2 s jump is ~9), and
rows sharing a frame number show different images, so no frames were replayed: only the metadata
is wrong. The Harp column is right as written; the old QC left it unchanged. **Decision: correct**
(`frame_numbers_increase` and `camera_time_increases` fail, `no_frames_lost` passes: Harp as written).

This led to the no-frames-lost rule in "Correction" below: frame numbers are only needed to locate
lost frames, so when none were lost they are not used.

The two glitch sessions run (`751766_2025-02-11`, `751769_2025-01-16`) differed from `main` on one
row by 8 µs (the row after the glitch, which the old QC nudged), with movements, trials and licks
identical. The two `ok` sessions were identical. The header-row session
(`behavior_781166_2025-05-15_14-20-48`) crashed on `main` and ran on the branch.

Separately, `763590`'s MP4 has one frame more than its CSV (1,932,898 vs 1,932,897). Not yet
checked whether keypoints are offset by a row; see Open questions.

## What the old QC did (replaced in Phase 2)

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

## Design

One module of plain functions, in the style of contraqctor's `CameraTestSuite`: each check asks
one question and returns `passed`, `count`, `message` and the offending `rows`. Errors are
`ValueError`. numpy/pandas only (contraqctor itself is not a dependency).

```python
from aind_dynamic_foraging_behavior_video_analysis import video_timing_qc as vtq

timing = vtq.load_video_timing(csv_path)      # harp_time_raw, frame_number, camera_time (s)
checks = vtq.check_video_timing(timing)       # one row per check
vtq.timing_action(checks)                     # what the correction will do
fixed = vtq.correct_video_timing(timing)      # + harp_time, harp_source; raises if refused
fixed = vtq.correct_video_timing(timing, trigger_times=vtq.read_harp_trigger_log(log_path))
vtq.check_session(behavior_videos_path)       # one row per camera: action, failed checks, ...

times, source = vtq.correct_frame_times(frame_number, camera_time, trigger_times)  # arrays only
```

### Checks

Three sources, each used only for what it can answer:

| Source | Answers | Cannot see |
|---|---|---|
| Harp column | *When* trigger *n* fired. Arrival-order pairing makes it the trigger sequence in order, one per row, evenly spaced whether or not frames were lost | Which exposure a row is; a constant offset |
| Frame number + camera time (camera metadata; fails together) | *Which exposure* a row is. Needed only when frames were lost | Anything about Harp |
| Trigger log (optional) | The full trigger sequence, and how many exposures the board triggered | — |

| Check | Question | Fails on |
|---|---|---|
| `no_frames_lost` | Do the frame numbers span exactly as many exposures as rows? | drops (`count` = frames lost; `rows` = skips) |
| `frame_numbers_increase` | Does every frame number step forward? | corrupted metadata |
| `camera_time_increases` | Does every camera time step forward? | corrupted metadata |
| `no_duplicate_frames` | Is no frame saved twice (same frame number and camera time on consecutive rows)? | replayed frame |
| `harp_has_no_glitches` | Is every Harp value in line with its neighbours? | isolated glitches (fixable) |
| `harp_evenly_spaced` | After fixing glitches, is every Harp step one frame interval? | clock steps, lost triggers, runs of bad rows |
| `clock_rates_agree` | Over the session, do Harp and camera advance at the same rate per exposure (≤ 100 ppm)? | drift no single step shows (e.g. many small clock steps); corrupted first/last row |
| `video_frame_count` | Does the video have one frame per row? (skipped without a count) | transcode problems |

Tolerances: half a frame per step (`STEP_TOLERANCE`); 100 ppm overall (`MAX_CLOCK_RATE_PPM`;
normal is +6.6 to +30.2 ppm on 17 cameras, 2025-01 to 2026-01). Frame interval: `frame_interval`,
the mean of the Harp steps within 25% of the median (Harp steps alternate by one 32 µs tick);
measured on the raw Harp column, where it is reliable at any drop rate. Glitch rule
(`find_glitch_rows`): a value off from its neighbours' midpoint while the neighbours are two frames
apart; runs of bad values are not glitches; Harp column only, so it also works in drop sessions.

### Decision and correction

`timing_action` decides from the check table; `correct_frame_times` runs the same checks
(`input_checks`) and follows it, so the decision exists in one place.

1. Fix isolated Harp glitches. **Every session** must pass `no_duplicate_frames`,
   `harp_evenly_spaced` and `clock_rates_agree` (validate the Harp source first).
2. `no_frames_lost` passes: row *n* was exposed by trigger *n*; use Harp (`use harp as written` or
   `fix glitches`). Camera metadata is not needed.
3. Otherwise require `frame_numbers_increase` and `camera_time_increases`; move each Harp value to
   its exposure's row (`reindex_to_exposures`); estimate rows past the last recorded trigger from
   a linear fit on camera time over the last 600 s (`estimate_missing_from_camera`), or read them
   from the trigger log; require each corrected step to match the camera step
   (`check_harp_matches_camera`, the one use of camera time as an independent witness). Action
   `re-index`.

Any failure is `refuse: <check>` and raises `ValueError` naming the check.

Trigger log (`correct_video_timing(timing, trigger_times=...)`; the LP pipeline uses it by default
when `Event_94.bin` exists): must have exactly one event per exposure (first to last frame number)
and match the CSV's Harp column to one 32 µs tick. Equality held on every session checked (7).

`harp_source` per row: `original`, `glitch_interpolated`, `reindexed`, `estimated_camera_fit`, or
`trigger_log`.

### Coverage

| Scenario | Outcome |
|---|---|
| ok / glitch / corrupted metadata without drops (763590) | Harp (glitches fixed) |
| Drops, drops + glitches, heavy drops (≥ 50%) | re-indexed |
| Clock step, with or without drops | refused: `harp_evenly_spaced` |
| Lost trigger, any drop rate | refused: `harp_evenly_spaced` |
| Many sub-threshold clock steps | refused: `clock_rates_agree` |
| Duplicated frame + a drop elsewhere (counts cancel) | refused: `no_duplicate_frames` |
| Drops + corrupted metadata | refused: `frame_numbers_increase` |
| More rows than exposures | refused (duplicate or non-increasing frame numbers) |
| Forward jump in corrupted metadata that never returns | **CSV alone: re-indexed, wrong by the jump** (looks exactly like a burst of drops); with the log: refused (event count) |
| Frames lost before the first saved row | **CSV alone: undetectable constant offset**; with the log: refused (event count) |
| Block of frames replayed (not seen) | **undetectable from CSV or log**: looks like corrupted metadata that returns to its start; only the video tells them apart (done by hand for 763590) |
| Glitch on the first or last row | refused (`harp_evenly_spaced`); acceptable |
| A single clock step < 1 ms | passes by design |

The last two bolded rows are the known limits; the trigger log (default in the LP pipeline)
removes the first two.

Changes from revision 4: `harp_evenly_spaced` is required on every session (it was only on the
no-drop path, so a lost trigger in a session with ≥ 50% drops was accepted); `no_duplicate_frames`
and `clock_rates_agree` added; the raw Harp-vs-camera comparison left the report (on raw Harp it
only restates drops) and is used only on re-indexed times; trigger-log count must equal exposures
(was ≥); the LP pipeline uses the log by default. Revision 4 replaced `qc_class` with the check
table and `timing_action`, and dropped a "residual from a camera-time fit ≤ 1 ms" check.

### Integration (Phase 2)

`integrate_keypoints_with_video_time(video_csv_path, keypoint_dfs, trigger_log_path=None)` now
loads with `load_video_timing` (fixing the new-layout crash), runs `check_video_timing` and
`correct_video_timing`, prints the action and any failed checks, and sets keypoint `time_raw`
from `harp_time`.
It returns the corrected timing frame in place of the old `Behav_Time`/`Frame`/`Camera_Time`
frame. Untrustworthy sessions raise `ValueError`, which `run_batch_analysis` already logs per
session and skips. `generate_tongue_dfs` and `run_batch_analysis` take `use_trigger_log`: None
(default) uses the session's `Event_94.bin` if it exists, True requires it, False uses the CSV
alone.
`generate_tongue_dfs` also now raises `FileNotFoundError` when no video CSV is found (it used to
fail with `AttributeError` on `None.exists()`).

## Consequence for `video_alignment`

`behavior_time_to_video_time(t, first_frame_behavior_time)` computes video position by subtraction.
That assumes Harp time advances one IFI per saved row. In a drop session with **uncorrected** Harp,
that assumption happens to hold (the pairing makes row *n*'s Harp ≈ first + *n* × IFI), so today's
clip positions are consistent with the file even though the events are mapped to the wrong frames.
Once Harp is corrected, subtraction is wrong in both senses: corrected Harp advances by more than
one IFI across each drop, but the file has no frame there.

So:

- Don't change `video_alignment`'s functions. The caveat is in the module docstring.
- New code that maps behavior events to video frames (the `video_clips.py` plan, which already
  chooses the frame index first) should `searchsorted` the event times into corrected
  `harp_time`, then convert frame index to file position (`index / fps` for a CFR file). A helper
  for this (`frame_index_for_harp_time`) is deferred to Phase 4.

## Consumers

| Consumer | Where | Today | With this package |
|---|---|---|---|
| Lightning Pose tongue kinematics | `kinematics/tongue_analysis.py::generate_tongue_dfs` → `integrate_keypoints_with_video_time` | Uses `Behav_Time` after the 2× fix as keypoint `time_raw`. Drop sessions pass silently with shifted times; drops of ≥ 2 frames get camera times overwritten. | **Done (Phase 2).** Check + correct; keypoint `time_raw` from corrected `harp_time`; untrusted sessions raise and are skipped by the batch. Optional trigger-log mode. |
| Motion-energy table | `kinematics_analysis/code/fip_me_aligned_table_plan.md` (branch `fip-motion-energy`) | Planned | Store corrected `harp_time`, `harp_source`, QC class per row / camera; ME is row-aligned with the CSV, so no other change. |
| FIP motion energy in analysis | `kinematics_analysis/code/fip_utils.py::motion_energy_to_session` | Reads CSV via `read_video_csv`, Harp time via `TIME_COLUMN_ALIASES` | Read the ME table's corrected time instead. |
| Video clips | `VIDEO_CLIPS_MIGRATION_PLAN.md` | Planned | Event → frame by `searchsorted` on corrected `harp_time`. |
| BEAST latents | `aind-BEAST-train-test/code/analyze_latents.ipynb` | `compute_video_session_offset`, `session_time_to_video_time` | Same as clips for drop sessions. |
| Notebooks | `kinematics_analysis/code/{model_quality,test_session_wrapper}.ipynb` call `integrate_keypoints_with_video_time`; `val_03`, `val_04` use `session_time_to_video_time` | Unchanged | `integrate_keypoints_with_video_time` callers get corrected times (and a changed returned CSV frame) once they move their pin. |

## Rollout

1. **Phase 1 (done): new module.** `video_timing_qc.py` + synthetic tests; validated on real data
   (below).
2. **Phase 2 (done): replace the old QC in the LP path.** `integrate_keypoints_with_video_time`
   uses the new module; the old `check_frame_monotonicity` / `qc_and_fix_timing` are removed
   (no legacy mode). Comparison against the old function on real sessions:

   | Session | Class | Keypoint `time_raw`, new vs old |
   |---|---|---|
   | `behavior_800886_2025-09-03_13-03-45` | ok | identical |
   | `behavior_809491_2025-10-02_09-23-46` | harp_glitch | 1 row differs by 8 µs (row 183625, which the old fix nudged; both fix 183624 identically) |
   | `behavior_816212_2025-12-05_13-47-41` | frame_drops | 2,746,049 rows differ, up to 374 s |
   | `behavior_818586_2026-01-21_09-43-54` | harp_glitch, new layout | old function crashes (`TypeError` on the header row); new works |

3. **Phase 3: re-run the LP batch** on a set covering ok, glitch, drop and new-layout sessions,
   once with `main` and once with this branch (`extract_clips=False`), and diff the outputs with
   `python scripts/compare_batch_outputs.py out_old out_new`. It reports, per session and
   intermediate parquet, `identical` or which columns changed and by how much. Expect ok sessions
   identical, glitch sessions to differ only slightly in time columns, drop sessions to differ in
   time and in everything matched by time (trials, licks), and new-layout sessions to appear
   only in the new run. Then release (minor version, changelog note: `time_raw` changes for drop
   and glitch sessions; the returned video frame has new column names).
4. **Phase 4: migrate other consumers** (ME table, clips, BEAST) to the corrected time.

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

`tests/test_video_timing_qc.py` (`unittest`, in CI). CSVs are simulated with arrival-order pairing
(the camera exposes and numbers every frame, some exposures are not saved, row *n* gets trigger
*n*), on the 32 µs Harp tick grid, with 15 ppm camera drift and jitter:

- clean; glitch (−983 ms; +3 ms); two consecutive bad rows (refused);
- drops of 1, 2, 3, 5 frames (exact re-indexing, tail within 0.1 ms, legacy threshold misses
  drops); a drop every 8 frames; glitch after a 2-frame drop and on a dropped exposure;
- lost trigger in a drop session (post-check refuses); repeated frame number; video frame count
  mismatch; unknown header; both layouts load identically; `check_session` on a mixed folder;
- trigger log: write and read `Event_94.bin`, exact correction including the tail, refusal for a
  log that does not match or is too short;
- `integrate_keypoints_with_video_time` on both layouts with drops, and refusal of untrusted
  timing.

No real-data fixtures are committed. `examples/video_timing_qc_validation.ipynb` (executed, with
outputs) shows the check and correction on an ok, a glitch, and a drops + glitch session against
the trigger log, with figures of the glitch and of Harp − camera time across a drop session. It
downloads what it needs. More sessions were checked by hand (CSVs and trigger logs
from `s3://aind-open-data`, public over HTTPS), both correction modes, against the trigger log:

| Session | Camera(s) | Class | Rows | Drops | Glitch rows | Tail rows | Re-indexed vs log | Tail vs log |
|---|---|---|---|---|---|---|---|---|
| `behavior_800886_2025-09-03_13-03-45` | bottom, side | ok | 2,672,569 | 0 | — | 0 | 0 | — |
| `behavior_809491_2025-10-02_09-23-46` | bottom, side | harp_glitch | 2,566,316 | 0 | 183624 | 0 | 0 | — |
| `behavior_816212_2025-12-05_13-47-41` | bottom | frame_drops | 2,748,281 | 187,024 | — | 175,161 | 0 µs | ≤ 0.062 ms |
| | side | frame_drops | 2,749,302 | 186,003 | — | 174,194 | 0 µs | ≤ 0.065 ms |
| `behavior_816214_2025-12-02_08-28-39` | bottom | frame_drops | 2,492,931 | 172,631 | 554395, 1357400 | 161,410 | 0 µs | ≤ 0.058 ms |
| | side | frame_drops | 2,498,774 | 166,788 | 554395, 1357400 | 156,359 | 0 µs | ≤ 0.059 ms |
| `behavior_818586_2026-01-21_09-43-54` (new layout) | Bottom, SideRight | harp_glitch | 2,517,841 | 0 | 1282502 | 0 | 0 | — |
| `ecephys_786867_2025-09-25_12-43-56` | bottom | frame_drops | 2,520,209 | 46,723 (32,237 gaps) | — | 46,723 | 0 µs | ≤ 0.053 ms |
| | side left | frame_drops | 2,523,521 | 43,411 (30,503 gaps) | — | 43,411 | 0 µs | ≤ 0.059 ms |
| | side right | frame_drops | 2,526,738 | 40,194 (28,900 gaps) | — | 40,194 | 0 µs | ≤ 0.046 ms |

No unexplained clock flags in any camera; every correction passed its post-checks. The ecephys
session has multi-frame drops, unlike the FIP sessions; Harp ended 80–93 s early, matching the
colleague's numbers.

## Phases

| Phase | Deliverable | Status |
|---|---|---|
| 1 | `video_timing_qc` + synthetic tests + real-data validation | done |
| 2 | Replace QC in `integrate_keypoints_with_video_time`, trigger-log option in `tongue_analysis` | done |
| 3 | LP batch re-run on affected sessions, release | |
| 4 | ME table, clips, BEAST migrated | |

## Open questions

1. ~~Confirm register 94 is the camera-trigger event.~~ Yes: `Camera1Frame` in the Harp Behavior
   `device.yml`.
2. Whether any workflow version logs triggers differently (older rigs, `Aind.Behavior.JustFrames`).
3. Can frames be lost before the first saved row? Trigger-log event count equalled exposures on
   all 6 sessions checked; check across all 97 once.
4. Default action for `frame_drops` in consumers other than LP: correct, or exclude?
5. Harp clock steps (751181, 754897): confirm the mechanism with the Harp / rig engineers, and
   whether lickometers and sound card step with the Behavior board. Refused until then.
6. MP4 frame count vs CSV rows on the older sessions (763590 has one extra frame): is it at the
   start (keypoints off by a row) or the end?
7. Whether this should become its own repository later. It depends only on `video_alignment`,
   numpy and pandas, so that would be a move, not a rewrite.

## Out of scope

- Fixing the Bonsai workflow (`rx:Zip` pairing): report upstream to `dynamic-foraging-task` and
  `Aind.Behavior.JustFrames` owners.
- Re-transcoding videos.
- Changing NWB or FIP timestamps.
