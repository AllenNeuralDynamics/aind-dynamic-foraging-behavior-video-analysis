# Plan: `video_quality_qc` — image-quality QC of behavior videos

> **Status: revision 8 (2026-10-01). We are now following "Revision 8: simplification plan"
> below.** It replaces the code structure in "Design" (dataclasses, summary fields, named level
> checks, session helpers, batch PDF) and the task-window fallbacks; where it and older sections
> disagree, revision 8 wins. The evidence and calibration in "Findings" and "Decisions:
> calibration" still hold, and the check values do not change. Revision 8 is implemented (see
> "Revision 8: implemented"). Next: Phase 3.
>
> Revision 7 (2026-10-01): Phases 1–2 done. Surveys of 97, then all 301 curated FIP
> sessions set the stability tolerances (sharpness 0.45, similarity 0.7); revision 7 records the
> calibration decisions ("Decisions: calibration"): a `scene_moves` check, a side camera
> clipping cutoff of 3.75%, no other level cutoffs, border strips removed, dirty mirror a known
> gap. Next: code review, then Phase 3 (integration). Phase 4 not started. Revision 2 recorded the decisions on revision 1's open questions (see "Decisions").
> Revision 3 recorded what the first real session changed (see "Findings: first real session"):
> `view_stable` is dropped (a lick-spout move reads as a camera shift), `similarity` is a plain
> correlation, clipping is counted at the tagged range, luma is read from the coded plane.
> Revision 4 (see "Findings: task window and border strips") samples the task only, with trial
> times read from the raw asset (no NWB), and adds per-border-strip shifts, reported only, for a
> survey of whether cameras ever move. Written so a new
> contributor or agent can pick it up without the conversation that produced it. The evidence
> behind each choice is in "Measurements" and "Findings".

## Revision 8: simplification plan (current)

Agreed 2026-10-01 after review of the revision 7 code, which works but is too elaborate: three
dataclasses (one with 52 hand-written summary fields), seven bespoke check functions of which three
are always skipped, warning plumbing for task-window fallbacks, and session and batch helpers that
duplicate code elsewhere. The goal is a module a new reader understands in one pass: plain
functions passing a frame array and a samples table, and checks that are rows of data.

### Principles

- **No classes.** Data moves as a uint8 frame array and a pandas samples table.
- **Measure everything, check little.** Every metric is computed and saved per sample. The checks
  read only the columns they need and decide pass or fail.
- **A check is data**: a metric, what it is taken over, a direction, a value. Its name is generated
  from those (e.g. `similarity >= 0.7`), and its result row stores parameters, observed value and
  outcome together.
- **Timing lives with timing.** Placing the task on frames is alignment, not image quality.

### Task window, in `video_alignment`

```python
task_frame_window(behavior_json, video_csv, trigger_log=None) -> (start, end)
```

Frames `[start, end)` from the first trial start to the last trial end (`read_trial_times`,
`behavior_time_to_frame_index`, both already in `video_alignment`). The Harp time of each CSV row
comes from the first source that works:

1. the trigger log by frame number, `log[frame_number - first_frame_number]` (clipped to the log),
   when a log is given;
2. the corrected timing (`video_timing_qc.correct_video_timing`);
3. the raw Harp column, when no frames were lost (`video_timing_qc` `no_frames_lost` passes).

A running maximum is applied so a glitch cannot reorder it. Raises `ValueError` when no source
works or the JSON has no Harp trial times; it has no other fallback. `video_timing_qc` imports from
`video_alignment`, so import it inside the function to avoid a cycle.

Why this order: in the 301-session survey the strict correction refused 65 of 602 cameras (56 for
one Harp step off by more than half a frame, 9 for a trigger log one event off the frame-number
span). The window only needs to be right to a few frames, so those cameras were placed correctly by
the raw Harp column (44) or the trigger log by frame number (21). The trigger log by frame number is
right whether or not frames were lost, so it goes first. Only one camera had no window
(`808057_2025-09-03` side: video CSV with missing values).

### Quality module, `video_quality_qc`

Constants: `N_SAMPLES = 100`, `EDGE_FRACTION = 0.01`, `REFERENCE_SAMPLES = 10`,
`FALLBACK_FRACTION = 0.5`, `HTTP_OPTIONS`, the two output file names, and the checks table:

```python
# (metric, over, op, value, cameras). Evidence: "Findings: full survey" and
# "Decisions: calibration" (301 FIP sessions, 602 cameras).
CHECKS = [
    ("sharpness_dev",    "samples", "<=", 0.45,  "all"),   # was sharpness_stable
    ("mean_dev",         "samples", "<=", 0.15,  "all"),   # was brightness_stable
    ("similarity",       "samples", ">=", 0.7,   "all"),   # was scene_stable
    ("similarity",       "p5",      "<",  0.998, "all"),   # was scene_moves
    ("pct_clipped_high", "median",  "<=", 3.75,  "side"),  # was exposure_ok (side)
]
```

`over="samples"` tests each sample and fails on two or more consecutive violations (an isolated one
is counted but passes: a paw in front of the lens). `over` = `"median"` or `"p5"` tests one session
statistic. `cameras` is `"all"` or a camera view (`"bottom"`, `"side"`, from the camera name in
either folder layout: `bottom_camera` / `BottomCamera`).

Functions, in pipeline order:

1. `sample_window(behavior_json, video_csv, trigger_log) -> (window, note)`: calls
   `task_frame_window`; if an input is missing or it raises, returns `None` and
   `"middle 50%: <reason>"`; otherwise the window and `"task"`.
2. `sample_keyframes(path, window) -> (frames, samples, color_range)`: refuse non-MP4; read the
   frame index; choose up to `N_SAMPLES` keyframes evenly inside the window (never frame 0 or the
   `EDGE_FRACTION` edges; with `window=None`, the middle `FALLBACK_FRACTION` of the file); seek each
   in one open container, verify the decoded frame is the requested keyframe, read the coded luma
   plane (never `to_ndarray(format="gray")`, see Findings 3); HTTP timeouts for URLs. `frames` is
   `(n, h, w)` uint8; `samples` has `frame_index`, `video_time`.
3. `measure(frames, samples, color_range) -> samples`: the reference is the pixel-wise median of
   the first `REFERENCE_SAMPLES` frames. Adds, per sample: `mean`, `std`, `p1`, `p99`,
   `dynamic_range`, `contrast_rms`, `entropy_bits`, `pct_clipped_low`, `pct_clipped_high` (at or
   beyond the tagged floor / ceiling), `sharpness` (Laplacian variance, 2x downsampled), `noise_sigma`
   (Immerkær), `similarity` (correlation with the reference, 2x downsampled), `shift_x`, `shift_y`,
   `shift` (phase correlation, reported only), `sharpness_dev` and `mean_dev` (`|x / median - 1|`),
   and `histogram` (luma counts). Intensity statistics are computed here, not imported from
   `aind_video_utils.compute_frame_stats` (its extra columns were redundant or always zero on these
   files). `aind-video-utils` is still used for `probe`, `read_mp4_frame_index`, `luma_range`.
4. `run_checks(samples, camera) -> DataFrame`: one row per applicable check: `check` (generated
   name), `metric`, `over`, `op`, `value`, `observed` (the statistic, or the number of samples in
   runs), `passed` (bool), `samples` (offending sample numbers).
5. `quality_action(checks) -> str`: `"use"`, or `"exclude: <check>"` for the first failure.
6. `write_video_quality(samples, checks, out_dir, camera, note)`: `video_quality_<camera>.parquet`
   (samples with histograms) and `video_quality_<camera>.json` (camera, window note, package and
   `aind-video-utils` versions, checks, action).
7. `video_quality(path, camera, behavior_json=None, video_csv=None, trigger_log=None) ->
   (frames, samples, checks, note)`: chains 1 to 5, for the batch pipeline.

### Reporter, `video_quality_report`

One function, `session_card(frames, samples, checks, title)`: reference with clipped pixels marked
(blue at or below the floor, red at or above the ceiling), 8 evenly spaced thumbnails on the tagged
luma range, one time-course panel per `over="samples"` check with its threshold line and failed
samples in red, a shift panel (reported only), the luma histogram (median and p5–p95 over samples),
and the outlier frames (blurriest, darkest, brightest, least similar, most shifted, |last − first|).
Thumbnails and the reference are computed from `frames` when drawing.

### Removed or moved

- Removed: `FrameQualityStats`, `VideoQualityQc`, `VideoQualityResult`, `qc_result_fieldnames`,
  the `_med/_p5/_p95/_spread` summary fields (a summary is `samples.median()`), format fields
  beyond what the code needs, `pts` and `edge_fraction` in outputs, the always-skipped level
  checks and the "skipped" state, `RawHarpWindowWarning`, `task_window_or_whole_file`.
- Moved to `scripts/video_quality_survey.py`: `check_session`, `contact_sheet`, `batch_pdf`.
- Not reimplemented: session folder discovery. `kinematics.tongue_kinematics_utils` already has
  `find_video_path` and `find_video_csv_path` (one camera, both folder layouts); the batch pipeline
  (`tongue_analysis`) uses the CSV one. `find_session_videos` and `find_behavior_json` go away; the
  survey script keeps its own S3 listing.

### Size target

About 250 lines of code (excluding docstrings, comments and blanks) in the module, from 614, and
about 120 in the reporter, from 325.

### Tests and docs

Keep the synthetic MP4 / CSV / JSON / trigger-log fixtures in `tests/test_video_quality_qc.py`.
Cover: each metric on arrays; keyframe choice (frame 0, edges, window, middle-50% default, fewer
keyframes than requested) and seek correctness (lands on the keyframe, luma not range-converted,
unsafe edit list and wrong landing refused); each check passing and failing on a synthetic fault
(defocus, brightness step, occlusion, still scene, clipping on a side camera only) and the
isolated-sample rule; a 6 px translation measured in `shift` and not checked; each task-window
source and the `ValueError`; `sample_window`'s fallback notes; written files round-trip; the card
renders. 100% coverage of both modules. Update the README section, the survey script and the
example notebook (re-run it on `behavior_816212_2025-12-05_13-47-41`). Stored survey outputs keep
the old field and check names; note that in the README.

### Revision 8: implemented (2026-10-01)

Branch `refactor/video-quality-qc`. Lines of code (excluding docstrings, comments and blanks; by
this count the old module is 659, not the 614 above):

| File | Before | After |
|---|---|---|
| `video_quality_qc.py` | 659 | 255 |
| `video_quality_report.py` | 336 | 141 |
| `video_alignment.py` | 97 | 119 (`task_frame_window`) |
| `tests/test_video_quality_qc.py` | 564 | 483 |

The reporter is over the 120 target: most of it is matplotlib layout (black splits each gridspec
call over 8–10 lines), and squeezing further would cost readability.

**Verified on real data.** `behavior_800886_2025-08-18_13-14-52`, both cameras, over HTTPS with
the stored window (302,396–2,552,853): the same 100 frame indices; every shared metric identical
per sample (shift to 2e-16, float rounding), and identical histograms; action `use` on both, as
stored. `task_frame_window` gives the stored window both with the trigger log and without it
(corrected timing).

Decisions the plan left open (simplest option taken):

- **Trigger log**: when given, it is the only source tried. An unreadable log raises (no fall
  through to the correction), and `sample_window` then samples the middle 50% with the reason.
  The correction is called without trigger times, since it is only reached without a log.
- **`sample_window`** catches `ValueError` and `OSError` (missing CSV file, network errors).
  Missing inputs give one note, `"middle 50%: no behavior JSON or video CSV"`.
- **Whole file**: no flag; pass a window past the end, e.g. `(0, 10**9)`.
- **Display range on the card**: `session_card` has no colour-range argument, so `measure` adds a
  constant `color_range` column to the samples.
- **Reference frame**: a small public `reference_frame(frames)` shared by `measure` and the card.
  It keeps the old truncation to uint8, so similarity and shift match stored values.
- **8-bit only**: luma is read from 8-bit planar formats, so `luma_range(8, ...)` and 256-bin
  histograms. Intensity statistics are the plan's list only (p5, p50, p95, `pct_at_min/max`,
  `pct_below_floor/above_ceiling`, `pct_outside_tagged` dropped).
- **Check names**: `"<metric> <op> <value>"` for `over="samples"`, `"<metric> <over> <op>
  <value>"` otherwise (`similarity p5 < 0.998`, `pct_clipped_high median <= 3.75`). `samples`
  lists every failing sample, isolated ones too; `observed` counts those in runs.
- **Card time course**: the checked metric itself (`sharpness_dev`, `mean_dev`, `similarity`)
  with its threshold dashed, not % of median.
- **`check_session`, `contact_sheet`, `batch_pdf`**: deleted, not moved. The survey script never
  used them; it builds its index and contact sheets from saved thumbnails and cards. Script
  changes: `--samples` removed (`N_SAMPLES` is a constant); summary rows carry `<metric>_med`
  medians, `similarity_p5` and `n_samples`; threshold pages read either parquet name; frames for
  them come from `sample_keyframes` with a one-frame window.
- **Outputs**: the JSON no longer repeats the video path or the sample frame indices (they are in
  the parquet), and the parquet is `video_quality_<camera>.parquet`.
- **Lint**: flake8 is clean on every touched file. That meant reflowing old over-long docstring
  lines in `video_alignment.py` and removing black-style `a[x : y]` slices (E203) in the survey
  script and the test fixture.

## Summary

A new module, `aind_dynamic_foraging_behavior_video_analysis.video_quality_qc`, that takes one
behavior-video MP4 (or a whole `behavior-videos` folder) and:

1. **Measures** image quality on ~100 keyframes spread evenly across the file: sharpness,
   contrast, intensity (histogram, clipping), noise, and whether the camera view stayed put.
2. **Checks** the measurements. Each check answers one question and passes, fails, or is skipped
   (the `video_timing_qc` pattern). Two kinds:
   - *stability*: did anything change during the session? Measured against the session's own
     median, so it works without calibration.
   - *level*: is the session sharp, bright and contrasty enough? These need thresholds calibrated
     on the population (Phase 2), so they are skipped until thresholds exist.
3. **Decides** one action per camera: `use` or `exclude: <check>`. A session whose video fails any
   check is dropped from video analysis as a whole.
4. **Optionally reports**: a fixed-layout "session card" PNG per camera, built around
   representative frames, and a multi-session PDF for scrolling through many sessions by eye.

Measuring is cheap: seek to ~100 keyframes, decode one frame at each. This takes a few seconds on
a local file and about 25 s over HTTPS from a laptop. No frame other than the sampled keyframes is
decoded.

Frames are read with PyAV. The exposure statistics, format probing and frame index come from
`aind-video-utils` (pinned). No OpenCV: all metrics are numpy (slicing and FFT).

## Decisions (revision 1 open questions)

| Question | Decision |
|---|---|
| Which files | **MP4 only** (the AIND file-standard transcode the pipeline reads). Other containers raise `ValueError`. |
| Dense keyframe pass ("tier 2") | **Deferred.** A session with unstable video is excluded, so there is no need to locate the change. Revisit only if Phase 2 shows many sessions failing on brief events, where dropping trials instead of sessions would recover real data. |
| Where generic metrics live | **Build here, offer to `aind-video-utils` later.** Follow its structure where it fits (see "Conventions"). |
| `compute_frame_stats` | **Import `aind-video-utils`, pinned** (`aind-video-utils==0.7.0` in the extra). |
| Region | **Whole frame.** |
| Sampling window | **Whole file**, minus 1% at each end (the `aind-video-utils` default); never frame 0. Revision 4: **the task** (first trial start to last trial end) when the raw asset's session JSON and the camera's CSV are present (`check_session` does this by default; `task_frame_window` otherwise), falling back to the whole file with the reason recorded. The first real session's recording ran 9 min before and 12 min after the task (Findings). |
| Known-bad examples | None yet. Phase 2's population survey is where they will be found. |
| Session vs trial exclusion | **Whole sessions.** |
| Link to `video_timing_qc` | **None for the quality metrics.** Revision 4: the task window uses it, since only the corrected timing puts trial times on the right frames (315 s off otherwise on a session with drops). Quality samples carry frame index and video time from the MP4 index. The batch pipeline records both actions side by side, and either one excludes a session. |

## Prior art: what exists and what to reuse

| Where | What it does | Use here |
|---|---|---|
| `video_utils.py` (from `AllenNeuralDynamics/aind_video_qc`, 2024–25) | cv2; reads 1000 frames with `CAP_PROP_POS_FRAMES` (slow, and not frame-exact on long-GOP h264); Laplacian variance, RMS contrast (`std/mean`), mean intensity; FPDF report with example frames at percentiles. Thresholds (focus 60, intensity 30–40) are ad hoc. | Metric *ideas*. No code. |
| `contraqctor/qc/camera.py` | Timing and metadata checks (covered by `video_timing_qc`), plus one mid-video frame: channel histogram and a saturation overlay (red ≥ 250, blue ≤ 5). | Saturation-overlay idea for the report. |
| `aind-video-utils` v0.7.0 (numpy-only core, ffmpeg subprocess) | `qc_batch.compute_frame_stats` → frozen `FrameExposureStats` (mean, std, p1/p5/p50/p95/p99, % at 0 / max, % outside the tagged range, entropy); `VideoExposureQc` aggregate; `qc_result_fieldnames` for CSV. `probe`, `get_video_range_info`, `luma_range`, `read_mp4_frame_index` (keyframe flags and presentation times from the `moov` atom, local or HTTPS, without reading frame data). `plotting.imshow_clipping`. Nothing on sharpness, stability or view shift. | **Import** `compute_frame_stats`, `luma_range`, `probe`, `get_video_range_info`, `read_mp4_frame_index`. **Not** its frame extraction: one ffmpeg process per frame costs 1.1–1.6 s per MP4 frame over HTTPS (Measurements). |
| `aind-basic-behavior-video-qc` (2026-07) | Start, average and end frames side by side as an `aind_data_schema` `QCEvaluation` for the QC portal. Notes that **frame 0 carries embedded metadata**. | QC-portal output (Phase 4). Never sample frame 0. |
| `behavior-inference-preview-video` | 100 random frames via cv2: Michelson contrast, mean/std, Laplacian variance, intensity histogram, mean frames, QC-portal upload. | Confirms the metric set. Michelson (`(max−min)/(max+min)`) hinges on single pixels; not used. |
| `foraging-cameraQc` | Timing only. | Nothing new. |

An org search for `blurdetect`, `signalstats`, `freezedetect`, `skip_frame`, `laplacian variance`
and `focus_threshold` found nothing else relevant. No existing tool measures quality stability over
a session or view shift, or samples by keyframe.

## Findings: first real session (revision 3)

`behavior_816212_2025-12-05_13-47-41`, both cameras, 100 keyframes, MP4 over HTTPS (~30 s per
camera). Reproduced in `examples/video_quality_qc_validation.ipynb`.

1. **The recording runs past the session.** From ~83 min to the end (92 min) both cameras show
   an empty rig. Over the whole file every stability check fails on those last 9 samples
   (similarity 0.04–0.16) and both cameras would be excluded. Over the first 82.5 min (read off
   the cards) both pass: sharpness spread 0.23 / 0.16, brightness spread 0.03 / 0.05,
   similarity p5 0.905 / 0.884. **Whole-file sampling will exclude every session with such a
   tail**, so the task window is needed (revision 4, below).
2. **A lick-spout move reads as a camera shift.** At ~71 min the bottom camera's phase-correlation
   shift jumps to (−9.8, −10.0) px and stays, constant to 0.02 px. An overlay of the frames before
   and after shows only the motorized spouts moved; the mouse and the rig are still registered.
   Whitened phase correlation locks onto the spouts' sharp edges. Tried and rejected:
   - partial whitening and 2× downsampling: still 8–14 px, or wrong on a synthetic shift;
   - a 4×4 tile grid with the median (or majority) shift: the textured tiles split evenly, five
     at the spout shift and five at ~0;
   - plain correlation with the reference: 0.91 for the spout move vs 0.86–0.90 for a real
     6 px camera bump, so it cannot separate them either.

   So `view_stable` is **dropped**; `shift` is kept as a reported metric (on the card, in the
   outputs) for Phase 2 to revisit, e.g. with a fixed-structure mask per camera.
   `scene_stable` (plain correlation ≥ 0.8) still catches gross changes: empty rig 0.04–0.16,
   half-frame occlusion, lights or IR off, bumps of ≳10 px.
3. **PyAV's `to_ndarray(format="gray")` stretches TV-range luma to 0–255** (swscale range
   conversion). On the bottom camera: coded Y 15–243, mean 61.3; converted 0–255, mean 51.8.
   Every exposure number, the clipping counts and the sharpness were on the wrong scale until
   `luma_plane` read the coded Y plane directly. A test now checks for it.
4. **Highlights clip at the TV ceiling, not at 255.** With the coded luma no pixel is at 0 or
   255, and none is below 16. The side camera's p99 is exactly 235 (1.5% of pixels at or above
   the ceiling): bright rig parts saturate at the top of the tagged range. The imported
   `pct_at_max` and `pct_above_ceiling` (strictly above) both miss this, so the module adds
   `pct_clipped_low` / `pct_clipped_high` (at or beyond the tagged floor / ceiling), and
   `exposure_ok` uses `pct_clipped_high`.

Task-window medians, for the Phase 2 survey to compare against:

| | bottom_camera | side_camera_right |
|---|---|---|
| sharpness | 256 | 384 |
| mean luma | 61.6 | 77.5 |
| dynamic range (p99 − p1) | 158.5 | 212 |
| % clipped at ceiling | 0.10 | 1.46 |
| noise σ | 1.08 | 1.49 |

## Findings: task window and border strips (revision 4)

**Trial times without an NWB.** The raw asset holds them. Ways of getting them, on
`behavior_816212_2025-12-05_13-47-41` (506 trials):

| Source | Cost | Agrees with the NWB? | Notes |
|---|---|---|---|
| `behavior/<subject>_<datetime>.json` (GUI session JSON, 7 MB) | 0.07 s parse, 0.36 s over HTTPS | **identical**, all trials, `start_time`, `goCue_start_time`, `stop_time` | The NWB is built from it: `TransferToNWB.bonsai_to_nwb` copies `B_TrialStartTimeHarp`, `B_TrialEndTimeHarp`, `B_GoCueTimeHarp` or `B_GoCueTimeSoundCard`. **Used.** |
| `behavior/raw.harp/ToBonsaiOSC/{TrialStartTime,GoCueTime,TrialEndTime}.csv` | 0.01 s | 2–6 ms off | Software (OSC) times, not the Harp values. |
| `bonsai_to_nwb` on the JSON, then `load_nwb` | 1.3 s + 0.2 s | identical | Needs the `kinematics` extra (pynwb and friends). |
| `foraging_nwb_bonsai` Code Ocean asset | — | not checked | Must be attached per capsule. The CO API key can search assets but file listing returned `forbidden`. |

`video_alignment.read_trial_times` reads the JSON (path or URL); files without Harp trial times
(older sessions with CPU times only) raise `ValueError`.

**Trial times to frames.** `task_frame_window` loads the camera's CSV, corrects its timing
(`video_timing_qc`, with the trigger log when present) and maps the first trial start and last
trial end with `behavior_time_to_frame_index` (`searchsorted` on the corrected Harp time). Cost
2.3 s per camera (1.5 s reading 2.7M CSV rows, 1.4 s correcting); the CSV must be local (4 s to
download 112 MB). The trigger log gave the same frames as the CSV alone. Subtracting the first
frame's Harp time instead puts the task end **315 s late** on this session, which dropped
187,024 frames, so the corrected timing is required. When the correction refuses a session,
`check_session` samples the whole file and records why in `window`.

The task ran from frame 280,988 to 2,388,169 of the bottom camera, **9.4 to 79.6 min** of a
92-min video: 9 minutes of setup before the first trial as well as the empty tail. Over the task
both cameras pass (sharpness spread 0.26 / 0.16, brightness spread 0.03 / 0.06, similarity p5
0.91 / 0.91).

**Border strips** (removed in revision 7). Shift per 48 px border strip, reported only (`edge_<side>_x/_y`,
`edge_<side>`, `edge_<side>_peak`, and `edges_shifted`, the count above 3 px), for a survey of
whether cameras ever move before designing a check:

| | top | bottom | left | right |
|---|---|---|---|---|
| side camera, max shift over the task (px) | 0.1 | 0.1 | 0.3 | 1.8 |
| side camera, median peak | 0.83 | 0.40 | 0.18 | 0.38 |
| bottom camera, max shift (px) | 2.0 | 184 | 100 | 13.7 (spouts) |
| bottom camera, median peak | 0.07 | 0.08 | 0.04 | 0.79 |

A synthetic 6 × 4 px bump is recovered by all four strips on the side camera and by top, bottom
and right on the bottom camera. On the side camera the strips are a clean signal
(`edges_shifted` 0 on all 100 samples). On the bottom camera they are not: the floor is dark and
the left strip holds the mouse, so their estimates are noise (`edges_shifted` 1–3 on 86 of 100
samples with no bump), and the one confident strip is the one the spouts cross. Gating on peak
height does not help there: a correctly recovered bump also has peaks 0.06–0.2 on those strips.
The survey should read the side camera's strips first.

## Findings: Phase 2 survey (revision 5)

97 FIP sessions (`kinematics_analysis/metadata/me_sessions_fip_curated.csv`, rows with
`used_in_fip05_07`), both cameras, 100 keyframes over the task window, read over HTTPS with
`scripts/video_quality_survey.py` (8 processes, ~30 min; ~60 s per camera including the CSV
download). Outputs in `video_quality_qc_data/survey_fip97/` (gitignored): `summary.csv`,
per-camera JSON/parquet/card, `video_quality_survey.pdf`.

**Result: all 194 cameras `use`.** No session in this population has an image-quality problem
the checks can see, so there are still no bad examples to calibrate against.

**Task window: 194 of 194.** 178 from the corrected timing; 12 from the raw Harp column (timing
refused, no frames lost); 4 from the trigger log by frame number (frames lost and timing
refused: two logs one event longer than the frame numbers span, one session with two bad Harp
steps). Both fallbacks were added during the survey: before them those cameras sampled the
whole file, and **all 4 first-run exclusions were these whole-file cameras failing on the
post-session tail** (empty rig from ~77–81 min). The correction stays strict for per-frame
analysis; the window only needs to be right to a few frames.

**Operational.** A worker hung for 10 minutes on a stalled HTTPS read (fixed: 60 s
`rw_timeout` with reconnect for URLs, 120 s socket timeout for downloads). Two side cameras
raised `InvalidDataError` mid-decode on the first run and passed on re-run (transient).

**Stability margins** (passing cameras, over the task; tolerances unchanged):

| | bottom (97) | side (97) | tolerance |
|---|---|---|---|
| max \|sharpness / median − 1\| per camera, p95 (max) | 0.33 (0.40) | 0.20 (0.24) | 0.30 |
| same, 95th percentile of samples, max over cameras | 0.31 | 0.17 | |
| max \|mean / median − 1\|, p95 (max) | 0.05 (0.13) | 0.06 (0.06) | 0.15 |
| min similarity, p5 (min) | 0.85 (0.80) | 0.81 (0.78) | 0.80 |

No camera had two consecutive samples out of tolerance, so the tolerances cause no false
exclusions here. They are close to natural variation in two places: 9 of 97 bottom cameras
have an isolated sample beyond 30% sharpness (the mouse's face and tongue move), and a few side
cameras dip below 0.8 similarity on single samples. The empty-rig tail (positive control)
sits far outside: sharpness 0.40–0.94 off the median, brightness 0.45–0.60, similarity
0.04–0.27. **Recommendation, not applied:** keep brightness at 0.15; loosen the bottom
camera's sharpness tolerance to ~0.45 and the similarity floor to ~0.7 if Phase 3 shows
false exclusions; both still catch the empty rig.

**Level thresholds: not set.** Whole-frame levels differ by subject and rig more than any
plausible failure would move them. Bottom-camera median sharpness per subject ranges 88–260
among healthy sessions. Subject 818586's bottom camera dropped from ~210 to ~85 between
2026-01-02 and 2026-01-05 and stayed there; frames on both sides show the mouse, whiskers and
spouts equally sharp, but a bright, finely dotted background replaced by a dark smooth one.
Whole-frame Laplacian variance tracks that background, not focus. An absolute sharpness
threshold would need an ROI on the mouse, or a per-subject baseline (a step against the
subject's own previous sessions). Side-camera clipping at 235 is 1.5–5.7% everywhere (bright
rig parts), so `max_pct_clipped` also needs a rig-aware value. Folder layouts name the same
camera differently (`bottom_camera` vs `BottomCamera`), so thresholds need a camera alias.

**Camera shift: it happens, rarely.** One clear case in 97 side cameras:
`behavior_818585_2025-12-22_13-10-50`, where at 58.5 min all four border strips and the whole
frame shift by +3.1 to +4.0 px in x and stay there (the rig's vertical edges double in an
overlay of the samples before and after). Similarity barely moves (0.85–0.9), so no current
check sees it. No other side camera has two samples with all four strips moved. Bottom
cameras show many 3-strip samples (noise from dark or mouse-filled strips, and spout moves,
as expected) and no clear case. Whether 3–4 px matters depends on the consumer (keypoint
models trained on a fixed view); a side-camera check (all four strips > 2 px on ≥ 2
consecutive samples) would catch this case with no false positives in this survey.

## Findings: full survey (revision 6)

All 301 sessions in `me_sessions_fip_curated.csv` (602 cameras), same method, in
`video_quality_qc_data/survey_fip/` (113 min with 8 workers; 4 cameras hit transient network
errors and passed on re-run). Task windows: 532 corrected timing, 44 raw Harp, 21 trigger log by
frame number, 1 whole file (`808057_2025-09-03` side camera: its video CSV has missing values,
which `load_video_timing` refuses).

At the first-guess tolerances 8 cameras were excluded. Their cards:

| Camera | Check | What it is | Verdict |
|---|---|---|---|
| `820688_2026-01-27` bottom and side | sharpness, brightness, scene | Video nearly black for the first 14 min of the task (9–23 min), then normal: IR light off at the start, both cameras | **real** |
| `808057_2025-09-03` side | sharpness | Empty rig at the end; no task window (CSV with missing values) | window failure |
| `800886_2025-09-08` bottom | scene | Spouts move 24 px at 20 min; similarity to the pre-move reference then sits at 0.79–0.80 | false positive |
| `815334_2025-10-23`, `818585_2026-01-27` bottom | scene | Similarity hovering at 0.78–0.80 for 10–87 samples, the same pattern | false positive |
| `808057_2025-08-22` side | scene | Posture change (paws up) in the last 10 min, similarity 0.70–0.80 | false positive |
| `813929_2025-11-04` bottom | sharpness | 40% low for the first 3 min of the task, with a 72 px (spout) shift at 11:49 | ambiguous |

Exclusions across all 602 cameras by tolerance (two consecutive samples beyond it):

| Setting | Excluded |
|---|---|
| sharpness 0.30 / 0.45 | 4 / 3 (0.45 drops 813929) |
| similarity 0.8 / 0.75 / 0.7 | 7 / 4 / 3 |
| brightness 0.15 | 3 |

At sharpness 0.45 and similarity 0.7 every check excludes exactly the same 3 cameras (820688
both, 808057 09-03 side), so these are now the defaults. The synthetic faults in the tests are
still caught.

**A failure no check sees: a camera not looking at the mouse.** `816212_2025-12-23`'s bottom
camera shows a static scene with no mouse (a bar and a dark ring) for the whole task, while its
side camera is normal. Every check passes because nothing changes: similarity p5 0.9996 (next
highest of 301 bottom cameras 0.991; side cameras at most 0.947), shift 0.05 px, and sharpness
23 (next lowest 61). Found on the threshold pages, not by the checks. A "does anything move?"
check (similarity p5 above ~0.995 means a still scene) would catch it; one example so far.

Note for the card: the reference frame is the median of the first 10 samples, so when the
first minutes are abnormal (820688's dark start) every later, normal sample reads as dissimilar.
The exclusion is still right, but the time course points at the wrong part of the session.

## Decisions: calibration (revision 7)

Made on 2026-10-01 from the full survey and its threshold pages
(`video_quality_survey.py --thresholds`: per metric and camera, frames from sessions across the
distribution, for picking a cutoff by eye).

- **`scene_moves` (new check).** Fails when the similarity p5 to the reference is at least 0.998:
  nothing in view moves. On the 602 cameras it fails only `816212_2025-12-23`'s bottom camera
  (0.9996, no mouse in view). The stillest camera on a live mouse is `813929_2025-10-30`'s
  bottom camera at 0.991: a mouse barely in frame (snout and whiskers at the left edge). The
  margin is narrower than the numbers suggest; a mouse further out of frame could approach the
  cutoff, which would arguably still deserve a look.
- **Side camera clipping: `max_pct_clipped` 3.75%.** On the threshold page, below ~1.5% only rig
  hardware clips (a fixed ~1.4%); at 2–3% the paws saturate; from ~3.9% the jaw, mouth and ear
  saturate in large patches, losing tongue and jaw detail. 36 of 301 side cameras exceed 3.75%,
  mostly two subjects (818586: 19, 809487: 14), so it is largely per-subject lighting or
  positioning. Thresholds are keyed by camera view (`camera_view`: bottom or side), since the
  two folder layouts name the same camera differently.
- **No other level cutoffs.** Across sessions, sharpness, brightness, black level, contrast,
  dynamic range and entropy track the scene (background, rig, subject) more than quality, and
  the threshold pages showed no level where frames turn bad. Bottom camera clipping is ~0
  everywhere.
- **Border strips removed.** They answered their question: cameras do move, but rarely (one case
  in 301 side cameras) and by 3–4 px, which is not visible by eye. The whole-frame `shift` stays,
  reported only.
- **Dirty bottom mirror: known gap, no check.** It looks like bokeh speckles across the
  background plus haze. Within-rig ranks of entropy, mean luma and black level rising together
  find it (809491 Oct 1–13; on rig 446_8D 800886 2025-09-09, 813929 2025-10-21, 818580
  2025-12-01, where it seems to build up and get cleaned in cycles), but also rank bright
  backgrounds (a card on rig 447_3D), the lights-off session and dark soft sessions as highly.
  Extremes of mean luma correlate with it but are not robust. Rig ids come from each raw asset's
  `session.json` (`rig_id`, e.g. `446_7D_20251007` = room 446, box 7D). **Idea to try:**
  measure contrast in a fixed region, e.g. the bottom right corner of every frame (background
  seen through the mirror, away from the mouse and spouts), instead of the whole frame, so the
  mouse and rig hardware do not dilute it.

## Background: the MP4

Checked on `behavior_816212_2025-12-05_13-47-41` (public, `s3://aind-open-data`),
`bottom_camera.mp4`: h264 High with B-frames, yuv420p 720×540, **TV range, BT.709 transfer**,
**GOP 250 (a keyframe every 0.5 s)**, 2,748,281 frames, 5.5 GB, 92 min. Chroma is flat (signalstats
`SATAVG` 0.7): the cameras are monochrome IR, so only luma is used.

- **Keyframes are free to seek to.** A seek lands on an I-frame and one decode returns it; there
  is no forward decoding. A 92-min video has about 11,000 keyframes to choose from.
- **Pick keyframes from the index, not from time.** `r_frame_rate` is 1000/1 while
  `avg_frame_rate` is 500, so frame index from time × rate is wrong. `read_mp4_frame_index` gives
  each keyframe's presentation index and time exactly. Choose the ~100 keyframes from it, seek to
  their presentation times, and check that the decoded frame is a keyframe with the expected
  `pts`.
- Values are gamma-encoded, TV range: floor 16, ceiling 235 (`luma_range(8, False)`). The
  original AVIs are linear light and would give different numbers; they are out of scope.
- Do not assume 720×540, 500 Hz or GOP 250. Read them from the file and record them.

## Measurements (2026-09-30, MacBook, ffmpeg 8.1.1, PyAV)

Sampling cost, `bottom_camera.mp4`:

| Method | Cost |
|---|---|
| ffmpeg process per frame, HTTPS (`aind-video-utils` style) | 1.1–1.6 s/frame (index re-read every time) |
| PyAV, one open container, 100 keyframe seeks, HTTPS | open 1.1 s, then **230 ms/frame** (network-bound; far less locally or in-region) |
| Every frame decoded, local (for comparison) | ~9.7 min per 92-min video |
| Every keyframe (deferred dense pass), PyAV + numpy metrics, local | 17 ms/frame, ~3.1 min per video |

Sharpness depends on frame type. Laplacian variance over 3,000 consecutive frames:

| Frame type | n | Full resolution | 2× downsampled |
|---|---|---|---|
| I (keyframe) | 12 | 191 | 360 |
| P | 756 | 141 | 336 |
| B | 2,232 | 139 | 333 |

At full resolution, keyframes score 35% higher than P/B frames because the encoder keeps more
fine detail and noise in them. At 2× downsample the gap is 7%. Frame-to-frame CV was 0.15. Two
rules follow: **sample only keyframes**, so every sample is the same frame type, and **compute
sharpness on the 2× downsampled image**, which tracks optical focus more than codec detail or
sensor noise.

## Design

### Conventions (following `aind-video-utils` where it fits)

- Results are **frozen dataclasses**: `FrameQualityStats` (one keyframe) holding the imported
  `FrameExposureStats` plus the metrics below, and `VideoQualityQc` (one video: format info and
  session summary), with `qc_result_fieldnames()` for flat CSV rows, like `VideoExposureQc`.
- `from __future__ import annotations`, type hints throughout, NumPy-style docstrings, a
  `probe_json` argument so a caller that already probed does not probe again.
- This repo's lint config stays in force (black 79, flake8, isort, interrogate 100%, coverage
  100%). Its check pattern (`_result` rows, an action string) is kept, because consumers already
  use it for timing.
- Functions and names are kept generic (no camera names or foraging terms in the metric code),
  so moving them to `aind-video-utils` later is a copy, not a rewrite.

### Frame access

1. `probe` once and refuse anything that is not an MP4 (`format_name` contains `mp4`).
2. `read_mp4_frame_index` once. Choose `n_samples` keyframes (default 100) evenly across the file,
   skipping the first and last 1% and frame 0. If the file has fewer eligible keyframes, use each
   one once.
3. One open PyAV container: seek to each chosen keyframe's presentation time, decode one frame,
   confirm `key_frame` and `pts`, and read the coded Y plane (`luma_plane`). Not
   `to_ndarray(format="gray")`, which range-converts (Findings 3). The demuxer's PTS is the
   index PTS minus the edit list's `media_time`; checked on the real file (every seek landed
   on the requested keyframe, first decoded frame).
4. Keep a downsized copy of each sample (e.g. 360×270, uint8) for the report, so reporting never
   decodes again. At 100 samples this is ~10 MB in memory and is not written to disk by default.

### Metrics (per sampled keyframe)

`d` = luma 2× downsampled by 2×2 mean, as float32.

| Metric | Definition | Catches |
|---|---|---|
| exposure stats (imported) | `compute_frame_stats(luma, color_range, bit_depth)`: mean, std, p1–p99, % at 0 / 255, % below floor / above ceiling, entropy | too dark or too bright, clipping, washed out |
| `contrast_rms` | `std / mean` | lost contrast (the old `generate_contrast_report` computed this, despite its "variance" label) |
| `dynamic_range` | `p99 − p1` | same, robust to a few extreme pixels |
| `pct_clipped_low`, `pct_clipped_high` | % of pixels at or beyond the tagged floor / ceiling | saturation; a TV-range file clips at 235, not 255 (Findings 4) |
| `sharpness` | variance of the 4-neighbour Laplacian of `d` | defocus, smeared or fogged lens |
| `noise_sigma` | Immerkær fast noise estimate on full-res luma | gain change, low light; explains a noisy frame that scores as "sharp" |
| `similarity` | Pearson correlation of `d` with the reference | occlusion, lens cap, lights or IR off, empty rig, large bumps |
| `shift_x`, `shift_y`, `shift` | phase correlation (Hann window, parabolic sub-pixel peak) of the full-res frame against the reference | **reported only**: a spout move reads the same as a camera bump (Findings 2) |
| `histogram` | 256 luma counts | report histogram; re-deriving any percentile later |

**Reference frame**: the pixel-wise median of the first 10 samples. The median ignores a moving
mouse. It is stored and shown in the report.

**Deferred**: flicker and frozen frames (need consecutive frames; duplicates are already caught
from metadata by `video_timing_qc`); compression blockiness; ROI-restricted metrics.

### Checks (one question each)

`check_video_quality(qc) -> DataFrame`, one row per check: `check`, `passed`
(True / False / None when skipped), `message`, `count`, `samples` (indices of offending samples).

Stability, relative to the session median (tolerances are first guesses, to be set in Phase 2):

- `sharpness_stable`: is every sample's sharpness within ±30% of the median?
- `brightness_stable`: is every sample's mean luma within ±15% of the median?
- `scene_stable`: is every sample's similarity to the reference ≥ a floor (e.g. 0.8)?

One sample out of line with both neighbours is reported in `count` and `samples` but does not fail
the check (a paw in front of the lens for a moment). Two or more consecutive samples fail it. This
is the same distinction as an isolated Harp glitch vs a run.

Level (skipped until a threshold exists for this camera):

- `sharp_enough`: median sharpness ≥ threshold.
- `exposure_ok`: median mean luma within [low, high], and median % clipped at the ceiling ≤ limit.
- `contrast_ok`: median dynamic range ≥ threshold.

Thresholds live in a table in the module keyed by camera name, with the evidence (sessions, date)
in a comment, like `MAX_CLOCK_RATE_PPM` in `video_timing_qc`. Callers can pass their own.

**Known limit**: a camera bump of a few pixels is not detected (Findings 2); only bumps large
enough to lower `similarity` below 0.8 are.

**Known limit**: with 100 samples ~55 s apart, a problem shorter than about two sample intervals
(~2 min) can be missed. `n_samples` can be raised. Seeks are cheap, and 300 samples still take
seconds locally.

### Decision

`quality_action(checks) -> str`: the first failed check gives `exclude: <check>`; otherwise
`use`. When level checks were skipped the message says so ("level checks not calibrated for
<camera>"), so a `use` is not mistaken for an absolute verdict. The module never removes data;
consumers act on the action.

### API sketch

```python
from aind_dynamic_foraging_behavior_video_analysis import video_quality_qc as vqq

window = vqq.task_frame_window("behavior/<subject>_<datetime>.json",
                               "behavior-videos/BottomCamera/metadata.csv", trigger_log=None)
qc = vqq.measure_video_quality("behavior-videos/BottomCamera/video.mp4", n_samples=100,
                               frame_window=window)  # [start, end) frames = CSV rows
qc.samples     # DataFrame: one row per keyframe: frame_index, video_time, metrics
qc.reference   # uint8 reference frame
qc.summary     # VideoQualityQc: format info + median/p5/p95/spread per metric
checks = vqq.check_video_quality(qc, camera="BottomCamera")
vqq.quality_action(checks)             # "use" / "exclude: scene_stable"
vqq.check_session("behavior-videos/")  # one row per camera, like video_timing_qc.check_session
vqq.write_video_quality(qc, checks, out_dir)
```

Reporting is a separate module, `video_quality_report`, so measuring never imports matplotlib:

```python
from aind_dynamic_foraging_behavior_video_analysis import video_quality_report as vqr
fig = vqr.session_card(qc, checks)
vqr.batch_pdf(results, "video_quality.pdf", sort_by="sharpness")
```

### Standardized output

Per camera, a JSON (`video_quality_<camera>.json`, file name declared as a constant next to the
writer, like `TONGUE_QUALITY_STATS_FILENAME`) holds: library and `aind-video-utils` versions; file
format (codec, pix_fmt, range, transfer, size, fps, frame count, GOP); sampling (n, edges skipped,
frame indices); per-metric session summary (median, p5, p95, spread `(p95 − p5) / median`); every
check; the action. The per-sample table is written beside it as parquet, so figures can be redrawn
without decoding. For batches, a flat CSV with one row per camera (`qc_result_fieldnames` order),
for sorting and for the Phase 2 survey.

### Report: built for scanning sessions by eye

The unit is a **session card**: one fixed-layout PNG per camera, the same size for every session,
so a folder or PDF of them can be flipped through quickly and differences stand out.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ behavior_816212_2025-12-05  BottomCamera  h264 yuv420p bt709  500 Hz     │
│ ACTION: use            sharp 336 · mean 53 · DR 180 · shift ≤1 px        │
├──────────────┬───────────────────────────────────────────────────────────┤
│ reference    │  8 thumbnails evenly spaced across the file, each         │
│ (median)     │  labelled mm:ss, same display scaling (no per-frame       │
│ + clipping   │  auto-contrast, so brightness changes are visible)        │
│ overlay      │                                                           │
├──────────────┴───────────────────────────────────────────────────────────┤
│ time course: sharpness, mean, shift, similarity vs video time,           │
│ each as % of session median, tolerance bands shaded, failed samples red  │
├──────────────────────────────┬───────────────────────────────────────────┤
│ luma histogram (median over  │ outlier frames: blurriest / darkest /      │
│ samples, p5–p95 band, floor  │ brightest / most shifted / least similar,  │
│ and ceiling marked)          │ each labelled with time and value          │
└──────────────────────────────┴───────────────────────────────────────────┘
```

- **Same display scaling for every thumbnail** (fixed 16–235), or dimming is hidden.
- **Clipping overlay** on the reference: pixels at or below the floor in blue, at or above the
  ceiling in red (the same pixels `pct_clipped_low` / `pct_clipped_high` count). Drawn in the
  report module rather than with `imshow_clipping`, whose colours stretch with its limits.
- **Outlier frames**, not percentile frames. The blurriest and most shifted frames are where
  problems show; a 50th-percentile frame says little.
- **First vs last**: a small `|first sample − last sample|` panel shows a view shift at a glance.
- **Batch view**: `batch_pdf` puts one card per page (matplotlib `PdfPages`; no FPDF), after an
  index page listing sessions sorted by action and by a chosen metric. A contact-sheet variant
  (one row per session: action and 4 thumbnails) fits about 10 sessions per page.

### Dependencies

- Core stays `numpy`, `pandas`.
- New extra `video-qc = ["av", "aind-video-utils==0.7.0", "matplotlib", "pyarrow"]`. PyAV wheels
  bundle FFmpeg. `aind-video-utils` calls the system `ffprobe` for `probe`, so ffmpeg must be on
  `PATH` (it is in the Code Ocean capsules that already clip videos). If that is a problem, the
  few probe fields needed can be read from PyAV instead.
- `aind-video-utils` needs Python ≥ 3.10; this repo needs ≥ 3.11. Compatible.
- No OpenCV, FPDF or seaborn.

## Tests

`tests/test_video_quality_qc.py` (`unittest`, in CI, 100% coverage). No real-data fixtures.
Synthetic MP4s are encoded in the test with PyAV (h264 with B-frames, yuv420p, e.g. 160×120, GOP 25,
a few hundred frames) from a textured pattern with a moving blob. Cases:

- clean: all stability checks pass, action `use`, level checks skipped with the message;
- blur from frame k to the end: `sharpness_stable` fails;
- one isolated blurred keyframe: counted, check passes;
- brightness step: `brightness_stable` fails;
- 6 × 4 px translation from frame k: measured shift within 0.5 px, and no check fails on it;
- occlusion (half the frame black): `scene_stable` fails;
- forced clipping: exposure stats reflect it; `exposure_ok` fails with a threshold;
- level checks pass and fail with thresholds supplied;
- sampling: every sample is a keyframe at the index-predicted frame; frame 0 and the 1% edges are
  never sampled; a file with fewer keyframes than `n_samples` uses each once;
- non-MP4 input raises `ValueError`;
- `check_session` on a folder with both layouts (`<Camera>/video.mp4`, `<camera>.mp4`);
- JSON and parquet round-trip; report functions render (Agg) and `batch_pdf` writes a PDF.

## Phases

1. **Measure, check, report (done, revisions 3–4).** Everything above except calibrated level
   thresholds. `tests/test_video_quality_qc.py`: 46 tests, 100% line coverage of both modules,
   ~10 s. Validate on
   public sessions (e.g. `behavior_816212_2025-12-05_13-47-41`, both cameras) in an executed
   example notebook (`examples/video_quality_qc_validation.ipynb`).
2. **Population survey and calibration (FIP survey done, revision 5).** Run on every attached
   LP session and the FIP population (97 sessions × 2 cameras, from the timing plan), over the
   task window. Look for step changes
   in the border-strip shifts (side camera first) to learn whether cameras ever move. Scroll the batch PDF, collect the bad
   sessions found, set stability tolerances and level thresholds per camera from the
   distributions, and record the evidence here. Decide from the data whether the dense keyframe
   pass is needed.
3. **Integration.** `run_batch_analysis` records the quality action per camera next to the
   timing action, then excludes `exclude:` sessions (opt-in first, default after Phase 2's
   thresholds are in). Release note in the README "Changes".
4. **Optional.** Offer the generic metric code to `aind-video-utils`. Write results as an
   `aind_data_schema` `QCEvaluation` for the QC portal, matching `aind-basic-behavior-video-qc`.

## Open questions

- **Dirty mirror.** Try contrast in a fixed background region (bottom right corner) of the
  bottom camera; see "Decisions: calibration".
- **Sharpness as focus.** Whole-frame sharpness tracks the background; a region on the mouse
  (fixed, or from keypoints) would be needed for a focus check.
- **Phase 3:** opt-in exclusion first; watch for false exclusions from the tolerances and the
  clipping cutoff on sessions outside the surveyed set.

## Out of scope

- AVIs and other containers.
- Timing and frame drops (`video_timing_qc`).
- Locating bad segments and trial-level exclusion (the deferred dense pass).
- Keypoint-quality QC (`tongue_quality_stats.json`).
- Re-encoding and colour-pipeline QC (`aind-video-utils`).
