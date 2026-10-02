# Plan: video screening before analysis (Phase 3 of video timing + quality QC)

> **Status (2026-10-02): Part A merged (PR #9). Part B in progress on `feat/video-screen`:
> steps 1–4 done (timing module, quality renames, `video_screen`, survey script and docs);
> verification next.** Decisions so far are dated in place. Background:
> `VIDEO_QUALITY_QC_PLAN.md` (revisions 1–8) and `VIDEO_TIMING_QC_PLAN.md`.

## Context

Analyses start from a list of sessions. Only videos that pass both timing QC (`video_timing_qc`)
and quality QC (`video_quality_qc`) should be analyzed by anything: pose estimation (Lightning
Pose), motion energy, `run_batch_analysis`. So the QC must run **upstream** of every analysis, as a
screen, not inside `run_batch_analysis`. Today:

- quality QC writes per-camera files, but only the survey script loops over sessions, and only
  over HTTPS;
- timing QC writes nothing; `check_session` returns a table in memory; the pipeline raises on a
  refused camera and logs the error; the FIP motion-energy build (`kinematics_analysis-fip-me/
  code/build_me_table.py`) rolled its own screen (`me_dry_run_fip.csv`);
- the two modules are not parallel: different words (`refuse: <check>` vs `exclude: <check>`),
  different function names, and timing's `timing_action` mixes the verdict with the correction
  method.

Intended outcome: one screening table (one row per session × camera, with a `use` column)
that any analysis filters on, produced by a layered API that works as an in-loop gate or as an
up-front batch, writes files only when asked, and takes **file locations from the caller**:
local paths (Code Ocean data assets) or HTTPS URLs (S3), the caller's choice, never the
module's. The module does not search for files.

Order (decided 2026-10-02): **Part A** merges video quality QC as it stands now; **Part B**
(screening) starts afterwards on a new branch from the updated `main`. This file is the plan for both; Part B adds a pointer to it from
`VIDEO_QUALITY_QC_PLAN.md` (Phase 3) and `VIDEO_TIMING_QC_PLAN.md`.

## Part A: PR for video quality QC (now)

Branch `refactor/video-quality-qc` (pushed; local and remote match): 17 commits over `main`
(16 of code and docs, plus this plan), 12 files. `main` is an ancestor, so no conflicts. Nothing in it
changes existing behavior.

Steps:

1. `git fetch` (the local remote-tracking ref for the branch is missing; GitHub has it).
2. Before opening: run the full suite once more (`python -m unittest discover`, venv with the
   `kinematics,video-qc` extras), and black / isort / flake8 on the touched files.
3. No version bump (decided 2026-10-02): README "Changes" stays "Unreleased"; the release
   comes with the screening PR.
4. `gh pr create --base main --head refactor/video-quality-qc` with the title and body below.
5. Do not merge; the user reviews and merges.

Title: `feat: video quality QC (keyframe-sampled image quality for behavior MP4s)`

Body (draft):

```markdown
## Summary
New `video_quality_qc` and `video_quality_report` modules, in a new `video-qc` extra
(`aind-video-utils==0.7.0`, `av`, `matplotlib`, `pyarrow`), plus
`video_alignment.read_trial_times`, `behavior_time_to_frame_index` and `task_frame_window`.
Nothing existing changes.

- Samples ~100 keyframes over the task (first trial start to last trial end, from the raw
  session JSON; middle 50% of the file if that fails), reading only those keyframes (seconds
  locally, ~40 s per camera over HTTPS).
- Measures per keyframe: brightness and exposure, contrast, entropy, clipping at the tagged
  range, sharpness, noise, similarity to the session's typical frame, shift (reported only).
- Checks are rows of a table (`CHECKS`): sharpness and brightness stability, similarity,
  "does anything move", mean luma 50–150, side-camera clipping ≤ 3.75%. Verdict per camera:
  `use` or `exclude: <check>`.
- Optional session card (one PNG per camera) for review by eye.

## Evidence
Thresholds set on a survey of all 301 curated FIP sessions (602 cameras); see
`VIDEO_QUALITY_QC_PLAN.md` ("Findings", "Decisions: calibration", "Survey with the revision 8
code"). Result: 563 use, 36 side cameras excluded for clipping (mostly two subjects), 2 for an
IR-off start (820688 2026-01-27), 1 camera not looking at the mouse (816212 2025-12-23).

## Testing
- `tests/test_video_quality_qc.py`: 45 tests on synthetic MP4s (each check fails on its fault),
  100% line coverage of both modules; full suite (82) passes.
- Re-measured `behavior_800886_2025-08-18_13-14-52` over HTTPS: identical to the stored survey.
- `examples/video_quality_qc_validation.ipynb` re-run on `behavior_816212_2025-12-05_13-47-41`.

## Not in this PR
- Integration: screening timing + quality upstream of every analysis is the next PR (Phase 3,
  `VIDEO_SCREEN_PLAN.md`).
- The reference frame changed to the median of all samples after the survey; the next survey
  run (through the screen) confirms the similarity cutoffs under it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## Part B: screening (new branch after Part A merges)

New branch `feat/video-screen` from `main` once Part A is merged (if review takes long, branch
from `refactor/video-quality-qc` and rebase after the merge). Everything below is Part B.

## Design

### Symmetric modules (option 3, decided 2026-10-02)

Both QC modules expose the same three steps, with parallel names and the same verdict language:

| Step | Timing (`video_timing_qc`) | Quality (`video_quality_qc`) |
|---|---|---|
| Checks → table | `check_video_timing(timing, trigger_times=None, video_frame_count=None)` | `check_video_quality(samples, camera)` (renamed from `run_checks`) |
| Verdict | `timing_verdict(checks)` | `quality_verdict(checks)` (renamed from `quality_action`) |
| Record, optional | `write_video_timing(checks, out_dir, camera)` (new) | `write_video_quality(samples, checks, out_dir, camera, note)` |

- **Verdict** for both: `use` or `exclude: <first failed check>`. No other values.
- **How timing is corrected is internal** to `correct_video_timing` (decided from the checks;
  `as written`, `fix glitches` or `re-index`), not a public function. It is already visible per
  frame in `harp_source` and goes, as one field, in the written record. Nothing outside the
  correction needs it to decide anything.
- **The verdict is final from the checks table.** Today a camera can pass `timing_action` and
  still be refused inside the correction: the trigger log's length or values do not match the
  CSV, or the re-indexed times disagree with camera time (`harp_matches_camera`). These become
  rows of `check_video_timing` (it takes `trigger_times` and runs the re-index trial when
  frames were lost), and `video_frame_count` (when given) counts toward the verdict, so
  `build_me_table.py`'s own frame-count refusal is no longer needed. `correct_video_timing`
  calls `check_video_timing` and raises when the verdict is not `use`.
- `timing_action` (released in v0.1.0; printed by `integrate_keypoints_with_video_time`,
  stored by `kinematics_analysis/code/build_me_table.py`) stays one release as a deprecated
  alias returning the old strings (`DeprecationWarning`); the library's own caller moves to
  `timing_verdict`. `run_checks` / `quality_action` were never released: renamed outright.
  `check_session` (timing) gets a `verdict` column in place of `action`.
- **Screen row `use`** = both verdicts `use` (or a manual override).
- **`error: <text>`**: a camera that could not be screened (unreadable file, network). Not an
  exclusion; re-screened on the next run.

### Inputs: the caller passes file locations (decided 2026-10-02)

Finding files is **out of scope** for the library (as revision 8 already decided: no folder
discovery). The screen takes, per session × camera: `session`, `camera`, `mp4`, `video_csv`,
`behavior_json` (optional), `trigger_log` (optional). Each is a local path (Code Ocean data
asset, e.g. `/root/capsule/data/behavior_.../behavior-videos/bottom_camera.mp4`) or an HTTPS
URL; the caller chooses, per file, and the module only handles what it is given:

- MP4 (PyAV, `read_mp4_frame_index`, `probe`) and behavior JSON (`read_trial_times`): read in
  place, path or URL.
- Video CSV (`video_alignment._csv_has_header` uses `open`) and trigger log (`np.fromfile`):
  readers need a local file, so a URL is downloaded to a temporary file and deleted after
  (the survey script's `download`, with its socket timeout). This is opening the given input,
  not locating it.

Callers build the inputs with whatever suits them: Code Ocean paths, the existing helpers in
`kinematics/tongue_kinematics_utils.py` (`find_video_path`, `find_video_csv_path`,
`find_behavior_videos_folder`), or the survey script's own S3 listing (`list_keys`,
`session_files`, which stay in `scripts/video_quality_survey.py`).

### Layers (each usable alone; nothing writes unless asked)

1. **Per-camera checks and verdicts, in memory**: the two modules' checks and verdict
   functions above (quality also keeps its one-call `video_quality(...)`).
2. **Writers, optional**:
   - new `write_video_timing(checks, out_dir, camera)` → `video_timing_<camera>.json`: verdict,
     correction method, every check row (frames lost, glitch rows, clock rate, frame count vs
     CSV rows, trigger log consistency), whether a trigger log was used, versions. Corrected
     times are **not** written (deterministic, ~2 s to recompute).
   - existing `write_video_quality(...)`; card PNG via `video_quality_report.session_card`
     (imported only when cards are requested, so screening without cards never imports
     matplotlib).
3. **`screen_camera(session, camera, mp4, video_csv, behavior_json=None, trigger_log=None,
   out_dir=None, quality=True, cards=False) -> dict`**: one table row. Timing: `load_video_timing`, `read_harp_trigger_log` (unreadable log → none),
   `check_video_timing(timing, trigger_times, video_frame_count=<MP4 index n_samples>)`,
   `timing_verdict`. Quality: `video_quality(...)`, `quality_verdict`. Writes the detail files only with `out_dir`. Never raises: failures
   become `error: ...`.
4. **`screen_sessions(inputs, out_dir=None, quality=True, cards=False, workers=1) ->
   DataFrame`**: `inputs` is a DataFrame (or list of dicts) with the input columns above, one
   row per session × camera; loops over it (process pool when `workers > 1`, as in the survey
   script). Screening only some cameras = passing only those rows. With `out_dir`:
   `video_screen.jsonl` appended per session (resumable), `video_screen.csv` rebuilt at the end,
   detail files under `<out_dir>/<session>/`; sessions already present with the same library
   version and no error are skipped, so the call doubles as a cache.
5. **`load_screen(path) -> DataFrame`**: reads `video_screen.csv`, applies
   `screen_overrides.csv` beside it if present (`session, camera, verdict, note, reviewer, date`;
   an override replaces `use` and records `override_note`).

### The table (`video_screen.csv`), one row per session × camera

`session`, `subject`, `camera`, `view` (`bottom`/`side`, `video_quality_qc.camera_view`),
`mp4` (the input as given), `use`, `reason` (first failure: `timing: exclude: harp_evenly_spaced`,
`quality: exclude: ...`, or `error: ...`), `timing` (verdict), `timing_method` (from the
correction, for information), `frames_lost`,
`glitch_rows`, `frame_count_diff`, `trigger_log` (used or not), `quality` (verdict), `window`,
a few medians (`sharpness`, `mean`, `pct_clipped_high`, `similarity_p5`), `versions`,
`screened_at`. Per camera: an analysis needing only the bottom camera filters
`view == 'bottom' and use`.

### Module layout

- `video_timing_qc.py`: `timing_verdict`, `write_video_timing`; `check_video_timing` gains
  `trigger_times` and the trigger-log and re-index rows; `correct_video_timing` follows the
  verdict and keeps the method internal; `timing_action` deprecated; `check_session` reports
  `verdict`.
- `video_quality_qc.py`: `run_checks` → `check_video_quality`, `quality_action` →
  `quality_verdict` (and their callers: report, survey script, tests, README, notebook).
- New `video_screen.py`: `screen_camera`, `screen_sessions`, `load_screen`. Plain functions,
  no classes (revision 8 principle). No file discovery.
- `scripts/video_quality_survey.py`: keeps its S3 listing to build the inputs table for the
  public bucket, then calls `screen_sessions`; keeps `--report` and `--thresholds`, reading the
  screen's files.
- Optional, last: `run_batch_analysis(..., screen=None)` skips sessions whose camera row is not
  `use`, so the existing pipeline can consume the table without other changes.

### Usage (goes in the README)

```python
from aind_dynamic_foraging_behavior_video_analysis import video_screen as vs

# up front: one row per session x camera, paths or URLs, built however suits the caller
inputs = pd.DataFrame([{
    "session": s, "camera": "bottom_camera",
    "mp4": f"/root/capsule/data/{s}/behavior-videos/bottom_camera.mp4",        # or an https URL
    "video_csv": f"/root/capsule/data/{s}/behavior-videos/bottom_camera.csv",
    "behavior_json": ..., "trigger_log": ...,
} for s in sessions])
screen = vs.screen_sessions(inputs, out_dir="/root/capsule/results/screen", workers=8)

# later, in any analysis
screen = vs.load_screen("screen/video_screen.csv")
for session in screen.query("view == 'bottom' and use").session: ...

# or as a gate inside an existing loop, nothing written
row = vs.screen_camera(session, camera, mp4, video_csv, behavior_json, trigger_log)
```

## Decisions during implementation (2026-10-02)

Where the design above left a choice open, the simplest option was taken:

- **Verdict order (timing).** `timing_verdict` keeps the old decision order, then the new rows:
  `ALWAYS_REQUIRED`; `REQUIRED_TO_REINDEX` when frames were lost; then `trigger_log_count`,
  `trigger_log_matches_csv`, `harp_matches_camera`, `video_frame_count` when they ran. Lost
  frames, glitches, and out-of-order frame numbers with no frames lost (corrupted metadata) still
  do not exclude, as before. When several checks fail, the reason reported can differ from the
  old error message (old: a trigger log mismatch was raised before the input checks); the
  outcome (use or not) is the same.
- **Re-index trial row.** `harp_matches_camera` is always in the table; skipped (`passed` None)
  with a reason when no frames were lost, when re-indexing is not allowed, or when the trigger
  log does not fit the CSV. The input checks run on the CSV's Harp column (the log, when it
  passes `trigger_log_matches_csv`, is equal within one tick).
- **Method names.** The internal method is `_correction_method(checks)`: `as written`,
  `fix glitches`, `re-index`, or None when excluded. `video_screen` reads it for
  `timing_method` (same package).
- **`check_session`** reports `exclude: unreadable` where it said `refuse: unreadable`.
- **Quality record.** The `video_quality_<camera>.json` field `action` is now `verdict`, and the
  session card says `VERDICT:`, matching timing. The notebook's stored outputs still say
  `action` (not re-run).
- **Quality runs when timing excludes**, so both verdicts are always recorded (`reason` names
  timing first).
- **`quality=False`**: `quality`, `window` and the medians are empty; `use` follows timing. A
  later run with `quality=True` screens those cameras again.
- **Cache key.** A camera is skipped when its last row in `video_screen.jsonl` has the same
  `versions` string (this package and `aind-video-utils`), no error, and quality measured if
  asked for. `cards` does not count.
- **Work unit.** `screen_sessions` groups input rows by session: one worker task per session,
  so files given as URLs (the trigger log) download once per session, and the session's rows
  are appended to `video_screen.jsonl` together. `video_screen.csv` holds the last row per
  session × camera of everything in the log, not only the current inputs.
- **Errors.** `reason` is `error: <type>: <message> (<file>:<line>)`; the timing and quality
  columns stay empty.
- **Frame count** comes from the MP4 index (`read_mp4_frame_index(mp4).n_samples`), read once
  for timing and again by quality sampling.
- **Trigger log column** is a boolean: True when a log was given and readable.
- **Version 0.2.0.** `__version__` is 0.2.0 in this branch (the release comes with this PR,
  decided 2026-10-02 for Part A); README "Changes" says "0.2.0 (unreleased)". The screen's
  cache key includes it.
- **Survey script.** Lists sessions in parallel threads, builds all-URL inputs, and calls
  `screen_sessions(..., cards=True)`. The thumbnails and contact sheets are gone (the session
  card already shows 8 evenly spaced frames); `--report` is the index plus every card. Its
  output folder is the screen's (`video_screen.csv`, not `summary.csv`). matplotlib's `Agg`
  backend is selected in `main()` so the imports stay at the top.

## Steps (commits)

1. `video_timing_qc`: complete checks table (trigger log rows, re-index trial,
   `video_frame_count` in the verdict), `timing_verdict`, method internal to the correction,
   `write_video_timing`, `timing_action` deprecated, `check_session` verdict; update
   `integrate_keypoints_with_video_time` to print the verdict; tests in
   `tests/test_video_timing_qc.py` (every existing case keeps its outcome: old `refuse: X` ==
   new `exclude: X`).
2. `video_quality_qc`: rename to `check_video_quality` / `quality_verdict` everywhere.
3. `video_screen`: input handling (temp download of a CSV or log given as a URL), then
   `screen_camera`, `screen_sessions` (incremental by session × camera and library version,
   workers, quality on/off, cards), `load_screen` with overrides.
4. Survey script builds inputs from its S3 listing and calls `screen_sessions`; README section and "Changes"; plan docs
   (pointers to this file from `VIDEO_QUALITY_QC_PLAN.md` Phase 3 and `VIDEO_TIMING_QC_PLAN.md`;
   this file's status updated as steps land).
5. Optional: `run_batch_analysis(screen=...)`.
6. Follow-up outside this repo, after release: `kinematics_analysis/code/build_me_table.py`
   moves from `timing_action` and its own frame-count refusal to `timing_verdict`.

## Verification

- Unit tests (`tests/test_video_screen.py`, reusing the synthetic MP4/CSV/JSON/trigger-log
  fixtures from `tests/test_video_quality_qc.py`): local inputs; URL inputs with the download
  mocked; each timing case's verdict (old `refuse: X` → `exclude: X`, the three old "use"
  actions → `use`); trigger-log mismatch and failed re-index now in the checks table; frame-count
  mismatch excludes; `timing_action` alias unchanged with a warning; the timing
  correction's outputs unchanged on every existing test; unreadable log ignored; errors recorded and re-screened; incremental skip;
  `quality=False`; writes nothing without `out_dir`; overrides in `load_screen`.
  100% line coverage of `video_screen.py`, `video_quality_qc.py`, `video_quality_report.py` and
  the changed timing code; full suite passes;
  black, isort, flake8 clean.
- Real data, local and URL inputs give the same rows: screen
  `behavior_800886_2025-08-18_13-14-52` once with all-URL inputs and once with the CSVs, JSON and
  trigger log downloaded to local paths (MP4 still a URL; the module treats each input
  independently); compare every column but the input locations.
- Full re-run of the 301 FIP sessions through `screen_sessions` (8 workers, ~2 h, new folder
  `video_quality_qc_data/screen_fip/`). This also confirms the similarity cutoffs under the
  all-samples reference (pending from revision 8). Expected: quality verdicts as in the last
  survey unless the reference change moves a camera across 0.7 / 0.998; timing verdicts match
  `me_dry_run_fip.csv` on its 97 sessions (178 use, 16 exclude), plus the 56 clock-step cameras
  excluded across all 301.
- Then open the PR from `feat/video-screen` to `main`.
