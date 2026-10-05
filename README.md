# aind-dynamic-foraging-behavior-video-analysis

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.11-blue?logo=python)



## Scope — what belongs in this library

One test: **would another AIND project doing tongue kinematics want this, unchanged?**

- **Here:** code that produces or annotates the per-session intermediates
  (`tongue_kins.parquet`, `tongue_movs.parquet`, `kps_raw_*.parquet`,
  `tongue_quality_stats.json`), runs in the batch pipeline, or is generic to
  tongue-kinematics sessions — keypoint I/O and filtering, segmentation,
  aggregation, trial/lick annotation, QC stats, lick detection, video/NWB
  lookup, video timing and quality QC, clip extraction, raster/PSTH
  primitives. It must
  stay stable: consuming capsules pin a commit and move the pin deliberately.
- **Not here:** analysis built *on top of* the intermediates for one
  scientific question — encoding models, per-unit result registries, spatial
  topography, manuscript figure styling. That lives in the consuming repo
  (e.g. `kinematics_analysis`) and is free to churn.
- **Plots:** the plot functions here are pipeline QC artefacts written to
  disk by `analyze_tongue_movement_quality`. They carry no styling contract;
  presentation figures are the consumer's job.
- **Contracts:** files this library writes and consumers read are declared
  next to the writer (see `TONGUE_QUALITY_STATS_FILENAME` in
  `kinematics/tongue_analysis.py`); read them through the accessor, not the
  path.

Module layering is spelled out in each module's docstring
(`kinematics/tongue_kinematics_utils.py`, `kinematics/tongue_lickometer_utils.py`,
`ephys/tongue_ephys.py`).


## Video timing QC

`video_timing_qc` checks and corrects the Harp (behavior-clock) time of every
frame in a behavior video CSV. The acquisition workflow pairs frames with
Harp triggers in arrival order, so when frames are dropped every later frame
carries an earlier trigger's time (minutes off by the end in affected
sessions); single Harp values can also be wrong (~983 ms glitches).

```python
from aind_dynamic_foraging_behavior_video_analysis import video_timing_qc as vtq

timing = vtq.load_video_timing("behavior-videos/bottom_camera.csv")  # either layout
checks = vtq.check_video_timing(timing)   # one row per check: passed, count, message, rows
vtq.timing_verdict(checks)                # "use" or "exclude: <check>"
fixed = vtq.correct_video_timing(timing)  # adds harp_time and harp_source; raises if excluded
log = vtq.read_harp_trigger_log("Event_94.bin")
checks = vtq.check_video_timing(timing, trigger_times=log, video_frame_count=n_frames)
fixed = vtq.correct_video_timing(timing, trigger_times=log)
vtq.write_video_timing(checks, "results/", "bottom_camera")  # video_timing_bottom_camera.json
```

The verdict is final from the checks table: with a trigger log it includes
whether the log has one event per exposure and matches the CSV, when frames
were lost whether the re-indexed times match the camera, and with a frame
count whether the video has one frame per CSV row. How the times are
corrected (as written, fixing glitches, or re-indexing) is decided inside
`correct_video_timing`.

The LP pipeline (`integrate_keypoints_with_video_time`, `generate_tongue_dfs`,
`run_batch_analysis`) uses it: keypoint `time_raw` is the corrected Harp time,
the session's trigger log is used when present, and sessions whose timing
cannot be trusted (e.g. a Harp clock step) raise `ValueError` and are skipped
by the batch. The checks, the decision, the evidence and the known limits are
in `VIDEO_TIMING_QC_PLAN.md`; `examples/video_timing_qc_validation.ipynb`
shows it on real sessions against the trigger log.


## Video quality QC

`video_quality_qc` measures image quality on about 100 keyframes spread
across a behavior-video MP4 and says whether to use the video. Only the
sampled keyframes are decoded (a few seconds locally, about 30-45 s per
camera over HTTPS). Every metric is saved per keyframe: brightness and
exposure statistics, contrast, entropy, clipping at the tagged range,
sharpness (Laplacian variance, 2× downsampled), noise, similarity to a
reference frame, and shift from it (reported only). Plain functions: frames
are a uint8 array, samples a pandas table, and checks are rows of the
`CHECKS` table. Needs the `video-qc` extra and `ffprobe` on `PATH`.

```python
from aind_dynamic_foraging_behavior_video_analysis import video_quality_qc as vqq
from aind_dynamic_foraging_behavior_video_analysis import video_quality_report as vqr

# All in one: the task window (first trial start to last trial end, from the
# session JSON and the camera's CSV; the middle 50% of the file if that
# fails), keyframes, metrics, checks.
frames, samples, checks, note = vqq.video_quality(
    "behavior-videos/bottom_camera.mp4", "bottom_camera",
    behavior_json="behavior/<subject>_<datetime>.json",
    video_csv="behavior-videos/bottom_camera.csv",
    trigger_log="behavior/raw.harp/BehaviorEvents/Event_94.bin",
)
vqq.quality_verdict(checks)   # "use" or "exclude: <check>"
vqq.write_video_quality(samples, checks, "results/", "bottom_camera", note)
vqr.session_card(frames, samples, checks, "<session>  bottom_camera")

# Or step by step:
window, note = vqq.sample_window(behavior_json, video_csv, trigger_log)
frames, samples, color_range = vqq.sample_keyframes(mp4, window)
samples = vqq.measure(frames, samples, color_range)
checks = vqq.check_video_quality(samples, camera)
```

Checks (`CHECKS`; values set on a survey of 301 FIP sessions, 602 cameras):

| Check | Fails when |
|---|---|
| `sharpness_dev <= 0.45` | two consecutive samples are more than 45% off the median sharpness |
| `mean_dev <= 0.15` | two consecutive samples are more than 15% off the median brightness |
| `similarity >= 0.7` | two consecutive samples correlate below 0.7 with the reference |
| `similarity p5 < 0.998` | nothing in view moves (a camera pointed away from the mouse) |
| `mean median >= 50`, `mean median <= 150` | the session's median brightness is too dark or too bright |
| `pct_clipped_high median <= 3.75` | side cameras only: the jaw, mouth and paws saturate |

One sample alone out of range is listed but passes (a paw in front of the
lens). The brightness limits are wide (survey medians 55–105, so none
excluded); level cutoffs on sharpness and contrast are not used: across
sessions they track the scene (background, rig) more than quality.
A dirty bottom mirror is not detected by any check.

The recording often runs past the session (and starts before it), which
fails the stability checks, so the task is sampled.
`video_alignment.task_frame_window` places it: trial times from the raw
session JSON (the same values as the NWB trials table, so no NWB is
needed), put on frames by the same timing QC correction the kinematics
pipeline uses (with the trigger log when present). Where that strict
correction refuses a camera for an error of a frame or two, the window falls
back to the trigger log by frame number, else the raw Harp column when no
frames were lost. The reference for `similarity` and `shift` is the
pixel-wise median of all samples, the session's typical view.

Outputs per camera: `video_quality_<camera>.parquet` (every metric per
keyframe, with histograms) and `video_quality_<camera>.json` (window note,
versions, checks, verdict). Survey outputs written before revision 8
(`video_quality_qc_data/survey_fip/`) use the older names:
`..._samples.parquet`, summary fields (`sharpness_med`, ...) and check names
(`sharpness_stable`, `brightness_stable`, `scene_stable`, and level checks
recorded as skipped; `scene_moves` and the clipping cutoff came later). Design, evidence and limits: `VIDEO_QUALITY_QC_PLAN.md`;
`examples/video_quality_qc_validation.ipynb` runs it on a public session;
`scripts/video_quality_survey.py` runs it on many.


## Video screening

`video_screen` runs timing QC and quality QC on each camera, upstream of any
analysis (pose estimation, motion energy, `run_batch_analysis`), and gives
one row per session × camera: `use` (both verdicts `use`) and `reason`
(`timing: exclude: <check>`, `quality: exclude: <check>`, or `error: ...`
for a camera that could not be screened, which is screened again next time).
The caller passes each file as a local path (e.g. a Code Ocean data asset)
or an HTTPS URL; the module never searches for files. Nothing is written
unless `out_dir` is given.

```python
from aind_dynamic_foraging_behavior_video_analysis import video_screen as vs

# up front: one row per session x camera, paths or URLs, built however suits the caller
inputs = pd.DataFrame([{
    "session": s, "camera": "bottom_camera",
    "mp4": f"/root/capsule/data/{s}/behavior-videos/bottom_camera.mp4",        # or an https URL
    "video_csv": f"/root/capsule/data/{s}/behavior-videos/bottom_camera.csv",
    "behavior_json": ..., "trigger_log": ...,                                  # optional
} for s in sessions])
screen = vs.screen_sessions(inputs, out_dir="/root/capsule/results/screen", workers=8)

# later, in any analysis
screen = vs.load_screen("screen/video_screen.csv")
for session in screen.query("view == 'bottom' and use").session: ...

# or as a gate inside an existing loop, nothing written
row = vs.screen_camera(session, camera, mp4, video_csv, behavior_json, trigger_log)
```

With `out_dir`, `screen_sessions` appends to `video_screen.jsonl` as each
session finishes (an interrupted run resumes), rebuilds `video_screen.csv`,
writes `video_timing_<camera>.json` and the quality files (and the session
card with `cards=True`) under `<out_dir>/<session>/`, and skips cameras
already screened by the same library versions. `quality=False` screens
timing only. Columns and decisions:
`VIDEO_SCREEN_PLAN.md`. `scripts/video_quality_survey.py` builds the inputs
from the public S3 bucket and screens many sessions.

### Using the screen

Read the table with `load_screen` (it keeps `subject` as text and empty
`reason`s as `""`) and decide with the `use` column, never by parsing
`reason`.

```python
screen = vs.load_screen("screen/video_screen.csv")

# Cameras to analyze: filter per camera, since an analysis may need only one view.
bottom = screen.query("view == 'bottom' and use")
for session in bottom.session: ...

# Sessions where every camera can be used.
all_ok = screen.groupby("session")["use"].all()
sessions = all_ok[all_ok].index

# Why cameras are dropped (first failure, timing before quality).
screen.loc[~screen.use, "reason"].value_counts()
```

Each camera's evidence is in `<out_dir>/<session>/`:

| File | Holds |
|---|---|
| `video_timing_<camera>.json` | timing verdict, correction method, trigger log used, frames lost, glitch rows, every check (`passed`, `count`, `message`, offending rows) |
| `video_quality_<camera>.json` | sampling window, every quality check with its observed value, verdict |
| `video_quality_<camera>.parquet` | every metric per sampled keyframe, with luma histograms |
| `session_card_<camera>.png` | one page for review by eye (with `cards=True`) |

`python scripts/video_quality_survey.py <sessions.csv> <out_dir> --report`
collects the cards into one PDF behind an index, cameras not in use first.

To screen new sessions, call `screen_sessions` again with the same `out_dir`:
cameras already screened by the same library versions are skipped, and
`error:` rows are screened again. Columns are listed in the `video_screen`
module docstring. Counts (`frames_lost`, ...) read back from the CSV as
floats, since cameras without timing checks leave them empty.


## Changes

### 0.2.0 (2026-10-05)

- **New:** `video_screen` (see above): `screen_camera`, `screen_sessions`,
  `load_screen`.
- **New:** `video_timing_qc.timing_verdict` (`use` or `exclude: <check>`) and
  `write_video_timing`. `check_video_timing` takes `trigger_times` and adds the
  rows `trigger_log_count`, `trigger_log_matches_csv`, `harp_matches_camera`
  (the re-index trial) and counts `video_frame_count` in the verdict.
- **Same outcomes:** `correct_video_timing` follows the verdict and raises in
  the same cases as before; every camera is used or refused as before
  (`refuse: X` is now `exclude: X`). New is that the verdict also excludes a
  video frame count that differs from the CSV, when a count is given.
- **Deprecated:** `video_timing_qc.timing_action` (warns; same strings as
  before); removed in a later release. `integrate_keypoints_with_video_time`
  prints the verdict instead.
- **Changed return value:** `video_timing_qc.check_session` has a `verdict`
  column in place of `action`.
- **Renamed (never released):** `video_quality_qc.run_checks` →
  `check_video_quality`, `quality_action` → `quality_verdict`; the quality
  record's `action` field → `verdict`.
- **New:** `video_alignment.read_trial_times` (trial start, go cue and end
  from the raw session JSON, as in the NWB),
  `video_alignment.behavior_time_to_frame_index` and
  `video_alignment.task_frame_window`.
- **New:** `video_quality_qc` and `video_quality_report` (see above), in a
  new `video-qc` extra (`aind-video-utils==0.7.0`, `av`, `matplotlib`,
  `pyarrow`). Nothing existing changes.

### 0.1.0 (2026-09-30)

- **New:** `video_timing_qc` (see above).
- **Changed behavior:** `integrate_keypoints_with_video_time` uses it instead of
  its old timing QC. Keypoint `time_raw` changes for sessions with dropped
  frames (up to minutes) or Harp glitches (the row after a glitch by µs);
  sessions with no problems are unchanged. Sessions with a Harp clock step are
  refused. Header-row (New/AIND) video CSVs now load instead of crashing.
- **Changed return value:** the second value returned by
  `integrate_keypoints_with_video_time` is the corrected timing table
  (`harp_time_raw`, `frame_number`, `camera_time`, `harp_time`,
  `harp_source`) instead of `Behav_Time`/`Frame`/`Camera_Time`/`Gain`/`Exposure`.
- **New arguments:** `generate_tongue_dfs` and `run_batch_analysis` take
  `use_trigger_log` (default: use `Event_94.bin` when present) and
  `camera_name` (default `"BottomCamera"`, as before).
- `generate_tongue_dfs` raises `FileNotFoundError` when no video CSV is found
  (was `AttributeError`).


## Usage
 - To use this template, click the green `Use this template` button and `Create new repository`.
 - After github initially creates the new repository, please wait an extra minute for the initialization scripts to finish organizing the repo.
 - To enable the automatic semantic version increments: in the repository go to `Settings` and `Collaborators and teams`. Click the green `Add people` button. Add `svc-aindscicomp` as an admin. Modify the file in `.github/workflows/tag_and_publish.yml` and remove the if statement in line 65. The semantic version will now be incremented every time a code is committed into the main branch.
 - To publish to PyPI, enable semantic versioning and uncomment the publish block in `.github/workflows/tag_and_publish.yml`. The code will now be published to PyPI every time the code is committed into the main branch.
 - The `.github/workflows/test_and_lint.yml` file will run automated tests and style checks every time a Pull Request is opened. If the checks are undesired, the `test_and_lint.yml` can be deleted. The strictness of the code coverage level, etc., can be modified by altering the configurations in the `pyproject.toml` file and the `.flake8` file.

## Installation
Supported Python: 3.11 and 3.12, each tested in CI.

For a Python 3.9 or 3.10 environment, install the last release that supports
it, tagged `py39-final`:
```bash
pip install "git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis.git@py39-final"
```

To use the software, in the root directory, run
```bash
pip install -e .
```

The core install (numpy, pandas) covers `video_alignment`,
`video_timing_qc` and `kinematics/tongue_lickometer_utils`. For the kinematics, ephys, NWB and
video-clip modules, install the `kinematics` extra:
```bash
pip install -e ".[kinematics]"
```

For video quality QC, install the `video-qc` extra (it also needs
`ffmpeg`/`ffprobe` on `PATH`):
```bash
pip install -e ".[video-qc]"
```

To develop the code, run
```bash
pip install -e ".[kinematics,video-qc,dev]"
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o docs/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html docs/source/ docs/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

### Read the Docs Deployment
Note: Private repositories require **Read the Docs for Business** account. The following instructions are for a public repo.

The following are required to import and build documentations on *Read the Docs*:
- A *Read the Docs* user account connected to Github. See [here](https://docs.readthedocs.com/platform/stable/guides/connecting-git-account.html) for more details.
- *Read the Docs* needs elevated permissions to perform certain operations that ensure that the workflow is as smooth as possible, like installing webhooks. If you are not the owner of the repo, you may have to request elevated permissions from the owner/admin. 
- A **.readthedocs.yaml** file in the root directory of the repo. Here is a basic template:
```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

# Required
version: 2

# Set the OS, Python version, and other tools you might need
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"

# Path to a Sphinx configuration file.
sphinx:
  configuration: docs/source/conf.py

# Declare the Python requirements required to build your documentation
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
```

Here are the steps for building docs in *Read the Docs*. See [here](https://docs.readthedocs.com/platform/stable/intro/add-project.html) for detailed instructions:
- From *Read the Docs* dashboard, click on **Add project**.
- For automatic configuration, select **Configure automatically** and type the name of the repo. A repo with public visibility should appear as you type. 
- Follow the subsequent steps.
- For manual configuration, select **Configure manually** and follow the subsequent steps

Once a project is created successfully, you will be able to configure/modify the project's settings; such as **Default version**, **Default branch** etc.
