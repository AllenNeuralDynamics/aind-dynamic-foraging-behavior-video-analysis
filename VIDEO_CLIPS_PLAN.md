# Plan: `video_clips` — frame-exact clips, and frames for labeling

> **Status: revision 9 (2026-10-08), Phases 1–3 implemented** (`video_clips.py`,
> `video_alignment.event_frame_ranges`, `tests/test_video_clips.py`, README,
> `examples/video_clips_example.ipynb`); results under "Verification", deviations under
> "Changes". Phase 4 (consumers) is outside this repo. Revision 8 built the module around
> one key, *(source video, frame index)*: it cuts and extracts by frame index and takes no
> times. Turning event times into frames is `video_alignment`'s job; which cameras to use is the
> screen's. Earlier revisions are summarized under "Changes". The file was
> `VIDEO_CLIPS_MIGRATION_PLAN.md` until revision 5. Written so a new contributor or agent can
> pick it up without the conversation that produced it.

## Use cases

1. **Cut video clips**, usually around behavior events: a lightweight, frame-exact tool.
2. **Frames for labeling**: clips around behavior events are the videos of a DeepLabCut
   project; frames are extracted from them and labeled in DLC; the labels are moved to Lightning
   Pose (with our own code, which reorganizes folders) for training and inference. Every label
   must trace back to an exact source frame, and from there to its behavior time.

## Design

### The key: source video and frame index

A clip is "frames `[start, start + n)` of this MP4". A labeled image is "frame *k* of that
clip", so source frame `start + k`. Both are exact and never go stale. Row *i* of the video CSV
is frame *i* of the MP4, also when frames were dropped (`VIDEO_TIMING_QC_PLAN.md`), so a source
frame index is also a CSV row and a Lightning Pose prediction row.

**Behavior time is joined on afterwards, not stored.** It comes from the camera's corrected
timing (`video_timing_qc.correct_video_timing`) when someone needs it. Timing correction may
improve; labels keyed on frames stay right.

### Separation

| Job | Where |
|---|---|
| Which cameras to use (timing + quality) | `video_screen` (caller filters on `use`) |
| Behavior time of each frame | `video_timing_qc.correct_video_timing` |
| Event times → frame ranges (for clips, or any per-frame signal) | `video_alignment.event_frame_ranges` (new) |
| Picking which events (go cues, licks, ...) | the caller; the example notebook shows how |
| Frame ranges → clips; clips → labeling frames; labels → source frames | `video_clips` (new) |
| DLC → LP folder reorganization | our own conversion code (outside the library), which calls `video_clips.add_context_frames` |

`video_clips` takes no times, runs no QC and imports no other module of this package. Its
dependencies: numpy, pandas, `aind-video-utils` (frame-exact seeking), scikit-learn (k-means
frame selection), and ffmpeg on `PATH`.

### Use case 1: clips

```python
from aind_dynamic_foraging_behavior_video_analysis import video_alignment as va
from aind_dynamic_foraging_behavior_video_analysis import video_clips as vc
from aind_dynamic_foraging_behavior_video_analysis import video_timing_qc as vtq

timing = vtq.correct_video_timing(
    vtq.load_video_timing(video_csv), trigger_times=vtq.read_harp_trigger_log(trigger_log)
)
go_cues = va.read_trial_times(behavior_json)["goCue_start_time"].to_numpy()
ranges = va.event_frame_ranges(go_cues[::20], timing["harp_time"], before=1.0, after=1.0)
clips = vc.cut_clips(mp4, ranges, out_dir="clips/", prefix=f"{session}_{camera}")
```

**`video_alignment.event_frame_ranges(event_times, harp_time, before, after)`** returns a table,
one row per event: `event_time`, `start_frame` (first frame at or after `event - before`),
`n_frames` (up to the first frame at or after `event + after`), and `in_video` (False when the
window runs past either end; `cut_clips` skips such rows). It is `behavior_time_to_frame_index`
twice.

It is a general frame-window helper, not a clipping one: any per-frame signal is row-aligned
with the CSV (motion energy, BEAST latents, Lightning Pose predictions), so the same ranges
slice event-aligned windows of those for averages or PSTHs, with no video involved. That is why
it lives in `video_alignment` (core dependencies only, next to `behavior_time_to_frame_index`,
and the job that module's docstring already names) and not in `video_clips`.
Its docstring says plainly: pass **corrected** `harp_time`. The raw CSV column is wrong for
whole sessions with dropped frames (18 of 97 recent FIP sessions, up to 374 s by the end), so a
clip placed by it shows the wrong moment. Across a drop a window has fewer frames and still spans
the requested time.

**`video_clips.cut_clip(mp4, start_frame, n_frames, out_path, index=None)`**: one clip.

```python
seek = index.presentation_seconds(start_frame)
ffmpeg -accurate_seek -ss f"{seek:.9f}" [http input flags] -i <mp4> \
       -frames:v <n_frames> -fps_mode passthrough -c:v libx264 ... <out_path>
```

- `aind_video_utils.read_mp4_frame_index` gives each frame's real container timestamp, so the
  seek is frame-exact with no nominal frame rate. This is `extract_frame_by_index`'s seek plus a
  frame count and an encoder. Checked in revision 4 on a variable-frame-rate synthetic video
  (11 of 11 frames exact; an `(i - 0.5) / fps` seek got 3 wrong). `mp4_index.py` and
  `frames.py` are unchanged in `aind-video-utils` 0.8.0.
- `-ss` before `-i` with a re-encode lands on the first frame at or after the seek;
  `-frames:v` is exact (not `-t`); `-fps_mode passthrough` neither duplicates nor drops frames.
- **The MP4 may be an HTTPS URL**: the index and ffmpeg read only the bytes they need, so clips
  come straight from `s3://aind-open-data`. HTTP input flags as in `aind-video-utils`
  (`http_input_flags` is private there: copy the flags, or ask upstream to export it).
- Raises `ValueError` if the MP4's edit list is not frame-addressing-safe
  (`index.is_frame_addressing_safe()`), or if `presentation_seconds(start_frame)` raises
  (non-monotonic container timestamps there). This is about seeking correctly, not QC.

**`video_clips.cut_clips(mp4, ranges, out_dir, prefix)`**: many clips from one MP4. Reads the
index once; cuts each row of `ranges` (any table with `start_frame` and `n_frames`; rows with
`in_video` False are skipped). Returns `ranges` plus `clip` and `status` (`cut`, `exists`,
`skipped: <why>`); one bad seek skips that row only.

Names and sidecars:

```
clips/
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_f0174266.mp4
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_f0174266.json
```

- **The stem is `<prefix>_f<start_frame:07d>`**: unique, stable, and it carries the start
  frame, so re-running never renumbers or overwrites another clip, and two events with the same
  window give one clip.
- **Sidecar** (`SIDECAR_FILE`): `source_video` (as given; a URL stays a URL), `start_frame`,
  `n_frames`, `versions` (this package, `aind-video-utils`). Nothing else.
- Written last (temp file, rename) after ffmpeg succeeds. **Resume**: a clip is `exists` only if
  its sidecar has the same `source_video` and `n_frames`; otherwise re-cut.
- **A clip that spans drops does not play in real time**: missing frames are absent and the MP4
  keeps even timestamps. Behavior time per frame comes from the timing join, never from a frame
  rate.

### Use case 2: labeling

The clips are the videos of the DLC project (added to `video_sets` in `config.yaml`, as DLC
expects; `labeled-data/<clip stem>/` matches them).

**`select_frames(clip, num_frames, labeled_data_dir, algorithm="uniform", seed=0, margin=2)`**
writes PNGs to `labeled_data_dir/<clip stem>/` and returns the clip-frame indices.

- **Named exactly as DLC's own frame extraction would:** `"img" + str(k).zfill(indexlength)`,
  `indexlength = ceil(log10(n_frames))` (DLC `frame_extraction.py`; `img042.png` in a 1000-frame
  clip). A labeler running DLC's extraction on the same clip gets the same files, not
  duplicates. (The capsule's "overflow past frame 999" was this convention with a longer clip;
  computing the width from the clip fixes it.)
- **Never picks the first or last `margin` frames**, so every labeled frame has the ±2
  neighbours Lightning Pose's context models need.
- Algorithms: `uniform` (`np.linspace`), `random` (`np.random.default_rng(seed)`), `kmeans` (one
  ffmpeg decode of the clip, downscaled to gray with explicit dimensions;
  `MiniBatchKMeans(random_state=seed, n_init=3)`; each cluster's member nearest its centroid).
- One ffmpeg call per PNG (`-vf select=eq(n\,K) -frames:v 1 -fps_mode passthrough`), counting
  frames. No cv2.

**`add_context_frames(labeled_data_dir, clips_dir, offsets=(-2, -1, 1, 2))`**, for the DLC → LP
step. Lightning Pose's temporal context network needs frames *t−2 … t+2* as PNGs in the same
folder as each labeled frame *t* (e.g. `img007`, `img008`, `img010`, `img011` for a labeled
`img009`), with the labels CSV unchanged (LP docs, "Temporal Context Network"). They are added
**after labeling**, not by `select_frames`: DLC's labeling GUI shows every PNG in a folder, so
context frames present during labeling would appear as frames to label.

**Lightning Pose has no function that writes them** (checked on LP `main` at `0b5ebf4`,
2026-09-21: package, `scripts/`, docs). The context-frames docs page gives a copy-and-paste
snippet, `get_frames_from_idxs(cap, idxs)`, that loads frames by index with OpenCV
(`cap.set` + `cap.read`, grayscale) and warns to check it returns the correct frames; it does not
read the labels or write PNGs, and `cap.set` seeking is not frame-exact on compressed video (the
capsule's k-means problem). `add_context_frames` uses the module's frame-exact extraction
instead. `litpose convert` (DLC import) only merges the
`CollectedData` files and copies `labeled-data/` and `videos/`. Training reads context frames
by name (`utils/io.get_context_img_paths`: the digits in the image name, ±2, same width, floored
at 0) and **silently uses the centre frame for any that are missing** (`data/datasets.py`). So
without this step a context model trains without real context and nothing warns. The separate
LP labeling app was not checked; it does not apply to the DLC → LP path.

- Reads each `CollectedData*.csv` under `labeled_data_dir`, finds the labeled images, and writes
  the missing neighbours from the clip with the same name width. Skips files that exist;
  never touches the CSV. Returns a table of frames written.
- Works on either side of our conversion code (the DLC project's or LP's `labeled-data/`), as
  long as the folder names are the clip stems. Our conversion code calls it.
- Across a dropped frame the neighbours are ±2 *saved* frames, so 4 ms apart instead of 2 ms
  there. Stated in the docstring; not corrected.

**`labeled_frames_table(labeled_data_dir)`**: one row per labeled image in every
`CollectedData*.csv`: `image` (path as in the CSV), `clip` (stem), `prefix`, `clip_frame` (from
the PNG name), `start_frame` (from the stem), `source_frame = start_frame + clip_frame`. Parsed
from names alone, so it needs neither clips nor sidecars. Behavior time is one more join:

```python
labels = vc.labeled_frames_table("labeled-data/")
labels["behavior_time"] = timing["harp_time"].to_numpy()[labels["source_frame"]]
```

`source_frame` also indexes Lightning Pose predictions and kinematics for the same camera.

### Picking events

Not in the library. Which go cues or licks to clip is a choice per labeling round, and each is a
line or two of numpy over arrays the caller already has (go cues from
`video_alignment.read_trial_times`, licks from the NWB). The example notebook shows go cues
spread over the session and the first lick after each go cue. If a selection gets reused across
projects, it moves into `video_alignment` then.

## Module API

`video_clips.py`, roughly 200 lines:

| Function | Role |
|---|---|
| `cut_clip(mp4, start_frame, n_frames, out_path, index=None)` | One frame-exact clip (MP4 only) |
| `cut_clips(mp4, ranges, out_dir, prefix)` | Many clips, named by start frame, with sidecars and resume |
| `read_clip_info(clip)` | The sidecar |
| `select_frames(clip, num_frames, labeled_data_dir, algorithm="uniform", seed=0, margin=2)` | DLC-named PNGs for labeling |
| `add_context_frames(labeled_data_dir, clips_dir, offsets=(-2, -1, 1, 2))` | LP context frames after labeling |
| `labeled_frames_table(labeled_data_dir)` | Labels → source frames |

`video_alignment.py` gains `event_frame_ranges(event_times, harp_time, before, after)`.

Private: the ffmpeg command builders (cut, gray decode, PNG), kept apart from `subprocess.run`;
`_png_name(k, n_frames)`; `_parse_stem(stem)` → `(prefix, start_frame)`.

Conventions as in the other modules: plain functions, DataFrames, `ValueError`, file-name
constants next to the writer, `versions` in records, NumPy-style docstrings, no file discovery.

### Dependencies

New extra **`video-clips`**: `aind-video-utils==0.7.0` (the `video-qc` pin; move both
together), `scikit-learn`. CI installs `.[kinematics,video-qc,video-clips,dev]` (it already has
ffmpeg). README "Installation" currently says the `kinematics` extra covers "video-clip modules";
point it at the new extra. `event_frame_ranges` stays core (numpy, pandas).

## Phases

0. **Python upgrade.** Library side done (`PYTHON_311_UPGRADE_PLAN.md`). The re-encoding capsule
   (Python 3.10.9, pinned to `py39-final`) needs 3.11+ to use the module (Phase 4); building
   does not wait on it. (Status as of 2026-09-26.)
1. **Clips.** `video_alignment.event_frame_ranges`; `video-clips` extra and CI line;
   `cut_clip`, `cut_clips`, sidecars, `read_clip_info`.
2. **Labeling.** `select_frames`, `add_context_frames`, `labeled_frames_table`.
3. **Docs and release.** README section (screen → timing → ranges → clips → frames → labels);
   "Changes"; `examples/video_clips_example.ipynb` on a session with dropped frames, including
   event picking.
4. **Consumers** (outside this repo): the re-encoding capsule moves to 3.11+ and calls the
   module; our DLC → LP conversion code calls `add_context_frames`.

## Out of scope

- **Any QC, and any time input to `video_clips`.** The screen and timing QC are upstream.
- **`kinematics/video_clip_utils.py` stays** (`tongue_analysis.py` uses it). Its
  `extract_trial_clip` seeks by a constant offset and is off in drop sessions; tracked in
  `TODO.md`. Moving it onto `event_frame_ranges` + `cut_clip` is the fix.
- **Event selection strategies** (see "Picking events"), finding files, NWB loading.
- **Re-encoding-era helpers dropped**: `run_aind_behavior_video_transformation`,
  `copy_nonvideo_and_metadata_files`, `copy_if_exists`, `is_video_file`, `find_top_level_folders`,
  `process_behavior_video_dry_run`, `extract_frames_only_from_existing_clips`.
- **Merging overlapping clips**: deferred; identical windows already give one clip.

## Verification

1. **Unit tests, no ffmpeg** (`tests/test_video_clips.py`, plus `event_frame_ranges` in the
   alignment tests): window edges, a window across a drop has fewer frames, out-of-range rows
   flagged; command builders (9-decimal seek, `-accurate_seek`, `-frames:v`,
   `-fps_mode passthrough`); unsafe edit list and non-monotonic seek raise (stub
   `Mp4FrameIndex`), and `cut_clips` skips only that row; sidecar and resume; `_png_name` matches
   DLC's rule at 999/1000/1001 frames; `margin` respected; k-means choice on synthetic features;
   `add_context_frames` writes exactly the missing neighbours and leaves the CSV unchanged;
   `labeled_frames_table` on a fake `CollectedData.csv`. Subprocess wrappers by mocking
   `subprocess.run`.
2. **Round trip with ffmpeg** (CI has it): an MP4 whose frame *N* has brightness
   `(N mod 64) * 4` (adapt `write_mp4` in `tests/test_video_quality_qc.py`). Clip frame *k* is
   source frame `start + k` for every *k*; each PNG from `select_frames` and
   `add_context_frames` is the source frame its name says; `labeled_frames_table` gives the
   right `source_frame`. Variants: B-frames on; variable container frame rate (no `i / fps`);
   unsafe edit list (raises). Plus `event_frame_ranges` on a timing table from `simulate` in
   `tests/test_video_timing_qc.py` with drops, through `correct_video_timing`.
3. **Real sessions, once** (recorded here): `behavior_816212_2025-12-05_13-47-41` bottom
   (frames dropped), MP4 over HTTPS, clips at late licks with corrected times: clip frame 0
   equals the source frame decoded by count, pixel for pixel; by eye the tongue reaches the spout
   near clip centre. `behavior_800886_2025-08-18_13-14-52`: local MP4 and URL give identical
   clips. One small DLC project from these clips opens in DLC's labeling GUI, and after
   `add_context_frames` and our conversion, LP loads it with a context model.
4. black, isort, flake8, interrogate; 100% coverage of `video_clips` and the new alignment
   function.

### Results (2026-10-08)

- **1, 2, 4:** 21 tests in `tests/test_video_clips.py`, all passing with ffmpeg 8.1.1; 100%
  coverage of `video_clips.py` and of `event_frame_ranges`; black, isort, flake8 and
  interrogate clean on the new files. Round trips cover B-frames, a variable container frame
  rate and an unsafe edit list (a stream copy from 0.5 s); the drop test uses `simulate` with
  10 dropped exposures (a window across them has 10 fewer frames; the raw column puts a later
  event 10 frames late). `_png_name` matches DLC's rule at 999/1000/1001 frames.
- **3, `behavior_816212_2025-12-05_13-47-41` bottom** (MP4 over HTTPS, 187,024 exposures not
  saved; raw Harp time 374 s behind at the end): three clips at late lick-bout starts, ±0.1 s.
  Clip frames 0–3 were compared with source frames −3…+6, decoded by counting from the
  preceding keyframe with PyAV: every clip frame *k* matched source `start + k` best (mean
  |Δ| ≈ 1.1 grey levels, the CRF 18 re-encode; neighbours 1.5–3). Not pixel for pixel, since
  clips are re-encoded. Note for anyone repeating it: PyAV (libavformat) reports PTS with the
  edit list's `media_time` subtracted (64 ticks = 2 frames here), the raw index does not.
  By eye: in the clearest clip the mouth opens and the tongue meets the lower spout in the
  second half of the clip (about frames 64–88 of 96, 30–80 ms after the lick time), not
  exactly at the centre. The raw column would have put the clips 162,000 frames late.
- **3, not done:** `behavior_800886_2025-08-18_13-14-52` local-vs-URL (needs the 4.4 GB MP4
  locally); the DLC GUI and the LP context-model load (no DLC/LP install here).

## Changes

### From revision 8 (revision 9, 2026-10-08): implementation

- **Clips get even timestamps** (`-vf setpts=N/FRAME_RATE/TB`, the source's nominal rate, for
  playback only). Without it a variable-rate source gave repeated timestamps in the clip.
  Seeking still uses no frame rate.
- **`http_input_flags` is public** in `aind_video_utils.utils` (0.7.0); imported, not copied.
- **Frame counts of clips** (`select_frames`, `add_context_frames`) come from the clip's MP4
  index (`n_samples`), so neither needs a sidecar.
- **`clip` is empty (NA) for skipped rows**, not None: pandas 3 stores the column as strings.
  Read it as `clips["clip"]` (`clips.clip` is `DataFrame.clip`).
- **Labels CSVs**: both index layouts are read (DLC 2.3+ three columns; older DLC and LP one
  path); images listed in several CSVs count once. `add_context_frames` skips neighbours outside
  the clip rather than flooring at 0.
- k-means thumbnails are 40×30 gray (`KMEANS_SIZE`); clips are CRF 18 (`ENCODE_ARGS`).

### From revision 7 (2026-10-08)

- **Frame index is the key; `video_clips` takes no times.** No timing table input, no per-frame
  times in sidecars; behavior time is a join at lookup. Cameras whose timing QC excludes them can
  be clipped by frame (choosing their frames is then the caller's call).
- **Events → frames moves to `video_alignment.event_frame_ranges`**, with a before/after window.
- **Labeling follows DLC and LP:** DLC's PNG naming; `margin` so each labeled frame has
  neighbours; new `add_context_frames` for LP's context models, run after labeling (LP has no
  such function and silently falls back to the centre frame).
- `event_frame_ranges` documented as a general frame-window helper (any per-frame signal).
- **`labeled_frames_table` parses names only**; sidecars are just source video, start, count,
  versions.
- **Event picking leaves the library** (example notebook).

### From revision 6 (revision 7, 2026-10-07)

Removed QC from the module: it took the table `correct_video_timing` returns instead of a CSV and
trigger log, with only input validation left.

### From revision 5 (revision 6, 2026-10-06)

Frame times from `video_timing_qc` instead of the raw CSV column (revision 5 placed every event
after a drop on the wrong frame); `video-clips` extra with the `video-qc` pin; per-frame
container-timestamp check dropped (timing QC answers it); Phase 0 no longer blocks building the
module; file renamed; branch based on `main` at `5f11a6c`.

### Earlier revisions

- **5** (2026-09-25): library-side Python upgrade recorded as done.
- **4**: seeking through `aind-video-utils` (`read_mp4_frame_index`, `presentation_seconds`,
  `is_frame_addressing_safe`).
- **3**: no nominal frame rate anywhere.
- **2**: after an adversarial review: start index chosen before cutting, `cut_clip` in frames,
  one ffmpeg call per PNG, deterministic k-means, stems encode the start frame, sidecars written
  last and atomically, out-of-range events dropped.
