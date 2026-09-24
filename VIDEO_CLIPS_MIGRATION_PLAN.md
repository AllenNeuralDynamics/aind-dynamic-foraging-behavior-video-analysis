# Plan: `video_clips.py` — clips around behavioral events, frames for labeling

> Status: pre-implementation, revision 4. Revision 2 followed an adversarial review; revision 3
> removed every use of a nominal frame rate; revision 4 hands frame-accurate seeking to
> `aind-video-utils` and makes a Python upgrade a prerequisite. See "Changes" at the end.

## Summary

Move the clip-and-extract pipeline out of the Code Ocean re-encoding capsule into one new module,
`video_clips.py`, next to `video_alignment.py`. It does three things:

1. **Pick event times**: go cues and licks, as plain arrays, on the behavior (harp) clock.
2. **Cut clips**: one short mp4 per event, plus a small JSON *sidecar* recording which source
   frame the clip starts at.
3. **Extract frames**: sample about 20 frames per clip into DeepLabCut `labeled-data/` layout.

The point of the rewrite is **provenance**: a hand-labeled `labeled-data/<stem>/img00042.png` must
resolve to an exact source frame, and from there to a behavior time. The current code can't do
this.

## Prerequisite: Python upgrade (Phase 0)

Frame-accurate seeking comes from `aind-video-utils`, which declares `requires-python >= 3.10`.
Its code does import and run on 3.9 today (checked against 0.7.0), but nothing guarantees that,
and installing it on 3.9 needs `--ignore-requires-python`. So the upgrade comes first:

1. **Move the capsule image to Python 3.11 or 3.12.** Not 3.10, which reaches end of life in
   October 2026. 3.9 has been end of life since October 2025.
2. **Move every other consumer that installs this library.** Consumers install from `main` with
   no version pin (README "Scope"), so once `requires-python` is raised, a 3.9 environment's
   `pip install` fails outright instead of falling back to an older version. Known consumers: the
   clip/re-encoding capsule and the analysis repos built on the kinematics intermediates (e.g.
   `kinematics_analysis`). Audit for others before raising the floor.
3. **Then raise the floor here**: `requires-python = ">=3.11"`, black `target_version = ['py311']`,
   and the README badge (which already says `>=3.10` and disagrees with the pyproject).
4. **Recommended alongside: start pinning consumers to tags** (`@v0.x`), so future breaking changes
   are opt-in for consumers rather than a surprise on rebuild.

The code itself needs no changes for the upgrade: it uses no 3.9-specific workarounds, and its
AIND dependencies (`aind-dynamic-foraging-data-utils`, `-basic-analysis`) declare `>=3.9`.

**If work should start before the upgrade:** everything except the seek (event selection,
sidecars, frame selection, `labeled_frames_table`) is independent of `aind-video-utils` and can be
built first, with the seek isolated in `cut_clip`.

## Why rewrite instead of copy

| Problem in the capsule code | Effect |
|---|---|
| Clips cut with `-c copy`, which starts at the previous keyframe | The clip's start frame is unknown, so labeled frames can't be traced back |
| Acquisition CSV read with hardcoded headerless column names | Wrong times on new-format `metadata.csv`. `video_alignment.read_video_csv` already handles both layouts |
| `timestamp_strategy="lick"` calls a function that was never written | That path raises `NameError` |
| `cap.set(POS_FRAMES)` before every `read()` in k-means | Re-decodes from a keyframe for every frame, roughly O(n²) |
| k-means representative chosen with unseeded `random.choice` | Frame selection isn't reproducible |
| PNGs named `img{:03d}` | The name overflows past frame 999 |

## Design

### Provenance: choose the frame index first, then seek to it

The acquisition CSV is a frame table: row `i` holds frame `i`'s behavior time. So a clip only
needs to record the source frame it starts at:

```
labeled-data/<stem>/img00042.png
  -> clip frame 42
  -> source frame   start_source_index + 42
  -> behavior time  csv_times[start_source_index + 42]
```

**No nominal frame rate is used anywhere.** Two tables stand in for it:

- **The CSV** maps frame index to behavior time. It answers "which frames cover this event?"
- **The container's per-frame index** (`aind_video_utils.read_mp4_frame_index`) maps frame index
  to the frame's real presentation timestamp. It answers "where do I seek to reach frame `i`?"

Neither table is converted into the other with `i / fps`. AIND videos can have irregular
timestamps (the `aind-video-utils` docs describe PTS glitches at concatenation seams), so an
`i / fps` seek would land on the wrong frame on some real sessions.

#### The frame index: `aind-video-utils`

`read_mp4_frame_index(video_path)` parses the mp4's `moov` sample tables directly (pure
`struct` + numpy, no decoding; it reads a few MB even for multi-GB files, and works over HTTP with
range requests). It returns an `Mp4FrameIndex` with each frame's presentation timestamp,
keyframes and edit list. The parts used here:

| `Mp4FrameIndex` member | Used for |
|---|---|
| `presentation_seconds(i)` | The exact input-side `-ss` for presentation-order frame `i`. Subtracts the edit list's `media_time`, so files that don't start at 0 are handled. Raises if frame `i`'s timestamp isn't strictly greater than its predecessor's (a seek there would be ambiguous) |
| `is_frame_addressing_safe()` | Whether the edit list can be ignored for frame addressing. `False` for empty edits, multi-entry lists, non-unit rates, or trims that drop frames |
| `n_samples` | The exact frame count, for `frames_verified` |
| `pts` | The constant-frame-rate check: `constant_frame_rate` is whether `np.diff(np.sort(pts))` is all one value. Recorded, not relied on |

The index is read **once per source video** in `cut_clips_at_events` and passed to each
`cut_clip`. If `is_frame_addressing_safe()` is `False`, `cut_clips_at_events` raises before
cutting anything, naming the file: a clip from such a file couldn't be traced back reliably.

Checked on a synthetic variable-frame-rate video with timestamp jumps (ffmpeg 8.1,
`aind-video-utils` 0.7.0): `extract_frame_by_index` matched a frame-count decode
(`select=eq(n,i)`) on all 11 frames tested, including frames right at the jumps. The revision 2
seek, `(i - 0.5) / fps`, got 3 of the 11 wrong.

#### From event and duration to frames

Users ask for clips in seconds. Seconds are converted to frames through the **CSV's own
timestamps**, so the conversion uses the camera's real frame times, not a nominal rate:

```python
start_source_index = searchsorted(csv_times, event_behavior_time - duration / 2)
end_source_index   = searchsorted(csv_times, event_behavior_time + duration / 2)
n_frames           = end_source_index - start_source_index
```

The clip covers behavior times `[event - duration/2, event + duration/2)`. If the camera dropped
frames in that window, the clip is a few frames shorter, which is correct: it still spans the
requested time. `n_frames` is recorded in the sidecar.

#### From frame index to clip

The index is **decided before cutting**, never derived from a time afterwards:

```python
seek = index.presentation_seconds(start_source_index)
ffmpeg -accurate_seek -ss f"{seek:.9f}" -i source.mp4 \
       -frames:v <n_frames> -fps_mode passthrough -c:v libx264 ...
```

This is the same seek `aind_video_utils.extract_frame_by_index` uses for single frames, with a
frame count and an encoder added. The library has no clip-cutting function, so the ffmpeg call
stays here.

Why each part matters:

- **`-ss` before `-i` with a re-encode** makes ffmpeg decode from the previous keyframe and output
  the first frame whose timestamp is at or after the seek. Seeking to the frame's exact timestamp
  lands on it: ffmpeg truncates `-ss` to microseconds, which only ever moves the seek earlier, and
  the previous frame is milliseconds earlier still.
- **The index comes from the CSV, the seek from the container.** Converting harp time to video time
  with `(t - t0)` drifts by about 1 s over a session if the camera's real rate differs from
  nominal by 0.02%.
- **`-frames:v n_frames`** instead of `-t duration`. `-t` gave 999 or 1000 frames depending on
  where the start fell; `-frames:v` is exact.
- **`-fps_mode passthrough`** stops ffmpeg from duplicating or dropping frames.

An event whose start frame makes `presentation_seconds` raise (non-monotonic timestamps at that
frame) is skipped with a logged warning; the rest of the video is still cut.

**Remaining assumption, stated in the sidecar:**

- *Encoded frame `i` is acquired frame `i`*: every acquired frame reached the file. This is what
  the CSV lookup relies on. `frames_verified` records whether `len(csv) == index.n_samples`. A
  matching count can still hide one dropped plus one duplicated frame; a per-frame check is
  deferred (see "Out of scope").

### Clip names and sidecars

Clip stems encode the start frame, so a name is unique and stable by construction:

```
clips/
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_f0174266.mp4
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_f0174266.json
labeled-data/
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_f0174266/
    img00014.png  img00130.png  ...
```

Naming by frame rather than by a counter (`_001`) means re-running with different event
parameters never renumbers or overwrites an unrelated clip, and an event picked by two strategies
produces one clip.

Sidecar (one per clip):

```json
{
  "source_video": ".../SideCameraLeft/video.mp4",
  "source_csv": ".../SideCameraLeft/metadata.csv",
  "start_source_index": 174266,
  "n_frames": 1000,
  "frames_verified": true,
  "constant_frame_rate": true,
  "event_behavior_time": 3805.642,
  "requested_duration": 2.0,
  "session_name": "behavior_751004_2024-12-21_13-28-28",
  "camera_name": "SideCameraLeft",
  "strategy": "go_cue",
  "aind_video_utils_version": "0.7.0"
}
```

Rules that keep sidecars from going stale:

- **Written last**, to a temp file and then renamed, only after ffmpeg succeeds. A crashed cut
  leaves no sidecar, so it's retried next run.
- **Resume**: a clip is skipped only if its sidecar exists *and* has the same `source_video` and
  `n_frames`. Otherwise it's re-cut.
- **No `clip_path` key.** The sidecar is found by stem, so a key pointing at itself would only go
  stale on rename.
- Concurrent runs writing to the same `clips/` folder are not supported.

### Frame selection

Frames are named by **clip-local** index (`img00042.png` = clip frame 42), matching DLC's
convention of sparse frame numbers within one video folder.

| Algorithm | How indices are chosen |
|---|---|
| `uniform` | `np.linspace` over the clip |
| `random` | seeded `np.random.default_rng(seed)` |
| `kmeans` | one ffmpeg decode of the whole clip, downscaled to gray (`scale=W:H,format=gray -f rawvideo`, both dimensions passed explicitly). Then `MiniBatchKMeans(random_state=seed, n_init=3)`. Each cluster is represented by the member **nearest its centroid**, so the choice is deterministic |

Writing the PNGs: **one ffmpeg call per chosen frame**
(`-vf select=eq(n\,K) -frames:v 1 -fps_mode passthrough imgKKKKK.png`). This counts frames, so it
needs no timestamps, and decoding from the start of a ~1000-frame clip about 20 times is fast.
Each file is named directly, so there's no positional renaming step to get wrong.
(`extract_frame_by_index` would also work, but returns an array and would need an image writer.)

This replaces cv2. sklearn is kept for k-means; hand-writing it isn't worth it.

### Dependencies

`pyproject.toml` currently declares `dependencies = []`, although `video_alignment.py` already
imports pandas. Add `numpy`, `pandas`, `scikit-learn`, `aind-video-utils>=0.7` (core only, no
extras; its only required dependency is numpy).

ffmpeg and ffprobe are system binaries, found on `PATH`.

## Module API

One module, `src/aind_dynamic_foraging_behavior_video_analysis/video_clips.py`, roughly 250 lines.

**Event selection** (pure, array in, array out, behavior clock):

| Function | Returns |
|---|---|
| `evenly_spaced(values, n)` | `n` values picked with `np.linspace` over the sorted input (all of them if `n >= len`) |
| `first_licks_per_trial(go_cue_times, left_licks, right_licks)` | `(first_left, first_right)` |
| `go_cue_event_times(go_cue_times, n_go_cues, n_session)` | sorted event times |
| `lick_event_times(go_cue_times, left_licks, right_licks, n_licks, n_first_licks, n_session)` | sorted event times |

`n_session` adds evenly spaced times between the first and last go cue, for coverage. This is
deterministic; the old random sampling is dropped.

**Cutting:**

| Function | Role |
|---|---|
| `cut_clips_at_events(video_path, csv_path, event_behavior_times, duration, out_dir, session_name, camera_name, strategy)` | **The user-facing call.** `duration` is in seconds (the whole clip, centred on each event). Reads the frame index once and checks it's frame-addressing-safe, converts each event and duration to a start index and frame count via the CSV's timestamps, drops events whose clip would run past either end of the CSV or video, cuts and writes sidecars. Returns the clip paths |
| `cut_clip(video_path, start_source_index, n_frames, out_path, index=None)` | The primitive, for callers who already think in frames. Takes a frame index and count, not times, so there's no clock to mix up. Reads the frame index if `index` isn't passed. Returns the path |

A notebook call looks like:

```python
clips = cut_clips_at_events(
    video_path, csv_path, go_cue_event_times(go_cues, 20, 10),
    duration=2.0, out_dir=clips_dir, session_name=..., camera_name=..., strategy="go_cue",
)
```

**Frames and lookup:**

| Function | Role |
|---|---|
| `select_frames(clip_path, num_frames, out_dir, algorithm="uniform", resize_width=50, seed=0)` | Writes `out_dir/img#####.png`, returns the clip-local indices |
| `read_clip_info(clip_path)` | Loads the sidecar |
| `labeled_frames_table(labeled_data_dir, clips_dir)` | Reads `CollectedData*.csv`, joins each image path to its sidecar, returns a DataFrame with `source_frame` and `behavior_time` per labeled image |

`labeled_frames_table` is what makes hand labels joinable to trials, licks and kinematics.

Private helpers:

- `_event_frame_range(csv_times, event_behavior_time, duration)`: `(start_source_index, n_frames)`
  from the CSV. Pure, so it's tested directly.
- The ffmpeg command builders (cut, gray decode, PNG), kept separate from `subprocess.run` so they
  can be tested without ffmpeg. Width and height for the gray decode come from
  `aind_video_utils.probe` / `get_frame_dimensions`.

## Phases

0. **Python upgrade** (see "Prerequisite" above).
1. **Dependencies and event selection.** Update `pyproject.toml`. Port the event functions as pure
   functions and write the missing lick strategy.
2. **Cutting and sidecars.** `_event_frame_range`, `cut_clip`, `cut_clips_at_events`, sidecar
   write and resume. The CSV is read with `video_alignment.read_video_csv`, which fixes the
   column-name bug.
3. **Frame selection.** `select_frames` with the three algorithms.
4. **Lookup, example and TODO.** `read_clip_info`, `labeled_frames_table`, an example notebook
   modelled on `examples/video_alignment_example.ipynb`, and a `TODO.md` entry for the deferred
   check below.

## Out of scope

- **`kinematics/video_clip_utils.py` stays as it is.** `tongue_analysis.py` uses its
  `extract_trial_clip` (10 s trial clips, a different job), and moviepy overlays live there too.
  Callers move to the new module when convenient; no deprecation shim.
- **NWB loading and path discovery stay in the capsule.** The library takes resolved paths and
  plain arrays. The capsule can reuse `find_video_path` / `find_video_csv_path` from
  `tongue_kinematics_utils.py`.
- **Re-encoding-era helpers are dropped:** `run_aind_behavior_video_transformation`,
  `copy_nonvideo_and_metadata_files`, `copy_if_exists`, `is_video_file`, `find_top_level_folders`.
- **Also dropped:** `process_behavior_video_dry_run` (compute event times without cutting instead)
  and `extract_frames_only_from_existing_clips` (loop `select_frames` over existing clips).
- **Merging overlapping clips** (`min_separation`) is deferred. Stable clip names already remove
  exact duplicates.
- **Deferred TODO: a per-frame check.** Compare the frame index's `pts` intervals against the CSV's
  frame intervals to catch a dropped-plus-duplicated pair that a count can't. The data is already
  in hand once `aind-video-utils` is a dependency. It is deferred because it only helps if the
  recording software stamps frames with real acquisition times. If it stamps `i / fps` regardless
  (likely for Bonsai writing through an ffmpeg pipe), the container intervals are constant and
  carry no information. Check which it is on real data first (Verification step 4).

## Constraints

- Python 3.11+ after Phase 0.
- black and isort at line length 79, flake8, NumPy-style docstrings.
- `interrogate` and `coverage` both at `fail-under = 100`. There is no CI workflow in the repo, so
  these are enforced locally.

## Verification

1. **Unit tests, no ffmpeg.** Event functions; `_event_frame_range` (window edges, a CSV with a
   gap giving a shorter clip); ffmpeg command builders (seek value from `presentation_seconds`
   formatted to 9 decimals, `-accurate_seek`, `-frames:v`, flags); `cut_clips_at_events` raising
   on an index that isn't frame-addressing-safe and skipping an event whose
   `presentation_seconds` raises (both with a stub `Mp4FrameIndex`); bounds dropping; sidecar
   resume rules; k-means index choice on synthetic feature arrays; `labeled_frames_table` on a
   fake `CollectedData.csv` plus sidecars. The subprocess wrappers are covered by mocking
   `subprocess.run`, and `read_mp4_frame_index` by a stub. This is what reaches 100% coverage.
2. **Round trip with ffmpeg** (skipped if ffmpeg isn't installed). Synthesize a video whose frame
   `N` has brightness `(N mod 64) * 4`, plus a matching fake CSV. Cut at several events, including
   times computed as `(t0 + k/fps) - t0`, and assert clip frame `k` has source frame `start + k`'s
   brightness for every `k`, and `n_frames` is exact. Run it on three variants:
   - constant 500 fps, B-frames on;
   - **variable frame rate** (timestamp jumps written with `setpts` and `-fps_mode vfr`), where any
     `i / fps` seek lands on the wrong frame. This is what proves no nominal rate sneaks back in;
   - a file with an unsafe edit list (e.g. written with `-output_ts_offset`, which adds an empty
     edit): `cut_clips_at_events` must raise, not cut.
3. **Both CSV layouts** give identical sidecars (headerless `bottom_camera.csv` and headered
   `metadata.csv` with `ReferenceTime`).
4. **One real session, once.** Check `is_frame_addressing_safe()` on the real file. Cut a clip,
   then decode source frame `start_source_index` directly (`select=eq(n\,idx)` on the source,
   which counts frames and needs no timestamps) and compare it to clip frame 0 pixel by pixel.
   Also record whether the file is constant-rate and whether its `pts` intervals track the CSV's
   (this decides whether the deferred per-frame check is worth building), and confirm
   `labeled_frames_table` behavior times land within `duration / 2` of the requested events.
   Ideally repeat on a session known to have concatenation seams.
5. **Guard test.** A CSV one row short gives `frames_verified: false`.
6. black, isort, flake8, interrogate, coverage.

## Changes from revision 3

- **Seeking uses `aind-video-utils`** (`read_mp4_frame_index`, `presentation_seconds`,
  `is_frame_addressing_safe`) instead of a hand-rolled ffprobe packet table and midpoint seek. The
  library handles edit lists properly and refuses files where frame addressing isn't safe.
  `_probe`, `_frame_pts` and `_seek_time` are removed.
- **Phase 0, a Python upgrade to 3.11+,** is added as a prerequisite, because `aind-video-utils`
  requires 3.10+, and consumers install this library from `main` unpinned.
- `cut_clips_at_events` raises on an unsafe file; an event at a non-monotonic timestamp is skipped
  with a warning.
- Sidecar adds `aind_video_utils_version`.
- The deferred per-frame check no longer needs any new code to get the data.

## Changes from revision 2

- No nominal frame rate anywhere: the seek uses the container's per-frame timestamps instead of
  `(i - 0.5) / fps`.
- `duration` (seconds) stays the user-facing input. It is converted to a start index and frame
  count through the CSV's timestamps, not `duration * fps`. `cut_clip` still takes frames.
- `frames_verified` compares against the container's frame count instead of `nb_frames`.
- Sidecar: `fps` removed; `constant_frame_rate` and `requested_duration` added.
- Verification adds a variable-frame-rate round trip.

## Changes from revision 1

- The start index is chosen before cutting (from the CSV) and the seek is aimed at it. This
  replaces the `"fps"` method, which had a float off-by-one, and the `"csv"` method, which compared
  harp time against container time. `index_method` is removed.
- `cut_clip` takes a frame index and a frame count instead of a video time and a duration.
- PNGs are named by clip-local index (revision 1 said both clip-local and source index in
  different places).
- One ffmpeg call per saved frame instead of a `select` pass plus positional renaming.
- k-means picks the member nearest each centroid (the old code picked one at random, unseeded).
- Clip stems encode the start frame; sidecars are written last, atomically, with a resume check,
  and have no `clip_path` key.
- Out-of-range events are dropped rather than clamped. `min_separation`, the strategy registry,
  `verify=True`, `clip_table`, `frame_source_index`, `frame_behavior_time` and the
  `video_clip_utils` shim are removed.
- Session-coverage times are deterministic only; `rng_seed` is removed.
- Coverage comes from mocked subprocess tests, not skipped integration tests.
