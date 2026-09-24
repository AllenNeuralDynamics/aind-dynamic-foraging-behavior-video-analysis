# Plan: `video_clips.py` — clipping behavioral-event video into labeled-data

> Status: pre-implementation. To be reviewed by a second model before any code is written.

## Context

Clip extraction for pose-training data currently lives in a Code Ocean re-encoding capsule
(`reencoding_utils.py` / `clip_and_extract.py`) whose main job — re-encoding — is now handled
elsewhere. The clipping half is worth keeping: it turns behavioral events (go cues, licks) into
short clips, then samples representative frames into DeepLabCut `labeled-data/` layout for
hand-labeling.

Three problems motivate a rewrite rather than a copy:

1. **No provenance.** Clips are cut with `-c copy`, whose start snaps back to the previous
   keyframe, and frames are named by index *within the clip*. So a hand-labeled `img014.png`
   cannot be mapped back to a behavior time — the link to trials, licks and kinematics is lost.
2. **A clock bug.** `process_behavior_video` hardcodes the old headerless 5-column acquisition
   CSV (`names=["Behav_Time", ...], header=None`) and takes `t_zero` from row 0. On new-format
   `metadata.csv` (header row, column named `ReferenceTime`) this silently produces wrong times.
   `video_alignment.py` in this repo already solves this properly.
3. **Dead and duplicated code.** `timestamp_strategy="lick"` calls `_compute_clip_timestamps_lick`,
   which was never written — that path raises `NameError`. Meanwhile `kinematics/video_clip_utils.py`
   holds two more clip cutters, and `kinematics_analysis` notebooks hold several more variants.

Outcome: **one** dependency-light module in this library, sibling to `video_alignment.py`, where a
labeled frame resolves back to an exact behavior time.

## Design

### Provenance is one integer

The acquisition CSV *is* a frame index — row `i` holds frame `i`'s `Behav_Time`. So if a clip
records the source frame index it started at, everything else is a lookup:

```
img00042.png  ->  clip frame 42
              ->  source frame  start_source_index + 42
              ->  behavior_time = video_csv[time_col].iloc[source_frame]
```

No fps arithmetic, no accumulated drift, no frame-level manifest to store. Three implementation
requirements make this hold, and each is easy to get wrong:

- **`-ss` before `-i` with a re-encode** (accurate seek), not `-c copy`.
- **`-fps_mode passthrough`** (`-vsync 0`). ffmpeg's default CFR behaviour duplicates or drops
  frames to hit a target rate, which breaks the 1:1 index mapping.
- **A frame-count guard**: compare `len(video_csv)` to `ffprobe`'s `nb_frames`. If they disagree,
  frames were dropped at encode time and the CSV-as-index assumption is invalid — record
  `frames_verified: false` rather than emit confident wrong times.

### One sidecar JSON per clip — no dataclasses, no run-level state

A *sidecar* is a companion file sharing a main file's name but not its extension, carrying
metadata about it (as `.xmp` does for camera RAW, or `.srt` for video). Here each clip gets a
`.json` beside it:

```
clips/
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_go_cue_001.mp4
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_go_cue_001.json
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_go_cue_002.mp4
  behavior_751004_2024-12-21_13-28-28_SideCameraLeft_go_cue_002.json
```

A single clip costs the same as a hundred — there is no run-level table that must exist, be found
or be kept in sync — and provenance travels with a clip that gets copied elsewhere. A directory of
them globs into a DataFrame in one line (`clip_table`) when batch work wants the table view.

Base keys, written by every clip (see "Two layers" below):

```json
{
  "clip_path": "video_349.033s_to_351.033s.mp4",
  "source_video": "/root/capsule/data/.../SideCameraLeft/video.mp4",
  "start_video_time": 349.033,
  "duration": 2.0,
  "start_source_index": 174516,
  "n_frames": 1000,
  "index_method": "fps",
  "frames_verified": true
}
```

Clips cut through the foraging layer add `source_csv`, `event_behavior_time`, `session_name` and
`camera_name`, and carry `"index_method": "csv"`.

### No cv2, no aind-video-utils; sklearn kept for k-means

`aind-video-utils` declares `requires-python = ">=3.10"` against this library's `>=3.9`, and the
bulk-decode path needed here has no API there anyway (`extract_frame_by_index` is one ffmpeg
subprocess *per frame* — fine for ~20 selected frames, unusable for a 2000-frame feature pass).
Skipped for v1.

Its `mp4_index` achieves frame accuracy by parsing the MP4 `moov` box's sample tables (`stts`
durations, `ctts` composition offsets, `stss` keyframes, `elst` edits) into a per-frame PTS array,
then seeking by a frame's *true* PTS rather than an assumed frame rate — reading only box headers
and the moov, never `mdat`, so it costs a few KB even on a 3 GB file. ffprobe cannot substitute
cheaply: `-show_frames` yields timestamps only by demuxing the whole file (~5M frames here), and
`-read_intervals` returns a timestamp rather than an index, since an index is a count from the
start.

**Not used in v1** — see the deferred upgrade below. The analysis capsule is pinned to Python 3.9
(`codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04`) and imports this library, so
`requires-python` stays `>=3.9` and there is no guarded import or second code path to test. v1
takes the simplest thing that works: `index_method` of `"csv"` or `"fps"`, and `frames_verified`
from the `len(csv)` vs `nb_frames` count check.

**Deferred upgrade (TODO for this package).** Three sources measure different things:

| Source | Knows |
|---|---|
| container (`mp4_index`, or `fps` arithmetic) | the *encoded* timeline — which frames are in the file, and when each is shown |
| acquisition CSV | the *acquired* timeline — when the camera grabbed each frame, on the harp clock |
| the bridge | encoded frame *i* == acquired frame *i* |

Every v1 path depends on that bridge, including the CSV `searchsorted`, and a re-encode is exactly
what can break it (hence `fail_on_frame_drop` / `normalize_cfr` in aind-video-utils' transcode).
Once this library can move to >=3.10, adopting `read_mp4_frame_index` buys two things: an exact
`start_source_index` with no CFR assumption, and a real bridge check comparing per-frame PTS
against CSV intervals — where the count check passes happily if frames were both dropped and
duplicated. Record this in `TODO.md`; it is blocked on the capsule image, which is due for a
rebuild anyway since Python 3.9 reached end-of-life in October 2025.

**cv2 is replaced by two ffmpeg passes.** This concerns *frame grabbing only* — the clip itself
stays an ordinary full-resolution h264 mp4, and nothing about Stage A (cutting) is downsampled.

```
clip_003.mp4 (2 s, full res)
      |  pass 1: decode ALL frames, downsampled -> choose ~20 indices
      |  pass 2: re-read THOSE frames at FULL res -> write PNGs
      v
labeled-data/clip_003/img00014.png, img00130.png, ...   (~20 PNGs, full res)
```

The PNGs are the end product for hand-labeling — the DLC `labeled-data/` convention, lossless so
no compression artifacts appear under a click-to-label workflow.

- *Pass 1 (decide)* — one sequential decode of the whole clip, downsampled:
  `ffmpeg -i clip.mp4 -vf scale=W:H,format=gray -f rawvideo -pix_fmt gray -`
  then `np.frombuffer(...).reshape(-1, h, w)`. Same idiom as `_rawvideo.py` in aind-video-utils.
  Downsampling is not new — the current code already does it via `kmeans_resize_width=50` and
  `cv2.resize`; this just moves the resize into the decoder, which is cheaper than decoding full
  res and shrinking afterwards. Clustering only needs "how different do these frames look":
  50 px-wide gray is 3.8 MB and 1900 dims for a 2000-frame clip, against ~2.3 GB and 1.1M dims at
  full res. Pass **both** scale dimensions explicitly, computed from ffprobe, rather than
  `scale=W:-1` — otherwise the reshape depends on guessing ffmpeg's rounding.
- *Pass 2 (save)* — one ffmpeg call using the `select` filter with the chosen indices, writing
  full-resolution PNGs directly. The image2 muxer numbers outputs sequentially (1..N) rather than
  by source index, so rename to `img{source_index:05d}.png` afterwards; `select` preserves order,
  so the mapping is positional.

`uniform` and `random` skip pass 1 entirely — they compute indices arithmetically.

This also fixes a performance bug, though the bug is not inherent to cv2. `kmeans_frame_selection`
currently calls `cap.set(CAP_PROP_POS_FRAMES, n)` before each `read()`, walking the clip *by
seeking*. h264 frames are inter-compressed, so each seek makes the decoder return to the previous
keyframe and decode forward — roughly O(n^2) work instead of O(n) (with a 250-frame GOP, ~250k
frame-decodes rather than 2k). Severity depends on GOP length and whether the OpenCV build caches
sequential seeks. Deleting the `cap.set()` line alone would fix it, since `read()` already
advances sequentially; the ffmpeg move is about dropping the dependency.

**sklearn's `MiniBatchKMeans` is kept**, unchanged from the current code — hand-rolling Lloyd's is
a maintenance liability for no real gain at this scale, and selected frames stay reproducible
against past runs. Pin `random_state` and pass `n_init` explicitly rather than relying on the
default, which has changed across sklearn releases.

This means declaring dependencies honestly: the library currently says `dependencies = []` while
`video_alignment.py` already imports pandas at module scope. Fix that in the same pass —
`numpy`, `pandas`, `scikit-learn` — and import normally rather than adding lazy-import machinery
to keep sklearn nominally optional, which would be more code, not less.

### DLC output contract (verified against real data)

Checked `/Users/mib/Downloads/labeled-data/`: the contract is `CollectedData.csv` column 0 holding
a literal relative path `<clip_stem>/imgNNN.png`. Filename width is **not** fixed — DLC's own
`extract_frames` derives it from the source frame count. What matters, and is already correct in
the old code: one folder per clip, and indices are *sparse clip-local frame numbers*
(`img014.png`, `img130.png`), not `000..N`.

Change `:03d` -> `:05d` (the old width silently overflows past 999 frames, which a 2 s clip at
these rates clears easily). Existing 3-digit folders stay valid in the same `CollectedData.csv`,
since each path is stored literally.

## Module

One new top-level module, sibling to `video_alignment.py` — not inside `kinematics/`. That
module's docstring states it is "decoupled from the kinematics pipeline so it can be reused on its
own"; clipping around behavioral events has the same property and is not kinematics-specific.

```
src/aind_dynamic_foraging_behavior_video_analysis/
  video_alignment.py        (exists — becomes a dependency)
  video_clips.py            NEW
```

~500 lines, unremarkable here (`mp4_index.py` is 710, `encoding.py` is 811). Frame selection lives
in the same file: with cv2 and sklearn both gone there is no dependency-weight argument for a
second module.

### Two layers: the primitive needs no CSV

`start_source_index` is an index into the *source video* and is derivable from the video alone.
The acquisition CSV is only needed for the last hop, source frame -> behavior time. So the plain
primitive takes a video, a start and a duration — no session, camera or behavior clock — and
behavior-time provenance is an optional upgrade layered on top. A caller outside this project can
cut a clip and still trace a frame back into the original video.

```python
clip = video_clips.cut_clip(video_path, start_video_time=349.033, duration=2.0, out_dir=out)
video_clips.frame_source_index(clip, 42)    # -> 174558, frame index in the source video

clips = video_clips.cut_clips_at_events(     # foraging layer: adds csv_path + behavior clock
    video_path, csv_path, event_behavior_times=go_cues, duration=2.0, out_dir=out)
video_clips.frame_behavior_time(clips[0], 42)   # -> 1546.887, same clock as goCue_start_time
```

**Clock discipline.** `cut_clip` takes `video_time` (it has to — with no CSV the file knows no
other clock); `cut_clips_at_events` takes `behavior_time` and converts internally via
`video_alignment.behavior_time_to_video_time()`. Both clocks are bare floats in the same plausible
range, so passing a go cue time to `cut_clip` would silently cut from the wrong place. Parameters
are therefore named `start_video_time` and `event_behavior_times`, never `start` or `times`, so
the clock is visible at every call site. Events on the *session* clock convert with the existing
`compute_video_session_offset` / `session_time_to_video_time` rather than a `clock=` parameter.

**How the index is derived.** The seek is ffmpeg's, unchanged: `-ss` before `-i` (container seek,
so it stays fast on an 84-minute source) with a re-encode, which makes ffmpeg decode forward from
the preceding keyframe and discard frames until the first whose PTS is at or after t. (*PTS*,
presentation timestamp, is the per-frame "when should this be displayed" value stored in the
container. A file carries it rather than deriving `frame / fps` because frame rate is not always
constant, and because h264 B-frames are stored out of display order.) What ffmpeg does *not*
report is which source frame number it landed on, and that number is what makes a labeled frame
traceable. `index_method` records how it was computed:

- `"fps"` — `ceil(start_video_time * fps)`. ffmpeg lands on the first frame at or after t, and for
  CFR frame *n* has PTS *n/fps*. Exact for a true CFR file, but an assumption.
- `"csv"` — `np.searchsorted(csv_times, target, side="left")` on the acquisition table. No fps
  assumption, so it survives jitter and non-uniform timelines. (aind-video-utils'
  `extract_frame_by_index` refuses to assume a frame rate for this reason, citing a "seam-glitch
  timeline" on these sources.)

They agree on a clean CFR file and diverge otherwise, so recording which was used tells a consumer
whether the index rests on an assumption. Note the deeper caveat: CSV row *i* is the *i*-th
**acquired** frame while the index needed is the *i*-th **encoded** frame — equal only if every
acquired frame reached the file, which is exactly what `frames_verified` (`len(csv)` vs
`nb_frames`) tests. When that fails neither method is trustworthy and the sidecar says so.

The index is also *checkable*: decode clip frame 0 and source frame `start_source_index` and
compare pixels. A few hundred ms, offered as an opt-in `verify=True` on `cut_clip`, and the same
check as the round-trip test under Verification.

| Function | Layer | Role |
|---|---|---|
| `cut_clip(video_path, start_video_time, duration, out_dir, stem=None)` | plain | **the primitive** — one clip + sidecar, returns `Path` |
| `frame_source_index(clip_path, frame_index)` | plain | clip frame -> source frame |
| `read_clip_info(clip_path)` | plain | load the sidecar dict |
| `clip_table(clip_dir)` | plain | glob sidecars -> `DataFrame` |
| `select_event_times(strategy, **kw)` | foraging | registry dispatch -> sorted `np.ndarray` of behavior times |
| `cut_clips_at_events(video_path, csv_path, event_behavior_times, ...)` | foraging | loops `cut_clip`, adds behavior-clock keys |
| `frame_behavior_time(clip_path, frame_index)` | foraging | the full provenance lookup |
| `select_frames(clip_path, num_frames, out_dir, algorithm=, resize_width=)` | either | `uniform`/`kmeans`/`random` -> `labeled-data/<stem>/img#####.png` |
| `labeled_frames_table(labeled_data_dir)` | either | join `CollectedData.csv` paths -> times via sidecars |

`select_frames` reads the sidecar itself, so it behaves identically however the clip was cut.

`labeled_frames_table` is the payoff and is ~15 lines: it is what makes hand-labeled keypoints
joinable to trials, licks and kinematics.

## Phases

**1. Event selection.** Port `get_first_licks_per_trial`, `select_evenly_spaced_events`,
`get_evenly_spaced_times`, `_compute_clip_timestamps_go_cue` as pure array-in/array-out functions
(no NWB — see "Out of scope"). Register strategies in a dict so dispatch is extensible.

Beyond a straight port:
- **Write the missing lick strategy**: N evenly-spaced licks from each of the left/right streams,
  plus M evenly-spaced first-lick-per-trial events from each side, plus K session-coverage times.
- **Unify the two sampling idioms** — `select_evenly_spaced_events` uses `int(len*i/(n+1))`
  (never hits endpoints, duplicates when `n >= len`); `_compute_clip_timestamps_go_cue` uses
  `np.linspace`. Standardize on linspace.
- **Deterministic session coverage** — replace `rng.uniform` with evenly-spaced by default
  (reproducible training sets), keeping `rng_seed` opt-in.

**2. Cutting + provenance.** Split a pure `build_clip_command()` from a thin subprocess runner; the
pure half is what gets unit-tested (see Verification). Reuse
`video_alignment.get_first_frame_behavior_time()` and `behavior_time_to_video_time()` instead of
the hand-rolled `t_zero` — this is the bug fix from Context item 2.

Also add **bounds clamping** (`t - clip_length/2` currently goes negative for early events and past
EOF for late ones, and ffmpeg emits a short clip in silence) and **overlap merging** via a
`min_separation` parameter (two events 0.5 s apart with `clip_length=2` currently produce
near-duplicate clips, wasting labeling effort and skewing the training set).

Clip stem: `{session}_{camera}_{strategy}_{NNN}` — all times live in the sidecar, not the name.
Re-encode settings: `-crf 15 -preset fast` with explicit color-metadata passthrough
(near-visually-lossless; these clips exist only to source frames for labeling).

**3. Frame selection.** `uniform` / `random` / `kmeans` over the two-ffmpeg-pass design above, plus
the numpy k-means. Output layout unchanged except the `:05d` width. Add `labeled_frames_table`.

**4. Wire up and shim.** `kinematics/video_clip_utils.py` becomes a deprecation shim delegating to
the new API. Known callers: `kinematics/tongue_analysis.py` (this repo), `lickometer_qc.py` and
several notebooks in `kinematics_analysis`, `reencoding_utils.py`, and
`examples/video_alignment_example.ipynb`.

Reuse the asset discovery already in `kinematics/tongue_kinematics_utils.py`:
`find_behavior_videos_folder`, `find_video_path`, `find_video_csv_path`,
`get_session_name_from_path`. Note the old `process_behavior_video` resolved `session_name` from
`input_folder` and then rebuilt `/root/capsule/data/<session_name>`, ignoring the input path — the
new code takes resolved paths, which is also what makes it testable off Code Ocean.

**5. Tests, example, docs.** See Verification. Example notebook mirroring
`examples/video_alignment_example.ipynb`. Add the deferred `mp4_index` upgrade to `TODO.md`,
noting it is blocked on this library moving to Python >=3.10, which is in turn blocked on the
capsule image.

## Out of scope / dropped

- **NWB coupling stays capsule-side.** The library takes plain arrays (`go_cue_times`,
  `left_licks`, `right_licks`); a thin capsule-side adapter pulls them out of NWB.
- **Dropped as re-encoding-era, now handled elsewhere**: `run_aind_behavior_video_transformation`,
  `copy_nonvideo_and_metadata_files`, `copy_if_exists`, `is_video_file`, `find_top_level_folders`.
- **Dropped as redundant**: `process_behavior_video_dry_run` (planning is now separable from
  cutting, so a dry run is free) and `extract_frames_only_from_existing_clips` (resume falls out
  of "sidecar exists -> skip").
- **moviepy keypoint overlays** (`create_labeled_video`, `make_cmap`, `process_and_label_clips`)
  stay in `video_clip_utils.py` for now — separate concern, heaviest dependency, and `clip.fl` is
  moviepy v1 API that no longer exists in v2. Worth its own pass later.

## Constraints

`pyproject.toml` enforces `coverage fail_under = 100` and `interrogate fail-under = 100`, so tests
and docstrings are mandatory, not cleanup. Target py39 (no `X | Y` annotations), black line length
79, isort, NumPy docstrings. `video_clip_utils.py` imports cv2, moviepy, matplotlib and seaborn at
module scope against an empty `dependencies` list — `video_clips.py` must not repeat that. Its
imports are stdlib, numpy, pandas, sklearn and `video_alignment`, and `pyproject.toml` gains
`numpy`, `pandas`, `scikit-learn` to match reality. `requires-python` stays `>=3.9` so the capsule
can still install the library.

## Verification

1. **Unit tests, no ffmpeg needed** — `build_clip_command()` is pure, so cut-mode flags, clamping
   and `min_separation` merging are tested directly. Event strategies are pure array-in/array-out.
   Frame-index selection is tested by feeding synthetic feature arrays straight to the clustering
   step, bypassing decode. This is what satisfies the 100% coverage gate without a real video.
2. **Round-trip provenance test** (the one that matters) — synthesize a short video with
   `ffmpeg -f lavfi -i testsrc` plus a matching fake acquisition CSV, cut a clip at a known event
   time, then assert `frame_behavior_time(clip, k)` equals the CSV's time for source frame
   `start_source_index + k`, for several `k`. Mark it to skip cleanly where ffmpeg is absent.
3. **Both CSV layouts** — assert identical results for a headerless 5-column `bottom_camera.csv`
   and a headered `metadata.csv` with `ReferenceTime`, which is the Context item 2 regression.
4. **Guard test** — a CSV whose row count disagrees with `nb_frames` must yield
   `frames_verified: false`.
5. **End-to-end on one real session** — run against a session from `clip_and_extract.py`'s list,
   confirm `labeled-data/<stem>/img#####.png` appears with sparse indices, then check
   `labeled_frames_table` returns behavior times landing within `clip_length/2` of the requested
   go cues.
6. Lint and coverage: black, isort, flake8, interrogate, coverage.
