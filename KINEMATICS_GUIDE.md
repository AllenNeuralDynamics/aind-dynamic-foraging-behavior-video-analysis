# Guide: using the kinematics core

How to go from tracked keypoints to tables of movements, and how to describe what you want in
settings. Start with the picture, then copy the recipe closest to what you need. Every option is
listed in the reference at the end. Why the design is the way it is: `KINEMATICS_PLAN.md`.

```python
import dataclasses
import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
    features, intervals, landmarks, pipeline as pl, pose_io, segment, signal,
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tasks import (
    dynamic_foraging_tongue as dft,
)
```

## The picture

You hand `pl.process` two things: a **pose table** (where each keypoint was in each frame, and
when) and a **settings object** (what to do with each keypoint). It hands back a dictionary of
tables.

```
pose  ──┐
        ├──> pl.process(pose, settings) ──> {"kinematics": ..., "movements": ..., "record": ...}
settings┘
```

Settings are nested, one level per question:

```
pl.Settings                     the whole job
├── fs_hz                       frame rate of the video
├── landmarks                   keypoints used only as fixed reference points (e.g. the jaw)
└── keypoints: [ ... ]          one entry per keypoint you want processed
     └── pl.KeypointSettings    how to process ONE keypoint
          ├── name              which keypoint
          ├── min_likelihood    below this tracker confidence, a point counts as missing
          ├── smooth            pl.Smooth: how to smooth it
          ├── reference         a landmark to measure positions from (optional)
          └── segments: { kind: pl.Segmentation, ... }
                                how to cut it into pieces, and what to measure on each piece;
                                each entry becomes one output table, named after its kind + "s"
```

So the jaw line that started this guide reads, from the inside out: *smooth the jaw at 30 Hz*
(`pl.Smooth`), *that is how to process the keypoint named "jaw"* (`pl.KeypointSettings`), *that
is the only keypoint, at 500 frames per second* (`pl.Settings`):

```python
jaw_only = pl.Settings(
    fs_hz=500.0,
    keypoints=[pl.KeypointSettings("jaw", smooth=pl.Smooth(cutoff_hz=30.0))],
)
```

Anything you leave out keeps its default (reference at the end), so most settings are short.

## Getting a pose table

A pose table has one row per frame per keypoint: `frame, time, keypoint, x, y, likelihood`.

**From a session the batch already processed** (it has time on the session clock):

```python
SESSION_DIR = "/root/capsule/scratch/kinematics_v050/behavior_716325_2024-05-31_10-31-14"
pose = pd.read_parquet(f"{SESSION_DIR}/pose.parquet")
pose_10 = pose[(pose.time >= 0) & (pose.time < 600)]      # first 10 min: quicker to play with
```

**From a tracker's output file** (Lightning Pose or DeepLabCut CSV, DeepLabCut or SLEAP HDF5),
plus one time per video frame:

```python
pose = pose_io.read_pose("video_predictions.csv")         # frame, keypoint, x, y, likelihood
harp_time, timing = dft.frame_times("bottom_camera.csv")  # corrected Harp time of each frame
pose, frame_info = signal.add_time(pose, harp_time)       # adds `time`
```

Any clock works: `add_time` takes whatever times you give it, one per frame.

## Recipes

### 1. Smooth one keypoint and look at its speed

```python
jaw_only = pl.Settings(
    fs_hz=500.0,
    keypoints=[pl.KeypointSettings("jaw", smooth=pl.Smooth(cutoff_hz=30.0))],
)
result = pl.process(pose_10, jaw_only)
kin = result["kinematics"]                        # one row per frame of the jaw
print(kin.columns.tolist())                       # frame, time, keypoint, valid, x, y, vx, vy, speed, ...
print(kin.speed.quantile([0.5, 0.75, 0.9, 0.99])) # px/s; useful for choosing thresholds
```

With no `segments`, you get only the per-frame table: smoothed positions and velocities.

### 2. Cut a keypoint into movements

Two ways to decide where a movement is, depending on the body part:

- **It is only tracked while it moves** (a tongue outside the mouth): a movement is a run of
  tracked frames. Method `"tracked_runs"`.
- **It is always visible** (jaw, nose, paw): a movement is a stretch where speed is high. Method
  `"above_threshold"`, with two levels so noise around one level does not chop it up.

```python
tongue = pl.KeypointSettings(
    "tongue_tip_center",
    segments={"movement": pl.Segmentation("tracked_runs", {"max_gap_frames": 10})},
)
jaw = pl.KeypointSettings(
    "jaw",
    smooth=pl.Smooth(cutoff_hz=30.0),
    segments={"jaw_movement": pl.Segmentation(
        "above_threshold", {"on": 150.0, "off": 50.0, "min_frames": 10})},
)
result = pl.process(pose_10, pl.Settings(fs_hz=500.0, keypoints=[tongue, jaw]))
movements = result["movements"]               # kind "movement"      -> table "movements"
jaw_movements = result["jaw_movements"]       # kind "jaw_movement"  -> table "jaw_movements"
```

The dictionary key in `segments` is your name for that kind of segment. It names the output
table (`"<kind>s"`), its id column (`<kind>_id`), and the id column added to the frame table.

`above_threshold` reads: start where speed reaches `on`, end where it drops below `off`, drop
segments shorter than `min_frames`. Pick `on` and `off` from recipe 1's quantiles.

### 3. Measure each movement

List feature groups in `features`; each adds columns to the segment table.

```python
tongue = pl.KeypointSettings(
    "tongue_tip_center",
    segments={"movement": pl.Segmentation(
        "tracked_runs", {"max_gap_frames": 10}, features=["path"])},
)
movements = pl.process(pose_10, pl.Settings(fs_hz=500.0, keypoints=[tongue]))["movements"]
movements[["movement_id", "duration", "path_length", "peak_speed", "mean_speed"]].head()
```

Every segment table has `start_time, end_time, duration, n_frames, n_valid` without asking.

### 4. Measure relative to a landmark

Positions in pixels differ between sessions (camera placement). Give the keypoint a
`reference` and its measurements are taken from that landmark's position (the median of its
confident points over the session). This also adds `radial_velocity` (speed away from the
landmark) to the frame table, and enables the `excursion` and `outbound` feature groups.

```python
tongue = pl.KeypointSettings(
    "tongue_tip_center",
    reference="jaw",
    segments={"movement": pl.Segmentation(
        "tracked_runs", {"max_gap_frames": 10},
        features=["path", "excursion", "outbound"])},
)
result = pl.process(pose_10, pl.Settings(fs_hz=500.0, keypoints=[tongue]))
result["movements"][["movement_id", "amplitude", "end_x", "end_y", "angle_deg", "out_duration"]].head()
result["landmarks"]                           # where the jaw was taken to be
```

The reference does not need its own `KeypointSettings`; listing it as `reference` is enough.
Use `landmarks=[...]` in `pl.Settings` for other fixed points you want located (e.g. spouts).

### 5. Split movements further, and group them into bouts

Segments can be nested. `within` cuts each movement into pieces; `"bouts"` groups movements that
follow each other closely. Each refers to a kind listed **before** it.

```python
tongue = pl.KeypointSettings(
    "tongue_tip_center",
    reference="jaw",
    segments={
        "movement": pl.Segmentation("tracked_runs", {"max_gap_frames": 10},
                                    features=["path", "excursion"]),
        "submovement": pl.Segmentation("zero_crossings", within="movement",
                                       features=["path"]),
        "bout": pl.Segmentation("bouts", {"of": "movement", "max_gap_s": 0.5,
                                          "min_count": 2}),
    },
)
result = pl.process(pose_10, pl.Settings(fs_hz=500.0, keypoints=[tongue]))
sub = result["submovements"]      # movement_id, number_in_movement, direction (+1 out, -1 back)
bouts = result["bouts"]           # n_movements, rate_hz
movements = result["movements"]   # now also has bout_id
```

`zero_crossings` splits where radial velocity changes sign (out vs back), so it needs a
`reference`. `speed_minima` splits at dips in speed instead and needs none.

### 6. Rhythm: dominant frequency of a segment

The `spectrum` group finds the strongest frequency in a band, for continuous signals such as
jaw position. Options go in the keypoint's `feature_options`, under the group's name.

```python
jaw = pl.KeypointSettings(
    "jaw",
    smooth=pl.Smooth(cutoff_hz=30.0),
    segments={"jaw_movement": pl.Segmentation(
        "above_threshold",
        {"on": 150.0, "off": 50.0, "max_gap_frames": 50, "min_frames": 250},
        features=["spectrum"])},
    feature_options={"spectrum": {"column": "y", "band": (2.0, 20.0)}},
)
jm = pl.process(pose_10, pl.Settings(fs_hz=500.0, keypoints=[jaw]))["jaw_movements"]
jm[["jaw_movement_id", "duration", "dominant_hz", "band_fraction"]]
```

Segments shorter than 0.5 s (`min_duration_s`) get NaN. `max_gap_frames: 50` here bridges the
pauses between licks so a licking bout is one segment.

### 7. Start from the tongue settings and change one thing

Settings cannot be changed in place; `dataclasses.replace` makes a copy with some fields
changed. The task module's settings are a good starting point:

```python
tongue = dft.TONGUE_SETTINGS.keypoints[0]
looser = dataclasses.replace(tongue, min_likelihood=0.7)
settings = dataclasses.replace(dft.TONGUE_SETTINGS, keypoints=[looser])

# The settings that produced a saved session, as an object to change and rerun:
settings = pl.Settings.from_dict(pl.read_record(SESSION_DIR)["settings"])
```

### 8. Add your own measurement

A feature group is a function that receives one segment's frames as a dictionary of arrays
(`time, x, y, vx, vy, speed, valid, ...`; positions are smoothed pixels, NaN where not valid)
and returns a dictionary of new columns. Register it, then list it in `features`.

```python
@features.add_feature_group("height")
def height(seg, context):
    y = seg["y"][seg["valid"]]
    return {"min_y": float(np.min(y)) if len(y) else np.nan}

tongue = pl.KeypointSettings(
    "tongue_tip_center",
    segments={"movement": pl.Segmentation("tracked_runs", features=["path", "height"])},
)
pl.process(pose_10, pl.Settings(fs_hz=500.0, keypoints=[tongue]))["movements"][["movement_id", "min_y"]]
```

`context` holds `fs_hz`, `origin` (the reference landmark's position, or None), `flip_x`,
`flip_y` and `options` (this keypoint's `feature_options`).

### 9. Save and read back

```python
pl.write(result, "my_output/session_x")                     # one parquet per table + record.json
back = pl.read("my_output/session_x", tables=["movements"])  # ask only for what you need
```

### 10. Summarize per trial, or per any window

`intervals.summarize_by_window` puts each item (e.g. a movement, by its `start_time`) in the
window its time falls in, and summarizes per window. Windows are any table with `start_time`
and `end_time`, sorted and not overlapping.

```python
trials = pd.read_parquet(f"{SESSION_DIR}/trials.parquet")
movements = pd.read_parquet(f"{SESSION_DIR}/movements.parquet")
go = trials.goCue_start_time_in_session.sort_values().to_numpy()
windows = pd.DataFrame({"start_time": go, "end_time": np.minimum(go + 2.0, np.r_[go[1:], np.inf])})

per_trial = intervals.summarize_by_window(movements, windows, "start_time", {
    "n_movements": (None, "count"),              # how many
    "first_amplitude": ("amplitude", "first"),   # of the earliest
    "mean_peak_speed": ("peak_speed", "mean"),   # any pandas summary, or a function
})

per_10s = intervals.summarize_by_window(          # fixed bins are windows too
    movements, intervals.time_bins(0, movements.end_time.max(), 10.0), "start_time",
    {"n": (None, "count")})
```

Whole trials as windows: `dft.trial_windows(trials)`, the ones `trials.parquet` is summarized
over (go cue to next go cue; the last trial ends one longest go-cue interval after its go cue).

Other helpers: `intervals.interval_of(times, starts, ends)` (which interval each time is in),
`intervals.closest_interval(times, starts, ends)` (the closest, and how far),
`intervals.nearest(times, targets, tolerance)`.

## Reference

### Settings

| Object | Field | Default | Meaning |
|---|---|---|---|
| `pl.Settings` | `fs_hz` | required | Nominal frame rate |
| | `keypoints` | `[]` | List of `KeypointSettings` |
| | `landmarks` | `[]` | Keypoints to locate as fixed points (references are added automatically) |
| | `landmark_min_likelihood` | 0.9 | Confidence cutoff for landmarks without their own settings |
| `pl.KeypointSettings` | `name` | required | Keypoint name as in the pose table |
| | `min_likelihood` | 0.9 | Points below this confidence count as missing |
| | `smooth` | `pl.Smooth()` | Smoothing |
| | `reference` | None | Landmark to measure from; enables radial velocity, `excursion`, `outbound`, `zero_crossings` |
| | `flip_x`, `flip_y` | False | Mirror positions measured from the reference |
| | `segments` | `{}` | `{kind: pl.Segmentation}`, run in order |
| | `feature_options` | `{}` | `{group: {option: value}}`, e.g. for `spectrum` |
| `pl.Smooth` | `cutoff_hz` | 50.0 | Low-pass cutoff; None for no filtering |
| | `order` | 4 | Filter order (even) |
| | `max_gap_s` | 0.02 | Longest missing stretch filled within a run; longer gaps split runs |
| | `kind` | "cubic" | Interpolation used to fill gaps for filtering ("linear", "cubic") |
| `pl.Segmentation` | `method` | required | One of the methods below |
| | `params` | `{}` | The method's parameters |
| | `within` | None | Kind to split (only `zero_crossings`, `speed_minima`) |
| | `features` | `[]` | Feature group names |

### Segmentation methods

| Method | Use for | Parameters (default) |
|---|---|---|
| `tracked_runs` | Parts seen only while moving | `max_gap_frames` (10): missing frames bridged inside a movement |
| `above_threshold` | Parts always visible | `column` ("speed"), `on`, `off` (= `on`), `min_frames` (1), `max_gap_frames` (0) |
| `zero_crossings` | Splitting at changes of direction | `column` ("radial_velocity"), `deadband` (0), `max_gap_frames` (3), `min_samples` (1), `short` ("merge") |
| `speed_minima` | Splitting at dips in speed | `column` ("speed"), `min_prominence`, `max_minimum_speed`, `max_gap_frames` (3), `min_samples` (1), `short` ("merge") |
| `bouts` | Grouping close segments | `of` (kind to group), `max_gap_s`, `min_count` (1) |

### Feature groups

| Group | Columns | Needs |
|---|---|---|
| `path` | `path_length, displacement, peak_speed, mean_speed` | |
| `excursion` | `amplitude, end_x, end_y, start_x, start_y, angle_deg, extent_x, extent_y` | `reference` |
| `outbound` | `out_duration, out_path_length, out_peak_speed, out_mean_speed` | `reference` |
| `spectrum` | `dominant_hz, band_power, band_fraction` | options: `column` ("speed"), `band` ((2, 20)), `min_duration_s` (0.5) |

### Output tables

| Table | One row per | Columns |
|---|---|---|
| `pose` | frame × keypoint | your input |
| `kinematics` | frame × keypoint | `frame, time, keypoint, valid, x, y, vx, vy, speed, radial_velocity`, one `<kind>_id` per kind |
| `<kind>s` | segment | `<kind>_id, keypoint, start_frame, end_frame, start_time, end_time, duration, n_frames, n_valid`, then: `<within>_id, number_in_<within>` if nested; `direction` for `zero_crossings`; `<column>_sum, <column>_peak` for `zero_crossings` / `speed_minima`; `n_<of>s, rate_hz` for `bouts`; feature columns |
| `landmarks` | landmark | `keypoint, x, y, n_valid, stat` |
| `record` | session | settings, versions, counts per keypoint (a dict; `record.json` when saved) |

Units: pixels, seconds, pixels per second, degrees. Ids start at 1, in time order.

## Common mistakes

- **`pose needs a time column`**: run `signal.add_time` first, or use a saved `pose.parquet`.
- **`zero_crossings needs column 'radial_velocity'`**: give the keypoint a `reference`.
- **`this feature group needs a reference point`**: `excursion` and `outbound` need a
  `reference` too.
- **A `KeyError` naming a kind**: `within` or `of` refers to a kind that is not listed earlier
  in `segments`.
- **`keypoint ... is not in the pose`**: the name must match the pose table's `keypoint` column
  exactly (`pose.keypoint.unique()`).
- **Thresholds in the wrong units**: `on` / `off` are in the units of `column` (px/s for speed).
  Look at quantiles first (recipe 1).
- **`FrozenInstanceError` when setting a field**: settings are read-only; use
  `dataclasses.replace` (recipe 7).
- **One giant segment, or none**: for an always-visible part, `tracked_runs` gives one segment
  per tracked stretch (often the whole session); use `above_threshold`.
