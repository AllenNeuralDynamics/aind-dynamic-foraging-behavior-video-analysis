# TODO

## `kinematics/tongue_kinematics_utils.py` — New/AIND video CSV support

`video_alignment.read_video_csv` now reads both acquisition CSV layouts
(see PR #3). The kinematics reader still assumes the Old/flat one, so it
breaks on sessions that use the New/AIND layout
(`behavior-videos/<CameraName>/metadata.csv`, written with a header row).

`find_video_csv_path` already resolves *either* layout, so the pipeline
locates a New/AIND file correctly and then mis-parses it.

- [ ] **`integrate_keypoints_with_video_time` (~line 1187)** hardcodes the
  Old/flat names:

  ```python
  video_csv = pd.read_csv(video_csv_path,
                          names=['Behav_Time', 'Frame', 'Camera_Time',
                                 'Gain', 'Exposure'])
  ```

  Replace with `video_alignment.read_video_csv(video_csv_path)`.
  Note the dependency direction: `video_alignment` is deliberately
  pandas-only, so kinematics imports from it, never the reverse.

- [ ] **Alias the other two columns.** `video_alignment.TIME_COLUMN_ALIASES`
  covers behavior_time only; kinematics needs all three:

  | Old/flat | New/AIND | used by |
  |---|---|---|
  | `Behav_Time` | `ReferenceTime` | covered by `_resolve_time_column` |
  | `Frame` | `CameraFrameNumber` | `check_frame_monotonicity` |
  | `Camera_Time` | `CameraFrameTime` | the `/ 1e9` at ~line 1190 |

  This probably wants a general alias map rather than a time-specific
  resolver.

- [ ] **Verify units before trusting the `/ 1e9`.** Old/flat `Camera_Time`
  is nanoseconds. Whether New/AIND `CameraFrameTime` uses the same scale is
  *unconfirmed* — sample values look like `33512404772286128`, consistent
  with ns, but this has not been checked against a session. Getting it
  wrong scales the camera clock silently.

- [ ] **`qc_and_fix_timing`** takes `time_col='Behav_Time'` /
  `camera_col='Camera_Time'` defaults; these need to follow whatever the
  resolver returns.

### Current failure mode

A New/AIND file reaches `pd.read_csv(names=[...5 Old/flat names])`, so the
header row lands in data row 0 as strings and the columns come back as
dtype `object`:

```
line 1190  Camera_Time / 1e9   -> TypeError: unsupported operand type(s) for /: 'str' and 'float'
check_frame_monotonicity       -> TypeError: unsupported operand type(s) for -: 'str' and 'str'
```

Two steps removed from the real cause, unlike the `ValueError` that
`video_alignment` used to raise.

### Related, lower priority

- [ ] `kinematics/tongue_analysis.py` (~line 364) calls
  `find_video_csv_path(...)` then guards with `if not video_csv.exists()`.
  The finder only ever returns an existing path or `None`, so the guard is
  a no-op on success and raises `AttributeError: 'NoneType' object has no
  attribute 'exists'` on the case it was written to catch. It also does not
  pass through the `camera_name` argument, so it is pinned to
  `BottomCamera`.
