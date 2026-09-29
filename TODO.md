# TODO

## `kinematics/tongue_analysis.py`

- [ ] `generate_tongue_dfs` calls `find_video_csv_path(videos_folder)`
  without a `camera_name`, so it is pinned to `BottomCamera`.

New/AIND video CSV support in `integrate_keypoints_with_video_time` is done:
it now reads through `video_timing_qc.load_video_timing` (see
`VIDEO_TIMING_QC_PLAN.md`). New/AIND `CameraFrameTime` is confirmed to be
nanoseconds.
