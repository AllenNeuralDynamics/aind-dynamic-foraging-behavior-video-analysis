# TODO

## Retire `video_timing_qc.timing_action` (deprecated in 0.2.0)

`timing_action` still works in 0.2.0 (same strings, plus a `DeprecationWarning`); nothing in
this repo calls it. In order:

1. ~~**Release 0.2.0**~~ Done 2026-10-05 (PR #10, tag `v0.2.0`). Callers pinned to v0.1.0
   are unaffected until they move.
2. **Move `kinematics_analysis` to 0.2.0** (`environment/Dockerfile` pins v0.1.0, `41e5b59`):
   - `code/build_me_table.py`: store `timing_verdict` instead of `timing_action`, pass the
     trigger log to `check_video_timing(timing, trigger_times, video_frame_count=...)`, and drop
     its own frame-count refusal (the verdict now covers it).
   - `code/verify_video_timing_qc.ipynb` and `code/fip_me_aligned_table_plan.md`: replace the
     `timing_action` mentions.
   - Better still, consume the screen (`video_screen.load_screen`) instead of its own dry run.
3. **Remove `timing_action`** (PR open, merge after step 2 has landed; 0.3.0): delete the function,
   its use in `tests/test_video_timing_qc.py` (`assertOutcome`, `test_failed_reindex_in_checks`),
   and say so in the README "Changes".

Other screening follow-ups (optional `run_batch_analysis(screen=...)`, the one-extra-event
trigger log question) are in `VIDEO_SCREEN_PLAN.md`. Video timing QC follow-ups are in
`VIDEO_TIMING_QC_PLAN.md`.
