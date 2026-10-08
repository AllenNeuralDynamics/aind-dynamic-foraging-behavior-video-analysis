# TODO

`timing_action` was retired 2026-10-05: released 0.2.0 (tag `v0.2.0`),
`kinematics_analysis` moved to `timing_verdict` (`fip-motion-energy` `600ad99`, dry run identical
to v0.1.0 on Code Ocean), and the function removed in 0.3.0 (PR #12).

Optional, in `kinematics_analysis`: `build_me_table.py` could consume the screen
(`video_screen.load_screen`) instead of its own dry run.

`kinematics/video_clip_utils.py::extract_trial_clip` seeks by a constant session-to-video
offset (`get_video_time`, subtraction), so its clips are off in sessions with dropped frames,
more so later in the session. Fix by moving it onto `video_alignment.event_frame_ranges` (with
corrected `harp_time`) + `video_clips.cut_clip`, which now exist (`VIDEO_CLIPS_PLAN.md`,
"Out of scope").

`video_clips` follow-ups (`VIDEO_CLIPS_PLAN.md`, "Verification" results): the local-vs-URL
check on `behavior_800886_2025-08-18_13-14-52`; one small DLC project opened in DLC's labeling
GUI and loaded by LP with a context model after `add_context_frames`; Phase 4 consumers
(re-encoding capsule on 3.11+, our DLC → LP conversion code).

Other screening follow-ups (optional `run_batch_analysis(screen=...)`, the one-extra-event
trigger log question) are in `VIDEO_SCREEN_PLAN.md`. Video timing QC follow-ups are in
`VIDEO_TIMING_QC_PLAN.md`.
