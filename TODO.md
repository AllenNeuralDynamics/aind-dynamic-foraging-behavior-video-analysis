# TODO

`timing_action` was retired 2026-10-05: released 0.2.0 (tag `v0.2.0`),
`kinematics_analysis` moved to `timing_verdict` (`fip-motion-energy` `600ad99`, dry run identical
to v0.1.0 on Code Ocean), and the function removed in 0.3.0 (PR #12).

Optional, in `kinematics_analysis`: `build_me_table.py` could consume the screen
(`video_screen.load_screen`) instead of its own dry run.

Other screening follow-ups (optional `run_batch_analysis(screen=...)`, the one-extra-event
trigger log question) are in `VIDEO_SCREEN_PLAN.md`. Video timing QC follow-ups are in
`VIDEO_TIMING_QC_PLAN.md`.

## Repair Harp clock steps in `video_timing_qc`

Old Harp Behavior boards (atxmega core < 1.14,
[harp-tech/protocol#48](https://github.com/harp-tech/protocol/issues/48)) run fast for ~4–5 min
and then step back on one row, twice per session, 1,280 s apart (peak 0.9–5.7 ms). These cameras
are excluded today (`harp_evenly_spaced`): 56 of 602 in the FIP screen. Walkthrough and the
proposed repair: `examples/video_timing_clock_step.ipynb`.

- **`find_clock_steps` / `fix_clock_steps`** as in the notebook's section 4: keep the Harp column
  as written (it matches the board's other events) and spread each step's shortfall `P` over the
  `m` rows before it, with `m` the smallest number with `P / (m + 1) < frame / 2`, so every step
  is within half a frame. Return the rows moved and the peak per step.
- **In `correct_video_timing`**: after `fix_glitches`; with frames lost, repair the times after
  re-indexing (the trigger log has the same step). Mark moved rows (`harp_source =
  "clock_step_spread"`). `check_trigger_log_matches_csv` keeps comparing the raw column.
- **Verdict**: a new informational check (`harp_has_no_clock_steps`: count, peaks), and
  `harp_evenly_spaced` evaluated after the repair, so these cameras become `use`. At 120 Hz,
  peaks under half a frame already pass as written; repair them too.
- **Tests**: synthetic ramp + step at 500 Hz (P = 1, 3.3, 5.7 ms) and 120 Hz, alone and with
  dropped frames and a glitch.
- **Then**: re-run the FIP screen, update section 5 of `video_timing_failure_modes.ipynb`, the
  README, and release as a minor version.
- **Open question**: frames in a ramp window stay off by up to `P` from devices that weren't
  ramping. Check whether the sound card ramps at the same seconds in an affected session.
