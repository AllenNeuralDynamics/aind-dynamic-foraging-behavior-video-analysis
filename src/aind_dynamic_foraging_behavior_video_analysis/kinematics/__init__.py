"""Pose kinematics: from tracker output to tidy tables.

General steps, independent of any task (numpy, pandas, scipy):
``pose_io`` (read tracker output), ``signal`` (time, mask, smooth,
velocity), ``landmarks`` (fixed reference points), ``segment`` (movements,
submovements, bouts), ``features`` (per-segment measurements),
``intervals`` (which segment or window things fall in, summaries per
window) and ``pipeline`` (all of it from one ``Settings``, save and read).

Task modules in ``tasks`` add a task's files, clock and events, e.g.
``tasks.dynamic_foraging_tongue``. Import submodules directly; this package
imports nothing so that ``tongue_lickometer_utils`` stays usable with the
core install.
"""
