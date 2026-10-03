"""Check and correct the Harp time of each frame in a behavior video CSV.

A video CSV has one row per saved frame (row ``i`` is video frame ``i``)
holding the frame's Harp time (the behavior clock), the camera's frame
number, and the camera's own clock. The acquisition workflow pairs frames
with Harp triggers in arrival order, so the Harp column is always the
trigger sequence in order, one trigger per row, whether or not frames were
lost. If frames were lost, later rows carry the times of earlier triggers.

Checks, one question each (:func:`check_video_timing` runs them all):

- ``no_frames_lost``: do the frame numbers span exactly as many exposures
  as there are rows?
- ``frame_numbers_increase``: does every frame number step forward?
- ``camera_time_increases``: does every camera time step forward?
- ``no_duplicate_frames``: is no frame saved twice (same frame number and
  camera time on consecutive rows)?
- ``harp_has_no_glitches``: is every Harp value in line with its
  neighbours?
- ``harp_evenly_spaced``: after fixing glitches, is every Harp step one
  frame interval?
- ``clock_rates_agree``: over the whole session, do Harp and camera time
  advance at the same rate per exposure (within ``MAX_CLOCK_RATE_PPM``)?
- ``trigger_log_count``: given a trigger log, does it have one event per
  exposure?
- ``trigger_log_matches_csv``: given a trigger log, does it hold the CSV's
  Harp column?
- ``harp_matches_camera``: if frames were lost, after re-indexing, does
  each corrected step match the camera step?
- ``video_frame_count``: given a frame count, does the video have one
  frame per row?

Verdict (:func:`timing_verdict`, which :func:`correct_video_timing`
follows): ``use`` or ``exclude: <check>``, from the checks table alone.

1. Every session must pass ``no_duplicate_frames``,
   ``harp_evenly_spaced`` and ``clock_rates_agree``.
2. Frames lost: frame numbers and camera time must increase.
3. The trigger log, the re-index trial and the frame count must pass when
   they ran.

The correction then fixes isolated Harp glitches and, if frames were lost,
moves each Harp value to the row of its exposure, estimating the rows the
CSV has no trigger for from camera time (or reading them from the trigger
log). An excluded camera raises ``ValueError``; a partial correction is
never returned. :func:`write_video_timing` saves the checks and verdict.

Limits of the CSV alone (the trigger log removes the first): a forward
jump in corrupted camera metadata looks exactly like a burst of dropped
frames; frames lost before the first saved row shift every time by a
constant; a block of replayed frames looks like corrupted metadata that
returns to its start (only the video can tell).

Example::

    timing = load_video_timing("behavior-videos/bottom_camera.csv")
    checks = check_video_timing(timing)  # one row per check
    timing_verdict(checks)               # "use" or "exclude: <check>"
    fixed = correct_video_timing(timing)  # adds harp_time, harp_source
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis import __version__
from aind_dynamic_foraging_behavior_video_analysis.video_alignment import (
    read_video_csv,
)

# The New/AIND layout has a header row; the first three columns must be
# these. The Old/flat layout has no header and the same column order.
NEW_LAYOUT_COLUMNS = ["ReferenceTime", "CameraFrameNumber", "CameraFrameTime"]

# Harp timestamps are seconds plus ticks of 32 microseconds.
HARP_TICK_S = 32e-6

# Allowed step error, as a fraction of the frame interval.
STEP_TOLERANCE = 0.5

# Allowed whole-session rate difference between Harp and camera time.
# Normal: +6.6 to +30.2 ppm on 17 cameras (2025-01 to 2026-01).
MAX_CLOCK_RATE_PPM = 100

RECORD_FILE = "video_timing_{camera}.json"


# --- Loading -------------------------------------------------------------


def load_video_timing(video_csv_path) -> pd.DataFrame:
    """Read a video CSV (either layout) into three columns.

    Parameters
    ----------
    video_csv_path : str or pathlib.Path
        A behavior video CSV (see :func:`video_alignment.read_video_csv`).

    Returns
    -------
    pandas.DataFrame
        One row per saved frame: ``harp_time_raw`` (s), ``frame_number``
        (int64) and ``camera_time`` (s, converted from ns).

    Raises
    ------
    ValueError
        If the file is empty, has an unknown header, or has missing values.
    """
    raw = read_video_csv(video_csv_path)
    if len(raw) == 0:
        raise ValueError(f"Video CSV has no rows: {video_csv_path}")
    first_three = list(raw.columns[:3])
    if first_three[0] != "Behav_Time" and first_three != NEW_LAYOUT_COLUMNS:
        raise ValueError(
            f"Unknown video CSV columns {list(raw.columns)} in "
            f"{video_csv_path}; expected {NEW_LAYOUT_COLUMNS} or no header"
        )
    raw = raw.iloc[:, :3]
    if raw.isna().any().any():
        raise ValueError(f"Video CSV has missing values: {video_csv_path}")
    return pd.DataFrame(
        {
            "harp_time_raw": raw.iloc[:, 0].astype("float64"),
            "frame_number": raw.iloc[:, 1].astype("int64"),
            "camera_time": raw.iloc[:, 2].astype("float64") / 1e9,
        }
    )


def read_harp_trigger_log(path) -> np.ndarray:
    """Read camera trigger times from a Harp ``Event_94.bin`` file.

    Register 94 is ``Camera1Frame`` on the Harp Behavior board, which
    triggers every camera. Each 13-byte message holds seconds (uint32) and
    32 us ticks (uint16). Same result as ``harp.io.read(path).index``.

    Parameters
    ----------
    path : str or pathlib.Path
        ``behavior/raw.harp/BehaviorEvents/Event_94.bin``.

    Returns
    -------
    numpy.ndarray
        Trigger times in Harp seconds, in file order.
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0 or data.size % 13:
        raise ValueError(f"Not a stream of 13-byte Harp messages: {path}")
    messages = data.reshape(-1, 13)
    if not (messages[:, 1] == 11).all() or not (messages[:, 4] == 17).all():
        raise ValueError(f"Unexpected Harp message format in {path}")
    seconds = messages[:, 5:9].copy().view("<u4").ravel()
    ticks = messages[:, 9:11].copy().view("<u2").ravel()
    return seconds + ticks * HARP_TICK_S


# --- Helpers -------------------------------------------------------------


def frame_interval(times) -> float:
    """Return the typical step of an evenly spaced series.

    The mean of the steps within 25% of the median: Harp steps alternate by
    one 32 us tick, and outliers (glitches, drops) are left out.
    """
    step = np.diff(np.asarray(times, dtype="float64"))
    median = np.median(step)
    return float(step[np.abs(step - median) < 0.25 * median].mean())


def find_glitch_rows(times, tolerance=STEP_TOLERANCE) -> np.ndarray:
    """Return isolated values that disagree with both neighbours.

    Row ``r`` is a glitch if it is off from the midpoint of ``r - 1`` and
    ``r + 1`` by more than ``tolerance`` frames while those two are two
    frames apart. Runs of bad values are not glitches.
    """
    t = np.asarray(times, dtype="float64")
    ifi = frame_interval(t)
    tol = tolerance * ifi
    off_midpoint = np.abs(t[1:-1] - (t[:-2] + t[2:]) / 2) > tol
    neighbours_agree = np.abs(t[2:] - t[:-2] - 2 * ifi) <= tol
    rows = np.flatnonzero(off_midpoint & neighbours_agree) + 1
    isolated = ~np.isin(rows - 1, rows) & ~np.isin(rows + 1, rows)
    return rows[isolated]


def fix_glitches(times):
    """Replace each isolated glitch with the midpoint of its neighbours.

    Returns
    -------
    fixed : numpy.ndarray
        A copy of ``times`` with glitches replaced.
    rows : numpy.ndarray
        The glitch rows.
    """
    fixed = np.asarray(times, dtype="float64").copy()
    rows = find_glitch_rows(fixed)
    fixed[rows] = (fixed[rows - 1] + fixed[rows + 1]) / 2
    return fixed, rows


def _result(check, passed, message, rows=(), count=None) -> dict:
    """One check's outcome. ``passed`` is None when the check was skipped."""
    rows = np.asarray(rows, dtype=int)
    return {
        "check": check,
        "passed": passed,
        "count": int(len(rows) if count is None else count),
        "message": message,
        "rows": rows[:100].tolist(),
    }


# --- Checks: one question each -------------------------------------------


def check_no_frames_lost(frame_number) -> dict:
    """Do the frame numbers span exactly as many exposures as rows?

    ``count`` is exposures minus rows (negative: more rows than exposures);
    ``rows`` are where the frame number skips forward.
    """
    frames = np.asarray(frame_number)
    lost = int(frames[-1] - frames[0] + 1 - len(frames))
    skips = np.flatnonzero(np.diff(frames) > 1) + 1
    if lost == 0:
        message = "no frames lost"
    elif lost > 0:
        message = f"frames lost: {lost}"
    else:
        message = f"more rows than exposures: {-lost}"
    return _result("no_frames_lost", lost == 0, message, skips, count=lost)


def check_frame_numbers_increase(frame_number) -> dict:
    """Does every frame number step forward?"""
    rows = np.flatnonzero(np.diff(np.asarray(frame_number)) <= 0) + 1
    message = f"frame numbers stepping back or repeating: {len(rows)}"
    return _result("frame_numbers_increase", len(rows) == 0, message, rows)


def check_camera_time_increases(camera_time) -> dict:
    """Does every camera time step forward?"""
    rows = np.flatnonzero(np.diff(np.asarray(camera_time)) <= 0) + 1
    message = f"camera times stepping back or repeating: {len(rows)}"
    return _result("camera_time_increases", len(rows) == 0, message, rows)


def check_no_duplicate_frames(frame_number, camera_time) -> dict:
    """Is no frame saved twice (same frame number and camera time in a row)?"""
    same_frame = np.diff(np.asarray(frame_number)) == 0
    same_time = np.diff(np.asarray(camera_time)) == 0
    rows = np.flatnonzero(same_frame & same_time) + 1
    message = f"frames saved twice: {len(rows)}"
    return _result("no_duplicate_frames", len(rows) == 0, message, rows)


def check_harp_has_no_glitches(harp) -> dict:
    """Is every Harp value in line with its neighbours?"""
    rows = find_glitch_rows(harp)
    message = f"isolated glitches: {len(rows)}"
    return _result("harp_has_no_glitches", len(rows) == 0, message, rows)


def check_harp_evenly_spaced(harp, tolerance=STEP_TOLERANCE) -> dict:
    """Is every Harp step one frame interval? Fix glitches first."""
    harp = np.asarray(harp, dtype="float64")
    ifi = frame_interval(harp)
    rows = np.flatnonzero(np.abs(np.diff(harp) - ifi) > tolerance * ifi) + 1
    message = f"steps off by > {tolerance} frame: {len(rows)}"
    return _result("harp_evenly_spaced", len(rows) == 0, message, rows)


def check_clock_rates_agree(
    harp, camera_time, frame_number, max_ppm=MAX_CLOCK_RATE_PPM
) -> dict:
    """Over the session, do Harp and camera advance at the same rate?

    Harp time per row (one trigger per row) against camera time per
    exposure, from the first and last rows only, so drops do not matter.
    Catches drift that no single step shows, e.g. many small clock steps.
    ``count`` is the difference in ppm. Fix glitches first.
    """
    harp = np.asarray(harp, dtype="float64")
    camera = np.asarray(camera_time, dtype="float64")
    frames = np.asarray(frame_number)
    exposures_spanned = frames[-1] - frames[0]
    if exposures_spanned <= 0 or camera[-1] <= camera[0]:
        return _result(
            "clock_rates_agree", False, "first and last rows out of order"
        )
    harp_interval = (harp[-1] - harp[0]) / (len(harp) - 1)
    camera_interval = (camera[-1] - camera[0]) / exposures_spanned
    ppm = (harp_interval / camera_interval - 1) * 1e6
    message = f"Harp runs {ppm:+.1f} ppm against camera (limit {max_ppm})"
    return _result(
        "clock_rates_agree", abs(ppm) <= max_ppm, message, count=round(ppm)
    )


def check_harp_matches_camera(
    harp, camera_time, tolerance=STEP_TOLERANCE
) -> dict:
    """Does each corrected Harp step match the camera step?

    Used on re-indexed times, where it witnesses that each row moved to the
    right trigger. On raw Harp it fails wherever frames were dropped.
    """
    harp = np.asarray(harp, dtype="float64")
    step_error = np.abs(np.diff(harp) - np.diff(camera_time))
    rows = np.flatnonzero(step_error > tolerance * frame_interval(harp)) + 1
    message = (
        f"steps differing from camera by > {tolerance} frame: {len(rows)}"
    )
    return _result("harp_matches_camera", len(rows) == 0, message, rows)


def check_video_frame_count(n_rows, video_frame_count=None) -> dict:
    """Does the video have one frame per row? Skipped without a count."""
    if video_frame_count is None:
        return _result("video_frame_count", None, "skipped: no count given")
    difference = int(video_frame_count - n_rows)
    message = f"video has {video_frame_count} frames, CSV {n_rows} rows"
    return _result(
        "video_frame_count", difference == 0, message, count=difference
    )


def check_trigger_log_count(frame_number, trigger_times=None) -> dict:
    """Does the trigger log have one event per exposure, first to last
    frame number? Skipped without a log. A mismatch means frames lost
    before the first or after the last saved row, or corrupted frame
    numbers (a forward jump the CSV alone cannot tell from dropped
    frames). ``count`` is events minus exposures."""
    if trigger_times is None:
        return _result("trigger_log_count", None, "skipped: no trigger log")
    frames = np.asarray(frame_number)
    n_exposures = int(frames[-1] - frames[0] + 1)
    difference = len(trigger_times) - n_exposures
    message = (
        f"trigger log has {len(trigger_times)} events, frame numbers span "
        f"{n_exposures} exposures"
    )
    return _result(
        "trigger_log_count", difference == 0, message, count=difference
    )


def check_trigger_log_matches_csv(harp, trigger_times=None) -> dict:
    """Does the trigger log hold the CSV's Harp column (within one tick)
    on the rows both have? Skipped without a log; a log from another
    session fails. ``rows`` are the rows that differ."""
    if trigger_times is None:
        return _result(
            "trigger_log_matches_csv", None, "skipped: no trigger log"
        )
    n = min(len(harp), len(trigger_times))
    error = np.abs(np.asarray(trigger_times[:n]) - np.asarray(harp[:n]))
    rows = np.flatnonzero(error > HARP_TICK_S)
    message = (
        f"trigger log does not match the CSV's Harp column on {len(rows)} "
        f"rows (max difference {error.max() * 1e3:.3f} ms)"
    )
    return _result("trigger_log_matches_csv", len(rows) == 0, message, rows)


def input_checks(frame_number, camera_time, harp) -> pd.DataFrame:
    """Run the timing checks on plain arrays; one row per check."""
    frames = np.asarray(frame_number)
    camera = np.asarray(camera_time, dtype="float64")
    harp_fixed, _ = fix_glitches(harp)
    return pd.DataFrame(
        [
            check_no_frames_lost(frames),
            check_frame_numbers_increase(frames),
            check_camera_time_increases(camera),
            check_no_duplicate_frames(frames, camera),
            check_harp_has_no_glitches(harp),
            check_harp_evenly_spaced(harp_fixed),
            check_clock_rates_agree(harp_fixed, camera, frames),
        ]
    )


# Checks every session must pass, before anything else is decided.
ALWAYS_REQUIRED = [
    "no_duplicate_frames",
    "harp_evenly_spaced",
    "clock_rates_agree",
]
# Checks needed to locate lost frames.
REQUIRED_TO_REINDEX = ["frame_numbers_increase", "camera_time_increases"]
# Checks every session must pass when they ran (None: skipped).
REQUIRED_WHEN_RUN = [
    "trigger_log_count",
    "trigger_log_matches_csv",
    "harp_matches_camera",
    "video_frame_count",
]


def _failed(passed, check):
    """True if ``check`` ran and failed (``passed`` maps check to result)."""
    return passed.get(check) is not None and not passed[check]


def _reindex_trial(frames, camera, harp, triggers, tail_fit_window_s):
    """The ``harp_matches_camera`` row: re-index with ``triggers`` and
    compare each step with the camera step. Skipped when no frames were
    lost, when re-indexing is not allowed, or without usable triggers."""
    checks = input_checks(frames, camera, harp)
    passed = dict(zip(checks["check"], checks["passed"]))
    if passed["no_frames_lost"]:
        reason = "no frames lost"
    elif any(
        _failed(passed, c) for c in ALWAYS_REQUIRED + REQUIRED_TO_REINDEX
    ):
        reason = "cannot re-index"
    elif triggers is None:
        reason = "trigger log does not fit the CSV"
    else:
        times, _ = _corrected_times(
            frames, camera, triggers, True, tail_fit_window_s
        )
        return checks, check_harp_matches_camera(times, camera)
    return checks, _result("harp_matches_camera", None, f"skipped: {reason}")


def _correction_checks(frames, camera, harp, triggers, tail_fit_window_s):
    """Input checks on ``harp`` plus the re-index trial on ``triggers``."""
    checks, trial = _reindex_trial(
        frames, camera, harp, triggers, tail_fit_window_s
    )
    return pd.concat([checks, pd.DataFrame([trial])], ignore_index=True)


def check_video_timing(
    timing, trigger_times=None, video_frame_count=None, tail_fit_window_s=600
) -> pd.DataFrame:
    """Run every check on one camera; :func:`timing_verdict` decides from
    the result alone.

    Parameters
    ----------
    timing : pandas.DataFrame
        From :func:`load_video_timing`.
    trigger_times : array-like, optional
        From :func:`read_harp_trigger_log`: adds ``trigger_log_count`` and
        ``trigger_log_matches_csv``, and the re-index trial uses it.
    video_frame_count : int, optional
        Frames in the video file, for ``video_frame_count``.
    tail_fit_window_s : float
        Passed to the re-index trial (see :func:`correct_frame_times`).

    Returns
    -------
    pandas.DataFrame
        One row per check: ``check``, ``passed`` (None if skipped),
        ``count``, ``message`` and ``rows`` (first 100 offending rows).
        The checks of the module docstring, then ``trigger_log_count``,
        ``trigger_log_matches_csv``, ``harp_matches_camera`` (the re-index
        trial, run only when frames were lost) and ``video_frame_count``.
    """
    harp = timing["harp_time_raw"].to_numpy()
    frames = timing["frame_number"].to_numpy()
    camera = timing["camera_time"].to_numpy()
    log_rows = [check_trigger_log_count(frames, trigger_times)]
    log_rows.append(check_trigger_log_matches_csv(harp, trigger_times))
    if trigger_times is None:
        triggers = harp
    elif all(r["passed"] for r in log_rows):
        triggers = np.asarray(trigger_times, dtype="float64")
    else:
        triggers = None
    checks, trial = _reindex_trial(
        frames, camera, harp, triggers, tail_fit_window_s
    )
    frame_count = check_video_frame_count(len(timing), video_frame_count)
    return pd.concat(
        [checks, pd.DataFrame(log_rows + [trial, frame_count])],
        ignore_index=True,
    )


def timing_verdict(checks) -> str:
    """``"use"``, or ``"exclude: <check>"`` for the first required check
    that failed.

    Required, in order: ``ALWAYS_REQUIRED``; ``REQUIRED_TO_REINDEX`` when
    frames were lost; then ``REQUIRED_WHEN_RUN`` (trigger log, re-index
    trial, video frame count) when they ran. Lost frames and isolated
    glitches alone do not exclude: the correction handles them.
    """
    passed = dict(zip(checks["check"], checks["passed"]))
    required = list(ALWAYS_REQUIRED)
    if _failed(passed, "no_frames_lost"):
        required += REQUIRED_TO_REINDEX
    for check in required + REQUIRED_WHEN_RUN:
        if _failed(passed, check):
            return f"exclude: {check}"
    return "use"


def _correction_method(checks):
    """How :func:`correct_video_timing` corrects a camera whose verdict is
    ``use``: ``as written``, ``fix glitches`` or ``re-index``; None for an
    excluded camera."""
    if timing_verdict(checks) != "use":
        return None
    passed = dict(zip(checks["check"], checks["passed"]))
    if _failed(passed, "no_frames_lost"):
        return "re-index"
    if _failed(passed, "harp_has_no_glitches"):
        return "fix glitches"
    return "as written"


def timing_action(checks) -> str:
    """Deprecated: use :func:`timing_verdict`.

    One of ``use harp as written``, ``fix glitches``, ``re-index`` or
    ``refuse: <check>``, from the input checks only.
    """
    warnings.warn(
        "timing_action is deprecated; use timing_verdict",
        DeprecationWarning,
        stacklevel=2,
    )
    passed = dict(zip(checks["check"], checks["passed"]))
    for check in ALWAYS_REQUIRED:
        if not passed[check]:
            return f"refuse: {check}"
    if passed["no_frames_lost"]:
        if not passed["harp_has_no_glitches"]:
            return "fix glitches"
        return "use harp as written"
    for check in REQUIRED_TO_REINDEX:
        if not passed[check]:
            return f"refuse: {check}"
    return "re-index"


def _raise_if_excluded(checks):
    """Raise ValueError naming the check that excludes the camera."""
    verdict = timing_verdict(checks)
    if verdict != "use":
        failed = checks.set_index("check").loc[verdict.split(": ")[1]]
        raise ValueError(f"{failed.name} failed: {failed['message']}")


def write_video_timing(checks, out_dir, camera):
    """Write ``RECORD_FILE`` into ``out_dir`` and return its path.

    The record holds the camera, the verdict, the correction method (None
    if excluded), whether a trigger log was used, frames lost, glitch
    rows, the library version and every check row. Corrected times are
    not written: they are recomputed in seconds.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_name = checks.set_index("check")
    record = {
        "camera": camera,
        "verdict": timing_verdict(checks),
        "method": _correction_method(checks),
        "trigger_log": by_name.loc["trigger_log_count", "passed"] is not None,
        "frames_lost": int(by_name.loc["no_frames_lost", "count"]),
        "glitch_rows": list(by_name.loc["harp_has_no_glitches", "rows"]),
        "versions": {
            "aind_dynamic_foraging_behavior_video_analysis": __version__,
        },
        "checks": json.loads(checks.to_json(orient="records")),
    }
    path = out / RECORD_FILE.format(camera=camera)
    path.write_text(json.dumps(record, indent=2))
    return path


# --- Correction ----------------------------------------------------------


def reindex_to_exposures(frame_number, trigger_times) -> np.ndarray:
    """Give each frame the trigger of its own exposure.

    Frame ``i`` was exposed by trigger ``frame_number[i] - frame_number[0]``.
    Frames whose trigger is past the end of ``trigger_times`` get NaN.
    """
    k = np.asarray(frame_number) - frame_number[0]
    times = np.full(len(k), np.nan)
    known = k < len(trigger_times)
    times[known] = np.asarray(trigger_times)[k[known]]
    return times


def estimate_missing_from_camera(times, camera_time, window_s=600):
    """Fill NaN times from a linear fit of time on camera time.

    The fit uses the known frames in the last ``window_s`` seconds of
    camera time before the last known frame.
    """
    times = np.asarray(times, dtype="float64").copy()
    camera_time = np.asarray(camera_time, dtype="float64")
    missing = np.isnan(times)
    if missing.any():
        last_known = camera_time[~missing][-1]
        fit = ~missing & (camera_time >= last_known - window_s)
        slope, intercept = np.polyfit(
            camera_time[fit] - last_known, times[fit], 1
        )
        times[missing] = intercept + slope * (
            camera_time[missing] - last_known
        )
    return times


def _corrected_times(frames, camera, trigger_times, reindex, window_s):
    """Corrected time and source per frame, glitches fixed; re-indexed
    to exposures when ``reindex``, else the triggers in row order."""
    n = len(frames)
    triggers, glitches = fix_glitches(trigger_times)
    k = frames - frames[0]
    source = np.full(n, "original", dtype=object)
    if reindex:
        times = reindex_to_exposures(frames, triggers)
        source[k != np.arange(n)] = "reindexed"
        source[np.isnan(times)] = "estimated_camera_fit"
        times = estimate_missing_from_camera(times, camera, window_s)
    else:
        times = triggers[:n]
    source[np.isin(k, glitches)] = "glitch_interpolated"
    return times, source


def correct_frame_times(
    frame_number, camera_time, trigger_times, tail_fit_window_s=600
):
    """Give each saved frame the time of the trigger that exposed it.

    Plain arrays in, plain arrays out; knows nothing about CSVs or Harp.
    Runs :func:`input_checks` on ``trigger_times[:n]`` and the re-index
    trial, and raises unless :func:`timing_verdict` is ``use``. Assumes one
    trigger per exposure, and that the first frame is the first exposure.

    Parameters
    ----------
    frame_number : array-like of int
        Camera exposure counter per saved frame.
    camera_time : array-like of float
        Camera clock per saved frame, in seconds.
    trigger_times : array-like of float
        Trigger times in order, starting with the first frame's trigger; at
        least one per frame.
    tail_fit_window_s : float
        Seconds of camera time used to estimate frames past the last
        trigger.

    Returns
    -------
    times : numpy.ndarray
        Corrected time per frame.
    source : numpy.ndarray of str
        Per frame: ``original``, ``glitch_interpolated``, ``reindexed`` or
        ``estimated_camera_fit``.

    Raises
    ------
    ValueError
        If a required check fails (see the module docstring).
    """
    frames = np.asarray(frame_number)
    camera = np.asarray(camera_time, dtype="float64")
    n = len(frames)
    if len(camera) != n or n < 3 or len(trigger_times) < n:
        raise ValueError(
            "Need >= 3 frames, matching camera times, one trigger per frame"
        )
    triggers = np.asarray(trigger_times, dtype="float64")
    checks = _correction_checks(
        frames, camera, triggers[:n], triggers, tail_fit_window_s
    )
    _raise_if_excluded(checks)
    return _corrected_times(
        frames,
        camera,
        triggers,
        _correction_method(checks) == "re-index",
        tail_fit_window_s,
    )


def correct_video_timing(
    timing, trigger_times=None, tail_fit_window_s=600
) -> pd.DataFrame:
    """Return ``timing`` with corrected Harp time per frame.

    Runs :func:`check_video_timing` and raises unless
    :func:`timing_verdict` is ``use``; then corrects as written, by fixing
    glitches, or by re-indexing, from the CSV's Harp column or, if
    ``trigger_times`` is given, from the Harp trigger log (which must have
    one event per exposure and match the CSV's Harp column).

    Parameters
    ----------
    timing : pandas.DataFrame
        From :func:`load_video_timing`.
    trigger_times : array-like, optional
        From :func:`read_harp_trigger_log`.
    tail_fit_window_s : float
        Seconds of camera time used to estimate frames past the last
        trigger.

    Returns
    -------
    pandas.DataFrame
        ``timing`` plus ``harp_time`` and ``harp_source``; rows read from a
        trigger log are labelled ``trigger_log``.

    Raises
    ------
    ValueError
        Naming the check that excludes the camera.
    """
    checks = check_video_timing(
        timing, trigger_times, tail_fit_window_s=tail_fit_window_s
    )
    _raise_if_excluded(checks)
    if trigger_times is None:
        triggers = timing["harp_time_raw"].to_numpy()
    else:
        triggers = np.asarray(trigger_times, dtype="float64")
    harp_time, source = _corrected_times(
        timing["frame_number"].to_numpy(),
        timing["camera_time"].to_numpy(),
        triggers,
        _correction_method(checks) == "re-index",
        tail_fit_window_s,
    )
    if trigger_times is not None:
        source[source != "glitch_interpolated"] = "trigger_log"
    fixed = timing.copy()
    fixed["harp_time"] = harp_time
    fixed["harp_source"] = source
    return fixed


# --- Whole session -------------------------------------------------------


def check_session(behavior_videos_path) -> pd.DataFrame:
    """Check every camera CSV in a ``behavior-videos`` folder.

    Parameters
    ----------
    behavior_videos_path : str or pathlib.Path
        Holds ``<camera>.csv`` (Old/flat) or ``<Camera>/metadata.csv``
        (New/AIND) files.

    Returns
    -------
    pandas.DataFrame
        One row per camera: ``camera``, ``csv_path``, ``verdict`` (from
        :func:`timing_verdict`), ``failed_checks``, ``frames_lost`` and
        ``glitch_rows``. Unreadable CSVs get ``verdict``
        ``exclude: unreadable`` and the error text in ``error``.
    """
    folder = Path(behavior_videos_path)
    csv_paths = sorted(folder.glob("*.csv")) + sorted(
        folder.glob("*/metadata.csv")
    )
    rows = []
    for csv_path in csv_paths:
        is_new_layout = csv_path.name == "metadata.csv"
        camera = csv_path.parent.name if is_new_layout else csv_path.stem
        info = {"camera": camera, "csv_path": str(csv_path)}
        try:
            checks = check_video_timing(load_video_timing(csv_path))
        except ValueError as e:
            info.update({"verdict": "exclude: unreadable", "error": str(e)})
        else:
            by_name = checks.set_index("check")
            info.update(
                {
                    "verdict": timing_verdict(checks),
                    "failed_checks": checks.loc[
                        checks["passed"].eq(False), "check"
                    ].tolist(),
                    "frames_lost": by_name.loc["no_frames_lost", "count"],
                    "glitch_rows": by_name.loc["harp_has_no_glitches", "rows"],
                }
            )
        rows.append(info)
    return pd.DataFrame(rows)
