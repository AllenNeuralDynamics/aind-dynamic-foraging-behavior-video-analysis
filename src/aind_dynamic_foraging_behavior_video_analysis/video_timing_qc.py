"""Check and correct the Harp time of each frame in a behavior video CSV.

A video CSV has one row per saved frame (row ``i`` is video frame ``i``)
holding the frame's Harp time (the behavior clock), the camera's frame
number, and the camera's own clock. The acquisition workflow pairs frames
with Harp triggers in arrival order, so the Harp column is always the
trigger sequence in order. If frames were lost, later rows carry the times
of earlier triggers.

Checks, one question each (:func:`check_video_timing` runs them all):

- ``no_frames_lost``: do the frame numbers span exactly as many exposures
  as there are rows?
- ``frame_numbers_increase``: does every frame number step forward?
- ``camera_time_increases``: does every camera time step forward?
- ``harp_has_no_glitches``: is every Harp value in line with its
  neighbours?
- ``harp_evenly_spaced``: after fixing glitches, is every Harp step one
  frame interval?
- ``harp_matches_camera``: after fixing glitches, does each Harp step match
  the camera step?
- ``video_frame_count``: does the video have one frame per row?

Correction (:func:`correct_frame_times`):

1. Fix isolated Harp glitches.
2. No frames lost: row ``n`` was exposed by trigger ``n``, so the Harp
   column is right. It must be evenly spaced.
3. Frames lost: frame numbers and camera time must increase. Move each
   Harp value to the row of its exposure, estimate the rows the CSV has no
   trigger for from camera time (or read them from the trigger log), and
   require the result to match camera time.

Anything else raises ``ValueError``; a partial correction is never returned.

Example::

    timing = load_video_timing("behavior-videos/bottom_camera.csv")
    checks = check_video_timing(timing)  # one row per check
    timing_action(checks)                # what correction will do
    fixed = correct_video_timing(timing)  # adds harp_time, harp_source
"""

from pathlib import Path

import numpy as np
import pandas as pd

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


def _require(result):
    """Raise ValueError if a check failed."""
    if not result["passed"]:
        raise ValueError(f"{result['check']} failed: {result['message']}")


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


def check_harp_matches_camera(
    harp, camera_time, tolerance=STEP_TOLERANCE
) -> dict:
    """Does each Harp step match the camera step? Fix glitches first."""
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


def check_video_timing(timing, video_frame_count=None) -> pd.DataFrame:
    """Run every check on one camera.

    Parameters
    ----------
    timing : pandas.DataFrame
        From :func:`load_video_timing`.
    video_frame_count : int, optional
        Frames in the video file, for ``video_frame_count``.

    Returns
    -------
    pandas.DataFrame
        One row per check: ``check``, ``passed`` (None if skipped),
        ``count``, ``message`` and ``rows`` (first 100 offending rows).
    """
    frames = timing["frame_number"].to_numpy()
    camera = timing["camera_time"].to_numpy()
    harp = timing["harp_time_raw"].to_numpy()
    harp_fixed, _ = fix_glitches(harp)
    return pd.DataFrame(
        [
            check_no_frames_lost(frames),
            check_frame_numbers_increase(frames),
            check_camera_time_increases(camera),
            check_harp_has_no_glitches(harp),
            check_harp_evenly_spaced(harp_fixed),
            check_harp_matches_camera(harp_fixed, camera),
            check_video_frame_count(len(timing), video_frame_count),
        ]
    )


def timing_action(checks) -> str:
    """Say what :func:`correct_frame_times` will do, from the check table.

    One of ``use harp as written``, ``fix glitches``, ``re-index`` or
    ``refuse: <check>``. ``re-index`` can still be refused after
    correcting, if the result does not match camera time.
    """
    passed = dict(zip(checks["check"], checks["passed"]))
    if passed["no_frames_lost"]:
        if not passed["harp_evenly_spaced"]:
            return "refuse: harp_evenly_spaced"
        if not passed["harp_has_no_glitches"]:
            return "fix glitches"
        return "use harp as written"
    for check in ["frame_numbers_increase", "camera_time_increases"]:
        if not passed[check]:
            return f"refuse: {check}"
    return "re-index"


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


def correct_frame_times(
    frame_number, camera_time, trigger_times, tail_fit_window_s=600
):
    """Give each saved frame the time of the trigger that exposed it.

    Plain arrays in, plain arrays out; knows nothing about CSVs or Harp.
    Assumes one trigger per exposure, and that the first frame is the
    first exposure.

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
    triggers, glitches = fix_glitches(trigger_times)
    k = frames - frames[0]
    source = np.full(n, "original", dtype=object)

    if check_no_frames_lost(frames)["passed"]:
        times = triggers[:n]
        _require(check_harp_evenly_spaced(times))
    else:
        _require(check_frame_numbers_increase(frames))
        _require(check_camera_time_increases(camera))
        times = reindex_to_exposures(frames, triggers)
        source[k != np.arange(n)] = "reindexed"
        source[np.isnan(times)] = "estimated_camera_fit"
        times = estimate_missing_from_camera(times, camera, tail_fit_window_s)
        _require(check_harp_matches_camera(times, camera))
    source[np.isin(k, glitches)] = "glitch_interpolated"
    return times, source


def correct_video_timing(
    timing, trigger_times=None, tail_fit_window_s=600
) -> pd.DataFrame:
    """Return ``timing`` with corrected Harp time per frame.

    Runs :func:`correct_frame_times` on the CSV's Harp column, or on the
    Harp trigger log if ``trigger_times`` is given (after checking the log
    belongs to this CSV).

    Parameters
    ----------
    timing : pandas.DataFrame
        From :func:`load_video_timing`.
    trigger_times : array-like, optional
        From :func:`read_harp_trigger_log`.
    tail_fit_window_s : float
        Passed to :func:`correct_frame_times`.

    Returns
    -------
    pandas.DataFrame
        ``timing`` plus ``harp_time`` and ``harp_source``; rows read from a
        trigger log are labelled ``trigger_log``.
    """
    harp_raw = timing["harp_time_raw"].to_numpy()
    frames = timing["frame_number"].to_numpy()
    if trigger_times is None:
        triggers = harp_raw
    else:
        triggers = np.asarray(trigger_times, dtype="float64")
        n_needed = max(len(timing), int(frames[-1] - frames[0] + 1))
        if len(triggers) < n_needed:
            raise ValueError(
                f"Trigger log has {len(triggers)} events, fewer than the "
                f"{n_needed} exposures"
            )
        mismatch = np.abs(triggers[: len(timing)] - harp_raw).max()
        if mismatch > HARP_TICK_S:
            raise ValueError(
                f"Trigger log does not match the CSV's Harp column "
                f"(max difference {mismatch * 1e3:.3f} ms)"
            )
    harp_time, source = correct_frame_times(
        frames, timing["camera_time"].to_numpy(), triggers, tail_fit_window_s
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
        One row per camera: ``camera``, ``csv_path``, ``action`` (from
        :func:`timing_action`), ``failed_checks``, ``frames_lost`` and
        ``glitch_rows``. Unreadable CSVs get ``action``
        ``refuse: unreadable`` and the error text in ``error``.
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
            info.update({"action": "refuse: unreadable", "error": str(e)})
        else:
            by_name = checks.set_index("check")
            info.update(
                {
                    "action": timing_action(checks),
                    "failed_checks": checks.loc[
                        checks["passed"].eq(False), "check"
                    ].tolist(),
                    "frames_lost": by_name.loc["no_frames_lost", "count"],
                    "glitch_rows": by_name.loc["harp_has_no_glitches", "rows"],
                }
            )
        rows.append(info)
    return pd.DataFrame(rows)
