"""QC and correction of per-frame Harp times in behavior video CSVs.

Each behavior video CSV has one row per saved video frame, and row ``i`` is
frame ``i`` of the video. Each row carries three values:

``harp_time``
    Harp seconds (``Behav_Time`` / ``ReferenceTime``), the behavior clock.
``frame_number``
    The camera's exposure counter (``Frame`` / ``CameraFrameNumber``). It
    counts every exposure, including frames the host later lost.
``camera_time``
    The camera's hardware clock (``Camera_Time`` / ``CameraFrameTime``),
    written in nanoseconds and converted to seconds here.

The acquisition workflow pairs frames with Harp trigger times in arrival
order (``rx:Zip``), so after a dropped frame every later row carries the
Harp time of an earlier trigger. Separately, a single Harp value is
sometimes wrong (typically ~983 ms early). This module detects both, and
corrects them. Isolated Harp glitches get the midpoint of their
neighbours; then it depends on whether frames were lost, i.e. whether the
first and last frame numbers span more exposures than there are rows:

- **No frames lost:** row ``i`` was exposed by trigger ``i``, so the Harp
  column is already right. Frame numbers and camera time are not used
  (they can be corrupted while Harp is fine). The Harp column must be
  evenly spaced; if it is not (e.g. a Harp clock step), the session is
  refused.
- **Frames lost:** each row gets the trigger time of its own exposure,
  ``k = frame_number - frame_number[0]``. Rows whose trigger was never
  written to the CSV (the last ones) are estimated from camera time, or
  read from the Harp trigger log (``Event_94.bin``) if one is given. The
  result must agree with camera time step by step.

Only Harp time is ever changed. Corrections that fail their checks raise
``ValueError`` rather than return partially corrected times.

Example
-------
::

    timing = load_video_timing("behavior-videos/bottom_camera.csv")
    qc = check_video_timing(timing)
    fixed = correct_video_timing(timing)  # adds harp_time, harp_source

The correction itself, :func:`correct_frame_times`, takes plain arrays
(frame numbers, camera times, trigger times) and knows nothing about CSV
layouts or Harp; the other functions adapt the acquisition files to it.
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


def load_video_timing(video_csv_path) -> pd.DataFrame:
    """Read a behavior video CSV into Harp, frame-number and camera columns.

    Parameters
    ----------
    video_csv_path : str or pathlib.Path
        A video CSV in either layout (see
        :func:`video_alignment.read_video_csv`).

    Returns
    -------
    pandas.DataFrame
        One row per saved video frame, with columns ``harp_time_raw``
        (seconds), ``frame_number`` (int64) and ``camera_time`` (seconds).

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
    """Read the camera trigger times from a Harp ``Event_94.bin`` file.

    Each message is 13 bytes: type, length, address, port, payload type,
    seconds (uint32), ticks (uint16, 32 us each), payload (uint8), checksum.
    Register 94 is ``Camera1Frame`` in the Harp Behavior ``device.yml``;
    both cameras are triggered from it, so one log serves every camera.
    Gives the same times as ``harp.io.read(path).index`` from the
    ``harp-python`` package, without the dependency.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``behavior/raw.harp/BehaviorEvents/Event_94.bin``.

    Returns
    -------
    numpy.ndarray
        Trigger times in Harp seconds, one per event, in file order.
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


def frame_interval(times) -> float:
    """Return the typical step of an evenly spaced time series.

    The mean of the steps within 25% of the median step. Harp steps
    alternate by one 32 us tick around the true interval, so the median
    alone is off by half a tick; outliers (glitches, drops) are excluded.

    Parameters
    ----------
    times : array-like
        Times in seconds.

    Returns
    -------
    float
        Frame interval in seconds.
    """
    step = np.diff(np.asarray(times, dtype="float64"))
    median = np.median(step)
    return float(step[np.abs(step - median) < 0.25 * median].mean())


def find_glitch_rows(trigger_times, ifi, threshold=0.5) -> np.ndarray:
    """Return isolated bad values in an evenly spaced trigger sequence.

    Index ``r`` is a glitch if it is more than ``threshold * ifi`` from the
    midpoint of ``r - 1`` and ``r + 1``, and those two are ``2 * ifi``
    apart (within the same tolerance). Runs of two or more bad values are
    not returned. The CSV's Harp column qualifies as a trigger sequence:
    under arrival-order pairing it is evenly spaced whether or not frames
    were dropped.

    Parameters
    ----------
    trigger_times : array-like
        Trigger times in seconds.
    ifi : float
        Frame interval in seconds.
    threshold : float
        Tolerance as a fraction of ``ifi``.

    Returns
    -------
    numpy.ndarray
        Indices of the glitches.
    """
    t = np.asarray(trigger_times, dtype="float64")
    tol = threshold * ifi
    off_midpoint = np.abs(t[1:-1] - (t[:-2] + t[2:]) / 2) > tol
    neighbours_agree = np.abs(t[2:] - t[:-2] - 2 * ifi) <= tol
    rows = np.flatnonzero(off_midpoint & neighbours_agree) + 1
    # Drop adjacent detections: a run of bad values is not a glitch.
    isolated = ~np.isin(rows - 1, rows) & ~np.isin(rows + 1, rows)
    return rows[isolated]


def find_irregular_steps(trigger_times, ifi, threshold=0.5) -> np.ndarray:
    """Return indices whose step from the previous value is not ``ifi``.

    Parameters
    ----------
    trigger_times : array-like
        Trigger times in seconds, glitches already fixed.
    ifi : float
        Frame interval in seconds.
    threshold : float
        Allowed deviation as a fraction of ``ifi``.

    Returns
    -------
    numpy.ndarray
        Index ``r`` for each step ``r - 1 -> r`` off by more than
        ``threshold * ifi``.
    """
    step = np.diff(np.asarray(trigger_times, dtype="float64"))
    return np.flatnonzero(np.abs(step - ifi) > threshold * ifi) + 1


def check_video_timing(timing, threshold=0.5, video_frame_count=None) -> dict:
    """Check one camera's per-frame timestamps and classify them.

    Parameters
    ----------
    timing : pandas.DataFrame
        From :func:`load_video_timing`.
    threshold : float
        Clock-disagreement threshold as a fraction of the frame interval.
        0.5 flags every dropped frame; the legacy QC used 2.
    video_frame_count : int, optional
        Number of frames in the video file. If given and different from the
        number of CSV rows, the class is ``transcode_mismatch``.

    Returns
    -------
    dict
        JSON-serialisable summary. ``n_frames_dropped`` is exposures
        (from the first and last frame numbers) minus rows. ``qc_class``
        is the first match of:

        - ``transcode_mismatch``: video and CSV frame counts differ.
        - ``frame_order_error``: more rows than exposures, or frames were
          dropped and frame numbers or camera time step back or repeat.
        - ``harp_irregular``: no frames dropped, but the Harp column is
          not evenly spaced after fixing glitches (e.g. a Harp clock
          step). Not correctable.
        - ``camera_metadata_error``: no frames dropped and Harp is evenly
          spaced, but frame numbers or camera time are inconsistent.
          Correctable: the Harp column is used as written.
        - ``harp_glitch``: no frames dropped; isolated bad Harp rows only
          (correctable).
        - ``frame_drops``: frames dropped (correctable).
        - ``ok``.
    """
    harp = timing["harp_time_raw"].to_numpy()
    frames = timing["frame_number"].to_numpy()
    cam = timing["camera_time"].to_numpy()
    n_rows = len(timing)
    if n_rows < 3:
        raise ValueError(f"Need at least 3 rows, got {n_rows}")

    # Harp is evenly spaced whether or not frames were dropped, and does
    # not depend on the camera metadata, which can be corrupted.
    ifi = frame_interval(harp)
    frame_step = np.diff(frames)
    gap_rows = np.flatnonzero(frame_step > 1) + 1
    frame_order_rows = np.flatnonzero(frame_step <= 0) + 1
    harp_backward_rows = np.flatnonzero(np.diff(harp) < 0) + 1
    camera_backward_rows = np.flatnonzero(np.diff(cam) <= 0) + 1

    clock_diff = np.abs(np.diff(harp) - np.diff(cam))
    clock_flag_rows = np.flatnonzero(clock_diff > threshold * ifi) + 1
    glitch_rows = find_glitch_rows(harp, ifi, threshold)
    # A glitch row flags its own step and the step out of it.
    explained = np.concatenate([gap_rows, glitch_rows, glitch_rows + 1])
    unexplained_rows = np.setdiff1d(
        np.union1d(clock_flag_rows, harp_backward_rows), explained
    )

    harp_fixed = harp.copy()
    harp_fixed[glitch_rows] = (
        harp[glitch_rows - 1] + harp[glitch_rows + 1]
    ) / 2
    irregular_rows = find_irregular_steps(harp_fixed, ifi, threshold)

    n_exposures = int(frames[-1] - frames[0] + 1)
    n_lost = n_exposures - n_rows
    camera_metadata_bad = bool(
        len(frame_order_rows)
        or len(camera_backward_rows)
        or len(gap_rows)
        or len(unexplained_rows)
    )
    if video_frame_count is not None and video_frame_count != n_rows:
        qc_class = "transcode_mismatch"
    elif n_lost < 0 or (
        n_lost > 0 and (len(frame_order_rows) or len(camera_backward_rows))
    ):
        qc_class = "frame_order_error"
    elif n_lost > 0:
        qc_class = "frame_drops"
    elif len(irregular_rows):
        qc_class = "harp_irregular"
    elif camera_metadata_bad:
        qc_class = "camera_metadata_error"
    elif len(glitch_rows):
        qc_class = "harp_glitch"
    else:
        qc_class = "ok"

    return {
        "qc_class": qc_class,
        "n_rows": n_rows,
        "n_exposures": n_exposures,
        "ifi_s": ifi,
        "fps": 1 / ifi,
        "n_frame_gaps": len(gap_rows),
        "n_frames_dropped": n_lost,
        "first_gap_row": int(gap_rows[0]) if len(gap_rows) else None,
        "n_frame_order_errors": len(frame_order_rows),
        "n_harp_backward": len(harp_backward_rows),
        "n_camera_backward": len(camera_backward_rows),
        "glitch_rows": glitch_rows.tolist(),
        "harp_irregular_rows": irregular_rows[:100].tolist(),
        "n_clock_flags": len(clock_flag_rows),
        "n_unexplained_flags": len(unexplained_rows),
        "unexplained_rows": unexplained_rows[:100].tolist(),
        "clock_diff_p99_ms": float(np.percentile(clock_diff, 99) * 1e3),
        "clock_diff_max_ms": float(clock_diff.max() * 1e3),
        "clock_slip_frames": float(
            ((cam[-1] - cam[0]) - (harp[-1] - harp[0])) / ifi
        ),
        "video_frame_count": video_frame_count,
        "threshold": threshold,
    }


def correct_frame_times(
    frame_number,
    camera_time,
    trigger_times,
    threshold=0.5,
    tail_fit_window_s=600,
):
    """Assign each saved frame the time of the trigger that exposed it.

    Hardware-agnostic core of the correction. First, isolated glitches in
    ``trigger_times`` are replaced by the midpoint of their neighbours.
    Then, with ``lost`` = exposures (from the first and last frame
    numbers) minus frames:

    - ``lost == 0``: frame ``i`` was exposed by trigger ``i``, so
      ``times = trigger_times[:n]``. Frame numbers and camera time are not
      used beyond the count. The times must be evenly spaced (see
      :func:`find_irregular_steps`).
    - ``lost > 0``: frame ``i`` was exposed by trigger
      ``k = frame_number[i] - frame_number[0]``, so
      ``times[i] = trigger_times[k]`` wherever ``trigger_times`` covers
      ``k``; the rest are estimated from a linear fit of time on
      ``camera_time`` over the last ``tail_fit_window_s`` of re-indexed
      frames. Frame numbers and camera time must strictly increase, and
      the result must track ``camera_time`` (see
      :func:`post_check_failures`).
    - ``lost < 0``: refused.

    Assumes one trigger per exposure and that the first frame is the first
    exposure.

    Parameters
    ----------
    frame_number : array-like of int
        Camera exposure counter per saved frame.
    camera_time : array-like of float
        Camera clock per saved frame, in seconds.
    trigger_times : array-like of float
        Trigger times in order, in seconds on the target clock, starting
        with the first frame's trigger. May be shorter than the number of
        exposures (e.g. only as many as saved frames).
    threshold : float
        Tolerance as a fraction of the frame interval, for glitch detection
        and the post-checks.
    tail_fit_window_s : float
        Seconds of camera time used for the tail fit.

    Returns
    -------
    times : numpy.ndarray
        Corrected time per saved frame.
    source : numpy.ndarray of str
        Per frame: ``original`` (unchanged from ``trigger_times[i]``),
        ``reindexed``, ``glitch_interpolated`` or ``estimated_camera_fit``.

    Raises
    ------
    ValueError
        If there are more frames than exposures; if frames were lost and
        frame numbers or camera times do not strictly increase; or if the
        corrected times fail their checks.
    """
    frames = np.asarray(frame_number)
    cam = np.asarray(camera_time, dtype="float64")
    triggers = np.asarray(trigger_times, dtype="float64").copy()
    n_frames = len(frames)
    if len(cam) != n_frames or n_frames < 3:
        raise ValueError("Need matching frame_number and camera_time, >= 3")
    if len(triggers) < n_frames:
        raise ValueError("Need at least one trigger per frame")
    ifi = frame_interval(triggers[:n_frames])
    glitches = find_glitch_rows(triggers, ifi, threshold)
    triggers[glitches] = (triggers[glitches - 1] + triggers[glitches + 1]) / 2
    k = frames - frames[0]  # trigger index of each frame's exposure
    lost = int(k[-1]) + 1 - n_frames
    source = np.full(n_frames, "original", dtype=object)

    if lost < 0:
        raise ValueError(
            f"{-lost} more frames than exposures; frame numbers unusable"
        )
    if lost == 0:
        times = triggers[:n_frames].copy()
        source[glitches[glitches < n_frames]] = "glitch_interpolated"
        irregular = find_irregular_steps(times, ifi, threshold)
        if len(irregular):
            raise ValueError(
                f"Trigger times not evenly spaced at {len(irregular)} rows "
                f"(first at row {irregular[0]}), e.g. a Harp clock step"
            )
        return times, source

    if not (np.diff(frames) > 0).all() or not (np.diff(cam) > 0).all():
        raise ValueError(
            "Frames were lost, and frame numbers or camera times do not "
            "increase, so lost frames cannot be located"
        )
    times = np.full(n_frames, np.nan)
    known = k < len(triggers)
    times[known] = triggers[k[known]]
    source[known & (k != np.arange(n_frames))] = "reindexed"
    tail = ~known
    if tail.any():
        last_known = cam[known][-1]
        fit = known & (cam >= last_known - tail_fit_window_s)
        slope, intercept = np.polyfit(cam[fit] - last_known, times[fit], 1)
        times[tail] = intercept + slope * (cam[tail] - last_known)
        source[tail] = "estimated_camera_fit"
    source[np.isin(k, glitches)] = "glitch_interpolated"

    failures = post_check_failures(times, cam, ifi, threshold)
    if failures:
        raise ValueError(
            "Correction failed post-checks: " + "; ".join(failures)
        )
    return times, source


def correct_video_timing(
    timing, trigger_times=None, threshold=0.5, tail_fit_window_s=600
) -> pd.DataFrame:
    """Return a copy of ``timing`` with corrected Harp time per frame.

    Wraps :func:`correct_frame_times`. The trigger sequence is the CSV's
    own Harp column (row ``n`` holds the ``n``-th trigger), or the Harp
    trigger log if ``trigger_times`` is given, after checking the log
    belongs to this CSV.

    Parameters
    ----------
    timing : pandas.DataFrame
        From :func:`load_video_timing`.
    trigger_times : array-like, optional
        Harp trigger times from :func:`read_harp_trigger_log`. When given,
        every row's Harp time comes from the log.
    threshold : float
        Passed to :func:`correct_frame_times`.
    tail_fit_window_s : float
        Passed to :func:`correct_frame_times`.

    Returns
    -------
    pandas.DataFrame
        ``timing`` plus ``harp_time`` (seconds) and ``harp_source``, one of
        ``original``, ``glitch_interpolated``, ``reindexed``,
        ``estimated_camera_fit`` or ``trigger_log``.

    Raises
    ------
    ValueError
        If the trigger log does not cover or match the CSV, or
        :func:`correct_frame_times` raises.
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
        frames,
        timing["camera_time"].to_numpy(),
        triggers,
        threshold=threshold,
        tail_fit_window_s=tail_fit_window_s,
    )
    if trigger_times is not None:
        source[source != "glitch_interpolated"] = "trigger_log"
    fixed = timing.copy()
    fixed["harp_time"] = harp_time
    fixed["harp_source"] = source
    return fixed


def post_check_failures(times, camera_time, ifi, threshold=0.5) -> list:
    """Return why corrected frame times fail to track camera time, if so.

    Parameters
    ----------
    times, camera_time : numpy.ndarray
        Corrected time and camera time per frame, in seconds.
    ifi : float
        Frame interval in seconds.
    threshold : float
        Allowed step disagreement as a fraction of ``ifi``.

    Returns
    -------
    list of str
        One message per failed check; empty if all pass.
    """
    step = np.diff(times)
    bad_steps = np.abs(step - np.diff(camera_time)) > threshold * ifi
    t = camera_time - camera_time[0]
    residual = times - np.polyval(np.polyfit(t, times, 1), t)
    failures = []
    if not (step > 0).all():
        failures.append(f"{(step <= 0).sum()} non-increasing steps")
    if bad_steps.any():
        failures.append(
            f"{bad_steps.sum()} steps disagree with camera time (first at "
            f"row {np.flatnonzero(bad_steps)[0] + 1})"
        )
    if np.abs(residual).max() > 1e-3:
        failures.append("residual from camera-time fit over 1 ms")
    return failures


def check_session(behavior_videos_path, threshold=0.5) -> pd.DataFrame:
    """Check every camera CSV in a ``behavior-videos`` folder.

    Finds Old/flat CSVs (``<camera>.csv``) and New/AIND ones
    (``<Camera>/metadata.csv``).

    Parameters
    ----------
    behavior_videos_path : str or pathlib.Path
        The session's ``behavior-videos`` folder.
    threshold : float
        Passed to :func:`check_video_timing`.

    Returns
    -------
    pandas.DataFrame
        One row per camera: ``camera``, ``csv_path`` and the fields of
        :func:`check_video_timing`. A CSV that cannot be read gets
        ``qc_class`` ``unreadable`` and the error text in ``error``.
    """
    folder = Path(behavior_videos_path)
    csv_paths = sorted(folder.glob("*.csv")) + sorted(
        folder.glob("*/metadata.csv")
    )
    rows = []
    for csv_path in csv_paths:
        camera = (
            csv_path.parent.name
            if csv_path.name == "metadata.csv"
            else (csv_path.stem)
        )
        info = {"camera": camera, "csv_path": str(csv_path)}
        try:
            info.update(
                check_video_timing(load_video_timing(csv_path), threshold)
            )
        except ValueError as e:
            info.update({"qc_class": "unreadable", "error": str(e)})
        rows.append(info)
    return pd.DataFrame(rows)
