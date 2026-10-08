"""Helpers for aligning behavior video frames to behavior/session time.

This module is intentionally dependency-light (``numpy`` and ``pandas``) and
decoupled from the kinematics pipeline so it can be reused on its own. It
answers a single question: given the behavior video acquisition CSV and the
time of the first go cue, how do you convert event times from other data
streams (spikes, fiber photometry, behavior events) into seconds within the
recorded video so you can clip around them?

Three clocks are used throughout, with fixed names:

``behavior_time``
    Harp / reference time -- absolute acquisition seconds. The video CSV
    ``Behav_Time`` column, NWB ``goCue_start_time``, spike times and FIP
    times all live on this clock.
``video_time``
    Seconds within the recorded video file (first frame = 0.0). This is what
    ``ffmpeg -ss`` expects.
``session_time``
    Seconds relative to the first go cue (first go cue = 0.0).

Let ``first_frame_behavior_time`` be the ``behavior_time`` of the first video
frame and ``first_go_cue_time`` be the ``behavior_time`` of the first go
cue. The core relationships are::

    # video_time of the 1st go cue
    offset       = first_go_cue_time - first_frame_behavior_time
    # behavior_time event -> video_time
    video_time   = behavior_time     - first_frame_behavior_time
    # session_time event -> video_time
    video_time   = session_time      + offset
    # behavior_time -> session_time
    session_time = behavior_time     - first_go_cue_time

Caveat: dropped frames
----------------------
These conversions assume Harp time advances one frame interval per saved
video frame. That holds for the raw CSV Harp column even when frames were
dropped (the acquisition workflow pairs frames with triggers in arrival
order), but then events land on the wrong frames. After correcting Harp
time with :mod:`video_timing_qc`, map events to frames with
``numpy.searchsorted`` on the corrected ``harp_time`` instead of by
subtraction.

Trial times
-----------
:func:`read_trial_times` reads trial start, go cue and trial end straight
from the raw session JSON (``behavior/<subject>_<datetime>.json``), the
same Harp-clock values ``TransferToNWB.bonsai_to_nwb`` writes into the NWB
trials table, so no NWB is needed. :func:`behavior_time_to_frame_index`
puts behavior times on video frames through a corrected timing table.
:func:`task_frame_window` combines them: the frames from the first trial
start to the last trial end. :func:`event_frame_ranges` gives a window of
frames around each event, for clips (``video_clips``) or for slicing any
per-frame signal.

Example
-------
First frame at ``behavior_time`` 100.0 s, first go cue at 105.25 s, and a
spike of interest at 112.0 s::

    >>> compute_video_session_offset("metadata.csv", 105.25)  # -> 5.25
    >>> behavior_time_to_video_time(112.0, 100.0)  # -> 12.0
    >>> session_time_to_video_time(6.75, 5.25)  # -> 12.0
"""

import json
import urllib.request
from typing import List, Optional

import numpy as np
import pandas as pd

# Column layout of the Bonsai / AIND behavior video acquisition CSV. Two
# layouts exist in the wild; :func:`read_video_csv` tells them apart. These
# names apply to the Old/flat layout, which is written without a header row,
# so the order matters.
DEFAULT_COLUMNS = ["Behav_Time", "Frame", "Camera_Time", "Gain", "Exposure"]

# Name of the behavior_time column in each layout, in resolution order:
# Old/flat calls it "Behav_Time", New/AIND calls it "ReferenceTime". Both
# hold absolute harp seconds, on the same clock as the NWB event times.
TIME_COLUMN_ALIASES = ["Behav_Time", "ReferenceTime"]


def _csv_has_header(video_csv_path) -> bool:
    """Return True if the CSV's first line is a header rather than data.

    Both layouts put the behavior_time value in the first field, which parses
    as a float in a data row and does not in a header row.
    """
    with open(video_csv_path) as f:
        first_line = f.readline()
    if not first_line.strip():
        raise ValueError(f"Video CSV is empty: {video_csv_path}")
    try:
        float(first_line.split(",")[0].strip())
    except ValueError:
        return True
    return False


def read_video_csv(
    video_csv_path,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Read a behavior video acquisition CSV written in either AIND layout.

    Priority:
      1) New/AIND: written with a header row (e.g.
         ``<CameraName>/metadata.csv``, whose columns are
         ``ReferenceTime,CameraFrameNumber,CameraFrameTime``). The file's
         own header is used, so columns keep the recorded names.
      2) Old/flat: written without a header row (e.g.
         ``bottom_camera.csv``). ``columns`` is applied, in order.

    The layout is detected from the file's contents, not from its name. The
    compression pipeline passes the acquisition CSV through unchanged, so a
    ``metadata.csv`` holds whatever that session's rig wrote and the
    filename does not imply a schema.

    Parameters
    ----------
    video_csv_path : str or pathlib.Path
        Path to the behavior video acquisition CSV.
    columns : list of str, optional
        Column names to assign if the file is headerless, in order.
        Defaults to :data:`DEFAULT_COLUMNS`. Ignored for a file with a
        header, which carries its own names.
    nrows : int, optional
        Read only the first ``nrows`` data rows. Worth passing when only the
        first frame is needed; these files run to millions of rows.

    Returns
    -------
    pandas.DataFrame
        The CSV contents, under the recorded column names for a New/AIND
        file and under ``columns`` for an Old/flat one.
    """
    if _csv_has_header(video_csv_path):
        return pd.read_csv(video_csv_path, nrows=nrows)
    if columns is None:
        columns = DEFAULT_COLUMNS
    return pd.read_csv(video_csv_path, names=columns, nrows=nrows)


def _resolve_time_column(
    video_csv: pd.DataFrame,
    time_column: Optional[str] = None,
) -> str:
    """Return the name of the behavior_time column in ``video_csv``.

    Parameters
    ----------
    video_csv : pandas.DataFrame
        A frame returned by :func:`read_video_csv`.
    time_column : str, optional
        An explicit column name, returned unchanged if the frame carries it.
        When omitted, the first of :data:`TIME_COLUMN_ALIASES` present in
        the frame is used.

    Returns
    -------
    str
        The behavior_time column name.

    Raises
    ------
    KeyError
        If ``time_column`` is absent, or no alias matches.
    """
    if time_column is not None:
        if time_column not in video_csv.columns:
            raise KeyError(
                f"Column {time_column!r} not in video CSV; found "
                f"{list(video_csv.columns)}"
            )
        return time_column
    for alias in TIME_COLUMN_ALIASES:
        if alias in video_csv.columns:
            return alias
    raise KeyError(
        f"No behavior_time column in video CSV; looked for "
        f"{TIME_COLUMN_ALIASES}, found {list(video_csv.columns)}"
    )


def get_first_frame_behavior_time(
    video_csv_path,
    time_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> float:
    """Return the behavior_time of the first video frame.

    Parameters
    ----------
    video_csv_path : str or pathlib.Path
        Path to the behavior video acquisition CSV (e.g. ``metadata.csv``
        or ``bottom_camera.csv``). Either layout is accepted; see
        :func:`read_video_csv`.
    time_column : str, optional
        Name of the column holding the behavior_time (harp) timestamp per
        frame. Defaults to auto-detection via :func:`_resolve_time_column`.
    columns : list of str, optional
        Column names to assign if the CSV is headerless, in order. Defaults
        to :data:`DEFAULT_COLUMNS`.

    Returns
    -------
    float
        The behavior_time of the first frame, i.e. ``Behav_Time.iloc[0]``.
    """
    video_csv = read_video_csv(video_csv_path, columns=columns, nrows=1)
    time_column = _resolve_time_column(video_csv, time_column)
    return float(video_csv[time_column].iloc[0])


def compute_video_session_offset(
    video_csv_path,
    first_go_cue_time: float,
    time_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> float:
    """Return the offset between session_time and video_time.

    The offset is ``first_go_cue_time - first_frame_behavior_time`` --
    equivalently, the ``video_time`` (seconds into the video) at which the
    first go cue occurs. Add it to a ``session_time`` to get a
    ``video_time``; subtract it to go back.

    Parameters
    ----------
    video_csv_path : str or pathlib.Path
        Path to the behavior video acquisition CSV.
    first_go_cue_time : float
        The behavior_time of the first go cue (e.g.
        ``float(nwb.trials["goCue_start_time"][0])``). Passed as a plain
        float so this module stays independent of any NWB / pynwb dependency.
    time_column : str, optional
        Name of the behavior_time column in the CSV. Defaults to
        auto-detection via :func:`_resolve_time_column`.
    columns : list of str, optional
        Column names to assign if the CSV is headerless. Defaults to
        :data:`DEFAULT_COLUMNS`.

    Returns
    -------
    float
        The session-to-video offset, in seconds.
    """
    first_frame_behavior_time = get_first_frame_behavior_time(
        video_csv_path, time_column=time_column, columns=columns
    )
    return first_go_cue_time - first_frame_behavior_time


def session_time_to_video_time(session_times, offset):
    """Convert session_time to video_time by adding the offset.

    Parameters
    ----------
    session_times : float, numpy.ndarray, or pandas.Series
        Times relative to the first go cue.
    offset : float
        The session-to-video offset from :func:`compute_video_session_offset`.

    Returns
    -------
    Same type as ``session_times``
        ``session_times + offset``.
    """
    return session_times + offset


def video_time_to_session_time(video_times, offset):
    """Convert video_time to session_time by subtracting the offset.

    Parameters
    ----------
    video_times : float, numpy.ndarray, or pandas.Series
        Times in seconds within the video file.
    offset : float
        The session-to-video offset from :func:`compute_video_session_offset`.

    Returns
    -------
    Same type as ``video_times``
        ``video_times - offset``.
    """
    return video_times - offset


def behavior_time_to_video_time(behavior_times, first_frame_behavior_time):
    """Convert behavior_time (harp) to video_time.

    Use this for events already on the raw behavior clock (e.g. spike or
    FIP times pulled straight from NWB) so you do not have to re-zero them
    to the go cue.

    Parameters
    ----------
    behavior_times : float, numpy.ndarray, or pandas.Series
        Times on the behavior_time (harp / reference) clock.
    first_frame_behavior_time : float
        The behavior_time of the first video frame, from
        :func:`get_first_frame_behavior_time`.

    Returns
    -------
    Same type as ``behavior_times``
        ``behavior_times - first_frame_behavior_time``.
    """
    return behavior_times - first_frame_behavior_time


def video_time_to_behavior_time(video_times, first_frame_behavior_time):
    """Convert video_time back to behavior_time (harp).

    Parameters
    ----------
    video_times : float, numpy.ndarray, or pandas.Series
        Times in seconds within the video file.
    first_frame_behavior_time : float
        The behavior_time of the first video frame, from
        :func:`get_first_frame_behavior_time`.

    Returns
    -------
    Same type as ``video_times``
        ``video_times + first_frame_behavior_time``.
    """
    return video_times + first_frame_behavior_time


def read_trial_times(behavior_json_path) -> pd.DataFrame:
    """Read per-trial Harp times from the raw foraging session JSON.

    Uses the fields ``TransferToNWB.bonsai_to_nwb`` copies into the NWB
    trials table, so the values equal the NWB's ``start_time``,
    ``goCue_start_time`` and ``stop_time``: ``B_TrialStartTimeHarp``,
    ``B_TrialEndTimeHarp``, and ``B_GoCueTimeHarp`` (older files) or
    ``B_GoCueTimeSoundCard``.

    Parameters
    ----------
    behavior_json_path : str or pathlib.Path
        Local path or http(s) URL of ``behavior/<subject>_<datetime>.json``.

    Returns
    -------
    pandas.DataFrame
        One row per trial: ``start_time``, ``goCue_start_time``,
        ``stop_time`` (Harp seconds, the behavior_time clock).

    Raises
    ------
    ValueError
        If the file has no Harp trial times (older sessions recorded CPU
        times only, which are not on the video's clock).
    """
    path = str(behavior_json_path)
    if path.startswith(("http://", "https://")):
        with urllib.request.urlopen(path) as response:
            obj = json.load(response)
    else:
        with open(path) as f:
            obj = json.load(f)
    if not obj.get("B_TrialEndTimeHarp"):
        raise ValueError(f"No Harp trial times in {behavior_json_path}")
    go_cue_field = (
        "B_GoCueTimeHarp"
        if "B_GoCueTimeHarp" in obj
        else "B_GoCueTimeSoundCard"
    )
    n_trials = len(obj["B_TrialEndTime"])
    return pd.DataFrame(
        {
            "start_time": obj["B_TrialStartTimeHarp"][:n_trials],
            "goCue_start_time": obj[go_cue_field][:n_trials],
            "stop_time": obj["B_TrialEndTimeHarp"][:n_trials],
        },
        dtype="float64",
    )


def behavior_time_to_frame_index(behavior_times, harp_time):
    """Return the first video frame at or after each behavior time.

    Parameters
    ----------
    behavior_times : float or array-like
        Harp (behavior_time) seconds.
    harp_time : array-like
        Harp time of every video frame, increasing: the ``harp_time``
        column of ``video_timing_qc.correct_video_timing``. Do not use the
        raw CSV column when frames were dropped; its rows carry the times
        of earlier triggers (see the module docstring).

    Returns
    -------
    numpy.ndarray or int
        Frame indices (= CSV rows); ``len(harp_time)`` for a time after the
        last frame.
    """
    return np.searchsorted(np.asarray(harp_time), behavior_times, side="left")


def event_frame_ranges(event_times, harp_time, before, after):
    """Return the video frames around each event, one row per event.

    The window of an event at ``t`` is the frames whose Harp time lies in
    ``[t - before, t + after)``: :func:`behavior_time_to_frame_index` at
    both ends. Frame indices are CSV rows, so the ranges slice any per-frame
    signal row-aligned with the video CSV (motion energy, pose predictions,
    latents) as well as the video itself (``video_clips.cut_clips``).

    Pass the **corrected** ``harp_time`` (``video_timing_qc.
    correct_video_timing``). The raw CSV column is wrong for whole sessions
    with dropped frames (by minutes late in some sessions), so a window
    placed by it shows the wrong moment. Across a drop a window has fewer
    frames and still spans the requested time.

    Parameters
    ----------
    event_times : float or array-like
        Event times on the Harp (behavior_time) clock.
    harp_time : array-like
        Harp time of every video frame, increasing.
    before, after : float
        Seconds before and after each event.

    Returns
    -------
    pandas.DataFrame
        ``event_time``; ``start_frame`` (first frame at or after
        ``event_time - before``); ``n_frames`` (frames up to, not
        including, the first frame at or after ``event_time + after``);
        ``in_video`` (False when the window runs past either end of the
        video, or the event time is NaN).
    """
    harp_time = np.asarray(harp_time, dtype=float)
    event_times = np.atleast_1d(np.asarray(event_times, dtype=float))
    start = behavior_time_to_frame_index(event_times - before, harp_time)
    end = behavior_time_to_frame_index(event_times + after, harp_time)
    if len(harp_time):
        in_video = (event_times - before >= harp_time[0]) & (
            event_times + after <= harp_time[-1]
        )
    else:
        in_video = np.zeros(len(event_times), dtype=bool)
    return pd.DataFrame(
        {
            "event_time": event_times,
            "start_frame": start.astype("int64"),
            "n_frames": (end - start).astype("int64"),
            "in_video": in_video,
        }
    )


def task_frame_window(behavior_json, video_csv, trigger_log=None):
    """Return the frames from the first trial start to the last trial end.

    Trial times come from the raw session JSON (:func:`read_trial_times`).
    They are put on frames through the Harp time of each CSV row, found the
    way the kinematics pipeline finds it: the timing QC correction
    (``video_timing_qc.correct_video_timing``), with the trigger log when
    one is given and readable (an unreadable log is ignored).

    The correction is strict because per-frame analysis needs every frame's
    time; it refuses a camera for errors of a frame or two (a Harp step off
    by more than half a frame, a log one event longer than the frame
    numbers span: 65 of 602 cameras in the survey of
    ``VIDEO_QUALITY_QC_PLAN.md``). The window only needs to be right to a
    few frames, so when the correction refuses, it falls back to:

    1. the trigger log by frame number, ``log[frame_number -
       first_frame_number]`` (clipped to the log), right whether or not
       frames were lost;
    2. the raw Harp column, when no frames were lost (row ``n`` is then
       trigger ``n``).

    A running maximum is applied so a glitch cannot reorder the times.

    Parameters
    ----------
    behavior_json : str or pathlib.Path
        ``behavior/<subject>_<datetime>.json`` (path or URL).
    video_csv : str or pathlib.Path
        The camera's video CSV (local), either layout.
    trigger_log : str or pathlib.Path, optional
        ``behavior/raw.harp/BehaviorEvents/Event_94.bin``.

    Returns
    -------
    (start, end)
        ``[start, end)`` video frame indices (= CSV rows).

    Raises
    ------
    ValueError
        If the JSON has no Harp trial times, or the correction is refused,
        frames were lost and there is no readable trigger log.
    """
    # video_timing_qc imports this module.
    from aind_dynamic_foraging_behavior_video_analysis import (
        video_timing_qc as vtq,
    )

    trials = read_trial_times(behavior_json)
    timing = vtq.load_video_timing(video_csv)
    log = None
    if trigger_log is not None:
        try:
            log = vtq.read_harp_trigger_log(trigger_log)
        except (ValueError, OSError):
            pass
    try:
        harp = vtq.correct_video_timing(timing, trigger_times=log)
        harp = harp["harp_time"].to_numpy()
    except ValueError:
        if log is not None:
            exposure = timing["frame_number"].to_numpy()
            harp = log[np.clip(exposure - exposure[0], 0, len(log) - 1)]
        else:
            checks = vtq.check_video_timing(timing).set_index("check")
            if checks.loc["no_frames_lost", "passed"] is not True:
                raise
            harp = timing["harp_time_raw"].to_numpy()
    harp = np.maximum.accumulate(harp)
    start = behavior_time_to_frame_index(trials["start_time"].min(), harp)
    end = behavior_time_to_frame_index(trials["stop_time"].max(), harp)
    return int(start), int(end)
