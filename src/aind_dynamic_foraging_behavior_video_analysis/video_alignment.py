"""Standalone helpers for aligning behavior video frames to behavior/session time.

This module is intentionally dependency-light (``pandas`` only) and decoupled from
the kinematics pipeline so it can be reused on its own. It answers a single
question: given the behavior video acquisition CSV and the time of the first go
cue, how do you convert event times from other data streams (spikes, fiber
photometry, behavior events) into seconds within the recorded video so you can
clip around them?

Three clocks are used throughout, with fixed names:

``behavior_time``
    Harp / reference time -- absolute acquisition seconds. The video CSV
    ``Behav_Time`` column, NWB ``goCue_start_time``, spike times and FIP times all
    live on this clock.
``video_time``
    Seconds within the recorded video file (first frame = 0.0). This is what
    ``ffmpeg -ss`` expects.
``session_time``
    Seconds relative to the first go cue (first go cue = 0.0).

Let ``first_frame_behavior_time`` be the ``behavior_time`` of the first video
frame and ``first_go_cue_time`` be the ``behavior_time`` of the first go cue. The
core relationships are::

    offset       = first_go_cue_time - first_frame_behavior_time   # video_time of the 1st go cue
    video_time   = behavior_time     - first_frame_behavior_time   # behavior_time event -> video_time
    video_time   = session_time      + offset                      # session_time event  -> video_time
    session_time = behavior_time     - first_go_cue_time           # behavior_time        -> session_time

Example
-------
First frame at ``behavior_time`` 100.0 s, first go cue at 105.25 s, and a spike of
interest at 112.0 s::

    >>> offset = compute_video_session_offset("metadata.csv", 105.25)  # -> 5.25
    >>> behavior_time_to_video_time(112.0, 100.0)                       # -> 12.0
    >>> session_time_to_video_time(6.75, offset)                        # -> 12.0
"""

from typing import List, Optional

import pandas as pd

# Column layout of the Bonsai / AIND behavior video acquisition CSV. The file is
# written without a header row, so the order matters.
DEFAULT_COLUMNS = ["Behav_Time", "Frame", "Camera_Time", "Gain", "Exposure"]


def get_first_frame_behavior_time(
    video_csv_path,
    time_column: str = "Behav_Time",
    columns: Optional[List[str]] = None,
) -> float:
    """Return the behavior_time of the first video frame.

    Parameters
    ----------
    video_csv_path : str or pathlib.Path
        Path to the behavior video acquisition CSV (e.g. ``metadata.csv``). The
        file is expected to be headerless with column order ``columns``.
    time_column : str, optional
        Name of the column holding the behavior_time (harp) timestamp per frame.
        Defaults to ``"Behav_Time"``.
    columns : list of str, optional
        Column names to assign to the headerless CSV, in order. Defaults to
        :data:`DEFAULT_COLUMNS`.

    Returns
    -------
    float
        The behavior_time of the first frame, i.e. ``Behav_Time.iloc[0]``.
    """
    if columns is None:
        columns = DEFAULT_COLUMNS
    video_csv = pd.read_csv(video_csv_path, names=columns)
    return float(video_csv[time_column].iloc[0])


def compute_video_session_offset(
    video_csv_path,
    first_go_cue_time: float,
    time_column: str = "Behav_Time",
    columns: Optional[List[str]] = None,
) -> float:
    """Return the offset between session_time and video_time.

    The offset is ``first_go_cue_time - first_frame_behavior_time`` -- equivalently,
    the ``video_time`` (seconds into the video) at which the first go cue occurs.
    Add it to a ``session_time`` to get a ``video_time``; subtract it to go back.

    Parameters
    ----------
    video_csv_path : str or pathlib.Path
        Path to the behavior video acquisition CSV.
    first_go_cue_time : float
        The behavior_time of the first go cue (e.g.
        ``float(nwb.trials["goCue_start_time"][0])``). Passed as a plain float so
        this module stays independent of any NWB / pynwb dependency.
    time_column : str, optional
        Name of the behavior_time column in the CSV. Defaults to ``"Behav_Time"``.
    columns : list of str, optional
        Column names for the headerless CSV. Defaults to :data:`DEFAULT_COLUMNS`.

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

    Use this for events already on the raw behavior clock (e.g. spike or FIP times
    pulled straight from NWB) so you do not have to re-zero them to the go cue.

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
