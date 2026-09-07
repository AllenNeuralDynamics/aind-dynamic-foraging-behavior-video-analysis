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
