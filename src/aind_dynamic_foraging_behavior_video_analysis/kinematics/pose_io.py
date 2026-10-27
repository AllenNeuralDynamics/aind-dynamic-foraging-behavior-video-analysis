"""Read pose-tracking output into one long table.

Every reader returns the same table, one row per frame per keypoint::

    frame  keypoint  x  y  likelihood

``frame`` is the row of the tracker output, which is the frame of the video
it was run on. Coordinates are image pixels as the tracker wrote them. Time
is joined on later (``signal.add_time``), so the readers know nothing about
clocks or tasks.

Supported: Lightning Pose and DeepLabCut CSV and DeepLabCut HDF5 (one
animal), SLEAP analysis HDF5 (one track), an ndx-pose ``PoseEstimation``
object, and plain arrays (``pose_table``) for anything else.
"""
from pathlib import Path

import numpy as np
import pandas as pd

POSE_COLUMNS = ["frame", "keypoint", "x", "y", "likelihood"]


def pose_table(xy, likelihood, keypoints) -> pd.DataFrame:
    """Build the pose table from arrays.

    Parameters
    ----------
    xy : array-like, shape (n_frames, n_keypoints, 2)
        Pixel positions.
    likelihood : array-like, shape (n_frames, n_keypoints), or None
        Tracker confidence per point; None fills 1.0.
    keypoints : sequence of str
        One name per keypoint.

    Returns
    -------
    pandas.DataFrame
        ``frame, keypoint, x, y, likelihood``, sorted by keypoint then frame.
    """
    xy = np.asarray(xy, dtype=float)
    keypoints = [str(k) for k in keypoints]
    if xy.ndim != 3 or xy.shape[2] != 2 or xy.shape[1] != len(keypoints):
        raise ValueError(
            "xy must have shape (n_frames, n_keypoints, 2) with one name "
            f"per keypoint; got {xy.shape} and {len(keypoints)} names"
        )
    n_frames, n_kp = xy.shape[:2]
    if likelihood is None:
        likelihood = np.ones((n_frames, n_kp))
    likelihood = np.asarray(likelihood, dtype=float)
    if likelihood.shape != (n_frames, n_kp):
        raise ValueError(
            f"likelihood must have shape {(n_frames, n_kp)}, "
            f"got {likelihood.shape}"
        )
    return pd.DataFrame(
        {
            "frame": np.tile(np.arange(n_frames), n_kp),
            "keypoint": np.repeat(keypoints, n_frames),
            "x": xy[:, :, 0].T.ravel(),
            "y": xy[:, :, 1].T.ravel(),
            "likelihood": likelihood.T.ravel(),
        }
    )


def _from_dlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pose table from a DLC-style frame with (scorer, bodypart, coord)
    column levels."""
    cols = df.columns
    if cols.nlevels == 4:  # multi-animal DLC: keep the first individual
        first = cols.get_level_values(1)[0]
        df = df.xs(first, axis=1, level=1)
        cols = df.columns
    bodyparts = list(dict.fromkeys(cols.get_level_values(-2)))
    coords = cols.get_level_values(-1)
    lik_name = "likelihood" if "likelihood" in set(coords) else None
    n = len(df)
    xy = np.empty((n, len(bodyparts), 2))
    lik = np.ones((n, len(bodyparts)))
    for j, bp in enumerate(bodyparts):
        sub = df.xs(bp, axis=1, level=-2)
        sub.columns = sub.columns.get_level_values(-1)
        xy[:, j, 0] = pd.to_numeric(sub["x"], errors="coerce")
        xy[:, j, 1] = pd.to_numeric(sub["y"], errors="coerce")
        if lik_name:
            lik[:, j] = pd.to_numeric(sub[lik_name], errors="coerce")
    return pose_table(xy, lik, bodyparts)


def read_dlc_csv(path) -> pd.DataFrame:
    """Read a Lightning Pose or DeepLabCut predictions CSV (3 header rows:
    scorer, bodyparts, coords; first column is the frame index)."""
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0, low_memory=False)
    return _from_dlc_frame(df)


def read_dlc_h5(path) -> pd.DataFrame:
    """Read a DeepLabCut predictions HDF5 file (needs ``tables``)."""
    return _from_dlc_frame(pd.read_hdf(path))


def read_sleap_h5(path, track=0) -> pd.DataFrame:
    """Read a SLEAP analysis HDF5 file (needs ``h5py``); one track."""
    import h5py

    with h5py.File(path, "r") as f:
        tracks = f["tracks"][:]  # (n_tracks, 2, n_nodes, n_frames)
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f["node_names"][:]]
        scores = (f["point_scores"][:] if "point_scores" in f else None)
    xy = np.transpose(tracks[track], (2, 1, 0))  # (n_frames, n_nodes, 2)
    lik = None if scores is None else scores[track].T
    return pose_table(xy, lik, names)


def read_ndx_pose(pose_estimation) -> pd.DataFrame:
    """Pose table from an ndx-pose ``PoseEstimation`` object (already read
    from NWB). Its own timestamps are ignored: time comes from the video's
    frame times, as for every other reader."""
    series = list(pose_estimation.pose_estimation_series.values())
    if not series:
        raise ValueError("PoseEstimation has no series")
    names = [s.name for s in series]
    xy = np.stack([np.asarray(s.data[:])[:, :2] for s in series], axis=1)
    lik = np.stack(
        [np.ones(len(xy)) if s.confidence is None
         else np.asarray(s.confidence[:]) for s in series],
        axis=1,
    )
    return pose_table(xy, lik, names)


def read_pose(path, fmt=None) -> pd.DataFrame:
    """Read a tracker output file into the pose table.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.
    fmt : {'dlc_csv', 'dlc_h5', 'sleap_h5'}, optional
        Format; guessed from the extension when omitted (``.csv`` is
        Lightning Pose / DeepLabCut; ``.h5`` is SLEAP if the name contains
        "analysis", else DeepLabCut).
    """
    path = Path(path)
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            fmt = "dlc_csv"
        elif suffix in (".h5", ".hdf5"):
            fmt = "sleap_h5" if "analysis" in path.name else "dlc_h5"
        else:
            raise ValueError(f"cannot guess the pose format of {path}")
    readers = {
        "dlc_csv": read_dlc_csv,
        "dlc_h5": read_dlc_h5,
        "sleap_h5": read_sleap_h5,
    }
    if fmt not in readers:
        raise ValueError(f"fmt must be one of {sorted(readers)}, got {fmt!r}")
    return readers[fmt](path)
