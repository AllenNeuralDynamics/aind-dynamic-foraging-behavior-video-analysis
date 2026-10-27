"""Frame-level steps: add time, mask, smooth, velocities.

All functions take and return tables with one row per frame (per keypoint)
and leave other columns alone. Time is whatever clock the caller supplies;
nothing here shifts it.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt


def add_time(pose: pd.DataFrame, frame_times):
    """Add ``time`` to the pose table from one time per video frame.

    Frame ``i`` gets ``frame_times[i]``. When the tracker output and the
    frame times differ in length, both are cut to the shorter, as frames
    past either end have no time or no pose.

    Parameters
    ----------
    pose : pandas.DataFrame
        Pose table (``frame`` column), e.g. from ``pose_io.read_pose``.
    frame_times : array-like
        Time of each video frame, on the caller's clock (e.g. corrected
        Harp time from ``video_timing_qc.correct_video_timing``).

    Returns
    -------
    pose : pandas.DataFrame
        Rows with a frame time, with ``time`` after ``frame``.
    info : dict
        ``n_frames_pose``, ``n_frames_timing`` and ``n_frames`` (kept).
    """
    frame_times = np.asarray(frame_times, dtype=float)
    n_pose = int(pose["frame"].max()) + 1 if len(pose) else 0
    n = min(n_pose, len(frame_times))
    out = pose[pose["frame"] < n].copy()
    out.insert(1, "time", frame_times[out["frame"].to_numpy()])
    info = {
        "n_frames_pose": n_pose,
        "n_frames_timing": int(len(frame_times)),
        "n_frames": int(n),
    }
    return out.reset_index(drop=True), info


def mask(pose: pd.DataFrame, min_likelihood) -> pd.DataFrame:
    """Mark points the tracker was unsure of.

    Adds ``valid`` (bool): finite ``x`` and ``y`` with ``likelihood`` at or
    above ``min_likelihood``. ``x`` and ``y`` become NaN where not valid.

    Parameters
    ----------
    pose : pandas.DataFrame
        Pose table.
    min_likelihood : float or dict
        One threshold, or ``{keypoint: threshold}`` (keypoints not listed
        keep every finite point).
    """
    out = pose.copy()
    if isinstance(min_likelihood, dict):
        thr = out["keypoint"].map(min_likelihood).fillna(-np.inf)
    else:
        thr = float(min_likelihood)
    valid = (
        np.isfinite(out["x"]) & np.isfinite(out["y"])
        & (out["likelihood"] >= thr)
    )
    out["valid"] = valid.to_numpy(bool)
    out.loc[~out["valid"], ["x", "y"]] = np.nan
    return out


def tracked_runs_by_time(time, valid, max_gap_s, fs_hz):
    """Runs of valid frames, bridging missing stretches of up to
    ``max_gap_s`` seconds.

    Returns a list of ``(first, last)`` row positions; both are valid frames.
    Frames dropped by the camera count as missing, since the test is on time.
    """
    idx = np.flatnonzero(np.asarray(valid, bool))
    if len(idx) == 0:
        return []
    t = np.asarray(time, float)[idx]
    new_run = np.r_[True, np.diff(t) > max_gap_s + 1.5 / fs_hz]
    starts = np.flatnonzero(new_run)
    ends = np.r_[starts[1:] - 1, len(idx) - 1]
    return list(zip(idx[starts], idx[ends]))


def _fill(t_known, v_known, t_new, kind):
    """Interpolate with ``kind`` when there are enough points, else lower."""
    n = len(t_known)
    if n == 1:
        return np.full(len(t_new), v_known[0])
    need = {"linear": 2, "quadratic": 3, "cubic": 4}.get(kind, 2)
    k = kind if n >= need else "linear"
    return interp1d(t_known, v_known, kind=k, assume_sorted=True)(t_new)


def _smooth_run(t, x, y, valid, fs_hz, sos, kind):
    """Fill, filter and differentiate one run; return values on its rows."""
    # Add samples where the camera dropped frames, so the filter sees an
    # evenly spaced signal.
    dt = np.diff(t)
    n_add = np.maximum(np.rint(dt * fs_hz).astype(int) - 1, 0)
    if n_add.any():
        extra = np.concatenate([
            t[i] + dt[i] * np.arange(1, k + 1) / (k + 1)
            for i, k in enumerate(n_add) if k
        ])
        grid = np.concatenate([t, extra])
        order = np.argsort(grid, kind="stable")
        grid = grid[order]
        is_row = order < len(t)
    else:
        grid, is_row = t, np.ones(len(t), bool)
    xs = _fill(t[valid], x[valid], grid, kind)
    ys = _fill(t[valid], y[valid], grid, kind)
    if sos is not None and len(grid) > 1:
        pad = min(3 * (2 * len(sos) + 1), len(grid) - 1)
        xs = sosfiltfilt(sos, xs, padlen=pad)
        ys = sosfiltfilt(sos, ys, padlen=pad)
    if len(grid) > 1:
        vx, vy = np.gradient(xs, grid), np.gradient(ys, grid)
    else:
        vx = vy = np.full(1, np.nan)
    return xs[is_row], ys[is_row], vx[is_row], vy[is_row]


def smooth(frames: pd.DataFrame, fs_hz, cutoff_hz=50.0, order=4,
           max_gap_s=0.02, kind="cubic") -> pd.DataFrame:
    """Smooth one keypoint's positions and add velocities.

    Each run of valid frames (missing stretches of up to ``max_gap_s``
    bridged) is filled across its gaps by interpolation, low-pass filtered
    forward and backward (Butterworth, no phase shift) and differentiated
    on time. Runs are processed separately, so nothing is interpolated
    across a longer gap and one run's edges do not leak into the next.
    Frames dropped by the camera are filled before filtering.

    Parameters
    ----------
    frames : pandas.DataFrame
        One keypoint, in frame order, with ``time, x, y`` and ``valid``
        (from ``mask``; NaN positions count as not valid if it is absent).
    fs_hz : float
        Nominal frame rate.
    cutoff_hz : float or None
        Low-pass cutoff; None (or at least ``fs_hz / 2``) skips filtering.
    order : int
        Filter order after the forward-backward pass (even; the
        Butterworth design has ``order / 2``).
    max_gap_s : float
        Longest missing stretch filled inside a run, in seconds.
    kind : str
        Interpolation for filling gaps ('linear', 'quadratic', 'cubic').

    Returns
    -------
    pandas.DataFrame
        Copy with smoothed ``x, y`` and added ``vx, vy, speed`` (pixels/s).
        All are NaN on frames that are not valid: filled values are used
        for filtering only.
    """
    if order % 2:
        raise ValueError(f"order must be even, got {order}")
    out = frames.copy()
    t = out["time"].to_numpy(float)
    x = out["x"].to_numpy(float)
    y = out["y"].to_numpy(float)
    valid = (out["valid"].to_numpy(bool) if "valid" in out
             else np.isfinite(x) & np.isfinite(y))
    sos = None
    if cutoff_hz is not None and cutoff_hz < fs_hz / 2:
        sos = butter(order // 2, cutoff_hz, fs=fs_hz, output="sos")
    cols = {c: np.full(len(out), np.nan) for c in ("x", "y", "vx", "vy")}
    for a, b in tracked_runs_by_time(t, valid, max_gap_s, fs_hz):
        sl = slice(a, b + 1)
        vals = _smooth_run(t[sl], x[sl], y[sl], valid[sl], fs_hz, sos,
                           kind)
        for c, v in zip(("x", "y", "vx", "vy"), vals):
            cols[c][sl] = np.where(valid[sl], v, np.nan)
    for c, v in cols.items():
        out[c] = v
    out["speed"] = np.hypot(out["vx"], out["vy"])
    return out


def relative_to(x, y, origin, flip_x=False, flip_y=False):
    """Positions relative to ``origin``, optionally mirrored.

    Returns ``(x - ox, y - oy)`` with either axis negated when asked, e.g.
    to undo a mirror in the camera path so +x points the same way in every
    session.
    """
    ox, oy = (float(c) for c in origin)
    rx = np.asarray(x, float) - ox
    ry = np.asarray(y, float) - oy
    return (-rx if flip_x else rx), (-ry if flip_y else ry)


def radial_velocity(frames: pd.DataFrame, origin) -> np.ndarray:
    """Velocity along the line from ``origin``: positive moving away
    (e.g. tongue protrusion from the jaw), negative moving back.

    ``(vx * dx + vy * dy) / |(dx, dy)|`` with ``(dx, dy)`` the position
    relative to ``origin``. NaN where any input is NaN or the point is at
    the origin.
    """
    dx, dy = relative_to(frames["x"], frames["y"], origin)
    r = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = (frames["vx"].to_numpy(float) * dx
              + frames["vy"].to_numpy(float) * dy) / r
    rv[r == 0] = np.nan
    return rv
