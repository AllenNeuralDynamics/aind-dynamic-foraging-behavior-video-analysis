"""Fixed reference points (jaw, spouts, ...) as one position per session."""
import numpy as np
import pandas as pd


def landmarks(pose: pd.DataFrame, keypoints, stat="median") -> pd.DataFrame:
    """One position per keypoint over the session.

    Parameters
    ----------
    pose : pandas.DataFrame
        Pose table; uses ``valid`` when present (from ``signal.mask``),
        else every finite point.
    keypoints : sequence of str
        Keypoints to summarize; ones absent from ``pose`` get NaN.
    stat : {'median', 'mean'}
        Median (default) is robust to brief mistracking.

    Returns
    -------
    pandas.DataFrame
        ``keypoint, x, y, n_valid, stat``.
    """
    if stat not in ("median", "mean"):
        raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")
    rows = []
    for kp in keypoints:
        sub = pose[pose["keypoint"] == kp]
        ok = (sub["valid"].to_numpy(bool) if "valid" in sub
              else np.isfinite(sub["x"]) & np.isfinite(sub["y"]))
        pts = sub.loc[ok, ["x", "y"]]
        x, y = (getattr(pts, stat)() if len(pts) else (np.nan, np.nan))
        rows.append({"keypoint": kp, "x": float(x), "y": float(y),
                     "n_valid": int(len(pts)), "stat": stat})
    return pd.DataFrame(rows, columns=["keypoint", "x", "y", "n_valid",
                                       "stat"])


def origin(landmarks_table: pd.DataFrame, keypoint):
    """``(x, y)`` of one landmark; raises if it is missing or NaN."""
    row = landmarks_table[landmarks_table["keypoint"] == keypoint]
    if row.empty or not np.isfinite(row[["x", "y"]].to_numpy()).all():
        raise ValueError(f"no position for landmark {keypoint!r}")
    return float(row["x"].iloc[0]), float(row["y"].iloc[0])
