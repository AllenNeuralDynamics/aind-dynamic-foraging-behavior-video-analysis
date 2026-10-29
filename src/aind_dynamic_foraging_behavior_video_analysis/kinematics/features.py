"""Per-segment measurements, in named groups.

``describe`` computes the chosen groups for every segment and returns one
row per segment. A group is a function ``f(seg, context) -> dict``:

* ``seg`` : dict of numpy arrays over the segment's frames (``time, x, y,
  vx, vy, speed, valid`` and any other numeric frame columns). Positions
  are NaN on frames that are not valid.
* ``context`` : dict with ``fs_hz``, ``origin`` (reference point, or None),
  ``flip_x``, ``flip_y`` and ``options`` (``{group: {...}}``).

Built-in groups:

=========== ============================================================
``path``     ``path_length, displacement, peak_speed, mean_speed``
``excursion`` relative to the reference point: ``amplitude`` (largest
             distance from it), ``end_x, end_y`` (position at that
             moment), ``start_x, start_y``, ``angle_deg`` (direction of
             the end point), ``extent_x, extent_y`` (signed position
             farthest from the reference along each axis)
``extremes`` relative to the reference point: ``x_max, x_min, y_max,
             y_min`` (largest and smallest position along each axis),
             each with the other coordinate at that frame
             (``y_at_x_max``, ...), and ``y_at_extent_x, x_at_extent_y``
``outbound`` from the first frame to the moment of ``amplitude``:
             ``out_duration, out_path_length, out_peak_speed,
             out_mean_speed``
``spectrum`` power spectrum of one column (``options['spectrum']``:
             ``column`` default 'speed', ``band`` default (2, 20) Hz,
             ``min_duration_s`` default 0.5): ``dominant_hz,
             band_power, band_fraction``
=========== ============================================================

Positions are in pixels relative to the reference point; with no reference
point the groups that need one raise. Add a group with
``add_feature_group``.
"""
import numpy as np
import pandas as pd
import scipy.signal

from aind_dynamic_foraging_behavior_video_analysis.kinematics.signal import (
    relative_to,
)

FEATURE_GROUPS = {}


def add_feature_group(name):
    """Register ``f(seg, context) -> dict`` as feature group ``name``."""
    def register(fn):
        """Add ``fn`` to ``FEATURE_GROUPS``."""
        FEATURE_GROUPS[name] = fn
        return fn
    return register


def _valid_xy(seg):
    """Time, position and speed on valid frames."""
    v = seg["valid"]
    return seg["time"][v], seg["x"][v], seg["y"][v], seg["speed"][v]


def _speed_stats(speed):
    """Peak and mean speed, NaN when none is finite."""
    if np.isfinite(speed).any():
        return float(np.nanmax(speed)), float(np.nanmean(speed))
    return np.nan, np.nan


def _path_length(x, y):
    """Summed step lengths; NaN with fewer than two points."""
    if len(x) < 2:
        return np.nan
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def _relative(seg, context):
    """Valid frames with positions relative to the reference point."""
    if context.get("origin") is None:
        raise ValueError("this feature group needs a reference point")
    t, x, y, speed = _valid_xy(seg)
    rx, ry = relative_to(x, y, context["origin"], context.get("flip_x"),
                         context.get("flip_y"))
    return t, rx, ry, speed


@add_feature_group("path")
def path(seg, context):
    """Path length, straight-line displacement and speed."""
    _, x, y, _ = _valid_xy(seg)
    peak, mean = _speed_stats(seg["speed"])
    disp = (float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if len(x) >= 2
            else np.nan)
    return {"path_length": _path_length(x, y), "displacement": disp,
            "peak_speed": peak, "mean_speed": mean}


_EXCURSION_NAN = dict.fromkeys(
    ["amplitude", "end_x", "end_y", "start_x", "start_y", "angle_deg",
     "extent_x", "extent_y"], np.nan)


@add_feature_group("excursion")
def excursion(seg, context):
    """How far, and where, the segment reaches from the reference point."""
    _, rx, ry, _ = _relative(seg, context)
    if len(rx) == 0:
        return dict(_EXCURSION_NAN)
    r = np.hypot(rx, ry)
    i = int(np.argmax(r))
    return {
        "amplitude": float(r[i]),
        "end_x": float(rx[i]), "end_y": float(ry[i]),
        "start_x": float(rx[0]), "start_y": float(ry[0]),
        "angle_deg": float(np.degrees(np.arctan2(ry[i], rx[i]))),
        "extent_x": float(rx[np.argmax(np.abs(rx))]),
        "extent_y": float(ry[np.argmax(np.abs(ry))]),
    }


_EXTREMES_NAN = dict.fromkeys(
    ["x_max", "y_at_x_max", "x_min", "y_at_x_min",
     "y_max", "x_at_y_max", "y_min", "x_at_y_min",
     "y_at_extent_x", "x_at_extent_y"], np.nan)


@add_feature_group("extremes")
def extremes(seg, context):
    """The segment's extreme points along each axis, relative to the
    reference point, with the other coordinate at the same frame.

    ``x_max, y_at_x_max``, ``x_min, y_at_x_min``, ``y_max, x_at_y_max``,
    ``y_min, x_at_y_min``: the largest and smallest position along each
    axis. ``y_at_extent_x``, ``x_at_extent_y``: the other coordinate at the
    frames of ``excursion``'s ``extent_x`` / ``extent_y`` (farthest from
    the reference in either direction). The farthest point overall is
    ``excursion``'s ``amplitude, end_x, end_y``. Ties go to the first frame.

    Before 0.5.0, ``max_x_distance`` was ``abs(x_max)``, ``max_y_distance``
    ``abs(extent_y)``, and ``max_x_from_jaw_y`` / ``max_y_from_jaw_x`` were
    ``y_at_x_max`` / ``x_at_extent_y`` in absolute pixels (add the
    landmark).
    """
    _, rx, ry, _ = _relative(seg, context)
    if len(rx) == 0:
        return dict(_EXTREMES_NAN)
    ix_max, ix_min = int(np.argmax(rx)), int(np.argmin(rx))
    iy_max, iy_min = int(np.argmax(ry)), int(np.argmin(ry))
    ix_ext, iy_ext = int(np.argmax(np.abs(rx))), int(np.argmax(np.abs(ry)))
    return {
        "x_max": float(rx[ix_max]), "y_at_x_max": float(ry[ix_max]),
        "x_min": float(rx[ix_min]), "y_at_x_min": float(ry[ix_min]),
        "y_max": float(ry[iy_max]), "x_at_y_max": float(rx[iy_max]),
        "y_min": float(ry[iy_min]), "x_at_y_min": float(rx[iy_min]),
        "y_at_extent_x": float(ry[ix_ext]),
        "x_at_extent_y": float(rx[iy_ext]),
    }


@add_feature_group("outbound")
def outbound(seg, context):
    """The outward part: first frame to the frame farthest from the
    reference point. A segment that starts at its farthest point has
    ``out_duration`` 0 and ``out_path_length`` NaN."""
    t, rx, ry, speed = _relative(seg, context)
    if len(rx) == 0:
        return dict.fromkeys(["out_duration", "out_path_length",
                              "out_peak_speed", "out_mean_speed"], np.nan)
    i = int(np.argmax(np.hypot(rx, ry)))
    peak, mean = _speed_stats(speed[:i + 1])
    return {"out_duration": float(t[i] - t[0]),
            "out_path_length": _path_length(rx[:i + 1], ry[:i + 1]),
            "out_peak_speed": peak, "out_mean_speed": mean}


@add_feature_group("spectrum")
def spectrum(seg, context):
    """Dominant frequency and power in a band, for rhythmic movement.

    Missing frames are filled linearly; segments shorter than
    ``min_duration_s`` or less than half valid give NaN.
    """
    opts = {"column": "speed", "band": (2.0, 20.0), "min_duration_s": 0.5}
    opts.update(context.get("options", {}).get("spectrum", {}))
    fs = float(context["fs_hz"])
    sig = np.asarray(seg[opts["column"]], float)
    ok = np.isfinite(sig)
    nan = {"dominant_hz": np.nan, "band_power": np.nan,
           "band_fraction": np.nan}
    if len(sig) < opts["min_duration_s"] * fs or ok.mean() < 0.5:
        return nan
    pos = np.arange(len(sig))
    sig = np.interp(pos, pos[ok], sig[ok])
    f, p = scipy.signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs)))
    lo, hi = opts["band"]
    band = (f >= lo) & (f <= hi)
    df = f[1] - f[0] if len(f) > 1 else fs
    total = float(p[f > 0].sum() * df)
    if not band.any() or total <= 0:
        return nan
    power = float(p[band].sum() * df)
    return {"dominant_hz": float(f[band][np.argmax(p[band])]),
            "band_power": power, "band_fraction": power / float(total)}


def describe(frames: pd.DataFrame, intervals: pd.DataFrame, groups,
             context) -> pd.DataFrame:
    """Compute feature ``groups`` for every segment.

    Parameters
    ----------
    frames : pandas.DataFrame
        One keypoint's frames in frame order (``time, x, y, vx, vy, speed,
        valid``; from ``signal.smooth``).
    intervals : pandas.DataFrame
        ``start``, ``end`` rows of ``frames`` (inclusive).
    groups : sequence of str
        Names in ``FEATURE_GROUPS``.
    context : dict
        See the module docstring.

    Returns
    -------
    pandas.DataFrame
        One row per interval (same order and index), one column per
        feature.
    """
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; "
                         f"known: {sorted(FEATURE_GROUPS)}")
    cols = {c: frames[c].to_numpy() for c in frames.columns
            if pd.api.types.is_numeric_dtype(frames[c])
            or pd.api.types.is_bool_dtype(frames[c])}
    cols["valid"] = frames["valid"].to_numpy(bool)
    for c in ("time", "x", "y", "vx", "vy", "speed"):
        cols[c] = frames[c].to_numpy(float)
    rows = []
    for a, b in zip(intervals["start"], intervals["end"]):
        seg = {c: v[a:b + 1] for c, v in cols.items()}
        row = {}
        for g in groups:
            row.update(FEATURE_GROUPS[g](seg, context))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=_column_names(groups, cols, context),
                            index=intervals.index, dtype=float)
    return pd.DataFrame(rows, index=intervals.index)


def _column_names(groups, cols, context):
    """Feature column names, from each group run on an empty segment, so
    that a kind with no segments still has its columns."""
    empty = {c: v[:0] for c, v in cols.items()}
    names = []
    for g in groups:
        try:
            names.extend(FEATURE_GROUPS[g](empty, context))
        except Exception:  # noqa: BLE001 - a group that cannot run empty
            pass
    return names
