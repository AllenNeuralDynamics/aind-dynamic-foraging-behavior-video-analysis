"""Work with segments and windows as intervals of time or frames.

Movements, bouts, trials and fixed time bins are all intervals, so one set
of helpers answers questions such as "which movement was each lick in",
"how many movements in each trial" and "first movement after each cue".
"""
import numpy as np
import pandas as pd


def label_frames(n_frames, intervals, ids):
    """Per frame, the id of the interval covering it (NA outside any).

    Parameters
    ----------
    n_frames : int
        Number of frames.
    intervals : pandas.DataFrame
        ``start``, ``end`` rows (inclusive), not overlapping.
    ids : array-like
        One id per interval.
    """
    out = pd.array([pd.NA] * n_frames, dtype="Int64")
    starts = intervals["start"].to_numpy(np.int64)
    ends = intervals["end"].to_numpy(np.int64)
    if len(starts) == 0:
        return out
    lengths = ends - starts + 1
    rows = np.repeat(starts, lengths) + (
        np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths,
                                             lengths))
    out[rows] = np.repeat(np.asarray(ids, np.int64), lengths)
    return out


def interval_of(times, starts, ends, closed="both"):
    """Per time, the row of the interval containing it, -1 if none.

    Intervals must be sorted by start and not overlap. ``closed`` is
    'both' (``start <= t <= end``) or 'left' (``start <= t < end``, for
    windows that tile time, such as trials or bins).
    """
    t = np.asarray(times, float)
    s = np.asarray(starts, float)
    e = np.asarray(ends, float)
    row = np.searchsorted(s, t, side="right") - 1
    ok = row >= 0
    inside = np.zeros(len(t), bool)
    if closed == "both":
        inside[ok] = t[ok] <= e[row[ok]]
    elif closed == "left":
        inside[ok] = t[ok] < e[row[ok]]
    else:
        raise ValueError(f"closed must be 'both' or 'left', got {closed!r}")
    return np.where(ok & inside, row, -1)


def closest_interval(times, starts, ends):
    """Per time, the closest interval and the signed time to it.

    Intervals must be sorted by start and not overlap.

    Returns
    -------
    row : numpy.ndarray of int
        Row of the closest interval (-1 if there are none). Inside an
        interval, that interval; between two, the nearer one (the earlier
        on a tie).
    offset : numpy.ndarray of float
        0 inside the interval; negative before its start (``t - start``),
        positive after its end (``t - end``).
    """
    t = np.asarray(times, float)
    s = np.asarray(starts, float)
    e = np.asarray(ends, float)
    if len(s) == 0:
        return np.full(len(t), -1, np.int64), np.full(len(t), np.nan)
    prev = np.searchsorted(s, t, side="right") - 1
    nxt = prev + 1
    after = np.where(prev >= 0, t - e[np.maximum(prev, 0)], np.inf)
    before = np.where(nxt < len(s), t - s[np.minimum(nxt, len(s) - 1)],
                      -np.inf)
    inside = after <= 0
    use_prev = inside | (after <= -before)
    row = np.where(use_prev, prev, nxt).astype(np.int64)
    offset = np.where(inside, 0.0, np.where(use_prev, after, before))
    return row, offset


def nearest(times, targets, tolerance):
    """Per time, the index of the nearest target within ``tolerance``,
    -1 if none."""
    t = np.asarray(times, float)
    g = np.asarray(targets, float)
    if len(g) == 0:
        return np.full(len(t), -1, np.int64)
    order = np.argsort(g, kind="stable")
    gs = g[order]
    if len(gs) == 1:
        pick = np.zeros(len(t), np.int64)
    else:
        hi = np.clip(np.searchsorted(gs, t), 1, len(gs) - 1)
        lo = hi - 1
        pick = np.where(np.abs(gs[lo] - t) <= np.abs(gs[hi] - t), lo, hi)
    ok = np.abs(gs[pick] - t) <= tolerance
    return np.where(ok, order[pick], -1)


def time_bins(start, stop, width):
    """Windows of ``width`` seconds from ``start`` to ``stop``:
    ``bin, start_time, end_time``."""
    edges = np.arange(float(start), float(stop) + 1e-12, float(width))
    if len(edges) < 2 or edges[-1] < stop:
        edges = np.append(edges, edges[-1] + width if len(edges) else stop)
    return pd.DataFrame({"bin": np.arange(len(edges) - 1),
                         "start_time": edges[:-1], "end_time": edges[1:]})


_COUNT = "count"
# 'first'/'last' mean the first/last item in time, even if its value is
# missing (pandas' own skip missing values).
_IN_TIME_ORDER = {"first": lambda s: s.iloc[0], "last": lambda s: s.iloc[-1]}


def summarize_by_window(items, windows, time_col, how,
                        window_start="start_time", window_end="end_time"):
    """One row per window, summarizing the items whose time falls in it.

    Parameters
    ----------
    items : pandas.DataFrame
        E.g. movements, with ``time_col`` (e.g. ``start_time``).
    windows : pandas.DataFrame
        E.g. trials or ``time_bins``, with ``window_start`` and
        ``window_end``; sorted, not overlapping, ``start <= t < end``.
    time_col : str
        Item time used to place each item in a window.
    how : dict
        ``{new_column: (item_column, summary)}``. ``summary`` is 'count'
        (``item_column`` may be None), 'first' or 'last' (in time order),
        any pandas aggregation name ('mean', 'median', 'max', 'sum', ...),
        or a function of a Series.

    Returns
    -------
    pandas.DataFrame
        ``windows`` with the new columns; counts are 0 for empty windows,
        other summaries NA.
    """
    items = items.sort_values(time_col, kind="stable")
    row = interval_of(items[time_col], windows[window_start],
                      windows[window_end], closed="left")
    inside = items[row >= 0].assign(_window=row[row >= 0])
    grouped = inside.groupby("_window", sort=True)
    out = windows.copy()
    n = len(windows)
    for name, (col, summary) in how.items():
        if summary == _COUNT:
            counts = grouped.size()
            vals = np.zeros(n, np.int64)
            vals[counts.index.to_numpy()] = counts.to_numpy()
            out[name] = vals
            continue
        fn = _IN_TIME_ORDER.get(summary, summary)
        res = grouped[col].agg(fn).reindex(range(n))
        out[name] = res.set_axis(out.index)
    return out
