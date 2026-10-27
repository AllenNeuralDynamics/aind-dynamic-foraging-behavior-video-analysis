"""Split one keypoint's frames into segments (movements, submovements,
bouts).

Every method returns an *intervals table*, one row per segment, in time
order:

* ``start``, ``end`` : int, first and last row (inclusive) of the segment
  in the frames it was given.
* ``within`` : int, row of the enclosing segment in the ``within`` table
  (only for methods run inside other segments).
* ``direction`` : int, +1 / -1 / 0 (``zero_crossings`` only).

Inputs are plain arrays over one keypoint's frames in frame order, so the
methods work on any signal. Choose by what the body part does:

* ``tracked_runs`` - the part is only tracked while it moves (a tongue out
  of the mouth): a segment is a run of tracked frames.
* ``above_threshold`` - the part is always visible (jaw, nose, paw): a
  segment is a stretch where a signal such as speed is high.
* ``zero_crossings`` / ``speed_minima`` - split segments further at
  changes of direction or dips in speed.
* ``group_into_bouts`` - group segments that follow each other closely.
"""
import numpy as np
import pandas as pd
import scipy.signal


def _table(starts, ends, **extra):
    """Intervals table from arrays."""
    data = {"start": np.asarray(starts, np.int64),
            "end": np.asarray(ends, np.int64)}
    for k, v in extra.items():
        data[k] = np.asarray(v, np.int64)
    return pd.DataFrame(data)


def _runs(mask, max_gap_frames=0):
    """First and last index of runs of True, bridging gaps of up to
    ``max_gap_frames`` False values."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if len(idx) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    new = np.r_[True, np.diff(idx) - 1 > max_gap_frames]
    s = np.flatnonzero(new)
    e = np.r_[s[1:] - 1, len(idx) - 1]
    return idx[s], idx[e]


def tracked_runs(valid, max_gap_frames=10):
    """Segments are runs of valid frames.

    A segment starts at a valid frame, continues across up to
    ``max_gap_frames`` missing frames, and ends at its last valid frame.
    """
    s, e = _runs(valid, max_gap_frames)
    return _table(s, e)


def above_threshold(signal, on, off=None, min_frames=1, max_gap_frames=0):
    """Segments are stretches where ``signal`` is high.

    A segment is a stretch above ``off`` that reaches ``on`` somewhere
    (two thresholds, so noise around one level does not chatter).
    Stretches separated by up to ``max_gap_frames`` frames are joined, then
    segments shorter than ``min_frames`` are dropped. NaN counts as low.

    Parameters
    ----------
    signal : array-like
        E.g. speed.
    on : float
        Level a segment must reach.
    off : float, optional
        Level below which a segment ends (default ``on``).
    min_frames : int
        Shortest segment kept.
    max_gap_frames : int
        Longest low stretch inside a segment.
    """
    x = np.asarray(signal, float)
    off = on if off is None else off
    if off > on:
        raise ValueError(f"off ({off}) must not exceed on ({on})")
    with np.errstate(invalid="ignore"):
        above_off = x > off
        above_on = x >= on
    s, e = _runs(above_off, 0)
    if len(s):
        csum = np.r_[0, np.cumsum(above_on)]
        hits = csum[e + 1] - csum[s]
        keep = hits > 0
        s, e = s[keep], e[keep]
    if len(s) and max_gap_frames > 0:
        join = np.r_[False, s[1:] - e[:-1] - 1 <= max_gap_frames]
        first = np.flatnonzero(~join)
        last = np.r_[first[1:] - 1, len(s) - 1]
        s, e = s[first], e[last]
    keep = (e - s + 1) >= min_frames
    return _table(s[keep], e[keep])


def _within_codes(n, within):
    """Per-frame row of the enclosing segment, -1 outside any."""
    codes = np.full(n, -1, np.int64)
    if within is None:
        return codes
    for j, (a, b) in enumerate(zip(within["start"], within["end"])):
        codes[a:b + 1] = j
    return codes


def _merge_short_segments(starts, lengths, blocks, min_samples):
    """Mask over segments: True where a segment still starts a new one
    after merging short segments into a neighbor in the same block.

    A short segment merges into the previous one; the first segment of a
    block merges into the next. A block with a single short segment keeps
    it.
    """
    keep = np.ones(len(starts), dtype=bool)
    acc = 0
    for j in range(len(starts)):
        if j == 0 or blocks[j] != blocks[j - 1]:
            acc = lengths[j]
        elif acc < min_samples or lengths[j] < min_samples:
            keep[j] = False
            acc += lengths[j]
        else:
            acc = lengths[j]
    return keep


def _segment_directions(seg, vals, deadband):
    """Per segment (seg >= 0, in order): the sign of the sum of its values
    outside the deadband (its net motion along the signal), or 0 if none is
    outside it.

    The sum, not the largest sample, so a short blip merged into a segment
    cannot flip its direction; values inside the deadband are left out, so
    an unmerged segment's direction is the sign it was split on.
    """
    kept = seg >= 0
    seg_k, vals_k = seg[kept], vals[kept]
    first = np.flatnonzero(np.r_[True, seg_k[1:] != seg_k[:-1]])
    strong = np.where(np.abs(vals_k) > deadband, vals_k, 0.0)
    total = np.add.reduceat(strong, first)
    return np.sign(total).astype(np.int64)


def _split_starts(vals, hard, block, method, deadband, min_prominence,
                  max_minimum_speed):
    """Where a new piece starts, before short pieces are handled."""
    start = hard.copy()
    if method == "zero_crossing":
        sign = np.where(np.abs(vals) > deadband, np.sign(vals), np.nan)
        # Frames inside the deadband keep the previous sign; the leading
        # ones take the first sign outside it.
        sign = (pd.Series(sign).groupby(block).ffill()
                .groupby(block).bfill().fillna(0).to_numpy())
        start[1:] |= (sign[1:] != sign[:-1]) & ~hard[1:]
    else:
        height = None if max_minimum_speed is None else -max_minimum_speed
        block_starts = np.flatnonzero(hard)
        for b0, b1 in zip(block_starts, np.append(block_starts[1:],
                                                  len(vals))):
            peaks, _ = scipy.signal.find_peaks(
                -vals[b0:b1], prominence=min_prominence, height=height)
            start[b0 + peaks] = True
    return start


def _handle_short(start, hard, block, vals, method, deadband, min_samples,
                  short):
    """Merge or drop pieces shorter than ``min_samples``; return the piece
    number per valid sample (-1 if dropped)."""
    starts = np.flatnonzero(start)
    lengths = np.diff(np.append(starts, len(vals)))
    seg = np.cumsum(start) - 1
    if min_samples <= 1:
        return seg
    if short == "drop":
        long_enough = lengths >= min_samples
        return np.where(long_enough[seg], np.cumsum(long_enough)[seg] - 1,
                        -1)
    keep = _merge_short_segments(starts, lengths, block[starts], min_samples)
    start = start.copy()
    start[starts[~keep]] = False
    if method == "zero_crossing":
        # A merged blip leaves same-sign neighbors: fuse them.
        seg = np.cumsum(start) - 1
        starts = np.flatnonzero(start)
        direction = _segment_directions(seg, vals, deadband)
        fuse = np.r_[False, direction[1:] == direction[:-1]]
        start[starts[fuse & ~hard[starts]]] = False
    return np.cumsum(start) - 1


def _split(values, within, method, max_gap_frames, deadband, min_prominence,
           max_minimum_speed, min_samples, short):
    """Shared core of ``zero_crossings`` and ``speed_minima``."""
    if short not in ("merge", "drop"):
        raise ValueError(f"short must be 'merge' or 'drop', got {short!r}")
    values = np.asarray(values, dtype=float)
    n_frames = len(values)
    if within is None:
        s, e = _runs(np.isfinite(values), max_gap_frames)
        within = _table(s, e)
    parent = _within_codes(n_frames, within)
    idx = np.flatnonzero(np.isfinite(values) & (parent >= 0))
    if len(idx) == 0:
        return _table([], [], within=[], direction=[])
    vals, par = values[idx], parent[idx]
    # Hard starts: new parent, or a gap too long to bridge.
    hard = np.ones(len(idx), dtype=bool)
    hard[1:] = (par[1:] != par[:-1]) | (np.diff(idx) - 1 > max_gap_frames)
    block = np.cumsum(hard) - 1
    start = _split_starts(vals, hard, block, method, deadband,
                          min_prominence, max_minimum_speed)
    seg = _handle_short(start, hard, block, vals, method, deadband,
                        min_samples, short)
    kept = seg >= 0
    if not kept.any():
        return _table([], [], within=[], direction=[])
    k_idx, k_seg = idx[kept], seg[kept]
    first = np.flatnonzero(np.r_[True, k_seg[1:] != k_seg[:-1]])
    last = np.r_[first[1:] - 1, len(k_seg) - 1]
    direction = (_segment_directions(seg, vals, deadband)
                 if method == "zero_crossing"
                 else np.zeros(len(first), np.int64))
    return _table(k_idx[first], k_idx[last], within=par[kept][first],
                  direction=direction)


def zero_crossings(values, within=None, deadband=0.0, max_gap_frames=3,
                   min_samples=1, short="merge"):
    """Split where a signed velocity changes sign.

    E.g. radial velocity (``signal.radial_velocity``) splits each tongue
    movement into protrusion (+1) and retraction (-1).

    Parameters
    ----------
    values : array-like
        Signed signal per frame, NaN where missing.
    within : intervals table, optional
        Segments to split; pieces never cross their boundaries and frames
        outside them get none. Default: runs of finite values, bridging
        gaps of up to ``max_gap_frames``.
    deadband : float
        Values with ``|v| <= deadband`` keep the previous sign, so noise
        near zero adds no splits.
    max_gap_frames : int
        Missing stretches of up to this many frames are bridged; a longer
        one always ends a piece.
    min_samples : int
        Fewest valid frames in a piece (1: no minimum).
    short : {'merge', 'drop'}
        Shorter pieces merge into the previous one (the next one if first),
        or are dropped.

    Returns
    -------
    pandas.DataFrame
        Intervals table with ``within`` and ``direction`` (sign of the
        piece's summed values outside the deadband; 0 if there are none).
    """
    return _split(values, within, "zero_crossing", max_gap_frames, deadband,
                  None, None, min_samples, short)


def speed_minima(values, within=None, min_prominence=None,
                 max_minimum_speed=None, max_gap_frames=3, min_samples=1,
                 short="merge"):
    """Split at local minima of speed (``scipy.signal.find_peaks`` on the
    negated signal).

    ``min_prominence`` sets how deep a dip must be; minima where speed is
    above ``max_minimum_speed`` are ignored. Other parameters are as in
    ``zero_crossings``; ``direction`` is 0.
    """
    return _split(values, within, "speed_minima", max_gap_frames, 0.0,
                  min_prominence, max_minimum_speed, min_samples, short)


def group_into_bouts(start_time, end_time, max_gap_s, min_count=1):
    """Group segments that follow each other closely into bouts.

    Consecutive segments join a bout when the next one starts within
    ``max_gap_s`` of the previous one's end. Bouts with fewer than
    ``min_count`` segments are not kept.

    Parameters
    ----------
    start_time, end_time : array-like
        Per segment, in time order.

    Returns
    -------
    bout_of : numpy.ndarray of int
        Per segment, its row in ``bouts`` (-1 if its bout was not kept).
    bouts : pandas.DataFrame
        ``first``, ``last`` (segment rows, inclusive) and ``n_segments``.
    """
    st = np.asarray(start_time, float)
    en = np.asarray(end_time, float)
    if len(st) == 0:
        return np.array([], np.int64), pd.DataFrame(
            {"first": [], "last": [], "n_segments": []}, dtype=np.int64)
    new = np.r_[True, st[1:] - en[:-1] > max_gap_s]
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(st) - 1]
    count = last - first + 1
    keep = count >= min_count
    row = np.full(len(first), -1, np.int64)
    row[keep] = np.arange(keep.sum())
    bout_of = row[np.cumsum(new) - 1]
    bouts = pd.DataFrame({"first": first[keep], "last": last[keep],
                          "n_segments": count[keep]})
    return bout_of, bouts
