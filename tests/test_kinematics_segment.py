"""Tests for kinematics.segment."""

import unittest

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics import segment
from aind_dynamic_foraging_behavior_video_analysis.kinematics.intervals import (  # noqa: E501
    label_frames,
)


def labels(iv, n, col=None):
    """Per-frame segment number (1-based) or ``col`` value; None outside."""
    values = (np.arange(1, len(iv) + 1) if col is None
              else iv[col].to_numpy())
    lab = label_frames(n, iv, np.arange(len(iv)))
    return [None if pd.isna(v) else int(values[v]) for v in lab]


def within(ids):
    """Intervals table of contiguous runs of equal non-null ids."""
    s = pd.Series(pd.array(ids, dtype="Int64"))
    rows = []
    key = s.ne(s.shift()).fillna(True).cumsum()
    for _, g in s.dropna().groupby(key[s.notna()]):
        rows.append((g.index[0], g.index[-1]))
    return pd.DataFrame(rows, columns=["start", "end"])


def reference_tracked_runs(valid, max_gap):
    """The pre-0.5.0 loop (``segment_movements_trimnans``)."""
    ids, mid, gap, inside = [None] * len(valid), 0, 0, False
    for i, v in enumerate(valid):
        if not v:
            gap += 1
        else:
            gap = 0
            if not inside:
                mid += 1
                inside = True
        if inside:
            if gap <= max_gap:
                ids[i] = mid
            else:
                inside = False
    last = {}
    for i, (m, v) in enumerate(zip(ids, valid)):
        if m is not None and v:
            last[m] = i
    return [m if m is not None and i <= last[m] else None
            for i, m in enumerate(ids)]


class TrackedRunsTest(unittest.TestCase):
    """tracked_runs matches the old segmentation."""

    def test_matches_old_loop_on_random_patterns(self):
        rng = np.random.default_rng(1)
        for p in (0.2, 0.5, 0.8):
            valid = rng.random(2000) < p
            for gap in (0, 3, 10):
                iv = segment.tracked_runs(valid, gap)
                self.assertEqual(labels(iv, len(valid)),
                                 reference_tracked_runs(valid, gap))

    def test_empty(self):
        self.assertEqual(len(segment.tracked_runs(np.zeros(5, bool))), 0)


class AboveThresholdTest(unittest.TestCase):
    """above_threshold: two levels, joining, minimum length."""

    def test_two_levels(self):
        x = np.array([0, 3, 6, 3, 0, 3, 3, 0, 6, np.nan, 0.0])
        iv = segment.above_threshold(x, on=5, off=2)
        # the 3-3 stretch never reaches 5; NaN counts as low
        self.assertEqual(labels(iv, len(x)),
                         [None, 1, 1, 1, None, None, None, None, 2, None,
                          None])

    def test_join_and_min_frames(self):
        x = np.array([6, 6, 0, 6, 0, 0, 0, 6.0])
        iv = segment.above_threshold(x, on=5, max_gap_frames=1)
        self.assertEqual(labels(iv, 8), [1, 1, 1, 1, None, None, None, 2])
        iv = segment.above_threshold(x, on=5, max_gap_frames=1,
                                     min_frames=2)
        self.assertEqual(labels(iv, 8), [1, 1, 1, 1] + [None] * 4)

    def test_off_above_on_raises(self):
        with self.assertRaises(ValueError):
            segment.above_threshold([1.0], on=1, off=2)


class ZeroCrossingsTest(unittest.TestCase):
    """zero_crossings on hand-built signals (ported from the submovement
    tests)."""

    def run_(self, rv, ids=1, **kw):
        rv = np.asarray(rv, float)
        w = within([ids] * len(rv) if np.isscalar(ids) else ids)
        return segment.zero_crossings(rv, within=w, **kw), len(rv)

    def test_protrusion_retraction_split(self):
        iv, n = self.run_([1, 2, 1, -1, -2, -1.0])
        self.assertEqual(labels(iv, n), [1, 1, 1, 2, 2, 2])
        self.assertEqual(labels(iv, n, "direction"), [1, 1, 1, -1, -1, -1])
        self.assertEqual(list(iv["within"]), [0, 0])

    def test_deadband_hysteresis(self):
        noisy = [1, 2, 0.1, -0.1, 0.1, -0.1, 2, -1, -2.0]
        iv, n = self.run_(noisy, deadband=0.5)
        self.assertEqual(labels(iv, n), [1, 1, 1, 1, 1, 1, 1, 2, 2])
        self.assertEqual(len(self.run_(noisy)[0]), 6)

    def test_leading_deadband_frames_join_first_segment(self):
        iv, n = self.run_([0.0, 0.1, 2, -2.0], deadband=0.5)
        self.assertEqual(labels(iv, n), [1, 1, 1, 2])

    def test_short_gap_bridged_with_boundary_at_gap(self):
        iv, n = self.run_([1, 1, np.nan, np.nan, -1, -1.0],
                          max_gap_frames=3)
        self.assertEqual(labels(iv, n), [1, 1, None, None, 2, 2])

    def test_gap_inside_a_segment_keeps_it(self):
        iv, n = self.run_([1, 1, np.nan, 1, -1.0], max_gap_frames=3)
        self.assertEqual(labels(iv, n), [1, 1, 1, 1, 2])

    def test_long_gap_splits_without_sign_change(self):
        iv, n = self.run_([1, 1, np.nan, np.nan, 1, 1.0], max_gap_frames=1)
        self.assertEqual(labels(iv, n), [1, 1, None, None, 2, 2])

    def test_respects_enclosing_segments(self):
        iv, n = self.run_([1, 1, 1, 1, -1, np.nan, 2.0],
                          ids=[1, 1, 2, 2, 2, None, None])
        self.assertEqual(labels(iv, n), [1, 1, 2, 2, 3, None, None])
        self.assertEqual(list(iv["within"]), [0, 1, 1])

    def test_standalone_uses_bridged_runs(self):
        rv = np.array([1, -1, np.nan, np.nan, np.nan, 1, -1.0])
        iv = segment.zero_crossings(rv, max_gap_frames=2)
        self.assertEqual(labels(iv, 7), [1, 2, None, None, None, 3, 4])
        self.assertEqual(list(iv["within"]), [0, 0, 1, 1])

    def test_min_samples_merge(self):
        # 1-frame blips at the start and in the middle; the neighbors of
        # the middle one share a sign and fuse
        iv, n = self.run_([-0.5, 1, 2, 1, -0.5, 1, 1, -1, -2, -1],
                          min_samples=2)
        self.assertEqual(labels(iv, n), [1] * 7 + [2] * 3)
        self.assertEqual(list(iv["direction"]), [1, -1])

    def test_merged_blip_keeps_direction(self):
        # a 1-frame spike of the other sign at the end (the tongue back at
        # the jaw) merges into the retraction without flipping it
        iv, n = self.run_([2, 2, 2, -2, -2, -2, 5.0], min_samples=2)
        self.assertEqual(labels(iv, n), [1, 1, 1, 2, 2, 2, 2])
        self.assertEqual(list(iv["direction"]), [1, -1])

    def test_min_samples_drop(self):
        iv, n = self.run_([1, 1, -1, 1, 1.0], min_samples=2, short="drop")
        self.assertEqual(labels(iv, n), [1, 1, None, 2, 2])

    def test_errors_and_empty(self):
        with self.assertRaises(ValueError):
            self.run_([1.0], short="nope")
        iv, _ = self.run_([np.nan, np.nan])
        self.assertEqual(len(iv), 0)
        self.assertIn("direction", iv)


class SpeedMinimaTest(unittest.TestCase):
    """speed_minima on two-bump speed profiles."""

    def setUp(self):
        t = np.linspace(0, 2, 101)
        self.v = np.abs(np.sin(np.pi * t)) + 0.1  # minimum at frame 50

    def test_splits_at_minimum(self):
        iv = segment.speed_minima(self.v)
        self.assertEqual(len(iv), 2)
        self.assertEqual(iv["start"].iloc[1], 50)
        self.assertEqual(list(iv["direction"]), [0, 0])

    def test_prominence_and_height_suppress_minima(self):
        shallow = np.r_[np.linspace(0.1, 1, 25), np.linspace(0.98, 0.9, 10),
                        np.linspace(0.92, 1, 10), np.linspace(0.98, 0.1, 25)]
        self.assertEqual(len(segment.speed_minima(shallow)), 2)
        self.assertEqual(
            len(segment.speed_minima(shallow, min_prominence=0.2)), 1)
        self.assertEqual(
            len(segment.speed_minima(self.v, max_minimum_speed=0.05)), 1)


class GroupIntoBoutsTest(unittest.TestCase):
    """group_into_bouts by end-to-start gap."""

    def test_groups_and_min_count(self):
        st = [0.0, 0.2, 0.4, 2.0, 5.0, 5.1]
        en = [0.1, 0.3, 0.5, 2.1, 5.05, 5.2]
        bout_of, bouts = segment.group_into_bouts(st, en, max_gap_s=0.15,
                                                  min_count=2)
        self.assertEqual(list(bout_of), [0, 0, 0, -1, 1, 1])
        self.assertEqual(list(bouts["n_segments"]), [3, 2])
        self.assertEqual(list(bouts["first"]), [0, 4])

    def test_empty(self):
        bout_of, bouts = segment.group_into_bouts([], [], 1.0)
        self.assertEqual(len(bout_of), 0)
        self.assertEqual(len(bouts), 0)


if __name__ == "__main__":
    unittest.main()
