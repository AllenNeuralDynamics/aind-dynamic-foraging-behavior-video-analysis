"""Tests for kinematics.features and kinematics.intervals."""

import unittest

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
    features,
    intervals,
)

DT = 0.002


def frames(xs, ys=None):
    """One segment's frames; speed is the per-frame step / DT."""
    xs = np.asarray(xs, float)
    ys = np.zeros(len(xs)) if ys is None else np.asarray(ys, float)
    v = np.full(len(xs), np.nan)
    if len(xs) > 1:
        v[1:] = np.hypot(np.diff(xs), np.diff(ys)) / DT
    return pd.DataFrame({"time": DT * np.arange(len(xs)), "x": xs, "y": ys,
                         "vx": np.nan, "vy": np.nan, "speed": v,
                         "valid": np.isfinite(xs)})


def one(df, groups, origin=(0.0, 0.0), **ctx):
    iv = pd.DataFrame({"start": [0], "end": [len(df) - 1]})
    context = {"fs_hz": 1 / DT, "origin": origin, **ctx}
    return features.describe(df, iv, groups, context).iloc[0]


class FeatureGroupTest(unittest.TestCase):
    """Built-in groups on hand-built trajectories."""

    def test_outbound_stops_at_farthest_frame(self):
        row = one(frames([1, 2, 3, 4, 3, 2, 1]), ["outbound"])
        self.assertAlmostEqual(row["out_duration"], 3 * DT)
        self.assertAlmostEqual(row["out_path_length"], 3.0)
        self.assertAlmostEqual(row["out_peak_speed"], 1 / DT)

    def test_outbound_starting_at_farthest_is_zero_length(self):
        row = one(frames([5, 4, 3]), ["outbound"])
        self.assertEqual(row["out_duration"], 0.0)
        self.assertTrue(np.isnan(row["out_path_length"]))
        self.assertTrue(np.isnan(row["out_peak_speed"]))

    def test_all_missing_gives_nan(self):
        groups = ["path", "excursion", "extremes", "outbound"]
        row = one(frames([np.nan, np.nan]), groups)
        self.assertTrue(row.isna().all())

    def test_path_and_excursion(self):
        row = one(frames([1, 2, 3, 2], [0, 1, 1, 1]), ["path", "excursion"],
                  origin=(1.0, 0.0))
        self.assertAlmostEqual(row["path_length"], np.sqrt(2) + 2)
        self.assertAlmostEqual(row["displacement"], np.sqrt(2))
        self.assertAlmostEqual(row["amplitude"], np.sqrt(5))
        self.assertEqual((row["end_x"], row["end_y"]), (2.0, 1.0))
        self.assertEqual((row["start_x"], row["start_y"]), (0.0, 0.0))
        self.assertAlmostEqual(row["angle_deg"],
                               np.degrees(np.arctan2(1, 2)))
        self.assertEqual((row["extent_x"], row["extent_y"]), (2.0, 1.0))
        flipped = one(frames([1, 2, 3, 2], [0, 1, 1, 1]), ["excursion"],
                      origin=(1.0, 0.0), flip_x=True)
        self.assertEqual(flipped["end_x"], -2.0)

    def test_extremes(self):
        # Relative to (1, 0): x = 0, 2, -3, 1; y = 0, 4, -1, -2
        row = one(frames([1, 3, -2, 2], [0, 4, -1, -2]), ["extremes"],
                  origin=(1.0, 0.0))
        self.assertEqual((row["x_max"], row["y_at_x_max"]), (2.0, 4.0))
        self.assertEqual((row["x_min"], row["y_at_x_min"]), (-3.0, -1.0))
        self.assertEqual((row["y_max"], row["x_at_y_max"]), (4.0, 2.0))
        self.assertEqual((row["y_min"], row["x_at_y_min"]), (-2.0, 1.0))
        self.assertEqual(row["y_at_extent_x"], -1.0)   # extent_x = -3
        self.assertEqual(row["x_at_extent_y"], 2.0)    # extent_y = 4

    def test_extremes_reproduce_the_old_columns(self):
        # aggregate_tongue_movements before 0.5.0: max_x_distance =
        # |x - jaw| at the most positive x offset; max_y_distance = largest
        # |y - jaw|; *_from_jaw_* = the other coordinate there, absolute.
        rng = np.random.default_rng(0)
        jaw = np.array([40.0, 25.0])
        for _ in range(50):
            xs = rng.normal(jaw[0], 10, 12)
            ys = rng.normal(jaw[1], 10, 12)
            row = one(frames(xs, ys), ["excursion", "extremes"],
                      origin=tuple(jaw))
            dx, dy = xs - jaw[0], np.abs(ys - jaw[1])
            ix, iy = int(np.argmax(dx)), int(np.argmax(dy))
            self.assertAlmostEqual(abs(row["x_max"]), abs(xs[ix] - jaw[0]))
            self.assertAlmostEqual(row["y_at_x_max"] + jaw[1], ys[ix])
            self.assertAlmostEqual(abs(row["extent_y"]), dy[iy])
            self.assertAlmostEqual(row["x_at_extent_y"] + jaw[0], xs[iy])

    def test_extremes_follow_flips(self):
        row = one(frames([1, 3, -2]), ["extremes"], origin=(0.0, 0.0),
                  flip_x=True)
        self.assertEqual((row["x_max"], row["x_min"]), (2.0, -3.0))

    def test_needs_reference_point(self):
        with self.assertRaises(ValueError):
            one(frames([1, 2]), ["excursion"], origin=None)
        with self.assertRaises(ValueError):
            one(frames([1, 2]), ["nope"])

    def test_spectrum_finds_the_rhythm(self):
        t = np.arange(1000) * DT
        df = frames(5 * np.sin(2 * np.pi * 7 * t))
        row = one(df, ["spectrum"],
                  options={"spectrum": {"column": "x", "band": (2, 20)}})
        self.assertAlmostEqual(row["dominant_hz"], 7.0, delta=1.0)
        self.assertGreater(row["band_fraction"], 0.9)
        self.assertLessEqual(row["band_fraction"], 1.0)
        short = one(frames(np.ones(10)), ["spectrum"])
        self.assertTrue(np.isnan(short["dominant_hz"]))

    def test_no_segments_keeps_columns(self):
        df = frames([1, 2, 3])
        iv = pd.DataFrame({"start": [], "end": []}, dtype=int)
        out = features.describe(df, iv, ["path", "excursion", "spectrum"],
                                {"fs_hz": 1 / DT, "origin": (0.0, 0.0)})
        self.assertEqual(len(out), 0)
        for c in ("path_length", "amplitude", "dominant_hz"):
            self.assertIn(c, out.columns)

    def test_add_feature_group(self):
        @features.add_feature_group("n_points")
        def n_points(seg, context):
            return {"n_points": int(seg["valid"].sum())}

        try:
            self.assertEqual(one(frames([1, np.nan, 2]),
                                 ["n_points"])["n_points"], 2)
        finally:
            del features.FEATURE_GROUPS["n_points"]


class IntervalsTest(unittest.TestCase):
    """label_frames, interval_of, nearest, time_bins,
    summarize_by_window."""

    def test_label_frames(self):
        iv = pd.DataFrame({"start": [1, 4], "end": [2, 4]})
        lab = intervals.label_frames(6, iv, [7, 8])
        self.assertEqual(list(lab.fillna(0)), [0, 7, 7, 0, 8, 0])
        self.assertTrue(intervals.label_frames(
            3, iv.iloc[:0], []).isna().all())

    def test_interval_of(self):
        t = [0.5, 1.0, 1.5, 2.0, 3.5]
        self.assertEqual(
            list(intervals.interval_of(t, [1.0, 3.0], [2.0, 4.0])),
            [-1, 0, 0, 0, 1])
        self.assertEqual(
            list(intervals.interval_of(t, [1.0, 3.0], [2.0, 4.0], "left")),
            [-1, 0, 0, -1, 1])
        with self.assertRaises(ValueError):
            intervals.interval_of(t, [1.0], [2.0], "right")

    def test_closest_interval(self):
        t = [0.5, 1.5, 2.2, 2.9, 3.5, 5.0]
        row, off = intervals.closest_interval(t, [1.0, 3.0], [2.0, 4.0])
        self.assertEqual(list(row), [0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(off, [-0.5, 0.0, 0.2, -0.1, 0.0, 1.0])
        # a tie between the end of one and the start of the next: earlier
        row, off = intervals.closest_interval([2.5], [1.0, 3.0], [2.0, 4.0])
        self.assertEqual((row[0], off[0]), (0, 0.5))
        row, off = intervals.closest_interval([1.0], [], [])
        self.assertEqual(row[0], -1)
        self.assertTrue(np.isnan(off[0]))

    def test_nearest(self):
        self.assertEqual(
            list(intervals.nearest([0.0, 1.04, 5.0], [2.0, 1.0], 0.05)),
            [-1, 1, -1])
        self.assertEqual(list(intervals.nearest([1.0], [1.01], 0.05)), [0])
        self.assertEqual(list(intervals.nearest([1.0], [], 0.05)), [-1])

    def test_time_bins(self):
        bins = intervals.time_bins(0, 2.5, 1.0)
        self.assertEqual(list(bins["start_time"]), [0.0, 1.0, 2.0])
        self.assertEqual(bins["end_time"].iloc[-1], 3.0)

    def test_summarize_by_window(self):
        items = pd.DataFrame({"t": [0.5, 0.2, 1.5, 3.5],
                              "v": [np.nan, 2.0, 3.0, 4.0]})
        win = pd.DataFrame({"start_time": [0.0, 1.0, 2.0],
                            "end_time": [1.0, 2.0, 3.0]})
        out = intervals.summarize_by_window(items, win, "t", {
            "n": (None, "count"),
            "first_v": ("v", "first"),
            "last_v": ("v", "last"),
            "mean_v": ("v", "mean"),
            "max_v": ("v", lambda s: s.max()),
        })
        self.assertEqual(list(out["n"]), [2, 1, 0])
        self.assertEqual(out["first_v"].iloc[0], 2.0)  # earliest item
        self.assertTrue(np.isnan(out["last_v"].iloc[0]))  # missing kept
        self.assertEqual(out["mean_v"].iloc[1], 3.0)
        self.assertTrue(np.isnan(out["max_v"].iloc[2]))


if __name__ == "__main__":
    unittest.main()
