"""Tests for kinematics.pose_io, kinematics.signal and
kinematics.landmarks."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
    landmarks as lm,
    pose_io,
    signal,
)

LP_CSV = """scorer,heatmap,heatmap,heatmap,heatmap,heatmap,heatmap
bodyparts,tongue,tongue,tongue,jaw,jaw,jaw
coords,x,y,likelihood,x,y,likelihood
0,1.0,2.0,0.9,10.0,20.0,1.0
1,1.5,2.5,0.1,10.0,20.0,1.0
2,,,0.0,11.0,21.0,0.95
"""


class PoseIoTest(unittest.TestCase):
    """Readers give the long pose table."""

    def test_lightning_pose_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.csv"
            path.write_text(LP_CSV)
            pose = pose_io.read_pose(path)
        self.assertEqual(list(pose.columns), pose_io.POSE_COLUMNS)
        self.assertEqual(list(pose["keypoint"].unique()), ["tongue", "jaw"])
        tongue = pose[pose["keypoint"] == "tongue"]
        self.assertEqual(list(tongue["frame"]), [0, 1, 2])
        self.assertEqual(list(tongue["x"].iloc[:2]), [1.0, 1.5])
        self.assertTrue(np.isnan(tongue["x"].iloc[2]))
        self.assertEqual(list(tongue["likelihood"]), [0.9, 0.1, 0.0])

    def test_pose_table_shapes(self):
        pose = pose_io.pose_table(np.zeros((4, 2, 2)), None, ["a", "b"])
        self.assertEqual(len(pose), 8)
        self.assertTrue((pose["likelihood"] == 1).all())
        with self.assertRaises(ValueError):
            pose_io.pose_table(np.zeros((4, 2, 2)), None, ["a"])
        with self.assertRaises(ValueError):
            pose_io.pose_table(np.zeros((4, 2, 2)), np.ones((4, 3)),
                               ["a", "b"])

    def test_ndx_pose_object(self):
        series = {
            "nose": SimpleNamespace(name="nose", data=np.ones((3, 2)),
                                    confidence=np.array([0.1, 0.5, 0.9])),
            "paw": SimpleNamespace(name="paw", data=np.zeros((3, 2)),
                                   confidence=None),
        }
        pose = pose_io.read_ndx_pose(
            SimpleNamespace(pose_estimation_series=series))
        self.assertEqual(list(pose["keypoint"].unique()), ["nose", "paw"])
        self.assertEqual(list(pose["likelihood"]),
                         [0.1, 0.5, 0.9, 1, 1, 1])

    def test_sleap_h5(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not installed")
        tracks = np.arange(2 * 2 * 3 * 4, dtype=float).reshape(2, 2, 3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.analysis.h5"
            with h5py.File(path, "w") as f:
                f["tracks"] = tracks
                f["node_names"] = [b"nose", b"ear", b"tail"]
                f["point_scores"] = np.full((2, 3, 4), 0.5)
            pose = pose_io.read_pose(path)
        self.assertEqual(list(pose["keypoint"].unique()),
                         ["nose", "ear", "tail"])
        ear = pose[pose["keypoint"] == "ear"]
        np.testing.assert_array_equal(ear["x"], tracks[0, 0, 1])
        np.testing.assert_array_equal(ear["y"], tracks[0, 1, 1])
        self.assertTrue((pose["likelihood"] == 0.5).all())

    def test_multi_animal_dlc_keeps_first(self):
        cols = pd.MultiIndex.from_tuples(
            [("dlc", ind, "nose", c) for ind in ("m1", "m2")
             for c in ("x", "y", "likelihood")])
        df = pd.DataFrame([[1, 2, 0.9, 5, 6, 0.8]], columns=cols)
        pose = pose_io._from_dlc_frame(df)
        self.assertEqual(list(pose[["x", "y", "likelihood"]].iloc[0]),
                         [1.0, 2.0, 0.9])

    def test_format_errors(self):
        with self.assertRaises(ValueError):
            pose_io.read_pose("preds.txt")
        with self.assertRaises(ValueError):
            pose_io.read_pose("preds.csv", fmt="nope")


class AddTimeMaskTest(unittest.TestCase):
    """add_time trims to the shorter input; mask sets valid."""

    def test_add_time_trims(self):
        pose = pose_io.pose_table(np.zeros((5, 1, 2)), None, ["a"])
        out, info = signal.add_time(pose, [10.0, 11.0, 12.0])
        self.assertEqual(list(out["time"]), [10.0, 11.0, 12.0])
        self.assertEqual(info, {"n_frames_pose": 5, "n_frames_timing": 3,
                                "n_frames": 3})
        self.assertEqual(list(out.columns[:2]), ["frame", "time"])

    def test_mask_per_keypoint(self):
        pose = pose_io.pose_table(np.ones((2, 2, 2)),
                                  [[0.5, 0.5], [0.95, 0.95]], ["a", "b"])
        out = signal.mask(pose, {"a": 0.9})
        self.assertEqual(list(out["valid"]), [False, True, True, True])
        self.assertTrue(np.isnan(out["x"].iloc[0]))
        self.assertEqual(signal.mask(pose, 0.9)["valid"].sum(), 2)


class SmoothTest(unittest.TestCase):
    """smooth: per-run filtering, gaps, dropped frames, velocity."""

    fs = 500.0

    def frames(self, x, valid=None, t=None):
        n = len(x)
        t = np.arange(n) / self.fs if t is None else t
        x = np.asarray(x, float)
        valid = np.isfinite(x) if valid is None else valid
        return pd.DataFrame({"time": t, "x": np.where(valid, x, np.nan),
                             "y": np.zeros(n), "valid": valid})

    def test_velocity_of_a_slow_sine(self):
        t = np.arange(2000) / self.fs
        x = 10 * np.sin(2 * np.pi * 2 * t)
        out = signal.smooth(self.frames(x), self.fs, cutoff_hz=50)
        inner = slice(200, -200)
        np.testing.assert_allclose(out["x"][inner], x[inner], atol=1e-3)
        np.testing.assert_allclose(
            out["vx"][inner], 10 * 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * t)
            [inner], rtol=0, atol=0.5)
        np.testing.assert_allclose(out["speed"], np.abs(out["vx"]))

    def test_no_leak_across_long_gaps(self):
        # two runs at different levels; a long gap between them
        x = np.r_[np.zeros(100), np.full(50, np.nan), np.full(100, 100.0)]
        out = signal.smooth(self.frames(x), self.fs, max_gap_s=0.02)
        np.testing.assert_allclose(out["x"][:100], 0, atol=1e-9)
        np.testing.assert_allclose(out["x"][150:], 100, atol=1e-9)
        self.assertTrue(out["x"][100:150].isna().all())

    def test_short_gap_filled_for_filtering_only(self):
        x = np.r_[np.arange(50.0), np.full(5, np.nan), np.arange(55, 100.0)]
        out = signal.smooth(self.frames(x), self.fs, max_gap_s=0.02)
        self.assertTrue(out["x"][50:55].isna().all())
        np.testing.assert_allclose(out["x"][60:90], x[60:90], atol=1e-3)

    def test_dropped_camera_frames(self):
        # a 10-frame hole in time; the line stays a line
        t = np.r_[np.arange(100), np.arange(110, 200)] / self.fs
        x = 1000 * t
        out = signal.smooth(self.frames(x, t=t), self.fs)
        np.testing.assert_allclose(out["vx"][20:-20], 1000, rtol=1e-5)

    def test_tiny_runs_and_options(self):
        x = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                      np.nan, np.nan, np.nan, np.nan, np.nan, 2.0, 3.0])
        out = signal.smooth(self.frames(x), self.fs, cutoff_hz=None,
                            max_gap_s=0.0)
        self.assertEqual(out["x"].iloc[0], 1.0)
        self.assertTrue(np.isnan(out["vx"].iloc[0]))
        self.assertAlmostEqual(out["vx"].iloc[13], self.fs)
        with self.assertRaises(ValueError):
            signal.smooth(self.frames(x), self.fs, order=3)

    def test_relative_and_radial(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 0.0, np.nan],
                           "y": [0.0, 0.0, 0.0, 1.0],
                           "vx": [3.0, -2.0, 1.0, 1.0],
                           "vy": [5.0, 0.0, 1.0, 1.0]})
        rv = signal.radial_velocity(df, (0, 0))
        self.assertAlmostEqual(rv[0], 3.0)   # away; tangential part ignored
        self.assertAlmostEqual(rv[1], -2.0)  # toward the origin
        self.assertTrue(np.isnan(rv[2]))     # at the origin
        self.assertTrue(np.isnan(rv[3]))     # missing
        rx, ry = signal.relative_to([3.0], [4.0], (1, 1), flip_x=True,
                                    flip_y=True)
        self.assertEqual((rx[0], ry[0]), (-2.0, -3.0))


class LandmarksTest(unittest.TestCase):
    """landmarks: median over valid points."""

    def test_median_and_missing(self):
        pose = pd.DataFrame({"keypoint": ["jaw"] * 4,
                             "x": [1.0, 2.0, 3.0, 100.0],
                             "y": [0.0, 0.0, 0.0, 0.0],
                             "valid": [True, True, True, False]})
        out = lm.landmarks(pose, ["jaw", "spout"])
        self.assertEqual(lm.origin(out, "jaw"), (2.0, 0.0))
        self.assertEqual(int(out["n_valid"].iloc[0]), 3)
        with self.assertRaises(ValueError):
            lm.origin(out, "spout")
        with self.assertRaises(ValueError):
            lm.landmarks(pose, ["jaw"], stat="mode")
        self.assertEqual(
            lm.landmarks(pose.drop(columns="valid"), ["jaw"], "mean")
            ["x"].iloc[0], 26.5)


if __name__ == "__main__":
    unittest.main()
