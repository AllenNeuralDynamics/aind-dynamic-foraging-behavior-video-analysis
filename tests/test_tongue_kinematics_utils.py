"""Tests for movement aggregation and outbound-phase metrics."""

import unittest

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (  # noqa: E501
    aggregate_tongue_movements,
    compute_outbound_metrics,
)

OUT_COLS = [
    "out_duration",
    "out_peak_velocity",
    "out_mean_velocity",
    "out_total_distance",
]


def _movement(mid, t0, xs, ys, dt=0.002, trial=1, lick=False):
    """Frame rows for one movement; velocity is the per-frame speed."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    v = np.full(len(xs), np.nan)
    if len(xs) > 1:
        v[1:] = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
    return pd.DataFrame({
        "movement_id": mid,
        "time_in_session": t0 + dt * np.arange(len(xs)),
        "x": xs,
        "y": ys,
        "xv": np.gradient(xs) if len(xs) > 1 else [0.0],
        "yv": np.gradient(ys) if len(ys) > 1 else [0.0],
        "v": v,
        "lick": lick,
        "lick_index": np.nan,
        "trial": trial,
    })


class ComputeOutboundMetricsTest(unittest.TestCase):
    """compute_outbound_metrics on hand-built trajectories."""

    def setUp(self):
        self.jaw = pd.DataFrame({"x": [0.0, 0.0], "y": [0.0, 0.0]})

    def test_outbound_stops_at_farthest_frame(self):
        # out 4 frames, back 3 frames; endpoint is frame index 3
        seg = _movement(1, 0.0, [1, 2, 3, 4, 3, 2, 1], [0] * 7)
        out = compute_outbound_metrics(seg, self.jaw)
        row = out.set_index("movement_id").loc[1]
        self.assertAlmostEqual(row["out_duration"], 3 * 0.002)
        self.assertAlmostEqual(row["out_total_distance"], 3.0)
        self.assertAlmostEqual(row["out_peak_velocity"], 1.0 / 0.002)
        self.assertAlmostEqual(row["out_mean_velocity"], 1.0 / 0.002)

    def test_endpoint_at_first_frame_is_zero_length(self):
        # tongue starts far out and only retracts
        seg = _movement(2, 1.0, [5, 4, 3], [0, 0, 0])
        row = compute_outbound_metrics(seg, self.jaw).iloc[0]
        self.assertEqual(row["out_duration"], 0.0)
        self.assertTrue(np.isnan(row["out_total_distance"]))
        # the single outbound frame has no velocity
        self.assertTrue(np.isnan(row["out_peak_velocity"]))

    def test_all_nan_movement_gives_nan_row(self):
        seg = _movement(3, 2.0, [np.nan, np.nan], [np.nan, np.nan])
        row = compute_outbound_metrics(seg, self.jaw).iloc[0]
        self.assertTrue(row[OUT_COLS].isna().all())

    def test_missing_column_raises(self):
        seg = _movement(1, 0.0, [1, 2], [0, 0]).drop(columns=["v"])
        with self.assertRaises(ValueError):
            compute_outbound_metrics(seg, self.jaw)

    def test_empty_input_has_columns(self):
        seg = _movement(1, 0.0, [1, 2], [0, 0]).iloc[0:0]
        out = compute_outbound_metrics(seg, self.jaw)
        self.assertEqual(list(out.columns), ["movement_id"] + OUT_COLS)
        self.assertEqual(len(out), 0)


class AggregateEmitsOutboundTest(unittest.TestCase):
    """aggregate_tongue_movements carries out_* natively."""

    def test_out_columns_match_standalone(self):
        seg = pd.concat([
            _movement(1, 0.0, [1, 2, 3, 2], [0, 1, 2, 1]),
            _movement(2, 1.0, [5, 4, 3], [0, 0, 0], trial=2),
            _movement(3, 2.0, [np.nan, 1, 2], [np.nan, 1, 2], trial=3),
        ], ignore_index=True)
        jaw = pd.DataFrame({"x": [0.0, 0.0], "y": [0.0, 0.0]})
        movs = aggregate_tongue_movements(seg, {"jaw": jaw})
        for c in OUT_COLS:
            self.assertIn(c, movs.columns)
        expected = compute_outbound_metrics(seg, jaw).set_index("movement_id")
        got = movs.set_index("movement_id")[OUT_COLS]
        pd.testing.assert_frame_equal(
            got.sort_index(), expected[OUT_COLS].sort_index(),
            check_names=False,
        )
        # the endpoint the two paths agree on
        self.assertEqual(movs.set_index("movement_id").loc[1, "endpoint_x"], 3)


if __name__ == "__main__":
    unittest.main()
