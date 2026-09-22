"""Tests for the tongue_quality_stats.json contract."""

import json
import tempfile
import unittest
from pathlib import Path

from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import (  # noqa: E501
    TONGUE_QUALITY_STATS_FILENAME,
    get_quality_summary,
    load_tongue_quality_stats,
    session_already_done,
)


class QualityStatsContractTest(unittest.TestCase):
    """Round-trip a stats file through the accessors."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.tmp.name) / "behavior_000000_2025-01-01"
        self.session_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file(self):
        self.assertFalse(session_already_done(self.session_dir))
        with self.assertRaises(FileNotFoundError):
            load_tongue_quality_stats(self.session_dir)

    def test_round_trip(self):
        stats = {
            "session_id": self.session_dir.name,
            "pred_csv": None,
            "total_licks": 100,
            "licks_with_movement": 95,
            "coverage_pct": 95.0,
            # json.dump turns float keys into strings, as the writer does
            "percentiles": {"duration": {0.5: 0.07, 0.9: 0.2}},
        }
        with open(self.session_dir / TONGUE_QUALITY_STATS_FILENAME, "w") as f:
            json.dump(stats, f)
        self.assertTrue(session_already_done(self.session_dir))
        summary = get_quality_summary(
            load_tongue_quality_stats(self.session_dir)
        )
        self.assertEqual(summary["session_id"], self.session_dir.name)
        self.assertEqual(summary["coverage_pct"], 95.0)
        self.assertAlmostEqual(summary["duration_p50"], 0.07)

    def test_malformed_fails_closed(self):
        summary = get_quality_summary({})
        self.assertEqual(summary["coverage_pct"], 0.0)
        self.assertEqual(summary["duration_p50"], 0.0)


if __name__ == "__main__":
    unittest.main()
