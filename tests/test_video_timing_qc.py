"""Tests for video_timing_qc, on CSVs simulated with arrival-order pairing.

The camera exposes every frame and numbers it; the host saves only some of
them; row n of the CSV gets the n-th Harp trigger time, as the acquisition
workflow's ``rx:Zip`` does.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis import (
    video_timing_qc as vtq,
)

IFI = 0.002


def simulate(
    folder,
    n_exposures=3000,
    dropped=(),
    trigger_errors=None,
    lost_trigger=None,
    layout="flat",
    name="bottom_camera",
):
    """Write a simulated video CSV; return its path and the true triggers.

    Parameters
    ----------
    dropped : exposure indices the host did not save.
    trigger_errors : {trigger index: seconds added} (Harp glitches).
    lost_trigger : trigger index missing from the Harp sequence.
    """
    # Harp triggers on the 32 us tick grid, as the trigger log stores them.
    true_triggers = np.round((100.0 + IFI * np.arange(n_exposures)) / 32e-6)
    true_triggers = true_triggers * 32e-6
    triggers = true_triggers.copy()
    for idx, err in (trigger_errors or {}).items():
        triggers[idx] += err
    if lost_trigger is not None:
        triggers = np.delete(triggers, lost_trigger)
    # Camera clock: other offset, 15 ppm drift, tiny jitter.
    rng = np.random.default_rng(0)
    exposure_ns = (
        5000.0 + IFI * (1 + 15e-6) * np.arange(n_exposures)
    ) * 1e9 + rng.normal(0, 20e3, n_exposures)
    saved = np.setdiff1d(np.arange(n_exposures), dropped)
    harp = triggers[: len(saved)]
    lines = [
        f"{h:.6f},{f},{int(c)}"
        for h, f, c in zip(harp, saved + 1000, exposure_ns[saved])
    ]
    folder = Path(folder)
    if layout == "flat":
        path = folder / f"{name}.csv"
    else:
        (folder / name).mkdir(exist_ok=True)
        path = folder / name / "metadata.csv"
        lines.insert(0, ",".join(vtq.NEW_LAYOUT_COLUMNS))
    path.write_text("\n".join(lines) + "\n")
    return path, triggers


def write_trigger_log(path, triggers):
    """Write triggers as 13-byte Harp messages (Event_94.bin)."""
    ticks_total = np.round(np.asarray(triggers) / 32e-6).astype(np.int64)
    msg = np.zeros((len(triggers), 13), dtype=np.uint8)
    msg[:, 0], msg[:, 1], msg[:, 2], msg[:, 4], msg[:, 11] = 3, 11, 94, 17, 1
    msg[:, 5:9] = (
        (ticks_total // 31250).astype("<u4").view(np.uint8).reshape(-1, 4)
    )
    msg[:, 9:11] = (
        (ticks_total % 31250).astype("<u2").view(np.uint8).reshape(-1, 2)
    )
    msg.tofile(path)


class VideoTimingQCTest(unittest.TestCase):
    """Checks and corrections on simulated sessions."""

    def setUp(self):
        """Temporary folder per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        """Remove the temporary folder."""
        self._tmp.cleanup()

    def load(self, **kwargs):
        """Simulate, load, and check; return timing, qc and triggers."""
        path, triggers = simulate(self.tmp, **kwargs)
        timing = vtq.load_video_timing(path)
        return timing, vtq.check_video_timing(timing), triggers

    def test_clean(self):
        """No problems: ok, and correction leaves Harp unchanged."""
        timing, qc, _ = self.load()
        self.assertEqual(qc["qc_class"], "ok")
        self.assertEqual(qc["n_clock_flags"], 0)
        self.assertAlmostEqual(qc["ifi_s"], IFI, places=6)
        fixed = vtq.correct_video_timing(timing)
        np.testing.assert_array_equal(
            fixed["harp_time"], timing["harp_time_raw"]
        )
        self.assertTrue((fixed["harp_source"] == "original").all())

    def test_layouts_load_the_same(self):
        """Header and headerless CSVs give the same timing frame."""
        flat, _ = simulate(self.tmp, layout="flat")
        new, _ = simulate(self.tmp, layout="new", name="BottomCamera")
        a = vtq.load_video_timing(flat)
        b = vtq.load_video_timing(new)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
        self.assertEqual(
            list(a.columns), ["harp_time_raw", "frame_number", "camera_time"]
        )

    def test_glitch(self):
        """A ~983 ms early row is found alone and interpolated."""
        timing, qc, triggers = self.load(trigger_errors={500: -0.983})
        self.assertEqual(qc["qc_class"], "harp_glitch")
        self.assertEqual(qc["glitch_rows"], [500])
        self.assertEqual(qc["n_unexplained_flags"], 0)
        fixed = vtq.correct_video_timing(timing)
        self.assertAlmostEqual(
            fixed["harp_time"].iloc[500], 100.0 + 500 * IFI, places=5
        )
        self.assertEqual(fixed["harp_source"].iloc[500], "glitch_interpolated")
        self.assertEqual((fixed["harp_source"] != "original").sum(), 1)

    def test_small_glitch(self):
        """A +3 ms blip is also a glitch."""
        _, qc, _ = self.load(trigger_errors={800: 0.003})
        self.assertEqual(qc["glitch_rows"], [800])

    def test_consecutive_bad_rows_refused(self):
        """Two bad rows in a row are not a glitch and are not corrected."""
        timing, qc, _ = self.load(trigger_errors={500: -0.983, 501: -0.983})
        self.assertEqual(qc["qc_class"], "clock_disagreement")
        self.assertEqual(qc["glitch_rows"], [])
        with self.assertRaises(ValueError):
            vtq.correct_video_timing(timing)

    def test_drops_reindexed_and_tail_estimated(self):
        """Drops of 1, 2, 3, 5 frames: exact rows and tail near truth."""
        dropped = [100, 400, 401, 900, 901, 902] + list(range(1500, 1505))
        timing, qc, triggers = self.load(dropped=dropped)
        self.assertEqual(qc["qc_class"], "frame_drops")
        self.assertEqual(qc["n_frame_gaps"], 4)
        self.assertEqual(qc["n_frames_dropped"], 11)
        self.assertEqual(qc["first_gap_row"], 100)
        # 0.5 flags every drop; the legacy threshold (2) misses the 1-frame
        # drop, and the 2-frame one sits on its edge.
        self.assertEqual(qc["n_clock_flags"], 4)
        legacy = vtq.check_video_timing(timing, threshold=2)
        self.assertLess(legacy["n_clock_flags"], 4)

        fixed = vtq.correct_video_timing(timing)
        k = (timing["frame_number"] - 1000).to_numpy()
        source = fixed["harp_source"].to_numpy()
        exact = np.isin(source, ["original", "reindexed"])
        np.testing.assert_allclose(
            fixed["harp_time"].to_numpy()[exact], triggers[k[exact]], atol=1e-6
        )
        self.assertEqual((source == "estimated_camera_fit").sum(), 11)
        self.assertTrue((source[:100] == "original").all())
        tail_error = (
            fixed["harp_time"].to_numpy()[~exact] - triggers[k[~exact]]
        )
        self.assertLess(np.abs(tail_error).max(), 1e-4)

    def test_many_drops(self):
        """A drop every 8 frames, as in the affected FIP sessions."""
        dropped = np.arange(5, 20000, 8)
        timing, qc, triggers = self.load(n_exposures=20000, dropped=dropped)
        self.assertEqual(qc["n_frames_dropped"], len(dropped))
        fixed = vtq.correct_video_timing(timing)
        k = (timing["frame_number"] - 1000).to_numpy()
        error = fixed["harp_time"].to_numpy() - triggers[k]
        self.assertLess(np.abs(error).max(), 1e-4)

    def test_glitch_with_drops(self):
        """Glitches next to a drop and on a dropped exposure are fixed."""
        # Trigger 405 lands on the row after the 2-frame drop at 400-401;
        # trigger 900's exposure was itself dropped.
        timing, qc, triggers = self.load(
            dropped=[400, 401, 900],
            trigger_errors={403: -0.983, 900: -0.983},
        )
        self.assertEqual(qc["qc_class"], "frame_drops")
        self.assertEqual(qc["glitch_rows"], [403, 900])
        fixed = vtq.correct_video_timing(timing)
        k = (timing["frame_number"] - 1000).to_numpy()
        truth = 100.0 + IFI * k
        self.assertLess(np.abs(fixed["harp_time"] - truth).max(), 1e-4)
        self.assertEqual(
            (fixed["harp_source"] == "glitch_interpolated").sum(), 1
        )

    def test_lost_trigger_fails_post_check(self):
        """A trigger missing from the Harp sequence is refused."""
        timing, qc, _ = self.load(dropped=[100, 200], lost_trigger=1000)
        self.assertEqual(qc["qc_class"], "frame_drops")
        with self.assertRaisesRegex(ValueError, "post-checks"):
            vtq.correct_video_timing(timing)

    def test_frame_order_error(self):
        """A repeated frame number is not correctable."""
        path, _ = simulate(self.tmp)
        lines = path.read_text().splitlines()
        lines[50] = lines[49]
        path.write_text("\n".join(lines) + "\n")
        timing = vtq.load_video_timing(path)
        self.assertEqual(
            vtq.check_video_timing(timing)["qc_class"], "frame_order_error"
        )
        with self.assertRaises(ValueError):
            vtq.correct_video_timing(timing)

    def test_transcode_mismatch(self):
        """Video frame count different from CSV rows."""
        timing, _, _ = self.load()
        qc = vtq.check_video_timing(timing, video_frame_count=2994)
        self.assertEqual(qc["qc_class"], "transcode_mismatch")

    def test_unknown_header(self):
        """A header that is not the known one raises."""
        path = self.tmp / "odd.csv"
        path.write_text("a,b,c\n1,2,3\n")
        with self.assertRaises(ValueError):
            vtq.load_video_timing(path)

    def test_trigger_log(self):
        """Correction from the trigger log is exact, tail included."""
        dropped = [100, 400, 401, 900]
        path, triggers = simulate(
            self.tmp, dropped=dropped, trigger_errors={700: -0.983}
        )
        log_path = self.tmp / "Event_94.bin"
        write_trigger_log(log_path, triggers)
        log = vtq.read_harp_trigger_log(log_path)
        np.testing.assert_allclose(log, triggers, atol=1e-9)

        timing = vtq.load_video_timing(path)
        fixed = vtq.correct_video_timing(timing, trigger_times=log)
        k = (timing["frame_number"] - 1000).to_numpy()
        truth = 100.0 + IFI * k
        self.assertLess(np.abs(fixed["harp_time"] - truth).max(), 32e-6)
        self.assertTrue(
            fixed["harp_source"]
            .isin(["trigger_log", "glitch_interpolated"])
            .all()
        )
        # A log from another session does not match the CSV.
        with self.assertRaisesRegex(ValueError, "does not match"):
            vtq.correct_video_timing(timing, trigger_times=log + 1.0)
        with self.assertRaisesRegex(ValueError, "fewer"):
            vtq.correct_video_timing(timing, trigger_times=log[:2000])

    def test_check_session(self):
        """Both layouts are found in a behavior-videos folder."""
        simulate(self.tmp, name="bottom_camera")
        simulate(self.tmp, layout="new", name="SideCameraRight", dropped=[9])
        table = vtq.check_session(self.tmp)
        self.assertEqual(
            table.set_index("camera")["qc_class"].to_dict(),
            {"bottom_camera": "ok", "SideCameraRight": "frame_drops"},
        )


class CorrectFrameTimesTest(unittest.TestCase):
    """The array-level correction, with no CSV or Harp involved."""

    def setUp(self):
        """8 exposures at 10 ms; exposures 2 and 5 were not saved."""
        self.frame_number = np.array([0, 1, 3, 4, 6, 7])
        self.camera_time = 50.0 + 0.01 * self.frame_number
        self.triggers = 7.0 + 0.01 * np.arange(8)

    def test_full_trigger_list_is_exact(self):
        """Every frame gets the trigger of its own exposure."""
        times, source = vtq.correct_frame_times(
            self.frame_number, self.camera_time, self.triggers
        )
        np.testing.assert_allclose(times, self.triggers[self.frame_number])
        self.assertEqual(
            list(source),
            ["original", "original"] + ["reindexed"] * 4,
        )

    def test_short_trigger_list_estimates_tail(self):
        """Triggers only as many as frames: the rest come from a fit."""
        times, source = vtq.correct_frame_times(
            self.frame_number, self.camera_time, self.triggers[:6]
        )
        np.testing.assert_allclose(
            times, self.triggers[self.frame_number], atol=1e-9
        )
        self.assertEqual(list(source[-2:]), ["estimated_camera_fit"] * 2)

    def test_frames_must_increase(self):
        """Out-of-order frame numbers are refused."""
        with self.assertRaises(ValueError):
            vtq.correct_frame_times(
                [0, 2, 1], [0.0, 0.02, 0.03], self.triggers
            )


class IntegrateKeypointsTest(unittest.TestCase):
    """integrate_keypoints_with_video_time uses the corrected Harp time."""

    def test_both_layouts_with_drops(self):
        """Keypoint time_raw is the corrected Harp time, in either layout."""
        from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (  # noqa: E501
            integrate_keypoints_with_video_time,
        )

        with tempfile.TemporaryDirectory() as tmp:
            for layout, name in [
                ("flat", "bottom_camera"),
                ("new", "BottomCamera"),
            ]:
                with self.subTest(layout=layout):
                    path, triggers = simulate(
                        tmp,
                        dropped=[100, 400, 401],
                        layout=layout,
                        name=name,
                    )
                    n_rows = 3000 - 3
                    kps = {"tongue": pd.DataFrame({"x": np.zeros(n_rows)})}
                    kps_trim, video_csv = integrate_keypoints_with_video_time(
                        path, kps
                    )
                    k = (video_csv["frame_number"] - 1000).to_numpy()
                    time_raw = kps_trim["tongue"]["time_raw"].to_numpy()
                    self.assertLess(
                        np.abs(time_raw - (100.0 + IFI * k)).max(), 1e-4
                    )
                    self.assertEqual(kps_trim["tongue"]["time"].iloc[0], 0)

    def test_uncorrectable_raises(self):
        """Timing that can't be trusted stops the pipeline."""
        from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (  # noqa: E501
            integrate_keypoints_with_video_time,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path, _ = simulate(tmp, trigger_errors={500: -0.983, 501: -0.983})
            kps = {"tongue": pd.DataFrame({"x": np.zeros(3000)})}
            with self.assertRaises(ValueError):
                integrate_keypoints_with_video_time(path, kps)


if __name__ == "__main__":
    unittest.main()
