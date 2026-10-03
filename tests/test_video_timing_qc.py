"""Tests for video_timing_qc, on CSVs simulated with arrival-order pairing.

The camera exposes every frame and numbers it; the host saves only some of
them; row n of the CSV gets the n-th Harp trigger time, as the acquisition
workflow's ``rx:Zip`` does.
"""

import json
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


def failed(checks):
    """Names of the checks that failed."""
    return set(checks.loc[checks["passed"].eq(False), "check"])


def check(checks, name):
    """One check's row as a dict."""
    return checks.set_index("check").loc[name].to_dict()


# The deprecated timing_action's strings and the verdict and correction
# method each now gives: every old outcome is kept (refuse: X -> exclude: X).
OLD_ACTIONS = {
    "use harp as written": ("use", "as written"),
    "fix glitches": ("use", "fix glitches"),
    "re-index": ("use", "re-index"),
}


class VideoTimingQCTest(unittest.TestCase):
    """Checks and corrections on simulated sessions."""

    def setUp(self):
        """Temporary folder per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        """Remove the temporary folder."""
        self._tmp.cleanup()

    def assertOutcome(self, checks, action):
        """The old action (with a warning), the verdict and the method."""
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(vtq.timing_action(checks), action)
        verdict, method = OLD_ACTIONS.get(
            action, (action.replace("refuse", "exclude"), None)
        )
        self.assertEqual(vtq.timing_verdict(checks), verdict)
        self.assertEqual(vtq._correction_method(checks), method)

    def load(self, **kwargs):
        """Simulate, load, and check; return timing, checks and triggers."""
        path, triggers = simulate(self.tmp, **kwargs)
        timing = vtq.load_video_timing(path)
        return timing, vtq.check_video_timing(timing), triggers

    def test_clean(self):
        """No problems: every check passes and Harp is used as written."""
        timing, checks, _ = self.load()
        self.assertEqual(failed(checks), set())
        self.assertOutcome(checks, "use harp as written")
        self.assertEqual(
            check(checks, "harp_matches_camera")["message"],
            "skipped: no frames lost",
        )
        self.assertAlmostEqual(
            vtq.frame_interval(timing["harp_time_raw"]), IFI, places=6
        )
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
        timing, checks, _ = self.load(trigger_errors={500: -0.983})
        self.assertEqual(failed(checks), {"harp_has_no_glitches"})
        self.assertEqual(check(checks, "harp_has_no_glitches")["rows"], [500])
        self.assertOutcome(checks, "fix glitches")
        fixed = vtq.correct_video_timing(timing)
        self.assertAlmostEqual(
            fixed["harp_time"].iloc[500], 100.0 + 500 * IFI, places=5
        )
        self.assertEqual(fixed["harp_source"].iloc[500], "glitch_interpolated")
        self.assertEqual((fixed["harp_source"] != "original").sum(), 1)

    def test_small_glitch(self):
        """A +3 ms blip is also a glitch."""
        _, checks, _ = self.load(trigger_errors={800: 0.003})
        self.assertEqual(check(checks, "harp_has_no_glitches")["rows"], [800])

    def test_consecutive_bad_rows_refused(self):
        """Two bad rows in a row are not a glitch and are not corrected."""
        timing, checks, _ = self.load(
            trigger_errors={500: -0.983, 501: -0.983}
        )
        self.assertNotIn("harp_has_no_glitches", failed(checks))
        self.assertOutcome(checks, "refuse: harp_evenly_spaced")
        with self.assertRaisesRegex(ValueError, "harp_evenly_spaced"):
            vtq.correct_video_timing(timing)

    def test_harp_clock_step_refused(self):
        """Harp shifting back ~2.2 ms and staying there is refused."""
        timing, _, _ = self.load()
        timing.loc[1500:, "harp_time_raw"] -= 0.0022
        checks = vtq.check_video_timing(timing)
        self.assertEqual(check(checks, "harp_evenly_spaced")["rows"], [1500])
        self.assertOutcome(checks, "refuse: harp_evenly_spaced")
        with self.assertRaisesRegex(ValueError, "harp_evenly_spaced"):
            vtq.correct_video_timing(timing)

    def test_drops_reindexed_and_tail_estimated(self):
        """Drops of 1, 2, 3, 5 frames: exact rows and tail near truth."""
        dropped = [100, 400, 401, 900, 901, 902] + list(range(1500, 1505))
        timing, checks, triggers = self.load(dropped=dropped)
        self.assertEqual(failed(checks), {"no_frames_lost"})
        lost = check(checks, "no_frames_lost")
        self.assertEqual(lost["count"], 11)
        self.assertEqual(lost["rows"], [100, 399, 897, 1494])
        self.assertOutcome(checks, "re-index")
        self.assertIs(check(checks, "harp_matches_camera")["passed"], True)
        # 0.5 frame flags every drop; the legacy tolerance (2) misses the
        # 1-frame drop, and the 2-frame one sits on its edge.
        harp = timing["harp_time_raw"].to_numpy()
        camera = timing["camera_time"].to_numpy()
        raw = vtq.check_harp_matches_camera(harp, camera)
        self.assertEqual(raw["count"], 4)
        legacy = vtq.check_harp_matches_camera(harp, camera, tolerance=2)
        self.assertLess(legacy["count"], 4)

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
        timing, checks, triggers = self.load(
            n_exposures=20000, dropped=dropped
        )
        self.assertEqual(
            check(checks, "no_frames_lost")["count"], len(dropped)
        )
        fixed = vtq.correct_video_timing(timing)
        k = (timing["frame_number"] - 1000).to_numpy()
        error = fixed["harp_time"].to_numpy() - triggers[k]
        self.assertLess(np.abs(error).max(), 1e-4)

    def test_glitch_with_drops(self):
        """Glitches next to a drop and on a dropped exposure are fixed."""
        # Trigger 405 lands on the row after the 2-frame drop at 400-401;
        # trigger 900's exposure was itself dropped.
        timing, checks, _ = self.load(
            dropped=[400, 401, 900],
            trigger_errors={403: -0.983, 900: -0.983},
        )
        self.assertOutcome(checks, "re-index")
        self.assertEqual(
            check(checks, "harp_has_no_glitches")["rows"], [403, 900]
        )
        fixed = vtq.correct_video_timing(timing)
        k = (timing["frame_number"] - 1000).to_numpy()
        truth = 100.0 + IFI * k
        self.assertLess(np.abs(fixed["harp_time"] - truth).max(), 1e-4)
        self.assertEqual(
            (fixed["harp_source"] == "glitch_interpolated").sum(), 1
        )

    def test_lost_trigger_refused(self):
        """A trigger missing from the Harp sequence is refused up front."""
        timing, checks, _ = self.load(dropped=[100, 200], lost_trigger=1000)
        self.assertEqual(check(checks, "harp_evenly_spaced")["rows"], [1000])
        self.assertOutcome(checks, "refuse: harp_evenly_spaced")
        with self.assertRaisesRegex(ValueError, "harp_evenly_spaced"):
            vtq.correct_video_timing(timing)

    def test_heavy_drops(self):
        """Half or more frames dropped: corrected; a lost trigger refused."""
        for dropped in [
            np.arange(1, 6000, 2),  # keep exposure 0: first row = first
            np.flatnonzero(np.arange(6000) % 3 != 0),
        ]:
            timing, checks, triggers = self.load(
                n_exposures=6000, dropped=dropped
            )
            self.assertOutcome(checks, "re-index")
            fixed = vtq.correct_video_timing(timing)
            k = (timing["frame_number"] - 1000).to_numpy()
            self.assertLess(
                np.abs(fixed["harp_time"].to_numpy() - triggers[k]).max(),
                1e-4,
            )
            timing, checks, _ = self.load(
                n_exposures=6000, dropped=dropped, lost_trigger=1000
            )
            with self.assertRaisesRegex(ValueError, "harp_evenly_spaced"):
                vtq.correct_video_timing(timing)

    def test_duplicate_frame_cancelling_a_drop_refused(self):
        """A frame saved twice plus one dropped: counts match, but refused."""
        _, triggers = simulate(self.tmp, n_exposures=3000, dropped=[2000])
        timing = vtq.load_video_timing(self.tmp / "bottom_camera.csv")
        # Frame 500 arrives twice; arrival order gives it the next trigger.
        meta = timing[["frame_number", "camera_time"]]
        meta = pd.concat([meta.iloc[:501], meta.iloc[[500]], meta.iloc[501:]])
        timing = pd.DataFrame(
            {
                "harp_time_raw": triggers[:3000],
                "frame_number": meta["frame_number"].to_numpy(),
                "camera_time": meta["camera_time"].to_numpy(),
            }
        )
        checks = vtq.check_video_timing(timing)
        self.assertTrue(check(checks, "no_frames_lost")["passed"])
        self.assertEqual(check(checks, "no_duplicate_frames")["rows"], [501])
        self.assertOutcome(checks, "refuse: no_duplicate_frames")
        with self.assertRaisesRegex(ValueError, "no_duplicate_frames"):
            vtq.correct_video_timing(timing)

    def test_many_small_clock_steps_refused(self):
        """Steps each under tolerance that add up are caught overall."""
        timing, _, _ = self.load(n_exposures=6000)
        for row in range(250, 6000, 250):
            timing.loc[row:, "harp_time_raw"] -= 0.0003
        checks = vtq.check_video_timing(timing)
        self.assertEqual(failed(checks), {"clock_rates_agree"})
        self.assertOutcome(checks, "refuse: clock_rates_agree")

    def test_forward_metadata_jump(self):
        """A lasting forward jump in metadata looks like a burst of drops.

        The CSV alone re-indexes it (a known limit); the trigger log's event
        count refuses it.
        """
        path, triggers = simulate(self.tmp, n_exposures=6000)
        timing = vtq.load_video_timing(path)
        timing.loc[1000:, "frame_number"] += 1048
        timing.loc[1000:, "camera_time"] += 1048 * IFI
        checks = vtq.check_video_timing(timing)
        self.assertOutcome(checks, "re-index")
        log_path = self.tmp / "Event_94.bin"
        write_trigger_log(log_path, triggers)
        log = vtq.read_harp_trigger_log(log_path)
        with self.assertRaisesRegex(ValueError, "exposures"):
            vtq.correct_video_timing(timing, trigger_times=log)
        checks = vtq.check_video_timing(timing, trigger_times=log)
        self.assertEqual(
            vtq.timing_verdict(checks), "exclude: trigger_log_count"
        )
        self.assertEqual(check(checks, "trigger_log_count")["count"], -1048)
        self.assertEqual(
            check(checks, "harp_matches_camera")["message"],
            "skipped: trigger log does not fit the CSV",
        )

    def corrupt_metadata(self, timing, start=1000, stop=1400, offset=1048):
        """Shift frame number and camera time together, as in 763590."""
        rows = timing.index[start:stop]
        timing.loc[rows, "frame_number"] += offset
        timing.loc[rows, "camera_time"] += offset * IFI
        return timing

    def test_corrupted_metadata_uses_harp(self):
        """Corrupted frame numbers, no frames lost: Harp as written."""
        timing, _, _ = self.load(trigger_errors={2000: -0.983})
        timing = self.corrupt_metadata(timing)
        checks = vtq.check_video_timing(timing)
        self.assertTrue(check(checks, "no_frames_lost")["passed"])
        self.assertIn("frame_numbers_increase", failed(checks))
        self.assertOutcome(checks, "fix glitches")
        fixed = vtq.correct_video_timing(timing)
        expected = timing["harp_time_raw"].to_numpy().copy()
        expected[2000] = (expected[1999] + expected[2001]) / 2
        np.testing.assert_array_equal(fixed["harp_time"], expected)
        self.assertEqual(
            fixed["harp_source"].value_counts().to_dict(),
            {"original": 2999, "glitch_interpolated": 1},
        )

    def test_corrupted_metadata_with_drops_refused(self):
        """Corrupted frame numbers plus real drops cannot be re-indexed."""
        timing, _, _ = self.load(dropped=[2500])
        timing = self.corrupt_metadata(timing)
        checks = vtq.check_video_timing(timing)
        self.assertOutcome(checks, "refuse: frame_numbers_increase")
        self.assertEqual(
            check(checks, "harp_matches_camera")["message"],
            "skipped: cannot re-index",
        )
        with self.assertRaisesRegex(ValueError, "frame_numbers_increase"):
            vtq.correct_video_timing(timing)

    def test_more_rows_than_exposures_refused(self):
        """A repeated frame (more rows than exposures) is refused."""
        path, _ = simulate(self.tmp)
        lines = path.read_text().splitlines()
        lines.insert(50, lines[49])
        path.write_text("\n".join(lines) + "\n")
        timing = vtq.load_video_timing(path)
        checks = vtq.check_video_timing(timing)
        self.assertEqual(check(checks, "no_frames_lost")["count"], -1)
        self.assertOutcome(checks, "refuse: no_duplicate_frames")
        with self.assertRaises(ValueError):
            vtq.correct_video_timing(timing)

    def test_video_frame_count(self):
        """Skipped without a count; fails when the count differs."""
        timing, checks, _ = self.load()
        self.assertIsNone(check(checks, "video_frame_count")["passed"])
        checks = vtq.check_video_timing(timing, video_frame_count=2994)
        self.assertEqual(failed(checks), {"video_frame_count"})
        self.assertEqual(check(checks, "video_frame_count")["count"], -6)
        self.assertEqual(
            vtq.timing_verdict(checks), "exclude: video_frame_count"
        )
        checks = vtq.check_video_timing(timing, video_frame_count=3000)
        self.assertEqual(vtq.timing_verdict(checks), "use")

    def test_failed_reindex_in_checks(self):
        """A lasting frame-number jump with no camera-time jump passes the
        input checks but not the re-index trial: excluded by the table."""
        timing, _, _ = self.load(n_exposures=20000)
        timing.loc[10000:, "frame_number"] += 1
        checks = vtq.check_video_timing(timing)
        self.assertEqual(
            failed(checks), {"no_frames_lost", "harp_matches_camera"}
        )
        self.assertEqual(check(checks, "harp_matches_camera")["rows"], [10000])
        with self.assertWarns(DeprecationWarning):  # old: passed, then raised
            self.assertEqual(vtq.timing_action(checks), "re-index")
        self.assertEqual(
            vtq.timing_verdict(checks), "exclude: harp_matches_camera"
        )
        with self.assertRaisesRegex(ValueError, "harp_matches_camera"):
            vtq.correct_video_timing(timing)

    def test_write_video_timing(self):
        """The record holds verdict, method, log use and every check."""
        timing, _, triggers = self.load(dropped=[100, 400])
        log = self.tmp / "Event_94.bin"
        write_trigger_log(log, triggers)
        checks = vtq.check_video_timing(
            timing, vtq.read_harp_trigger_log(log), video_frame_count=2998
        )
        path = vtq.write_video_timing(checks, self.tmp / "out", "bottom")
        self.assertEqual(path.name, "video_timing_bottom.json")
        record = json.loads(path.read_text(), parse_constant=self.fail)
        self.assertEqual(record["verdict"], "use")
        self.assertEqual(record["method"], "re-index")
        self.assertTrue(record["trigger_log"])
        self.assertEqual(record["frames_lost"], 2)
        self.assertEqual(record["glitch_rows"], [])
        self.assertEqual(len(record["checks"]), len(checks))
        self.assertIn(
            "aind_dynamic_foraging_behavior_video_analysis",
            record["versions"],
        )
        checks = vtq.check_video_timing(timing, video_frame_count=1)
        record = json.loads(
            vtq.write_video_timing(checks, self.tmp, "b").read_text()
        )
        self.assertEqual(record["verdict"], "exclude: video_frame_count")
        self.assertIsNone(record["method"])
        self.assertFalse(record["trigger_log"])

    def test_bad_inputs(self):
        """Empty or incomplete CSVs, malformed logs and out-of-order rows
        raise or fail; check_session records an unreadable CSV."""
        (self.tmp / "empty.csv").write_text(
            ",".join(vtq.NEW_LAYOUT_COLUMNS) + "\n"
        )
        (self.tmp / "gap.csv").write_text("1.0,1,1\n1.1,,2\n")
        for name in ["empty.csv", "gap.csv"]:
            with self.assertRaises(ValueError):
                vtq.load_video_timing(self.tmp / name)
        table = vtq.check_session(self.tmp).set_index("camera")
        self.assertEqual(table.loc["gap", "verdict"], "exclude: unreadable")
        log = self.tmp / "Event_94.bin"
        log.write_bytes(bytes(12))
        with self.assertRaisesRegex(ValueError, "13-byte"):
            vtq.read_harp_trigger_log(log)
        log.write_bytes(bytes(13))
        with self.assertRaisesRegex(ValueError, "format"):
            vtq.read_harp_trigger_log(log)
        result = vtq.check_clock_rates_agree([0, 1, 2], [0, 1, 2], [5, 4, 3])
        self.assertFalse(result["passed"])
        with self.assertRaisesRegex(ValueError, "one trigger per frame"):
            vtq.correct_frame_times([0, 1, 2], [0.0, 0.1, 0.2], [0.0])

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
        checks = vtq.check_video_timing(timing, trigger_times=log)
        self.assertEqual(vtq.timing_verdict(checks), "use")
        for name in ["trigger_log_count", "trigger_log_matches_csv"]:
            self.assertIs(check(checks, name)["passed"], True)
        # A log from another session does not match the CSV.
        with self.assertRaisesRegex(ValueError, "does not match"):
            vtq.correct_video_timing(timing, trigger_times=log + 1.0)
        checks = vtq.check_video_timing(timing, trigger_times=log + 1.0)
        self.assertEqual(
            vtq.timing_verdict(checks), "exclude: trigger_log_matches_csv"
        )
        # One event per exposure, no more and no fewer.
        for wrong in [log[:2000], np.append(log, log[-1] + IFI)]:
            with self.assertRaisesRegex(ValueError, "exposures"):
                vtq.correct_video_timing(timing, trigger_times=wrong)
            checks = vtq.check_video_timing(timing, trigger_times=wrong)
            self.assertEqual(
                vtq.timing_verdict(checks), "exclude: trigger_log_count"
            )

    def test_check_session(self):
        """Both layouts are found in a behavior-videos folder."""
        simulate(self.tmp, name="bottom_camera")
        simulate(self.tmp, layout="new", name="SideCameraRight", dropped=[9])
        table = vtq.check_session(self.tmp).set_index("camera")
        self.assertEqual(
            table["verdict"].to_dict(),
            {"bottom_camera": "use", "SideCameraRight": "use"},
        )
        self.assertEqual(table.loc["SideCameraRight", "frames_lost"], 1)


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
