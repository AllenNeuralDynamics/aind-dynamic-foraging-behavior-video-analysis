"""Tests for kinematics.pipeline and kinematics.tasks."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
    pipeline as pl,
    pose_io,
    signal,
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tasks import (
    dynamic_foraging_tongue as dft,
)
from tests.test_video_timing_qc import simulate

FS = 500.0
N = 3000
SESSION = "behavior_123456_2025-01-01_10-00-00"


def synthetic_xy(n=N):
    """A tongue out for 30 frames every 75 (radial out and back from the
    jaw), a jaw at (100, 200) wobbling at 7 Hz, two spouts."""
    t = np.arange(n) / FS
    phase = np.arange(n) % 75
    out = phase < 30
    r = np.where(out, 40 * np.sin(np.pi * phase / 29), 0.0)
    tongue = np.c_[100 + r, 200 + 0.5 * r]
    jaw = np.c_[100 + 2 * np.sin(2 * np.pi * 7 * t), np.full(n, 200.0)]
    xy = np.stack([tongue, jaw, np.tile([150, 180], (n, 1)),
                   np.tile([150, 220], (n, 1))], axis=1)
    lik = np.c_[np.where(out, 0.99, 0.01), np.ones((n, 3))]
    return xy, lik, ["tongue_tip_center", "jaw", "spout_l", "spout_r"]


def synthetic_settings():
    """Tongue as in TONGUE_SETTINGS plus a jaw segmented by speed."""
    tongue = dft.TONGUE_SETTINGS.keypoints[0]
    jaw = pl.KeypointSettings(
        "jaw", smooth=pl.Smooth(cutoff_hz=30.0),
        segments={"jaw_movement": pl.Segmentation(
            "above_threshold", {"on": 20.0, "off": 1.0, "max_gap_frames": 5},
            features=("path", "spectrum"))},
        feature_options={"spectrum": {"column": "x", "band": (2, 20)}})
    return pl.Settings(fs_hz=FS, keypoints=(tongue, jaw),
                       landmarks=("jaw", "spout_l", "spout_r"))


class ProcessTest(unittest.TestCase):
    """process on a synthetic two-keypoint session."""

    @classmethod
    def setUpClass(cls):
        xy, lik, names = synthetic_xy()
        pose, _ = signal.add_time(pose_io.pose_table(xy, lik, names),
                                  np.arange(N) / FS)
        cls.pose = pose
        cls.result = pl.process(pose, synthetic_settings(), session="s1")

    def test_tables_and_columns(self):
        r = self.result
        self.assertEqual(
            set(r), {"pose", "kinematics", "landmarks", "movements",
                     "submovements", "bouts", "jaw_movements", "record"})
        for name, t in r.items():
            if isinstance(t, pd.DataFrame):
                self.assertEqual(t.columns[0], "session", name)
        kin = r["kinematics"]
        for c in pl.KINEMATICS_COLUMNS:
            self.assertIn(c, kin)
        for c in ["movement_id"] + pl.SEGMENT_COLUMNS:
            self.assertIn(c, r["movements"])

    def test_movements(self):
        mv = self.result["movements"]
        self.assertEqual(len(mv), N // 75)
        self.assertEqual(list(mv["movement_id"]), list(range(1, 41)))
        np.testing.assert_allclose(mv["duration"], 29 / FS)
        self.assertTrue((mv["n_valid"] == 30).all())
        # relative to the median jaw: straight out along (1, 0.5)
        np.testing.assert_allclose(mv["angle_deg"],
                                   np.degrees(np.arctan2(0.5, 1)), atol=1.0)
        self.assertTrue((mv["amplitude"] > 40).all())
        self.assertTrue((mv["bout_id"] == 1).all())

    def test_submovements_and_bouts(self):
        sub = self.result["submovements"]
        # every reversal is kept: out, back, and the 1-frame flip at the
        # end where the tongue is back at the jaw
        first = sub[sub["movement_id"] == 1]
        self.assertEqual(list(first["direction"]), [1, -1, 1])
        self.assertEqual(list(first["n_frames"]), [15, 14, 1])
        self.assertEqual(list(first["number_in_movement"]), [1, 2, 3])
        self.assertEqual(list(np.sign(first["radial_velocity_sum"])),
                         [1, -1, 1])
        np.testing.assert_array_equal(
            np.sign(first["radial_velocity_peak"]), [1, -1, 1])
        bouts = self.result["bouts"]
        self.assertEqual(int(bouts["n_movements"].iloc[0]), 40)
        self.assertAlmostEqual(bouts["rate_hz"].iloc[0], FS / 75)

    def test_frame_labels(self):
        kin = self.result["kinematics"]
        tongue = kin[kin["keypoint"] == "tongue_tip_center"]
        self.assertEqual(int(tongue["movement_id"].notna().sum()), 40 * 30)
        jaw = kin[kin["keypoint"] == "jaw"]
        self.assertTrue(jaw["movement_id"].isna().all())
        self.assertTrue(jaw["radial_velocity"].isna().all())

    def test_jaw_rhythm(self):
        jm = self.result["jaw_movements"]
        self.assertEqual(len(jm), 1)  # speed dips are shorter than 5 frames
        self.assertAlmostEqual(jm["dominant_hz"].iloc[0], 7.0, delta=1.0)

    def test_record_and_round_trip(self):
        rec = self.result["record"]
        self.assertEqual(rec["keypoints"]["tongue_tip_center"]["n_valid"],
                         40 * 30)
        s = synthetic_settings()
        self.assertEqual(pl.Settings.from_dict(s.to_dict()).to_dict(),
                         s.to_dict())
        self.assertEqual(pl.Settings.from_dict(s.to_dict()).keypoints[0], s
                         .keypoints[0])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(pl.is_done(tmp))
            pl.write(self.result, tmp)
            self.assertTrue(pl.is_done(tmp))
            back = pl.read(tmp)
            self.assertEqual(set(back), set(self.result))
            pd.testing.assert_frame_equal(back["movements"],
                                          self.result["movements"])
            self.assertEqual(back["record"]["session"], "s1")
            only = pl.read(tmp, tables=["movements"])
            self.assertEqual(set(only), {"movements", "record"})

    def test_record_with_numpy_values(self):
        result = {"record": {"a": np.int64(1), "b": np.float32(0.5),
                             "c": np.arange(2)}}
        with tempfile.TemporaryDirectory() as tmp:
            pl.write(result, tmp)
            self.assertEqual(pl.read_record(tmp),
                             {"a": 1, "b": 0.5, "c": [0, 1]})
            with self.assertRaises(TypeError):
                pl.write({"record": {"x": object()}}, tmp)

    def test_errors(self):
        s = synthetic_settings()
        with self.assertRaises(ValueError):
            pl.process(self.pose.drop(columns="time"), s)
        missing = pl.Settings(fs_hz=FS, keypoints=(
            pl.KeypointSettings("nose"),))
        with self.assertRaises(ValueError):
            pl.process(self.pose, missing)
        no_ref = pl.Settings(fs_hz=FS, keypoints=(pl.KeypointSettings(
            "jaw", segments={"x": pl.Segmentation("zero_crossings")}),))
        with self.assertRaises(ValueError):
            pl.process(self.pose, no_ref)
        bad = pl.Settings(fs_hz=FS, keypoints=(pl.KeypointSettings(
            "jaw", segments={"x": pl.Segmentation("nope")}),))
        with self.assertRaises(ValueError):
            pl.process(self.pose, bad)
        minima = pl.Settings(fs_hz=FS, keypoints=(pl.KeypointSettings(
            "jaw", segments={"x": pl.Segmentation("speed_minima")}),))
        self.assertIn("xs", pl.process(self.pose, minima))


def write_pose_csv(path, xy, lik, names):
    """Write a Lightning Pose style CSV."""
    cols = pd.MultiIndex.from_tuples(
        [("lp", k, c) for k in names for c in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"])
    data = np.concatenate(
        [np.c_[xy[:, j], lik[:, j]] for j in range(len(names))], axis=1)
    pd.DataFrame(data, columns=cols).to_csv(path)


def task_tables():
    """Trials and licks on session time = Harp - 101 s.

    Movement k starts at session time -1 + 0.15 k and lasts 0.058 s. Go
    cues at 0, 1.5 and 2.03 s (the last inside movement 21). The last lick
    is 5 ms before movement 25 starts: not matched, but recorded as close.
    """
    go_harp = np.array([101.0, 102.5, 103.03])
    trials = pd.DataFrame({"trial": [1, 2, 3], "goCue_start_time": go_harp,
                           "goCue_start_time_in_session": go_harp - 101.0,
                           "animal_response": [0, 1, 0]})
    start = lambda k: -1 + 0.15 * k  # noqa: E731
    licks = pd.DataFrame({
        "timestamps": [start(8) + 0.03, start(9) + 0.03, 0.12,
                       start(18) + 0.03, start(25) - 0.005],
        "trial": [1, 1, 1, 2, 3],
        "cue_response": [True, False, False, True, False],
        "rewarded": [True, False, False, False, False],
        "event": ["left_lick_time", "left_lick_time", "right_lick_time",
                  "right_lick_time", "left_lick_time"],
    })
    return trials, licks


class TaskTest(unittest.TestCase):
    """dynamic_foraging_tongue end to end on simulated files."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data_root = root / "data"
        videos = cls.data_root / SESSION / "behavior-videos"
        videos.mkdir(parents=True)
        cls.video_csv, _ = simulate(videos)
        cls.pose_path = root / f"{SESSION}_preds.csv"
        write_pose_csv(cls.pose_path, *synthetic_xy())
        cls.trials, cls.licks = task_tables()
        cls.result = dft.run_session(cls.pose_path, cls.video_csv,
                                     cls.trials, cls.licks)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_clock_and_record(self):
        rec = self.result["record"]
        self.assertEqual(rec["session"], SESSION)
        self.assertAlmostEqual(rec["clock"]["time_zero"], 101.0)
        self.assertEqual(rec["timing"]["verdict"], "use")
        mv = self.result["movements"]
        self.assertAlmostEqual(mv["start_time"].iloc[0], -1.0, places=4)

    def test_movement_trial_and_licks(self):
        mv = self.result["movements"].set_index("movement_id")
        # ids are 1-based: movement k has id k + 1
        self.assertTrue(pd.isna(mv.loc[7, "trial"]))
        self.assertEqual(mv.loc[8, "trial"], 1)
        self.assertAlmostEqual(mv.loc[8, "latency_from_go"], 0.05, places=4)
        self.assertEqual(mv.loc[8, "number_in_trial"], 1)
        # movement 21 spans the go cue at 2.03 s: no trial
        self.assertTrue(mv.loc[21, "spans_go_cue"])
        self.assertTrue(pd.isna(mv.loc[21, "trial"]))
        self.assertTrue(pd.isna(mv.loc[21, "latency_from_go"]))
        self.assertTrue(pd.isna(mv.loc[21, "number_in_trial"]))
        self.assertFalse(mv.loc[20, "spans_go_cue"])
        self.assertEqual(mv.loc[20, "trial"], 2)
        self.assertEqual(mv.loc[22, "number_in_trial"], 1)
        # the last trial ends 1.5 s (the longest go-cue interval) after
        # its go cue, at 3.53 s: movement 31 starts at 3.5, 32 at 3.65
        self.assertEqual(mv.loc[31, "trial"], 3)
        self.assertTrue(pd.isna(mv.loc[32, "trial"]))
        self.assertEqual(mv.loc[9, "n_licks"], 1)
        self.assertTrue(mv.loc[9, "cue_response"])
        self.assertEqual(mv.loc[9, "lick_side"], "left")
        self.assertFalse(mv.loc[8, "cue_response"])
        self.assertEqual(str(mv["cue_response"].dtype), "boolean")
        licks = self.result["licks"]
        self.assertEqual(list(licks["movement_id"].astype(object)
                              .where(licks["movement_id"].notna(), None)),
                         [9, 10, None, 19, None])
        # the near miss: closest movement 26 (k = 25), 5 ms before it
        self.assertEqual(int(licks["closest_movement_id"].iloc[4]), 26)
        self.assertAlmostEqual(licks["closest_movement_offset_s"].iloc[4],
                               -0.005, places=4)
        self.assertEqual(list(licks["closest_movement_offset_s"].iloc[:2]),
                         [0.0, 0.0])
        self.assertEqual(int(mv.loc[26, "n_licks"]), 0)

    def test_submovements_get_licks_and_features(self):
        sub = self.result["submovements"]
        for c in ("amplitude", "out_duration", "n_licks", "lick_time"):
            self.assertIn(c, sub)
        licks = self.result["licks"]
        matched = licks.dropna(subset=["submovement_id"])
        self.assertEqual(len(matched), 3)  # the same licks as movements
        parent = sub.set_index("submovement_id")["movement_id"]
        self.assertEqual(
            list(parent.loc[matched["submovement_id"].astype(int)]),
            list(matched["movement_id"].astype(int)))
        self.assertEqual(int(sub["n_licks"].sum()), 3)

    def test_trial_summary_and_quality(self):
        tr = self.result["trials"].set_index("trial")
        self.assertEqual(tr.loc[1, "n_movements"], 10)
        self.assertEqual(tr.loc[1, "first_movement_id"], 8)
        self.assertAlmostEqual(tr.loc[1, "first_movement_latency"], 0.05,
                               places=4)
        self.assertEqual(tr.loc[1, "cue_response_movement_id"], 9)
        self.assertEqual(tr.loc[1, "cue_response_movement_number"], 2)
        self.assertAlmostEqual(tr.loc[1, "lick_latency"], 0.23, places=4)
        self.assertEqual(tr.loc[1, "n_licks"], 3)
        self.assertAlmostEqual(tr.loc[1, "lick_coverage"], 2 / 3)
        self.assertEqual(tr.loc[2, "n_lick_movements"], 1)
        self.assertEqual(tr.loc[2, "n_movements"], 3)  # not the spanning one
        self.assertEqual(tr.loc[3, "n_movements"], 10)  # capped window
        self.assertIn("animal_response", tr)
        q = self.result["record"]["quality"]
        self.assertAlmostEqual(q["lick_coverage"], 0.6)
        self.assertAlmostEqual(q["lick_coverage_within_10ms"], 0.8)
        self.assertAlmostEqual(q["movement_duration_p50"], 0.058, places=4)

    def test_time_zero_needs_both_go_cue_columns(self):
        with self.assertRaisesRegex(ValueError, "pass time_zero"):
            dft.session_time_zero(
                self.trials.drop(columns="goCue_start_time"))

    def test_find_inputs_and_names(self):
        files = dft.find_inputs(self.data_root / SESSION)
        self.assertEqual(files["video_csv"], self.video_csv)
        self.assertIsNone(files["trigger_log"])
        self.assertEqual(dft.session_name_from_path(self.pose_path), SESSION)
        with self.assertRaises(ValueError):
            dft.session_name_from_path("nothing_here.csv")
        with self.assertRaises(FileNotFoundError):
            dft.find_inputs(self.data_root / SESSION, camera="SideCamera")
        with self.assertRaises(FileNotFoundError):
            dft.find_inputs(Path(self.tmp.name) / "empty")

    def test_load_nwb_tables(self):
        try:
            import aind_dynamic_foraging_data_utils.nwb_utils as nwb_utils
            from aind_dynamic_foraging_basic_analysis.licks import annotation
            from aind_dynamic_foraging_behavior_video_analysis.kinematics import (  # noqa: E501
                kinematics_nwb_utils,
            )
        except ImportError as e:
            self.skipTest(f"AIND NWB libraries not installed: {e}")
        nwb = mock.Mock()
        nwb.trials = {"goCue_start_time": [101.0, 102.5]}
        calls = []

        def trials(n, **kw):
            calls.append(kw)
            if "cut_interruptions" in kw:
                raise TypeError("unexpected keyword argument "
                                "'cut_interruptions'")
            return self.trials

        with mock.patch.object(kinematics_nwb_utils, "get_nwb_file",
                               return_value=nwb), \
                mock.patch.object(nwb_utils, "create_df_events",
                                  return_value="events"), \
                mock.patch.object(nwb_utils, "create_df_trials",
                                  side_effect=trials), \
                mock.patch.object(annotation, "annotate_licks",
                                  return_value=self.licks):
            tr, lk, ev, t0 = dft.load_nwb_tables(SESSION, nwb_folder="x")
            dft.load_nwb_tables(SESSION, ignore_trial_errors=True)
        self.assertIs(tr, self.trials)
        self.assertIs(lk, self.licks)
        self.assertEqual(ev, "events")
        self.assertEqual(t0, 101.0)
        self.assertEqual(calls, [
            {"cut_interruptions": True}, {},
            {"cut_interruptions": True, "ignore_errors": True},
            {"ignore_errors": True}])

    def test_flag_reward_before_choice(self):
        trials = pd.DataFrame({
            "trial": [1, 2, 3, 4, 5],
            "earned_reward": [True, True, True, False, True],
            "extra_reward": [False, False, True, False, False],
            "choice_time_in_session": [1.0, 5.0, 9.0, 13.0, np.nan],
            "reward_time_in_session": [1.2, 4.0, 8.0, np.nan, 17.0]})
        out = dft.flag_reward_before_choice(trials)
        # 2: earned reward 1 s before the choice; 3 is a manual reward
        self.assertEqual(list(out["reward_before_choice"]),
                         [False, True, False, False, False])
        missing = dft.flag_reward_before_choice(trials[["trial"]])
        self.assertTrue(missing["reward_before_choice"].isna().all())
        self.assertEqual(
            self.result["record"]["quality"]["n_trials_reward_before_choice"],
            0)
        self.assertIn("reward_before_choice", self.result["trials"])

    def test_compare_old_new_script(self):
        path = Path(__file__).parents[1] / "scripts" / \
            "compare_old_new_kinematics.py"
        spec = importlib.util.spec_from_file_location("cmp", path)
        cmp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmp)
        root = Path(self.tmp.name) / "cmp"
        new_dir = root / "new" / SESSION
        pl.write(self.result, new_dir)
        mv = self.result["movements"]
        jaw = self.result["landmarks"].set_index("keypoint").loc["jaw"]
        old = pd.DataFrame({
            "start_time": mv["start_time"], "end_time": mv["end_time"],
            "duration": mv["duration"], "n_datapoints": mv["n_valid"],
            "total_distance": mv["path_length"],
            "peak_velocity": mv["peak_speed"] * 1.01,
            "endpoint_x": mv["end_x"] + jaw["x"],
            "has_lick": mv["n_licks"] > 0})
        inter = root / "old" / SESSION / "intermediate_data"
        inter.mkdir(parents=True)
        old.to_parquet(inter / "tongue_movs.parquet")
        licks = self.result["licks"].rename(
            columns={"movement_id": "nearest_movement_id"})
        licks.to_parquet(inter / "nwb_df_licks.parquet")
        (inter.parent / "tongue_quality_stats.json").write_text(
            json.dumps({"coverage_pct": 75.0}))
        (root / "old" / "behavior_1_2025-01-01_00-00-00").mkdir()
        table = cmp.compare_roots(root / "old", root / "new")
        rows = table[table["session"] == SESSION].set_index("item")
        self.assertEqual(rows.loc["movements matched", "note"],
                         "100.00% of old")
        self.assertEqual(rows.loc["duration", "note"],
                         "median 0.00%, p99 0.00%")
        self.assertTrue(rows.loc["peak_velocity", "note"]
                        .startswith("median 0.99%"))
        self.assertEqual(rows.loc["endpoint_x", "note"],
                         "median 0.00%, p99 0.00%")
        self.assertEqual(rows.loc["has_lick", "note"], "100.00% agree")
        self.assertEqual(rows.loc["lick coverage", "old"],
                         rows.loc["lick coverage", "new"])
        self.assertEqual(rows.loc["quality lick_coverage", "new"], 0.6)

    def test_run_batch(self):
        # as from data-utils: no Harp-time go cue column on trials
        tables = (self.trials.drop(columns="goCue_start_time"), self.licks,
                  pd.DataFrame({"event": ["x"]}), 101.0)
        out = Path(self.tmp.name) / "out"
        bad = Path(self.tmp.name) / "behavior_999999_2025-01-01_10-00-00.csv"
        with mock.patch.object(dft, "load_nwb_tables",
                               return_value=tables) as load:
            status = dft.run_batch([self.pose_path, bad], self.data_root,
                                   out)
            self.assertEqual(list(status["status"].str[:5]),
                             ["done", "error"])
            self.assertTrue((out / "errors.txt").exists())
            self.assertFalse(pl.read_record(out / SESSION)["nwb"]
                             ["ignore_trial_errors"])
            again = dft.run_batch([self.pose_path], self.data_root, out)
            dft.run_batch([self.pose_path], self.data_root, out, force=True,
                          ignore_trial_errors=True)
            self.assertEqual(load.call_args.args, (SESSION, None, True))
            self.assertTrue(pl.read_record(out / SESSION)["nwb"]
                            ["ignore_trial_errors"])
            self.assertEqual(list(again["status"]), ["skipped"])
            strict = dft.run_batch([self.pose_path], self.data_root, out,
                                   use_trigger_log=True, force=True)
            self.assertIn("Event_94", strict["status"].iloc[0])
            screen = pd.DataFrame({
                "session": [SESSION, "behavior_999999_2025-01-01_10-00-00"],
                "camera": ["bottom_camera", "bottom_camera"],
                "use": [False, True],
                "reason": ["timing: exclude: no_frames_lost", ""]})
            gated = dft.run_batch([self.pose_path, bad, self.pose_path],
                                  self.data_root, out, force=True,
                                  screen=screen, camera="SideCamera")
            self.assertEqual(list(gated["status"]),
                             ["excluded: not screened"] * 3)
            gated = dft.run_batch([self.pose_path], self.data_root, out,
                                  force=True, screen=screen)
            self.assertEqual(list(gated["status"]),
                             ["excluded: timing: exclude: no_frames_lost"])
        q = dft.quality_summary(out / SESSION)
        self.assertEqual(q["session"], SESSION)
        self.assertAlmostEqual(q["lick_coverage"], 0.6)
        saved = pl.read(out / SESSION, tables=["events", "trials"])
        self.assertEqual(saved["events"]["session"].iloc[0], SESSION)


if __name__ == "__main__":
    unittest.main()
