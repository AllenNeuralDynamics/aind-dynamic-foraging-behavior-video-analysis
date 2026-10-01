"""Tests for video_quality_qc, video_quality_report and
video_alignment.task_frame_window.

Videos are synthetic MP4s encoded here with PyAV (h264 with B-frames, a
fixed GOP, luma written directly so values are not range-converted): a
smooth random texture with a moving blob, plus the fault under test on
chosen frames.
"""

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import av
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from aind_video_utils import probe, read_mp4_frame_index  # noqa: E402

from aind_dynamic_foraging_behavior_video_analysis import (  # noqa: E402
    video_quality_qc as vqq,
)
from aind_dynamic_foraging_behavior_video_analysis import (  # noqa: E402
    video_quality_report as vqr,
)
from aind_dynamic_foraging_behavior_video_analysis.video_alignment import (  # noqa: E402,E501
    behavior_time_to_frame_index,
    read_trial_times,
    task_frame_window,
)
from tests.test_video_timing_qc import write_trigger_log  # noqa: E402

# Synthetic Harp clock: frame i at FIRST_HARP + i / FPS (on the 32 us grid).
FIRST_HARP = 1000.0

W, H = 160, 120
GOP = 10
FPS = 50


def _smooth(image, passes=3):
    """Repeated 3x3 box blur (edges wrap)."""
    out = image.astype(np.float64)
    for _ in range(passes):
        out = (
            sum(
                np.roll(np.roll(out, dy, 0), dx, 1)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            )
            / 9
        )
    return out


def _texture(seed=0):
    """Fixed textured background, values about 40-200."""
    rng = np.random.default_rng(seed)
    t = _smooth(rng.uniform(0, 255, (H, W)), passes=2)
    return 40 + 160 * (t - t.min()) / (t.max() - t.min())


def _frame(i, texture, fault=None):
    """Luma for frame ``i``: texture, a small moving blob, then ``fault``."""
    y = texture.copy()
    left = 30 + (i * 2) % 100
    blob = slice(left, left + 10)
    y[55:65, blob] += 40
    if fault is not None:
        y = fault(i, y)
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def write_mp4(path, n_frames=300, fault=None, container_format=None):
    """Encode a synthetic video; ``fault(i, luma) -> luma`` alters frames."""
    texture = _texture()
    options = {
        "x264-params": f"keyint={GOP}:min-keyint={GOP}:scenecut=0:bframes=2",
        "crf": "12",
    }
    with av.open(str(path), "w", format=container_format) as out:
        stream = out.add_stream("libx264", rate=FPS, options=options)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            yuv = np.full((H * 3 // 2, W), 128, dtype=np.uint8)
            yuv[:H] = _frame(i, texture, fault)
            frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return Path(path)


def write_video_csv(path, n_frames=300, harp_step_at=None, drop_at=None):
    """New/AIND layout CSV, one row per saved frame.

    ``harp_step_at``: Harp jumps 0.5 s from that row on (a clock step).
    ``drop_at``: the exposure at that row was lost (frame number and camera
    time skip one frame; Harp stays in arrival order).
    """
    i = np.arange(n_frames)
    exposure = i if drop_at is None else i + (i >= drop_at)
    harp = FIRST_HARP + i / FPS
    if harp_step_at is not None:
        harp = harp + 0.5 * (i >= harp_step_at)
    pd.DataFrame(
        {
            "ReferenceTime": harp,
            "CameraFrameNumber": 5000 + exposure,
            "CameraFrameTime": (7_000_000_000 + exposure * 1e9 / FPS).astype(
                "int64"
            ),
        }
    ).to_csv(path, index=False)
    return Path(path)


def write_behavior_json(path, first_frame, last_frame, harp=True):
    """A session JSON whose trials span ``first_frame`` .. ``last_frame``."""
    starts = FIRST_HARP + np.linspace(first_frame, last_frame - 20, 4) / FPS
    ends = starts + 20 / FPS
    obj = {
        "B_TrialEndTime": list(ends),
        "B_TrialStartTimeHarp": list(starts),
        "B_TrialEndTimeHarp": list(ends) if harp else [],
        "B_GoCueTimeSoundCard": list(starts + 0.1),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj))
    return Path(path)


def after(k, change):
    """A fault applied from frame ``k`` on."""
    return lambda i, y: change(y) if i >= k else y


def between(k, m, change):
    """A fault applied to frames ``k`` .. ``m - 1``."""
    return lambda i, y: change(y) if k <= i < m else y


def blur(y):
    """Heavy defocus."""
    return _smooth(y, passes=4)


class TempDirTest(unittest.TestCase):
    """Creates and removes a temporary folder per class."""

    @classmethod
    def setUpClass(cls):
        """Make the folder."""
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        """Remove the folder."""
        shutil.rmtree(cls.tmp)


def measure_mp4(path, window=(0, 300)):
    """Sample and measure a video (every keyframe of a 300-frame file)."""
    frames, samples, color_range = vqq.sample_keyframes(path, window)
    return frames, vqq.measure(frames, samples, color_range)


def check_names(checks, passed):
    """Names of the checks that passed (or failed)."""
    return checks.loc[checks["passed"] == passed, "check"].tolist()


# --- Per-frame metrics ---------------------------------------------------


class FrameMetricTest(unittest.TestCase):
    """Metric functions on arrays."""

    def test_downsample2_drops_odd_edges(self):
        """2x2 means; an odd row or column is dropped."""
        y = np.arange(15, dtype=np.uint8).reshape(3, 5)
        d = vqq.downsample2(y)
        self.assertEqual(d.shape, (1, 2))
        self.assertAlmostEqual(float(d[0, 0]), (0 + 1 + 5 + 6) / 4)

    def test_laplacian_variance_falls_with_blur(self):
        """Blurring lowers sharpness; a flat image has none."""
        t = _texture()
        self.assertGreater(
            vqq.laplacian_variance(t), 5 * vqq.laplacian_variance(blur(t))
        )
        self.assertEqual(vqq.laplacian_variance(np.full((9, 9), 7.0)), 0)

    def test_noise_sigma_recovers_added_noise(self):
        """Gaussian noise of sigma 5 on a flat image reads about 5."""
        rng = np.random.default_rng(1)
        y = 100 + rng.normal(0, 5, (200, 200))
        self.assertAlmostEqual(vqq.noise_sigma(y), 5, delta=0.3)

    def test_phase_shift_positive_and_negative(self):
        """Recovers whole shifts in both directions."""
        t = _texture()
        for dy, dx in [(3, 5), (-4, -2), (0, 0)]:
            moved = np.roll(t, (dy, dx), axis=(0, 1))
            got = vqq.phase_shift(t, moved)
            self.assertAlmostEqual(got[0], dx, delta=0.25)
            self.assertAlmostEqual(got[1], dy, delta=0.25)

    def test_parabolic_offset_flat_peak(self):
        """A flat neighbourhood gives no offset."""
        self.assertEqual(vqq._parabolic_offset(1.0, 1.0, 1.0), 0.0)

    def test_similarity(self):
        """1 for identical, 0 when either image is flat."""
        t = _texture()
        self.assertAlmostEqual(vqq.similarity(t, t), 1.0)
        self.assertEqual(vqq.similarity(t, np.ones_like(t)), 0.0)

    def test_measure_on_arrays(self):
        """Intensity statistics, the histogram and the deviations."""
        frames = np.stack(
            [np.full((4, 4), v, dtype=np.uint8) for v in (0, 100)]
        )
        frames[1, 0, :2] = (17, 235)
        samples = pd.DataFrame({"frame_index": [1, 2], "video_time": [0, 1]})
        out = vqq.measure(frames, samples, "tv")
        black, grey = out.iloc[0], out.iloc[1]
        self.assertEqual(black["contrast_rms"], 0.0)  # not 0 / 0
        self.assertEqual(black["pct_clipped_low"], 100)
        self.assertEqual(black["entropy_bits"], 0)
        self.assertEqual(grey["pct_clipped_high"], 100 / 16)
        self.assertEqual(grey["pct_clipped_low"], 0)  # 17 is above 16
        self.assertEqual(grey["histogram"][100], 14)
        self.assertEqual(out["histogram"].map(sum).tolist(), [16, 16])
        self.assertAlmostEqual(grey["mean"], (14 * 100 + 17 + 235) / 16)
        self.assertEqual(grey["dynamic_range"], grey["p99"] - grey["p1"])
        self.assertEqual(out["color_range"].tolist(), ["tv", "tv"])
        self.assertTrue((out["mean_dev"] > 0).all())

    def test_clipping_counts_the_tagged_range(self):
        """TV range clips at 16 / 235 inclusive; full range at 0 / 255."""
        frames = np.array([[[16, 17, 235, 234]]], dtype=np.uint8)
        samples = pd.DataFrame({"frame_index": [1], "video_time": [0.0]})
        tv = vqq.measure(frames, samples, "tv").iloc[0]
        pc = vqq.measure(frames, samples, "pc").iloc[0]
        self.assertEqual((tv.pct_clipped_low, tv.pct_clipped_high), (25, 25))
        self.assertEqual((pc.pct_clipped_low, pc.pct_clipped_high), (0, 0))


# --- Sampling ------------------------------------------------------------


class SamplingTest(TempDirTest):
    """Keyframe choice and seeks on a clean synthetic video."""

    @classmethod
    def setUpClass(cls):
        """Encode one clean video (keyframes every 10 frames)."""
        super().setUpClass()
        cls.path = write_mp4(cls.tmp / "clean.mp4")
        cls.index = read_mp4_frame_index(cls.path)

    def keys(self, window):
        """Frame indices sampled in ``window``."""
        return vqq.sample_keyframes(self.path, window)[1]["frame_index"]

    def test_file_has_b_frames_and_edit_list(self):
        """The case the frame mapping must handle."""
        self.assertTrue(self.index.edits)
        self.assertFalse(np.array_equal(self.index.pts, self.index.dts))

    def test_never_frame_zero_nor_edges(self):
        """Frame 0 and the 1% edges are never sampled; with a 10% edge
        nothing below frame 30 or from 270 on."""
        self.assertEqual(
            self.keys((0, 300)).tolist(), list(range(10, 300, 10))
        )
        with mock.patch.object(vqq, "EDGE_FRACTION", 0.1):
            self.assertEqual(
                self.keys((0, 300)).tolist(), list(range(30, 270, 10))
            )

    def test_window(self):
        """Only keyframes inside ``[start, end)``."""
        self.assertEqual(
            self.keys((50, 101)).tolist(), [50, 60, 70, 80, 90, 100]
        )

    def test_middle_fraction_by_default(self):
        """No window: the middle 50% of the file, frames 75 to 224."""
        self.assertEqual(self.keys(None).tolist(), list(range(80, 225, 10)))

    def test_evenly_spread_when_more_keyframes_than_samples(self):
        """Fewer samples than keyframes: evenly spread, ends included."""
        with mock.patch.object(vqq, "N_SAMPLES", 5):
            self.assertEqual(
                self.keys((0, 300)).tolist(), [10, 80, 150, 220, 290]
            )

    def test_empty_window_raises(self):
        """No keyframe in the window."""
        with self.assertRaisesRegex(ValueError, "No keyframe"):
            self.keys((1000, 2000))

    def test_seeks_return_the_keyframes(self):
        """Each frame is the keyframe asked for, at its video time, with
        the luma as coded (not stretched to full range, which would move
        the mean and minimum by several units)."""
        frames, samples, color_range = vqq.sample_keyframes(
            self.path, (10, 151)
        )
        self.assertEqual(frames.shape, (15, H, W))
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(color_range, "unknown")  # the encoder sets no tag
        np.testing.assert_allclose(
            samples["video_time"], samples["frame_index"] / FPS
        )
        texture = _texture()
        for luma, i in zip(frames, samples["frame_index"]):
            expected = _frame(i, texture)
            self.assertLess(np.abs(luma - expected.astype(float)).mean(), 1.5)
            self.assertAlmostEqual(luma.mean(), expected.mean(), delta=0.5)
            self.assertAlmostEqual(luma.min(), expected.min(), delta=4)

    def test_luma_plane_refuses_other_formats(self):
        """RGB frames have no luma plane."""
        with self.assertRaisesRegex(ValueError, "pixel format"):
            vqq.luma_plane(av.VideoFrame(16, 16, "rgb24"))

    def test_unsafe_edit_list_refused(self):
        """An edit list that drops frames cannot be frame-addressed."""
        unsafe = mock.Mock(is_frame_addressing_safe=lambda: False)
        with mock.patch.object(
            vqq, "read_mp4_frame_index", return_value=unsafe
        ):
            with self.assertRaisesRegex(ValueError, "frame-addressing"):
                self.keys((0, 300))

    def test_wrong_landing_refused(self):
        """A seek that lands elsewhere raises instead of mislabelling:
        an index claiming every frame is a keyframe asks for frame 1."""
        lying = dataclasses.replace(
            self.index, is_keyframe=np.ones_like(self.index.is_keyframe)
        )
        with mock.patch.object(
            vqq, "read_mp4_frame_index", return_value=lying
        ):
            with self.assertRaisesRegex(ValueError, "returned pts"):
                self.keys((0, 300))

    def test_urls_get_timeouts(self):
        """A URL is opened with HTTP_OPTIONS, a file with none."""
        url = "https://example.org/clean.mp4"
        opened, real_open = [], av.open

        def fake_open(path, options):
            """Open the local file instead; record the options."""
            opened.append(options)
            return real_open(str(self.path), options=options)

        with (
            mock.patch.object(vqq, "probe", return_value=probe(self.path)),
            mock.patch.object(
                vqq, "read_mp4_frame_index", return_value=self.index
            ),
            mock.patch.object(vqq.av, "open", side_effect=fake_open),
        ):
            vqq.sample_keyframes(url, (10, 21))
            vqq.sample_keyframes(self.path, (10, 21))
        self.assertEqual(opened, [vqq.HTTP_OPTIONS, {}])

    def test_not_mp4_refused(self):
        """Matroska is refused."""
        path = write_mp4(self.tmp / "clip.mkv", n_frames=30)
        with self.assertRaisesRegex(ValueError, "Not an MP4"):
            vqq.sample_keyframes(path)


# --- Checks on synthetic faults ------------------------------------------


class CheckTest(TempDirTest):
    """Each check passes on a clean video and fails on its fault."""

    def run_video(self, name, fault=None, camera="bottom_camera"):
        """Encode, measure and check one video; return samples, checks."""
        path = write_mp4(self.tmp / f"{name}.mp4", fault=fault)
        _, samples = measure_mp4(path)
        return samples, vqq.run_checks(samples, camera)

    def test_clean(self):
        """Every check passes; the side camera also gets the clipping
        check. Names come from the table."""
        samples, checks = self.run_video("clean")
        self.assertEqual(len(samples), 29)
        self.assertLess(samples["shift"].max(), 0.5)
        self.assertGreater(samples["similarity"].min(), 0.9)
        self.assertEqual(vqq.quality_action(checks), "use")
        self.assertEqual(
            check_names(checks, True),
            [
                "sharpness_dev <= 0.45",
                "mean_dev <= 0.15",
                "similarity >= 0.7",
                "similarity p5 < 0.998",
                "mean median >= 50",
                "mean median <= 150",
            ],
        )
        side = vqq.run_checks(samples, "SideCameraRight")
        self.assertEqual(
            side["check"].iloc[-1], "pct_clipped_high median <= 3.75"
        )
        self.assertTrue(side["passed"].all())
        row = side.iloc[-1]
        self.assertEqual(
            (row.metric, row.over, row.op, row.value),
            ("pct_clipped_high", "median", "<=", 3.75),
        )
        self.assertEqual(row.observed, samples["pct_clipped_high"].median())

    def test_camera_view(self):
        """Either folder layout's name; anything else matches no view."""
        self.assertEqual(vqq.camera_view("BottomCamera"), "bottom")
        self.assertEqual(vqq.camera_view("side_camera_right"), "side")
        self.assertIsNone(vqq.camera_view(None))

    def test_defocus(self):
        """Blur from frame 200 on: 10 samples in a run."""
        _, checks = self.run_video("defocus", after(200, blur))
        self.assertEqual(
            vqq.quality_action(checks), "exclude: sharpness_dev <= 0.45"
        )
        row = checks.set_index("check").loc["sharpness_dev <= 0.45"]
        self.assertEqual(row["observed"], 10)
        self.assertEqual(row["samples"], list(range(19, 29)))

    def test_isolated_sample_passes(self):
        """One blurred keyframe (frame 150) is listed but passes; two
        consecutive ones fail."""
        _, checks = self.run_video("blip", between(145, 155, blur))
        row = checks.set_index("check").loc["sharpness_dev <= 0.45"]
        self.assertTrue(row["passed"])
        self.assertEqual((row["observed"], row["samples"]), (0, [14]))
        _, checks = self.run_video("blip2", between(145, 165, blur))
        row = checks.set_index("check").loc["sharpness_dev <= 0.45"]
        self.assertFalse(row["passed"])
        self.assertEqual((row["observed"], row["samples"]), (2, [14, 15]))

    def test_brightness_step(self):
        """Lights brighten by 40% from frame 200."""
        _, checks = self.run_video("bright", after(200, lambda y: y * 1.4))
        self.assertIn("mean_dev <= 0.15", check_names(checks, False))

    def test_too_dark_or_too_bright(self):
        """Median mean luma below 50 or above 150, whole session; the
        stability checks see no change."""
        _, dark = self.run_video("dark", lambda i, y: y * 0.3)
        _, bright = self.run_video("bright_all", lambda i, y: y + 90)
        self.assertEqual(check_names(dark, False), ["mean median >= 50"])
        self.assertEqual(check_names(bright, False), ["mean median <= 150"])
        self.assertLess(
            dark.set_index("check").loc["mean median >= 50", "observed"], 50
        )

    def test_occlusion(self):
        """Left half of the frame black from frame 200."""

        def occlude(y):
            """Black out the left half."""
            y = y.copy()
            y[:, : W // 2] = 16
            return y

        _, checks = self.run_video("occluded", after(200, occlude))
        self.assertIn("similarity >= 0.7", check_names(checks, False))

    def test_still_scene(self):
        """Every frame the same: only the "does anything move" check
        fails."""
        still = _frame(0, _texture()).astype(float)
        _, checks = self.run_video("still", lambda i, y: still)
        self.assertEqual(check_names(checks, False), ["similarity p5 < 0.998"])

    def test_clipping_side_camera_only(self):
        """Saturated top quarter: fails on a side camera; a bottom camera
        has no clipping check."""

        def clip(y):
            """Saturate the top quarter."""
            y = y.copy()
            y[: H // 4] = 255
            return y

        samples, bottom = self.run_video("clipped", after(0, clip))
        self.assertGreaterEqual(samples["pct_clipped_high"].median(), 25)
        side = vqq.run_checks(samples, "side_camera_right")
        self.assertIn(
            "pct_clipped_high median <= 3.75", check_names(side, False)
        )
        self.assertNotIn("pct_clipped_high", bottom["metric"].tolist())

    def test_translation_is_measured_not_checked(self):
        """A 6 x 4 px shift is reported in ``shift``; no check reads it.
        (On this small frame it also lowers similarity.)"""
        samples, checks = self.run_video(
            "moved", after(200, lambda y: np.roll(y, (4, 6), axis=(0, 1)))
        )
        moved = samples.loc[samples["frame_index"] >= 200]
        self.assertAlmostEqual(moved["shift_x"].median(), 6, delta=0.5)
        self.assertAlmostEqual(moved["shift_y"].median(), 4, delta=0.5)
        self.assertNotIn("shift", checks["metric"].tolist())


# --- Task window ---------------------------------------------------------


class TaskWindowTest(TempDirTest):
    """Trial times from the session JSON, mapped to frames."""

    @classmethod
    def setUpClass(cls):
        """A session JSON whose trials span frames 40 to 260."""
        super().setUpClass()
        cls.json = write_behavior_json(cls.tmp / "s.json", 40, 260)

    def test_read_trial_times(self):
        """NWB column names; Harp go cue preferred when present."""
        path = write_behavior_json(self.tmp / "a" / "s.json", 30, 90)
        trials = read_trial_times(path)
        self.assertEqual(
            list(trials.columns),
            ["start_time", "goCue_start_time", "stop_time"],
        )
        self.assertEqual(len(trials), 4)
        obj = json.loads(path.read_text())
        obj["B_GoCueTimeHarp"] = [1.0, 2.0, 3.0, 4.0]
        path.write_text(json.dumps(obj))
        self.assertEqual(
            read_trial_times(path)["goCue_start_time"].tolist(),
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_read_trial_times_url(self):
        """A URL is fetched."""
        with mock.patch(
            "urllib.request.urlopen", return_value=open(self.json, "rb")
        ) as urlopen:
            trials = read_trial_times("https://example.org/s.json")
        urlopen.assert_called_once_with("https://example.org/s.json")
        self.assertEqual(len(trials), 4)

    def test_behavior_time_to_frame_index(self):
        """First frame at or after; past the end gives len."""
        harp = np.array([10.0, 10.5, 11.0])
        np.testing.assert_array_equal(
            behavior_time_to_frame_index([10.0, 10.2, 12.0], harp), [0, 1, 3]
        )

    def test_corrected_timing(self):
        """No log: the corrected timing, right despite a lost frame (the
        raw column would put the end one frame late)."""
        csv = write_video_csv(self.tmp / "drop.csv", 300, drop_at=100)
        self.assertEqual(task_frame_window(self.json, csv), (40, 259))

    def test_trigger_log_by_frame_number(self):
        """A lost frame and a Harp clock step (correction refused): each
        row takes its exposure's log time. The log goes first even when
        the correction would work."""
        csv = write_video_csv(
            self.tmp / "both.csv", 300, harp_step_at=280, drop_at=100
        )
        triggers = FIRST_HARP + np.arange(301) / FPS
        triggers[281:] += 0.5  # the same clock step, in the log
        log = self.tmp / "Event_94_step.bin"
        write_trigger_log(log, triggers)
        self.assertEqual(task_frame_window(self.json, csv, log), (40, 259))
        with mock.patch(
            "aind_dynamic_foraging_behavior_video_analysis.video_timing_qc"
            ".correct_video_timing"
        ) as correct:
            task_frame_window(self.json, csv, log)
        correct.assert_not_called()

    def test_raw_harp_without_lost_frames(self):
        """A clock step is refused by the correction; with no frames lost
        the raw column still places the task."""
        csv = write_video_csv(self.tmp / "step.csv", 300, harp_step_at=280)
        self.assertEqual(task_frame_window(self.json, csv), (40, 260))

    def test_refused_with_lost_frames_raises(self):
        """Lost frames, a refused correction and no log: no window."""
        csv = write_video_csv(
            self.tmp / "neither.csv", 300, harp_step_at=280, drop_at=100
        )
        with self.assertRaises(ValueError):
            task_frame_window(self.json, csv)

    def test_no_harp_trial_times_raises(self):
        """Older JSONs have CPU trial times only."""
        path = write_behavior_json(self.tmp / "b" / "s.json", 40, 260, False)
        csv = write_video_csv(self.tmp / "ok.csv", 300)
        with self.assertRaisesRegex(ValueError, "No Harp trial times"):
            task_frame_window(path, csv)

    def test_sample_window_notes(self):
        """The task, or the middle 50% with the reason."""
        csv = write_video_csv(self.tmp / "metadata.csv", 300)
        self.assertEqual(
            vqq.sample_window(self.json, csv), ((40, 260), "task")
        )
        self.assertEqual(
            vqq.sample_window(None, csv),
            (None, "middle 50%: no behavior JSON or video CSV"),
        )
        window, note = vqq.sample_window(self.json, self.tmp / "none.csv")
        self.assertIsNone(window)
        self.assertTrue(note.startswith("middle 50%: "))
        self.assertIn("none.csv", note)


# --- Output, pipeline and report -----------------------------------------


class OutputTest(TempDirTest):
    """video_quality end to end, written files, and the session card."""

    @classmethod
    def setUpClass(cls):
        """One passing and one failing camera through video_quality."""
        super().setUpClass()
        json_path = write_behavior_json(cls.tmp / "s.json", 40, 260)
        csv = write_video_csv(cls.tmp / "metadata.csv", 300)
        cls.good = vqq.video_quality(
            write_mp4(cls.tmp / "good.mp4"), "side_camera", json_path, csv
        )
        cls.bad = vqq.video_quality(
            write_mp4(cls.tmp / "bad.mp4", fault=after(200, blur)), "bottom"
        )

    def test_video_quality(self):
        """Task window when the files are given, middle 50% otherwise."""
        frames, samples, checks, note = self.good
        self.assertEqual(note, "task")
        self.assertEqual(
            samples["frame_index"].tolist(), list(range(40, 260, 10))
        )
        self.assertEqual(len(frames), len(samples))
        self.assertEqual(len(checks), 7)
        frames, samples, checks, note = self.bad
        self.assertTrue(note.startswith("middle 50%"))
        self.assertEqual(samples["frame_index"].iloc[0], 80)
        self.assertEqual(
            vqq.quality_action(checks), "exclude: sharpness_dev <= 0.45"
        )

    def test_write_and_read_back(self):
        """The parquet keeps every column and the histograms; the JSON is
        valid (no NaN) and holds the checks and the action."""
        _, samples, checks, note = self.bad
        record_path, samples_path = vqq.write_video_quality(
            samples, checks, self.tmp / "out", "bottom_camera", note
        )
        self.assertEqual(record_path.name, "video_quality_bottom_camera.json")
        self.assertEqual(
            samples_path.name, "video_quality_bottom_camera.parquet"
        )
        back = pd.read_parquet(samples_path)
        pd.testing.assert_frame_equal(
            back.drop(columns="histogram"), samples.drop(columns="histogram")
        )
        np.testing.assert_array_equal(
            np.stack(back["histogram"]), np.stack(samples["histogram"])
        )
        record = json.loads(record_path.read_text(), parse_constant=self.fail)
        self.assertEqual(record["camera"], "bottom_camera")
        self.assertEqual(record["window"], note)
        self.assertIn("aind_video_utils", record["versions"])
        self.assertEqual(record["action"], vqq.quality_action(checks))
        self.assertEqual(
            pd.DataFrame(record["checks"])["passed"].tolist(),
            checks["passed"].tolist(),
        )

    def test_session_card(self):
        """Renders for passing and failing cameras: reference, 8 frames,
        3 check panels and the shift, histogram, 6 outlier panels."""
        for frames, samples, checks, note in (self.good, self.bad):
            fig = vqr.session_card(frames, samples, checks, f"test ({note})")
            self.assertEqual(len(fig.axes), 20)
            fig.savefig(self.tmp / "card.png", dpi=50)
            matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    unittest.main()
