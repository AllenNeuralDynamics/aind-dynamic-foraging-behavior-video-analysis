"""Tests for video_quality_qc and video_quality_report.

Videos are synthetic MP4s encoded here with PyAV (h264 with B-frames, a
fixed GOP, luma written directly so values are not range-converted): a
smooth random texture with a moving blob, plus the fault under test on
chosen frames.
"""

import json
import re
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

from aind_video_utils import read_mp4_frame_index  # noqa: E402

from aind_dynamic_foraging_behavior_video_analysis import (  # noqa: E402
    video_quality_qc as vqq,
)
from aind_dynamic_foraging_behavior_video_analysis import (  # noqa: E402
    video_quality_report as vqr,
)

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
    cx = 30 + (i * 2) % 100
    y[55:65, cx : cx + 10] += 40
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

    def test_luma_histogram_counts_every_pixel(self):
        """One bin per value, summing to the pixel count."""
        y = np.array([[0, 255], [255, 7]], dtype=np.uint8)
        h = vqq.luma_histogram(y, 8)
        self.assertEqual(len(h), 256)
        self.assertEqual(h[255], 2)
        self.assertEqual(h.sum(), 4)

    def test_frame_quality_stats_black_frame(self):
        """A black frame has zero contrast rather than dividing by zero."""
        stats = vqq.frame_quality_stats(
            np.zeros((8, 8), dtype=np.uint8), "tv", 8
        )
        self.assertEqual(stats.contrast_rms, 0.0)
        self.assertEqual(stats.pct_clipped_low, 100.0)

    def test_clipping_counts_the_tagged_range(self):
        """TV range clips at 16 / 235; full range at 0 / 255."""
        y = np.array([[16, 17, 235, 234]], dtype=np.uint8)
        tv = vqq.frame_quality_stats(y, "tv", 8)
        pc = vqq.frame_quality_stats(y, "pc", 8)
        self.assertEqual((tv.pct_clipped_low, tv.pct_clipped_high), (25, 25))
        self.assertEqual((pc.pct_clipped_low, pc.pct_clipped_high), (0, 0))

    def test_phase_shift_positive_and_negative(self):
        """Recovers whole and fractional shifts in both directions."""
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


# --- Sampling ------------------------------------------------------------


class SamplingTest(TempDirTest):
    """Keyframe choice and decoding on a clean synthetic video."""

    @classmethod
    def setUpClass(cls):
        """Encode one clean video."""
        super().setUpClass()
        cls.path = write_mp4(cls.tmp / "clean.mp4")
        cls.index = read_mp4_frame_index(cls.path)

    def test_file_has_b_frames_and_edit_list(self):
        """The case the frame mapping must handle."""
        self.assertTrue(self.index.edits)
        self.assertFalse(np.array_equal(self.index.pts, self.index.dts))

    def test_keyframes_every_gop(self):
        """Presentation indices of keyframes are multiples of the GOP."""
        keys = vqq.keyframe_display_indices(self.index)
        np.testing.assert_array_equal(keys, np.arange(0, 300, GOP))

    def test_choose_skips_frame_zero_and_edges(self):
        """Never frame 0 nor the edge fraction."""
        chosen = vqq.choose_keyframes(self.index, 100, edge_fraction=0.1)
        self.assertGreaterEqual(chosen.min(), 30)
        self.assertLess(chosen.max(), 270)
        chosen = vqq.choose_keyframes(self.index, 100, edge_fraction=0)
        self.assertEqual(chosen.min(), GOP)

    def test_choose_fewer_than_available(self):
        """Evenly spread, unique, ascending."""
        chosen = vqq.choose_keyframes(self.index, 5, edge_fraction=0)
        self.assertEqual(len(chosen), 5)
        self.assertEqual(chosen[0], GOP)
        self.assertEqual(chosen[-1], 290)
        self.assertTrue((np.diff(chosen) > 0).all())

    def test_choose_time_window(self):
        """Only keyframes inside the window (video seconds)."""
        chosen = vqq.choose_keyframes(
            self.index, 100, edge_fraction=0, time_window=(1.0, 2.0)
        )
        np.testing.assert_array_equal(chosen, np.arange(50, 101, GOP))

    def test_choose_nothing_eligible(self):
        """An empty window raises."""
        with self.assertRaises(ValueError):
            vqq.choose_keyframes(self.index, 10, time_window=(100, 200))

    def test_read_keyframes_returns_the_keyframes(self):
        """Each decoded frame is the keyframe asked for."""
        texture = _texture()
        frames = vqq.read_keyframes(self.path, self.index, [10, 150])
        for (pts, luma), i in zip(frames, [10, 150]):
            self.assertEqual(luma.shape, (H, W))
            expected = _frame(i, texture).astype(float)
            self.assertLess(np.abs(luma - expected).mean(), 1.5)

    def test_luma_plane_is_not_range_converted(self):
        """Values come back as coded, not stretched to full range (which
        would move the mean by several units)."""
        luma = vqq.read_keyframes(self.path, self.index, [10])[0][1]
        expected = _frame(10, _texture())
        self.assertAlmostEqual(luma.mean(), expected.mean(), delta=0.5)
        self.assertAlmostEqual(luma.min(), expected.min(), delta=4)

    def test_luma_plane_refuses_other_formats(self):
        """RGB frames have no luma plane."""
        frame = av.VideoFrame(16, 16, "rgb24")
        with self.assertRaises(ValueError):
            vqq.luma_plane(frame)

    def test_read_keyframes_refuses_unsafe_edit_list(self):
        """An edit list that drops frames cannot be frame-addressed."""
        unsafe = mock.Mock(is_frame_addressing_safe=lambda: False)
        with self.assertRaises(ValueError):
            vqq.read_keyframes(self.path, unsafe, [10])

    def test_read_keyframes_refuses_wrong_landing(self):
        """A seek that lands elsewhere raises instead of mislabelling."""
        with self.assertRaisesRegex(ValueError, "returned pts"):
            vqq.read_keyframes(self.path, self.index, [15])


# --- Measuring and checks ------------------------------------------------


class MeasureTest(TempDirTest):
    """measure_video_quality and the checks on synthetic faults."""

    def measure(self, name, fault=None, **kwargs):
        """Encode and measure one video, sampling every keyframe."""
        path = write_mp4(self.tmp / f"{name}.mp4", fault=fault)
        kwargs.setdefault("n_samples", 100)
        kwargs.setdefault("edge_fraction", 0)
        return vqq.measure_video_quality(path, **kwargs)

    def action(self, qc, **kwargs):
        """Checks and action for a measured video."""
        checks = vqq.check_video_quality(qc, **kwargs)
        return checks.set_index("check"), vqq.quality_action(checks)

    def test_clean(self):
        """All stability checks pass; level checks skipped; action use."""
        qc = self.measure("clean")
        s = qc.summary
        self.assertEqual((s.width, s.height, s.codec), (W, H, "h264"))
        self.assertEqual((s.n_frames, s.n_keyframes, s.gop), (300, 30, 10))
        self.assertAlmostEqual(s.fps, FPS)
        self.assertEqual(s.n_samples, 29)
        self.assertIsNone(s.window_start)
        samples = qc.samples
        self.assertEqual(
            samples["frame_index"].tolist(), list(range(10, 300, 10))
        )
        np.testing.assert_allclose(
            samples["video_time"], samples["frame_index"] / FPS
        )
        self.assertLess(samples["shift"].max(), 0.5)
        self.assertGreater(samples["similarity"].min(), 0.9)
        self.assertEqual(qc.histograms.shape, (29, 256))
        self.assertEqual(qc.reference.shape, (H, W))
        self.assertEqual(qc.thumbnails.shape, (29, H // 2, W // 2))
        checks, action = self.action(qc, camera="BottomCamera")
        self.assertEqual(action, "use")
        self.assertTrue(checks.loc[vqq.STABILITY_CHECKS, "passed"].all())
        self.assertTrue(checks.loc[vqq.LEVEL_CHECKS, "passed"].isna().all())
        self.assertIn("BottomCamera", checks.loc["sharp_enough", "message"])

    def test_defocus_fails_sharpness(self):
        """Blur from frame 200 on."""
        qc = self.measure("defocus", after(200, blur))
        checks, action = self.action(qc)
        self.assertEqual(action, "exclude: sharpness_stable")
        self.assertEqual(checks.loc["sharpness_stable", "count"], 10)

    def test_isolated_blur_is_counted_but_passes(self):
        """One blurred keyframe (a GOP around frame 150)."""
        qc = self.measure("blip", between(145, 155, blur))
        checks, action = self.action(qc)
        self.assertEqual(action, "use")
        row = checks.loc["sharpness_stable"]
        self.assertTrue(row["passed"])
        self.assertEqual(row["samples"], [14])
        self.assertIn("isolated", row["message"])

    def test_brightness_step_fails_brightness(self):
        """Lights brighten by 40% from frame 200."""
        qc = self.measure("bright", after(200, lambda y: y * 1.4))
        checks, _ = self.action(qc)
        self.assertFalse(checks.loc["brightness_stable", "passed"])

    def test_occlusion_fails_scene(self):
        """Left half of the frame black from frame 200."""

        def occlude(y):
            """Black out the left half."""
            y = y.copy()
            y[:, : W // 2] = 16
            return y

        qc = self.measure("occluded", after(200, occlude))
        checks, _ = self.action(qc)
        self.assertFalse(checks.loc["scene_stable", "passed"])

    def test_translation_is_measured_not_checked(self):
        """A 6 px shift is reported in ``shift``; no check fails on it."""
        qc = self.measure(
            "moved", after(200, lambda y: np.roll(y, (4, 6), axis=(0, 1)))
        )
        moved = qc.samples.loc[qc.samples["frame_index"] >= 200]
        self.assertAlmostEqual(moved["shift_x"].median(), 6, delta=0.5)
        self.assertAlmostEqual(moved["shift_y"].median(), 4, delta=0.5)
        self.assertNotIn("view_stable", self.action(qc)[0].index)

    def test_level_checks_with_thresholds(self):
        """Clipped highlights fail exposure; other levels pass or fail."""

        def clip(y):
            """Saturate the top quarter."""
            y = y.copy()
            y[: H // 4] = 255
            return y

        qc = self.measure("clipped", after(0, clip))
        self.assertGreaterEqual(qc.summary.pct_clipped_high_med, 25)
        checks, action = self.action(
            qc,
            thresholds={
                "min_sharpness": 1e9,
                "max_pct_clipped": 1.0,
                "min_dynamic_range": 10,
            },
        )
        self.assertFalse(checks.loc["sharp_enough", "passed"])
        self.assertFalse(checks.loc["exposure_ok", "passed"])
        self.assertIn("clipped", checks.loc["exposure_ok", "message"])
        self.assertTrue(checks.loc["contrast_ok", "passed"])
        self.assertEqual(action, "exclude: sharp_enough")

    def test_exposure_limits(self):
        """Too dark, too bright, and within limits."""
        summary = mock.Mock(mean_med=50.0, pct_clipped_high_med=0.0)
        dark = vqq.check_exposure_ok(summary, {"min_mean": 60})
        bright = vqq.check_exposure_ok(summary, {"max_mean": 40})
        ok = vqq.check_exposure_ok(summary, {"min_mean": 40, "max_mean": 60})
        self.assertIn("too dark", dark["message"])
        self.assertIn("too bright", bright["message"])
        self.assertTrue(ok["passed"])

    def test_camera_thresholds_are_looked_up(self):
        """LEVEL_THRESHOLDS supplies a camera's thresholds."""
        qc = self.measure("lookup")
        with mock.patch.dict(
            vqq.LEVEL_THRESHOLDS, {"Cam": {"min_sharpness": 0}}
        ):
            checks, _ = self.action(qc, camera="Cam")
        self.assertTrue(checks.loc["sharp_enough", "passed"])

    def test_time_window_recorded(self):
        """The window limits samples and is kept in the summary."""
        qc = self.measure("window", time_window=(1.0, 3.0))
        self.assertEqual(qc.summary.window_start, 1.0)
        self.assertEqual(qc.summary.window_end, 3.0)
        self.assertEqual(qc.samples["frame_index"].min(), 50)
        self.assertEqual(qc.samples["frame_index"].max(), 150)

    def test_fewer_keyframes_than_samples(self):
        """A short file uses each eligible keyframe once."""
        path = write_mp4(self.tmp / "short.mp4", n_frames=40)
        qc = vqq.measure_video_quality(path, n_samples=100, edge_fraction=0)
        self.assertEqual(qc.samples["frame_index"].tolist(), [10, 20, 30])

    def test_not_mp4_refused(self):
        """Matroska is refused."""
        path = write_mp4(self.tmp / "clip.mkv", n_frames=30)
        with self.assertRaisesRegex(ValueError, "Not an MP4"):
            vqq.measure_video_quality(path)

    def test_fps_falls_back_to_r_frame_rate(self):
        """Without avg_frame_rate, r_frame_rate gives fps; without both,
        fps is None."""
        index = mock.Mock(n_samples=10, is_keyframe=np.ones(10, bool))
        probe_json = {
            "streams": [
                {
                    "pix_fmt": "yuv420p",
                    "width": 4,
                    "height": 2,
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "25/1",
                }
            ]
        }
        self.assertEqual(vqq._format_info(probe_json, index)["fps"], 25)
        del probe_json["streams"][0]["r_frame_rate"]
        self.assertIsNone(vqq._format_info(probe_json, index)["fps"])


# --- Output and session --------------------------------------------------


class OutputTest(TempDirTest):
    """Written files and check_session."""

    @classmethod
    def setUpClass(cls):
        """One measured video."""
        super().setUpClass()
        path = write_mp4(cls.tmp / "clean.mp4")
        cls.qc = vqq.measure_video_quality(path, edge_fraction=0)
        cls.checks = vqq.check_video_quality(cls.qc)

    def test_write_and_read_back(self):
        """JSON is valid (no NaN) and the parquet has the histograms."""
        json_path, parquet_path = vqq.write_video_quality(
            self.qc, self.checks, self.tmp / "out", camera="BottomCamera"
        )
        self.assertEqual(json_path.name, "video_quality_BottomCamera.json")
        record = json.loads(json_path.read_text(), parse_constant=self.fail)
        self.assertEqual(record["action"], "use")
        self.assertEqual(record["camera"], "BottomCamera")
        self.assertIn("aind_video_utils", record["versions"])
        self.assertEqual(
            set(record["summary"]), set(vqq.qc_result_fieldnames())
        )
        # Nothing reaches the floor, so that spread is undefined.
        self.assertIsNone(record["summary"]["pct_clipped_low_spread"])
        self.assertEqual(len(record["checks"]), 6)
        samples = pd.read_parquet(parquet_path)
        self.assertEqual(len(samples), len(self.qc.samples))
        self.assertEqual(len(samples["histogram"].iloc[0]), 256)

    def test_json_ready(self):
        """Numpy scalars and NaN convert; other values pass through."""
        self.assertIsNone(vqq._json_ready(np.float64("nan")))
        self.assertEqual(vqq._json_ready(np.int64(3)), 3)
        self.assertEqual(vqq._json_ready("x"), "x")

    def test_check_session(self):
        """Both layouts, and an unreadable file."""
        folder = self.tmp / "behavior-videos"
        (folder / "BottomCamera").mkdir(parents=True)
        write_mp4(folder / "BottomCamera" / "video.mp4", n_frames=120)
        write_mp4(folder / "side_camera.mp4", n_frames=120)
        (folder / "broken.mp4").write_bytes(b"not a video")
        table = vqq.check_session(folder, n_samples=5).set_index("camera")
        self.assertEqual(
            list(table.index), ["BottomCamera", "broken", "side_camera"]
        )
        self.assertEqual(table.loc["BottomCamera", "action"], "use")
        self.assertEqual(table.loc["BottomCamera", "n_samples"], 5)
        self.assertEqual(
            table.loc["BottomCamera", "skipped_checks"], vqq.LEVEL_CHECKS
        )
        self.assertEqual(table.loc["broken", "action"], "exclude: unreadable")
        self.assertTrue(table.loc["broken", "error"])


class ReportTest(TempDirTest):
    """Figures render; the PDF is written in the documented order."""

    @classmethod
    def setUpClass(cls):
        """A passing and a failing video."""
        super().setUpClass()
        good = write_mp4(cls.tmp / "good.mp4")
        bad = write_mp4(cls.tmp / "bad.mp4", fault=after(200, blur))
        cls.items = []
        for label, path, window in [
            ("good", good, (0.5, 5.0)),
            ("bad", bad, None),
        ]:
            qc = vqq.measure_video_quality(
                path, edge_fraction=0, time_window=window
            )
            cls.items.append((label, qc, vqq.check_video_quality(qc)))

    def test_session_card(self):
        """Renders for passing and failing videos."""
        for label, qc, checks in self.items:
            fig = vqr.session_card(qc, checks)
            # reference, 8 thumbnails, 4 time panels, histogram, 6 frames
            self.assertEqual(len(fig.axes), 20)
            matplotlib.pyplot.close(fig)

    def test_contact_sheet(self):
        """One row per video."""
        fig = vqr.contact_sheet(self.items)
        self.assertEqual(len(fig.axes), 2 * 9)
        matplotlib.pyplot.close(fig)

    def test_batch_pdf(self):
        """Excluded first; index, contact sheet, then one card each."""
        with mock.patch.object(vqr, "session_card", wraps=vqr.session_card):
            path = vqr.batch_pdf(self.items, self.tmp / "batch.pdf")
            order = [c.args[2] for c in vqr.session_card.call_args_list]
        self.assertEqual(order, ["bad", "good"])
        self.assertGreater(path.stat().st_size, 10_000)
        pages = re.findall(rb"/Type\s*/Page(?!s)", path.read_bytes())
        self.assertEqual(len(pages), 4)

    def test_failed_samples_unknown_check(self):
        """A check that is absent or passed marks nothing."""
        _, _, checks = self.items[0]
        self.assertEqual(vqr._failed_samples(checks, "nope"), [])
        self.assertEqual(vqr._failed_samples(checks, "scene_stable"), [])


if __name__ == "__main__":
    unittest.main()
