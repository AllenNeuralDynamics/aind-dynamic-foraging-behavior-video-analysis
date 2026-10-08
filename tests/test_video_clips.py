"""Tests for video_clips and video_alignment.event_frame_ranges.

Round trips use synthetic MP4s whose frame N has luma ``16 + 3 * (N mod
64)``, so every decoded frame (clip frame or PNG) names its source frame.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from aind_video_utils import read_mp4_frame_index

from aind_dynamic_foraging_behavior_video_analysis import video_clips as vc
from aind_dynamic_foraging_behavior_video_analysis import (
    video_timing_qc as vtq,
)
from aind_dynamic_foraging_behavior_video_analysis.video_alignment import (
    event_frame_ranges,
)
from tests.test_video_timing_qc import simulate

W, H = 64, 48
FPS = 30
N_SOURCE = 240
# Luma step between frames, after ffmpeg's TV-range to RGB conversion.
STEP = 3 * 255 / 219


def write_source(path, vfr=False, trim=False):
    """Encode the synthetic source video (B-frames on); return its path.

    ``vfr``: uneven container timestamps (no ``i / fps`` grid).
    ``trim``: then stream-copy it from 0.5 s, which leaves an edit list
    that skips displayed frames (not frame-addressing-safe).
    """
    path = Path(path)
    graph = (
        f"nullsrc=s={W}x{H}:r={FPS}:d={N_SOURCE / FPS},"
        "geq=lum='16+3*mod(N\\,64)':cb=128:cr=128"
    )
    if vfr:
        graph += f",settb=1/90000,setpts='(N+0.4*mod(N\\,3))/{FPS}/TB'"
    target = path.with_name("full.mp4") if trim else path
    subprocess.run(
        [
            *vc._ffmpeg(),
            "-f",
            "lavfi",
            "-i",
            graph,
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "demux",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-x264-params",
            "keyint=30:bframes=2",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "90000",
            str(target),
        ],
        check=True,
    )
    if trim:
        subprocess.run(
            [*vc._ffmpeg(), "-ss", "0.5", "-i", str(target)]
            + ["-c", "copy", str(path)],
            check=True,
        )
    return path


def frame_ids(path):
    """Decode a video or image; return each frame's source frame mod 64."""
    raw = subprocess.run(
        [*vc._ffmpeg(), "-i", str(path), "-fps_mode", "passthrough"]
        + ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, H * W * 3)
    return np.round(frames.mean(axis=1) / STEP).astype(int)


def write_labels(folder, images, layout="dlc"):
    """Write a CollectedData CSV labeling ``images`` (paths in the CSV).

    ``layout``: ``dlc`` (three index columns, DLC 2.3+) or ``path`` (one
    path column, older DLC and Lightning Pose).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lead = ["scorer", "", ""] if layout == "dlc" else ["scorer"]
    blank = [""] * (len(lead) - 1)
    lines = [
        lead + ["me", "me"],
        ["bodyparts"] + blank + ["tongue", "tongue"],
        ["coords"] + blank + ["x", "y"],
    ]
    for image in images:
        index = image.split("/") if layout == "dlc" else [image]
        lines.append(index + ["1.5", "2.5"])
    path = folder / "CollectedData_me.csv"
    path.write_text("\n".join(",".join(line) for line in lines) + "\n")
    return path


class StubIndex:
    """Stands in for aind_video_utils.Mp4FrameIndex."""

    def __init__(self, n_samples=100, safe=True, monotonic=True):
        """Frame count and which faults to stand in for."""
        self.n_samples = n_samples
        self.safe = safe
        self.monotonic = monotonic

    def is_frame_addressing_safe(self):
        """Whether the edit list allows frame addressing."""
        return self.safe

    def presentation_seconds(self, display_index):
        """Seek time of a frame; raises where timestamps repeat."""
        if not self.monotonic and display_index == 50:
            raise ValueError("non-monotonic presentation timeline")
        return display_index / 30


class EventFrameRangesTest(unittest.TestCase):
    """video_alignment.event_frame_ranges."""

    def test_window_edges(self):
        """Frames in [t - before, t + after); in_video at both ends."""
        harp = 100.0 + np.arange(100) * 0.1
        ranges = event_frame_ranges(
            [101.0, 101.05, 100.1, 109.8, np.nan], harp, 0.2, 0.3
        )
        self.assertEqual(list(ranges.start_frame[:2]), [8, 9])
        self.assertEqual(list(ranges.n_frames[:2]), [5, 5])
        self.assertEqual(
            list(ranges.in_video), [True, True, False, False, False]
        )
        self.assertEqual(ranges.start_frame.dtype, np.int64)
        one = event_frame_ranges(101.0, harp, 0.0, 0.1)
        self.assertEqual((one.start_frame[0], one.n_frames[0]), (10, 1))

    def test_no_frames(self):
        """An empty video puts every window outside it."""
        ranges = event_frame_ranges([1.0], [], 0.1, 0.1)
        self.assertFalse(ranges.in_video[0])

    def test_window_across_drop_has_fewer_frames(self):
        """Corrected times: same span, fewer frames, the right frames."""
        with tempfile.TemporaryDirectory() as tmp:
            dropped = np.arange(1000, 1010)
            path, triggers = simulate(tmp, dropped=dropped)
            timing = vtq.load_video_timing(path)
            harp = vtq.correct_video_timing(timing, trigger_times=triggers)[
                "harp_time"
            ].to_numpy()
        saved = np.setdiff1d(np.arange(3000), dropped)
        events = triggers[[500, 1005, 2000]]
        ranges = event_frame_ranges(events, harp, 0.05, 0.05)
        clean = event_frame_ranges(events, triggers, 0.05, 0.05)
        self.assertEqual(ranges.n_frames[0], clean.n_frames[0])
        self.assertEqual(ranges.n_frames[1], clean.n_frames[1] - len(dropped))
        # The frame at the event is the exposure at the event.
        first = saved[ranges.start_frame[2]]
        self.assertEqual(first, clean.start_frame[2])
        # The raw CSV column puts it 10 frames late.
        raw = event_frame_ranges(
            events, timing["harp_time_raw"].to_numpy(), 0.05, 0.05
        )
        self.assertEqual(saved[raw.start_frame[2]] - first, len(dropped))


class NamesTest(unittest.TestCase):
    """Private helpers: names and command lines."""

    def test_png_name_follows_dlc(self):
        """ceil(log10(n_frames)) digits."""
        self.assertEqual(vc._png_name(42, 999), "img042.png")
        self.assertEqual(vc._png_name(42, 1000), "img042.png")
        self.assertEqual(vc._png_name(42, 1001), "img0042.png")
        self.assertEqual(vc._png_name(7, 60), "img07.png")
        self.assertEqual(vc._png_name(0, 1), "img0.png")

    def test_parse_stem(self):
        """Prefix and start frame; bad names raise."""
        self.assertEqual(
            vc._parse_stem("behavior_1_2024_Side_f0174266"),
            ("behavior_1_2024_Side", 174266),
        )
        with self.assertRaises(ValueError):
            vc._parse_stem("clip")

    def test_cut_command(self):
        """9-decimal accurate seek, exact frame count, no frame-rate fit."""
        cmd = vc._cut_command("in.mp4", 1 / 3, 25, "out.mp4")
        seek = cmd.index("-ss")
        self.assertEqual(cmd[seek + 1], "0.333333333")
        self.assertLess(cmd.index("-accurate_seek"), seek)
        self.assertLess(seek, cmd.index("-i"))
        self.assertEqual(cmd[cmd.index("-frames:v") + 1], "25")
        self.assertEqual(cmd[cmd.index("-fps_mode") + 1], "passthrough")
        self.assertNotIn("-reconnect", cmd)
        url = vc._cut_command("https://x/in.mp4", 0.0, 1, "out.mp4")
        self.assertLess(url.index("-reconnect"), url.index("-i"))

    def test_png_and_gray_commands(self):
        """Select by decoded-frame count; gray at explicit dimensions."""
        cmd = vc._png_command("c.mp4", 9, "img09.png")
        self.assertEqual(cmd[cmd.index("-vf") + 1], "select=eq(n\\,9)")
        self.assertEqual(cmd[cmd.index("-frames:v") + 1], "1")
        cmd = vc._gray_command("c.mp4")
        self.assertIn("scale=40:30,format=gray", cmd)

    def test_kmeans_pick(self):
        """One frame per cluster, the one nearest its centroid."""
        centres = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
        ring = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1.0]])
        # Each cluster's centroid is its first point, at the centre.
        features = (centres[:, None, :] + ring[None]).reshape(-1, 2)
        picks = vc._kmeans_pick(features, 3, seed=0)
        np.testing.assert_array_equal(picks, [0, 5, 10])


class StubbedCutTest(unittest.TestCase):
    """cut_clip and cut_clips with ffmpeg and the MP4 index stubbed."""

    def setUp(self):
        """Temporary output folder."""
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        """Remove it."""
        self.tmp.cleanup()

    def fake_run(self, command, stdout=None, stderr=None):
        """Write the output file, as ffmpeg would."""
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def test_cut_clip_raises(self):
        """Empty, out of range, unsafe edit list, non-monotonic seek."""
        out = self.dir / "c.mp4"
        cases = [
            (StubIndex(), 0, 0),
            (StubIndex(), -1, 5),
            (StubIndex(), 90, 20),
            (StubIndex(safe=False), 0, 5),
            (StubIndex(monotonic=False), 50, 5),
        ]
        for index, start, n in cases:
            with self.subTest(start=start, n=n):
                with self.assertRaises(ValueError):
                    vc.cut_clip("in.mp4", start, n, out, index=index)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_ffmpeg_failure_leaves_nothing(self):
        """A failed encode raises with ffmpeg's message; no partial file."""

        def failing(command, stdout=None, stderr=None):
            """Write half a file, then fail."""
            Path(command[-1]).write_bytes(b"half")
            return subprocess.CompletedProcess(command, 1, b"", b"boom")

        with mock.patch("subprocess.run", failing):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                vc.cut_clip(
                    "in.mp4", 0, 5, self.dir / "c.mp4", index=StubIndex()
                )
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_cut_clips_skips_only_bad_rows_and_resumes(self):
        """Statuses, names, sidecars; a second run finds every clip."""
        ranges = pd.DataFrame(
            {
                "start_frame": [10, 50, 10, 0, 20],
                "n_frames": [5, 5, 5, 5, 5],
                "in_video": [True, True, True, False, True],
            }
        )
        patches = [
            mock.patch("subprocess.run", self.fake_run),
            mock.patch.object(
                vc,
                "read_mp4_frame_index",
                return_value=StubIndex(monotonic=False),
            ),
        ]
        with patches[0], patches[1]:
            out = vc.cut_clips("src.mp4", ranges, self.dir / "clips", "s_c")
            again = vc.cut_clips("src.mp4", ranges, self.dir / "clips", "s_c")
            other = vc.cut_clips(
                "other.mp4", ranges[:1], self.dir / "clips", "s_c"
            )
        self.assertEqual(list(out.status[[0, 2, 4]]), ["cut", "exists", "cut"])
        self.assertTrue(out.status[1].startswith("skipped: "))
        self.assertIn("non-monotonic", out.status[1])
        self.assertIn("past the video", out.status[3])
        self.assertTrue(pd.isna(out["clip_path"][1]))
        self.assertTrue(out["clip_path"][0].endswith("s_c_f0000010.mp4"))
        self.assertEqual(
            list(again.status[[0, 2, 4]]), ["exists", "exists", "exists"]
        )
        self.assertEqual(other.status[0], "cut")
        info = vc.read_clip_info(out["clip_path"][4])
        self.assertEqual(
            (info["source_video"], info["start_frame"], info["n_frames"]),
            ("src.mp4", 20, 5),
        )
        self.assertEqual(
            set(info["versions"]),
            {
                "aind_dynamic_foraging_behavior_video_analysis",
                "aind_video_utils",
            },
        )
        self.assertEqual(
            vc.read_clip_info(out["clip_path"][0])["source_video"], "other.mp4"
        )
        names = sorted(p.name for p in (self.dir / "clips").iterdir())
        self.assertEqual(
            names,
            [
                "s_c_f0000010.json",
                "s_c_f0000010.mp4",
                "s_c_f0000020.json",
                "s_c_f0000020.mp4",
            ],
        )

    def test_corrupt_or_missing_sidecar_recuts(self):
        """A clip without a readable sidecar is cut again."""
        clips = self.dir / "clips"
        clips.mkdir()
        (clips / "s_f0000010.mp4").write_bytes(b"old")
        (clips / "s_f0000020.mp4").write_bytes(b"old")
        (clips / "s_f0000020.json").write_text("{not json")
        ranges = pd.DataFrame({"start_frame": [10, 20], "n_frames": [5, 5]})
        with (
            mock.patch("subprocess.run", self.fake_run),
            mock.patch.object(
                vc, "read_mp4_frame_index", return_value=StubIndex()
            ),
        ):
            out = vc.cut_clips("src.mp4", ranges, clips, "s")
        self.assertEqual(list(out.status), ["cut", "cut"])

    def test_select_frames_rejects_unknown_algorithm(self):
        """Only uniform, random and kmeans."""
        with mock.patch.object(vc, "_clip_frame_count", return_value=50):
            with self.assertRaises(ValueError):
                vc.select_frames("c_f0000000.mp4", 5, self.dir, "best")

    def test_png_missing_frame_raises(self):
        """ffmpeg writes nothing for a frame past the clip's end."""

        def silent(command, stdout=None, stderr=None):
            """Succeed without writing anything."""
            return subprocess.CompletedProcess(command, 0, b"", b"")

        with mock.patch("subprocess.run", silent):
            with self.assertRaisesRegex(RuntimeError, "no frame 99"):
                vc._write_png("c.mp4", 99, self.dir / "img99.png")


class LabelsTableTest(unittest.TestCase):
    """labeled_frames_table from names alone."""

    def test_both_layouts(self):
        """DLC per-folder CSVs and an LP CSV give the same rows, once."""
        images = [
            "labeled-data/s1_Side_f0001000/img009.png",
            "labeled-data/s1_Side_f0001000/img042.png",
            "labeled-data/s2_Bottom_f0000020/img0007.png",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "labeled-data"
            write_labels(root / "s1_Side_f0001000", images[:2])
            write_labels(root / "s2_Bottom_f0000020", images[2:])
            dlc = vc.labeled_frames_table(root)
            write_labels(root, images, layout="path")
            both = vc.labeled_frames_table(root)
        self.assertEqual(list(dlc.image), images)
        self.assertEqual(list(dlc.source_frame), [1009, 1042, 27])
        self.assertEqual(list(dlc.clip_frame), [9, 42, 7])
        self.assertEqual(list(dlc.prefix), ["s1_Side", "s1_Side", "s2_Bottom"])
        self.assertEqual(len(both), 3)

    def test_empty_and_bad_names(self):
        """No CSVs: empty table. A name without digits raises."""
        with tempfile.TemporaryDirectory() as tmp:
            empty = vc.labeled_frames_table(tmp)
            write_labels(Path(tmp) / "x", ["labeled-data/s_f0000001/a.png"])
            with self.assertRaises(ValueError):
                vc.labeled_frames_table(tmp)
            with self.assertRaises(ValueError):
                vc.add_context_frames(tmp, tmp)
        self.assertEqual(len(empty), 0)
        self.assertIn("source_frame", empty.columns)


class RoundTripTest(unittest.TestCase):
    """Real ffmpeg: every frame written is the source frame it claims."""

    @classmethod
    def setUpClass(cls):
        """Encode the source videos once."""
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.src = write_source(cls.dir / "src.mp4")
        cls.vfr = write_source(cls.dir / "vfr.mp4", vfr=True)
        cls.trimmed = write_source(cls.dir / "trimmed.mp4", trim=True)

    @classmethod
    def tearDownClass(cls):
        """Remove them."""
        cls.tmp.cleanup()

    def test_source_ids(self):
        """The synthetic video encodes what the tests assume."""
        np.testing.assert_array_equal(
            frame_ids(self.src), np.arange(N_SOURCE) % 64
        )

    def test_clip_frames_are_source_frames(self):
        """Clip frame k is source frame start + k, B-frames and VFR."""
        for source in (self.src, self.vfr):
            for start, n in [(0, 7), (37, 20), (61, 30), (N_SOURCE - 9, 9)]:
                with self.subTest(source=source.name, start=start):
                    out = self.dir / f"{source.stem}_{start}.mp4"
                    vc.cut_clip(source, start, n, out)
                    np.testing.assert_array_equal(
                        frame_ids(out), np.arange(start, start + n) % 64
                    )
                    # Even timestamps, also from a variable-rate source.
                    pts = np.sort(read_mp4_frame_index(out).pts)
                    self.assertEqual(len(np.unique(np.diff(pts))), 1)

    def test_unsafe_edit_list_raises(self):
        """A stream copy that trims displayed frames cannot be cut."""
        with self.assertRaisesRegex(ValueError, "edit list"):
            vc.cut_clip(self.trimmed, 0, 5, self.dir / "never.mp4")

    def test_labeling_round_trip(self):
        """Clips -> PNGs -> labels -> context frames -> source frames."""
        work = self.dir / "labeling"
        clips_dir = work / "clips"
        labeled = work / "labeled-data"
        ranges = pd.DataFrame({"start_frame": [100, 30], "n_frames": [40, 12]})
        clips = vc.cut_clips(self.vfr, ranges, clips_dir, "sess_cam")
        self.assertEqual(list(clips.status), ["cut", "cut"])
        long_clip, short_clip = map(Path, clips["clip_path"])

        picked = {}
        for algorithm in ("uniform", "random", "kmeans"):
            picks = vc.select_frames(
                long_clip, 4, labeled, algorithm=algorithm, seed=3
            )
            self.assertEqual(len(picks), 4)
            self.assertTrue(np.all((picks >= 2) & (picks <= 37)))
            picked[algorithm] = picks
        np.testing.assert_array_equal(picked["uniform"], [2, 14, 25, 37])
        np.testing.assert_array_equal(
            vc.select_frames(long_clip, 4, labeled, "random", seed=3),
            picked["random"],
        )
        # k-means on a ramp: picks spread over the clip.
        self.assertGreater(np.ptp(picked["kmeans"]), 20)
        self.assertEqual(
            len(vc.select_frames(short_clip, 50, labeled, margin=5)), 2
        )
        self.assertEqual(
            len(vc.select_frames(short_clip, 3, labeled, "kmeans", margin=6)),
            0,
        )

        folder = labeled / long_clip.stem
        for png in folder.glob("*.png"):
            k = int(png.stem[3:])
            self.assertEqual(len(png.stem), 5)
            self.assertEqual(frame_ids(png)[0], (100 + k) % 64)

        # Label two frames, one near the clip's start.
        images = [
            f"labeled-data/{long_clip.stem}/img02.png",
            f"labeled-data/{long_clip.stem}/img26.png",
        ]
        csv = write_labels(folder, images)
        before = csv.read_text()
        before_pngs = {p.name for p in folder.glob("*.png")}
        written = vc.add_context_frames(labeled, clips_dir)
        self.assertEqual(csv.read_text(), before)
        expected = {f"img{k:02d}.png" for k in (0, 1, 3, 4, 24, 25, 27, 28)}
        self.assertEqual(
            set(Path(p).name for p in written.path), expected - before_pngs
        )
        for png in folder.glob("*.png"):
            self.assertEqual(frame_ids(png)[0], (100 + int(png.stem[3:])) % 64)
        self.assertEqual(len(vc.add_context_frames(labeled, clips_dir)), 0)
        # The last frame has no right-hand neighbours.
        vc._write_png(long_clip, 39, folder / "img39.png")
        write_labels(folder, [f"labeled-data/{long_clip.stem}/img39.png"])
        last = vc.add_context_frames(labeled, clips_dir)
        self.assertEqual(list(last.clip_frame), [38])  # 37 exists

        table = vc.labeled_frames_table(labeled)
        self.assertEqual(list(table.source_frame), [139])
        for row in table.itertuples():
            png = work / row.image
            self.assertEqual(frame_ids(png)[0], row.source_frame % 64)

    def test_sidecar_json(self):
        """The sidecar holds exactly the four fields."""
        out = vc.cut_clips(
            self.src,
            pd.DataFrame({"start_frame": [5], "n_frames": [3]}),
            self.dir / "sidecar",
            "p",
        )
        sidecar = Path(out["clip_path"][0]).with_suffix(".json")
        self.assertEqual(
            set(json.loads(sidecar.read_text())),
            {"source_video", "start_frame", "n_frames", "versions"},
        )


if __name__ == "__main__":
    unittest.main()


class ExtractTrialClipTest(unittest.TestCase):
    """kinematics.video_clip_utils.extract_trial_clip, across a drop."""

    @classmethod
    def setUpClass(cls):
        """A source video and frame-level kinematics with 20 frames lost."""
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.src = write_source(cls.dir / "src.mp4")
        rows = np.arange(N_SOURCE)
        # 20 exposures lost after row 60: later rows are 20 frames later.
        harp = 100.0 + (rows + 20 * (rows > 60)) / FPS
        first_go_cue = 101.0
        cls.kins = pd.DataFrame(
            {
                "time": harp - harp[0],
                "time_raw": harp,
                "time_in_session": harp - first_go_cue,
            }
        )
        cls.first_go_cue = first_go_cue

    @classmethod
    def tearDownClass(cls):
        """Remove the files."""
        cls.tmp.cleanup()

    def trial(self, go_cue_harp, name=7):
        """A trials row with its go cue at ``go_cue_harp``."""
        return pd.Series(
            {"goCue_start_time_in_session": go_cue_harp - self.first_go_cue},
            name=name,
        )

    def test_clip_is_the_frames_around_the_go_cue(self):
        """Frames [go - pad, go + duration + pad) on corrected times.

        The pad is off the frame grid, so no edge rounds either way.
        """
        from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
            video_clip_utils as vcu,
        )

        go_cue = self.kins.time_raw[150]
        out = vcu.extract_trial_clip(
            "s",
            self.trial(go_cue),
            self.kins,
            self.src,
            self.dir / "t",
            clip_duration_s=1.0,
            pad_s=0.11,
        )
        self.assertEqual(out.name, "trial_7_f0000147.mp4")
        np.testing.assert_array_equal(frame_ids(out), np.arange(147, 184) % 64)
        # The old constant offset put the start 20 frames late.
        with self.assertWarns(DeprecationWarning):
            video_time = vcu.get_video_time(
                go_cue - self.first_go_cue - 0.11, self.kins
            )
        self.assertEqual(round(video_time * FPS), 147 + 20)
        # An existing clip is kept.
        mtime = out.stat().st_mtime_ns
        again = vcu.extract_trial_clip(
            "s",
            self.trial(go_cue),
            self.kins,
            self.src,
            self.dir / "t",
            clip_duration_s=1.0,
            pad_s=0.11,
        )
        self.assertEqual(again.stat().st_mtime_ns, mtime)

    def test_window_clipped_to_video(self):
        """A window past the end is cut short; one outside gives None."""
        from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
            video_clip_utils as vcu,
        )

        late = self.kins.time_raw[N_SOURCE - 5]
        out = vcu.extract_trial_clip(
            "s",
            self.trial(late),
            self.kins,
            self.src,
            self.dir / "e",
            clip_duration_s=1.0,
            pad_s=0.0,
        )
        np.testing.assert_array_equal(
            frame_ids(out), np.arange(N_SOURCE - 5, N_SOURCE) % 64
        )
        self.assertIsNone(
            vcu.extract_trial_clip(
                "s", self.trial(500.0), self.kins, self.src, self.dir / "e"
            )
        )
