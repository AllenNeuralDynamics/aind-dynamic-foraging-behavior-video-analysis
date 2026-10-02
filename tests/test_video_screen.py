"""Tests for video_screen, on the synthetic videos of test_video_quality_qc.

Each session is a 300-frame MP4 with its video CSV, session JSON (trials
over frames 40 to 260) and Harp trigger log. URL inputs are served from
local files by a mocked ``urlopen``.
"""

import io
import json
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis import video_screen as vs
from tests.test_video_quality_qc import (
    FIRST_HARP,
    FPS,
    after,
    blur,
    write_behavior_json,
    write_mp4,
    write_video_csv,
)
from tests.test_video_timing_qc import write_trigger_log

SESSION = "behavior_123456_2026-01-02_03-04-05"
URL = "https://example.org/"
# Columns that differ between runs of the same inputs.
VOLATILE = ["mp4", "screened_at", "seconds"]


def write_session(folder, name="bottom_camera", fault=None, **csv_kwargs):
    """A camera's files in ``folder``; return the inputs row."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n_rows = csv_kwargs.pop("n_frames", 300)
    log = folder / "Event_94.bin"
    write_trigger_log(log, FIRST_HARP + np.arange(n_rows) / FPS)
    return {
        "session": SESSION,
        "camera": name,
        "mp4": str(write_mp4(folder / f"{name}.mp4", fault=fault)),
        "video_csv": str(
            write_video_csv(folder / f"{name}.csv", n_rows, **csv_kwargs)
        ),
        "behavior_json": str(write_behavior_json(folder / "s.json", 40, 260)),
        "trigger_log": str(log),
    }


def stable(row):
    """A row without the columns that change between runs."""
    return {k: v for k, v in row.items() if k not in VOLATILE}


class ScreenTest(unittest.TestCase):
    """Shared synthetic sessions, one temporary folder per class."""

    @classmethod
    def setUpClass(cls):
        """A clean camera, reused by most tests."""
        cls.tmp = Path(tempfile.mkdtemp())
        cls.clean = write_session(cls.tmp / "clean")

    @classmethod
    def tearDownClass(cls):
        """Remove the folder."""
        shutil.rmtree(cls.tmp)

    def screen(self, inputs, **kwargs):
        """screen_camera on an inputs row."""
        return vs.screen_camera(**inputs, **kwargs)


class ScreenCameraTest(ScreenTest):
    """One camera, in memory."""

    def test_local_inputs(self):
        """A clean camera is in use; nothing is written."""
        before = set(Path(tempfile.gettempdir()).glob("video_screen_*"))
        row = self.screen(self.clean)
        self.assertEqual(list(row), vs.SCREEN_COLUMNS)
        self.assertTrue(row["use"])
        self.assertEqual(row["reason"], "")
        self.assertEqual(row["subject"], "123456")
        self.assertEqual(row["view"], "bottom")
        self.assertEqual(
            (row["timing"], row["timing_method"], row["quality"]),
            ("use", "as written", "use"),
        )
        self.assertEqual(
            (row["frames_lost"], row["glitch_rows"], row["frame_count_diff"]),
            (0, 0, 0),
        )
        self.assertTrue(row["trigger_log"])
        self.assertEqual(row["window"], "task")
        self.assertEqual(row["versions"], vs.VERSIONS)
        self.assertEqual(
            set(Path(tempfile.gettempdir()).glob("video_screen_*")), before
        )
        self.assertEqual(
            sorted(p.name for p in (self.tmp / "clean").iterdir()),
            [
                "Event_94.bin",
                "bottom_camera.csv",
                "bottom_camera.mp4",
                "s.json",
            ],
        )

    def test_url_inputs(self):
        """CSV, JSON and log as URLs give the same row as local paths;
        each URL is downloaded once."""
        keys = ("video_csv", "behavior_json", "trigger_log")
        served = {URL + k: self.clean[k] for k in keys}
        inputs = {**self.clean, **{k: URL + k for k in keys}}

        def urlopen(url, *args, **kwargs):
            return open(served[url], "rb")

        with mock.patch("urllib.request.urlopen", side_effect=urlopen) as m:
            row = self.screen(inputs)
        self.assertEqual(stable(row), stable(self.screen(self.clean)))
        downloads = [c.args[0] for c in m.call_args_list]
        self.assertEqual(downloads.count(URL + "video_csv"), 1)
        self.assertEqual(downloads.count(URL + "trigger_log"), 1)

    def test_timing_excluded(self):
        """A Harp clock step and lost frames: timing excludes; quality is
        still measured (from the middle of the file)."""
        inputs = write_session(
            self.tmp / "step", harp_step_at=280, drop_at=100, n_frames=299
        )
        row = self.screen({**inputs, "trigger_log": None})
        self.assertFalse(row["use"])
        self.assertEqual(row["reason"], "timing: exclude: harp_evenly_spaced")
        self.assertIsNone(row["timing_method"])
        self.assertFalse(row["trigger_log"])
        self.assertEqual(row["quality"], "use")
        self.assertTrue(row["window"].startswith("middle 50%"))

    def test_frame_count_mismatch_excludes(self):
        """A CSV with one row fewer than the video has frames."""
        inputs = write_session(self.tmp / "short", n_frames=299)
        row = self.screen(inputs)
        self.assertEqual(row["frame_count_diff"], 1)
        self.assertEqual(row["reason"], "timing: exclude: video_frame_count")

    def test_quality_excluded(self):
        """Defocus from frame 100: quality excludes."""
        inputs = write_session(self.tmp / "blur", fault=after(100, blur))
        row = self.screen(inputs)
        self.assertEqual(row["timing"], "use")
        self.assertEqual(
            row["reason"], "quality: exclude: sharpness_dev <= 0.45"
        )
        self.assertFalse(row["use"])

    def test_unreadable_trigger_log_ignored(self):
        """A missing or corrupt log: screened from the CSV alone."""
        corrupt = self.tmp / "corrupt.bin"
        corrupt.write_bytes(b"not thirteen-byte messages")
        for log in (self.tmp / "missing.bin", corrupt):
            row = self.screen({**self.clean, "trigger_log": str(log)})
            self.assertTrue(row["use"])
            self.assertFalse(row["trigger_log"])

    def test_error_recorded(self):
        """A missing MP4 is an error, not an exclusion; nothing raises."""
        row = self.screen({**self.clean, "mp4": str(self.tmp / "none.mp4")})
        self.assertFalse(row["use"])
        self.assertTrue(row["reason"].startswith("error: "))
        self.assertIsNone(row["timing"])

    def test_unusable_csv_excluded(self):
        """A CSV with missing values is excluded, not an error; quality is
        still measured (middle of the file) and the reason is recorded,
        naming the CSV as given."""
        inputs = write_session(self.tmp / "gaps")
        lines = Path(inputs["video_csv"]).read_text().splitlines()
        lines[10] = lines[10].rsplit(",", 1)[0] + ","
        Path(inputs["video_csv"]).write_text("\n".join(lines) + "\n")
        out = self.tmp / "gaps_out"
        record_path = out / SESSION / "video_timing_bottom_camera.json"
        row = self.screen(inputs, out_dir=out)
        self.assertEqual(row["reason"], "timing: exclude: unreadable")
        self.assertFalse(row["use"])
        self.assertEqual(row["quality"], "use")
        self.assertTrue(row["window"].startswith("middle 50%"))
        self.assertTrue(row["trigger_log"])
        self.assertIsNone(row["frames_lost"])
        record = json.loads(record_path.read_text())
        self.assertEqual(record["verdict"], "exclude: unreadable")
        self.assertIn("missing values", record["error"])
        self.assertIn(inputs["video_csv"], record["error"])
        # Given as a URL: the URL is named, not its download.
        served = Path(inputs["video_csv"]).read_bytes()
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **k: io.BytesIO(served),
        ):
            row = self.screen(
                {**inputs, "video_csv": URL + "c.csv"}, out_dir=out
            )
        self.assertIn(URL + "c.csv", row["window"])
        self.assertIn(
            URL + "c.csv", json.loads(record_path.read_text())["error"]
        )

    def test_unreachable_json_is_an_error(self):
        """A session JSON URL that cannot be fetched is an error (screened
        again next time), not a silent fallback to the middle 50%."""
        failure = urllib.error.URLError("nodename nor servname")
        with mock.patch("urllib.request.urlopen", side_effect=failure):
            row = self.screen({**self.clean, "behavior_json": URL + "s.json"})
        self.assertTrue(row["reason"].startswith("error: URLError"))
        self.assertIsNone(row["window"])

    def test_quality_off(self):
        """Timing alone decides; no quality columns."""
        row = self.screen(self.clean, quality=False)
        self.assertTrue(row["use"])
        self.assertIsNone(row["quality"])
        self.assertIsNone(row["window"])

    def test_writes_with_out_dir(self):
        """Timing and quality records, and the card when asked for."""
        out = self.tmp / "camera_out"
        row = self.screen(self.clean, out_dir=out, cards=True)
        names = sorted(p.name for p in (out / SESSION).iterdir())
        self.assertEqual(
            names,
            [
                "session_card_bottom_camera.png",
                "video_quality_bottom_camera.json",
                "video_quality_bottom_camera.parquet",
                "video_timing_bottom_camera.json",
            ],
        )
        record = json.loads(
            (out / SESSION / "video_timing_bottom_camera.json").read_text()
        )
        self.assertEqual(record["verdict"], row["timing"])

    def test_optional_inputs_and_names(self):
        """Missing optional inputs (None, NaN, empty) and odd names."""
        self.assertIsNone(vs._given(float("nan")))
        self.assertIsNone(vs._given(""))
        self.assertEqual(vs._given("x"), "x")
        self.assertIsNone(vs._subject("odd-name"))
        row = self.screen(
            {**self.clean, "behavior_json": np.nan, "trigger_log": ""}
        )
        self.assertTrue(row["window"].startswith("middle 50%"))
        self.assertFalse(row["trigger_log"])


class ScreenSessionsTest(ScreenTest):
    """Many cameras, the files and the cache."""

    @classmethod
    def setUpClass(cls):
        """A clean camera and a blurred one in a second session."""
        super().setUpClass()
        blurred = write_session(cls.tmp / "blur2", fault=after(100, blur))
        cls.inputs = [
            cls.clean,
            {**blurred, "session": SESSION.replace("123456", "654321")},
        ]

    def test_in_memory(self):
        """Rows in input order; nothing written; a list of dicts works."""
        screen = vs.screen_sessions(self.inputs)
        self.assertEqual(list(screen.columns), vs.SCREEN_COLUMNS)
        self.assertEqual(screen["use"].tolist(), [True, False])
        self.assertEqual(screen["subject"].tolist(), ["123456", "654321"])

    def test_missing_columns(self):
        """mp4 and video_csv are required."""
        with self.assertRaisesRegex(ValueError, "video_csv"):
            vs.screen_sessions([{"session": "s", "camera": "c", "mp4": "m"}])

    def test_workers_match_serial(self):
        """Two processes give the same rows as one."""
        serial = vs.screen_sessions(self.inputs)
        parallel = vs.screen_sessions(self.inputs, workers=2)
        pd.testing.assert_frame_equal(
            serial.drop(columns=VOLATILE), parallel.drop(columns=VOLATILE)
        )

    def test_files_cache_and_rescreen(self):
        """Writes the table and detail files; a second run screens only
        errors, other library versions and cameras missing quality."""
        out = self.tmp / "out"
        broken = {
            **self.clean,
            "camera": "side_camera",
            "mp4": str(self.tmp / "none.mp4"),
        }
        inputs = pd.DataFrame(self.inputs + [broken])
        first = vs.screen_sessions(inputs, out_dir=out, quality=False)
        self.assertTrue((out / vs.SCREEN_FILE).exists())
        self.assertTrue(
            (out / SESSION / "video_timing_bottom_camera.json").exists()
        )
        self.assertTrue(first["reason"].iloc[2].startswith("error"))
        self.assertTrue(first["quality"].isna().all())

        with mock.patch.object(
            vs, "_screen_camera", wraps=vs._screen_camera
        ) as screened:
            vs.screen_sessions(inputs, out_dir=out, quality=False)
        self.assertEqual(
            [c.args[1] for c in screened.call_args_list], ["side_camera"]
        )
        with mock.patch.object(
            vs, "_screen_camera", wraps=vs._screen_camera
        ) as screened:
            again = vs.screen_sessions(inputs, out_dir=out)
        self.assertEqual(screened.call_count, 3)  # quality now asked for
        self.assertEqual(
            again["quality"].iloc[:2].tolist(),
            ["use", "exclude: sharpness_dev <= 0.45"],
        )
        with (
            mock.patch.object(vs, "VERSIONS", "other"),
            mock.patch.object(
                vs, "_screen_camera", wraps=vs._screen_camera
            ) as screened,
        ):
            vs.screen_sessions(inputs.iloc[:1], out_dir=out)
        self.assertEqual(screened.call_count, 1)

        table = pd.read_csv(out / vs.SCREEN_FILE)
        self.assertEqual(len(table), 3)  # the last row per camera
        lines = (out / vs.SCREEN_LOG).read_text().splitlines()
        self.assertEqual(len(lines), 3 + 1 + 3 + 1)

    def test_load_screen(self):
        """Types survive the CSV: subject as text, empty reason as ""."""
        out = self.tmp / "loaded"
        vs.screen_sessions(self.inputs, out_dir=out, quality=False)
        screen = vs.load_screen(out / vs.SCREEN_FILE)
        self.assertEqual(screen["use"].tolist(), [True, True])
        self.assertEqual(screen["reason"].tolist(), ["", ""])
        self.assertEqual(screen["subject"].tolist(), ["123456", "654321"])

    def test_url_download_cached_per_session(self):
        """Two cameras of a session share one download of the log."""
        log_url = URL + "Event_94.bin"
        inputs = [
            {**self.clean, "trigger_log": log_url},
            {**self.clean, "camera": "side_camera", "trigger_log": log_url},
        ]
        data = Path(self.clean["trigger_log"]).read_bytes()
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **k: io.BytesIO(data),
        ) as urlopen:
            screen = vs.screen_sessions(inputs, quality=False)
        self.assertEqual(urlopen.call_count, 1)
        self.assertTrue(screen["trigger_log"].all())


if __name__ == "__main__":
    unittest.main()
