"""Screen behavior videos before analysis: timing QC plus quality QC.

One row per session x camera says whether the camera's video may be
analyzed (``use``) and, if not, why (``reason``). The caller passes the
file locations, each a local path or an HTTPS URL; this module never
searches for files. Nothing is written unless ``out_dir`` is given.

- :func:`screen_camera`: one camera, in memory (a gate inside a loop).
- :func:`screen_sessions`: many cameras, optionally in parallel; with
  ``out_dir``, writes ``video_screen.csv`` and the per-camera detail files
  and skips cameras already screened by the same library version.
- :func:`load_screen`: reads ``video_screen.csv`` and applies the manual
  overrides in ``screen_overrides.csv`` beside it.

``reason`` is empty for a camera in use; otherwise ``timing: exclude:
<check>`` (``exclude: unreadable`` for a video CSV whose content cannot be
used), ``quality: exclude: <check>``, or ``error: <text>`` for a camera
that could not be screened (a file that cannot be opened, network), which
is not an exclusion and is screened again on the next run.

Example::

    inputs = pd.DataFrame([{
        "session": "behavior_816212_2025-12-05_13-47-41",
        "camera": "bottom_camera",
        "mp4": ".../behavior-videos/bottom_camera.mp4",  # or https://...
        "video_csv": ".../behavior-videos/bottom_camera.csv",
        "behavior_json": ".../behavior/816212_2025-12-05_13-47-41.json",
        "trigger_log": ".../behavior/raw.harp/BehaviorEvents/Event_94.bin",
    }])
    screen = screen_sessions(inputs, out_dir="screen/", workers=8)
    screen = load_screen("screen/video_screen.csv")
    screen.query("view == 'bottom' and use")
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
import tempfile
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from aind_video_utils import __version__ as aind_video_utils_version
from aind_video_utils import read_mp4_frame_index

from aind_dynamic_foraging_behavior_video_analysis import (
    __version__,
)
from aind_dynamic_foraging_behavior_video_analysis import (
    video_quality_qc as vqq,
)
from aind_dynamic_foraging_behavior_video_analysis import (
    video_timing_qc as vtq,
)

# Columns of the inputs table; the last two may be missing or empty.
INPUT_COLUMNS = [
    "session",
    "camera",
    "mp4",
    "video_csv",
    "behavior_json",
    "trigger_log",
]
SCREEN_COLUMNS = [
    "session",
    "subject",
    "camera",
    "view",
    "mp4",
    "use",
    "reason",
    "timing",
    "timing_method",
    "frames_lost",
    "glitch_rows",
    "frame_count_diff",
    "trigger_log",
    "quality",
    "window",
    "sharpness",
    "mean",
    "pct_clipped_high",
    "similarity_p5",
    "versions",
    "screened_at",
    "seconds",
]
SCREEN_FILE = "video_screen.csv"
SCREEN_LOG = "video_screen.jsonl"
OVERRIDES_FILE = "screen_overrides.csv"
CARD_FILE = "session_card_{camera}.png"
# Timing verdict of a video CSV whose content cannot be used.
UNREADABLE = "exclude: unreadable"
# A download gives up on a stalled read after this many seconds; the
# camera then records an error and is screened again on the next run.
DOWNLOAD_TIMEOUT_S = 120
VERSIONS = (
    f"aind_dynamic_foraging_behavior_video_analysis={__version__}; "
    f"aind_video_utils={aind_video_utils_version}"
)


def _is_url(location):
    """True for an http(s) URL."""
    return str(location).startswith(("http://", "https://"))


def _local(location, folder):
    """A local path for ``location``: itself, or for a URL a copy
    downloaded into ``folder`` (once per URL; the readers of video CSVs
    and trigger logs need local files, and a session JSON that cannot be
    fetched must be an error rather than a fallback window)."""
    if not _is_url(location):
        return Path(location)
    digest = hashlib.sha1(str(location).encode()).hexdigest()[:12]
    path = Path(folder) / f"{digest}_{Path(str(location)).name}"
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        with (
            urllib.request.urlopen(
                location, timeout=DOWNLOAD_TIMEOUT_S
            ) as response,
            partial.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)
        partial.rename(path)
    return path


def _given(value):
    """None for a missing optional input (None, NaN or empty)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value if str(value) else None


def _subject(session):
    """Subject ID from ``behavior_<subject>_<date>_<time>``; else None."""
    match = re.match(r"^[a-z]+_(\d+)_", str(session))
    return match.group(1) if match else None


def _write_unreadable(session_out, camera, message):
    """The timing record of a camera whose CSV content cannot be used."""
    session_out.mkdir(parents=True, exist_ok=True)
    record = {
        "camera": camera,
        "verdict": UNREADABLE,
        "error": message,
        "versions": {
            "aind_dynamic_foraging_behavior_video_analysis": __version__,
        },
    }
    path = session_out / vtq.RECORD_FILE.format(camera=camera)
    path.write_text(json.dumps(record, indent=2))


def _screen_timing(mp4, video_csv, trigger_log, tmp, session_out, camera):
    """Timing QC of one camera: ``(columns, local CSV, local trigger log
    or None)``.

    A CSV whose content cannot be used (no rows, unknown header, missing
    values) is excluded as ``unreadable``: it fails the same way on every
    run, so it is not an error. Failing to open it is an error.
    """
    csv = _local(video_csv, tmp)
    log_path, log_times = None, None
    if trigger_log is not None:
        try:
            log_path = _local(trigger_log, tmp)
            log_times = vtq.read_harp_trigger_log(log_path)
        except (ValueError, OSError):
            log_path = None  # unreadable: as without a log
    try:
        timing = vtq.load_video_timing(csv)
    except ValueError as e:
        if session_out is not None:
            # Name the CSV as given, not its temporary download.
            message = str(e).replace(str(csv), str(video_csv))
            _write_unreadable(session_out, camera, message)
        columns = {"timing": UNREADABLE, "trigger_log": log_times is not None}
        return columns, csv, log_path
    n_frames = read_mp4_frame_index(mp4).n_samples
    checks = vtq.check_video_timing(
        timing, log_times, video_frame_count=n_frames
    )
    if session_out is not None:
        vtq.write_video_timing(checks, session_out, camera)
    counts = checks.set_index("check")["count"]
    columns = {
        "timing": vtq.timing_verdict(checks),
        "timing_method": vtq._correction_method(checks),
        "frames_lost": int(counts["no_frames_lost"]),
        "glitch_rows": int(counts["harp_has_no_glitches"]),
        "frame_count_diff": int(counts["video_frame_count"]),
        "trigger_log": log_times is not None,
    }
    return columns, csv, log_path


def _screen_quality(
    row, behavior_json, csv, log_path, session_out, cards, given
):
    """Quality QC of one camera: its columns of the row. ``given`` maps
    each local copy to the location as given, for the window note."""
    frames, samples, checks, note = vqq.video_quality(
        row["mp4"], row["camera"], behavior_json, csv, log_path
    )
    for local, location in given.items():
        note = note.replace(local, location)
    if session_out is not None:
        vqq.write_video_quality(
            samples, checks, session_out, row["camera"], note
        )
        if cards:
            title = f"{row['session']}  {row['camera']}  ({note})"
            _write_card(
                frames, samples, checks, session_out, title, row["camera"]
            )
    return {
        "quality": vqq.quality_verdict(checks),
        "window": note,
        "sharpness": samples["sharpness"].median(),
        "mean": samples["mean"].median(),
        "pct_clipped_high": samples["pct_clipped_high"].median(),
        "similarity_p5": samples["similarity"].quantile(0.05),
    }


def _reason(row, quality):
    """The first failure, timing before quality; empty if in use."""
    if row["timing"] != "use":
        return f"timing: {row['timing']}"
    if quality and row["quality"] != "use":
        return f"quality: {row['quality']}"
    return ""


def _screen_camera(
    session,
    camera,
    mp4,
    video_csv,
    behavior_json,
    trigger_log,
    out_dir,
    quality,
    cards,
    tmp,
):
    """:func:`screen_camera`, with downloads cached in ``tmp``."""
    start = datetime.datetime.now(datetime.timezone.utc)
    row = dict.fromkeys(SCREEN_COLUMNS)
    row.update(
        {
            "session": session,
            "subject": _subject(session),
            "camera": camera,
            "view": vqq.camera_view(camera),
            "mp4": str(mp4),
            "versions": VERSIONS,
            "screened_at": start.isoformat(timespec="seconds"),
        }
    )
    session_out = None if out_dir is None else Path(out_dir) / str(session)
    try:
        json_path = None
        if behavior_json is not None:
            # Downloaded here so a network failure is an error, not a
            # fallback window.
            json_path = _local(behavior_json, tmp)
        timing, csv, log_path = _screen_timing(
            mp4, video_csv, trigger_log, tmp, session_out, camera
        )
        row.update(timing)
        if quality:
            given = {
                str(csv): str(video_csv),
                str(json_path): str(behavior_json),
                str(log_path): str(trigger_log),
            }
            row.update(
                _screen_quality(
                    row, json_path, csv, log_path, session_out, cards, given
                )
            )
        row["reason"] = _reason(row, quality)
    except Exception as e:  # noqa: BLE001 - recorded, never raised
        last = traceback.extract_tb(e.__traceback__)[-1]
        row["reason"] = (
            f"error: {type(e).__name__}: {e} "
            f"({Path(last.filename).name}:{last.lineno})"
        )
    row["use"] = row["reason"] == ""
    elapsed = datetime.datetime.now(datetime.timezone.utc) - start
    row["seconds"] = round(elapsed.total_seconds(), 1)
    return row


def _write_card(frames, samples, checks, session_out, title, camera):
    """Save the session card; matplotlib is imported only here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aind_dynamic_foraging_behavior_video_analysis import (
        video_quality_report as vqr,
    )

    fig = vqr.session_card(frames, samples, checks, title)
    fig.savefig(session_out / CARD_FILE.format(camera=camera), dpi=90)
    plt.close(fig)


def screen_camera(
    session,
    camera,
    mp4,
    video_csv,
    behavior_json=None,
    trigger_log=None,
    out_dir=None,
    quality=True,
    cards=False,
) -> dict:
    """Screen one camera: timing QC, then quality QC; never raises.

    Parameters
    ----------
    session, camera : str
        Names for the row (and the folder and file names in ``out_dir``).
    mp4, video_csv : str or pathlib.Path
        Local path or HTTPS URL of the camera's video and its CSV.
    behavior_json, trigger_log : str or pathlib.Path, optional
        The session JSON (for the task window) and the Harp camera trigger
        log (``Event_94.bin``). An unreadable log is ignored.
    out_dir : str or pathlib.Path, optional
        If given, writes ``video_timing_<camera>.json`` and the quality
        files into ``<out_dir>/<session>/``. Nothing is written otherwise.
    quality : bool
        Run quality QC; if False, ``use`` follows timing alone.
    cards : bool
        With ``out_dir``, also save the quality session card (PNG).

    Returns
    -------
    dict
        One row with the ``SCREEN_COLUMNS`` (see the module docstring).
    """
    tmp = Path(tempfile.mkdtemp(prefix="video_screen_"))
    try:
        return _screen_camera(
            session,
            camera,
            mp4,
            video_csv,
            _given(behavior_json),
            _given(trigger_log),
            out_dir,
            quality,
            cards,
            tmp,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _screen_session(rows, out_dir, quality, cards):
    """Screen one session's cameras, sharing downloads."""
    tmp = Path(tempfile.mkdtemp(prefix="video_screen_"))
    try:
        return [
            _screen_camera(
                r["session"],
                r["camera"],
                r["mp4"],
                r["video_csv"],
                _given(r.get("behavior_json")),
                _given(r.get("trigger_log")),
                out_dir,
                quality,
                cards,
                tmp,
            )
            for r in rows
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _read_log(path):
    """Rows of ``video_screen.jsonl``, the last per session x camera."""
    if not path.exists():
        return {}
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["session"], r["camera"]): r for r in rows}


def _is_current(row, quality):
    """True if ``row`` needs no re-screening: same versions, no error, and
    quality measured if it is asked for."""
    return (
        row["versions"] == VERSIONS
        and not row["reason"].startswith("error")
        and (not quality or row["quality"] is not None)
    )


def _inputs_table(inputs):
    """``inputs`` as a table of the ``INPUT_COLUMNS``, None where empty."""
    table = pd.DataFrame(inputs)
    missing = set(INPUT_COLUMNS[:4]) - set(table.columns)
    if missing:
        raise ValueError(f"inputs lack columns {sorted(missing)}")
    table = table.reindex(columns=INPUT_COLUMNS).astype(object)
    return table.where(table.notna(), None)


def _record(rows, results, log_path):
    """Keep one session's rows, append them to the log, print them."""
    for r in rows:
        results[(r["session"], r["camera"])] = r
    if log_path is not None:
        with log_path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    summary = ", ".join(f"{r['camera']}: {r['reason'] or 'use'}" for r in rows)
    print(f"  {rows[0]['session']}: {summary}", flush=True)


def screen_sessions(
    inputs, out_dir=None, quality=True, cards=False, workers=1
) -> pd.DataFrame:
    """Screen many cameras; one row per input row.

    Parameters
    ----------
    inputs : pandas.DataFrame or list of dict
        One row per session x camera with the ``INPUT_COLUMNS`` (the
        optional ``behavior_json`` and ``trigger_log`` may be missing).
        To screen only some cameras, pass only those rows.
    out_dir : str or pathlib.Path, optional
        If given: rows are appended to ``video_screen.jsonl`` as each
        session finishes (an interrupted run resumes), ``video_screen.csv``
        is rebuilt from it at the end, detail files go under
        ``<out_dir>/<session>/``, and cameras already screened by the same
        library versions without an error are not screened again.
    quality, cards : bool
        See :func:`screen_camera`.
    workers : int
        Sessions screened in parallel (processes) when > 1.

    Returns
    -------
    pandas.DataFrame
        The ``SCREEN_COLUMNS``, in the order of ``inputs``.
    """
    table = _inputs_table(inputs)
    log_path = None if out_dir is None else Path(out_dir) / SCREEN_LOG
    done = {}
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        done = {
            key: row
            for key, row in _read_log(log_path).items()
            if _is_current(row, quality)
        }
    todo = table[
        [(s, c) not in done for s, c in zip(table["session"], table["camera"])]
    ]
    groups = [
        g.to_dict("records") for _, g in todo.groupby("session", sort=False)
    ]
    print(
        f"video screen: {len(table)} cameras, {len(table) - len(todo)} "
        f"already screened, {len(todo)} to screen"
    )
    results = dict(done)
    if workers > 1 and len(groups) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_screen_session, g, out_dir, quality, cards)
                for g in groups
            ]
            for future in as_completed(futures):
                _record(future.result(), results, log_path)
    else:
        for g in groups:
            rows = _screen_session(g, out_dir, quality, cards)
            _record(rows, results, log_path)
    screen = pd.DataFrame(
        [results[(s, c)] for s, c in zip(table["session"], table["camera"])],
        columns=SCREEN_COLUMNS,
    )
    if log_path is not None:
        everything = pd.DataFrame(
            list(_read_log(log_path).values()), columns=SCREEN_COLUMNS
        )
        everything.to_csv(log_path.parent / SCREEN_FILE, index=False)
    return screen


def load_screen(path) -> pd.DataFrame:
    """Read ``video_screen.csv`` and apply the overrides beside it.

    ``screen_overrides.csv`` (optional, same folder) has one row per
    reviewed camera: ``session``, ``camera``, ``verdict`` (``use`` or
    ``exclude: <why>``), ``note``, ``reviewer``, ``date``. An override
    sets ``use`` from its verdict and records ``override_note``.
    """
    path = Path(path)
    screen = pd.read_csv(path, dtype={"subject": str})
    screen["reason"] = screen["reason"].fillna("")
    screen["override_note"] = None
    overrides_path = path.parent / OVERRIDES_FILE
    if overrides_path.exists():
        overrides = pd.read_csv(overrides_path, dtype=str)
        for _, o in overrides.iterrows():
            rows = screen["session"].eq(o["session"]) & screen["camera"].eq(
                o["camera"]
            )
            screen.loc[rows, "use"] = o["verdict"] == "use"
            screen.loc[rows, "override_note"] = (
                f"{o['verdict']} ({o['reviewer']}, {o['date']}): {o['note']}"
            )
    return screen
