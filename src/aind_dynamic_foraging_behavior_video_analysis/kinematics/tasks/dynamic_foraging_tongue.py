"""Tongue kinematics in the AIND dynamic foraging task.

Composes the general steps (``kinematics.pipeline``) with this task's
files, clock and events:

* **Frame times**: the camera's corrected Harp time from
  ``video_timing_qc`` (with the Harp trigger log when present). Sessions
  whose timing cannot be trusted raise ``ValueError``.
* **Clock**: session time, seconds from the first go cue (the clock of the
  NWB ``*_in_session`` trial columns and lick ``timestamps``). The record
  keeps ``time_zero``, the Harp time of session time 0.
* **Events**: licks are matched to the movement and the submovement they
  fall in; each movement gets its trial (last go cue at or before it
  starts) and lick details; each trial gets a summary of its movements.

Outputs (``pipeline.write``): ``pose, kinematics, landmarks, movements,
submovements, bouts`` as in ``pipeline``, plus ``trials`` (the NWB trials
table with movement summaries), ``licks`` (the NWB licks table with
``movement_id`` and ``submovement_id``), ``events`` when given, and
``record.json`` with the timing verdict and ``quality`` (``lick_coverage``,
``movement_duration_p50``). Read with ``pipeline.read`` and
``quality_summary``.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis import (
    video_clips,
    video_timing_qc,
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
    pipeline as pl,
    pose_io,
    signal,
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics.intervals import (  # noqa: E501
    closest_interval,
    summarize_by_window,
)

TONGUE = "tongue_tip_center"
JAW = "jaw"

TONGUE_SETTINGS = pl.Settings(
    fs_hz=500.0,
    landmarks=(JAW, "spout_l", "spout_r"),
    keypoints=(
        pl.KeypointSettings(
            name=TONGUE,
            min_likelihood=0.9,
            smooth=pl.Smooth(cutoff_hz=50.0, order=4, max_gap_s=0.02),
            reference=JAW,
            segments={
                "movement": pl.Segmentation(
                    "tracked_runs", {"max_gap_frames": 10},
                    features=("path", "excursion", "outbound")),
                # Every reversal is kept (no minimum length): which pieces
                # are real is decided downstream, from the saved
                # radial_velocity_sum / _peak per piece.
                "submovement": pl.Segmentation(
                    "zero_crossings", {"max_gap_frames": 3},
                    within="movement",
                    features=("path", "excursion", "outbound")),
                "bout": pl.Segmentation(
                    "bouts", {"of": "movement", "max_gap_s": 0.5,
                              "min_count": 2}),
            },
        ),
    ),
)
"""Settings matching the pipeline before 0.5.0 (confidence 0.9, 50 Hz
4th-order low-pass, movements bridging up to 10 missing frames), positions
relative to the session's median jaw position, submovements split at every
reversal of radial velocity (none merged), and bouts of movements less than
0.5 s apart."""


# --- Finding files ----------------------------------------------------------
def session_name_from_path(path) -> str:
    """``behavior_<subject>_<YYYY-MM-DD>_<HH-MM-SS>`` found in a path."""
    m = re.search(r"behavior_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
                  str(path))
    if not m:
        raise ValueError(f"no session name in path: {path}")
    return m.group(0)


def find_inputs(session_dir, camera="BottomCamera") -> dict:
    """Video CSV and Harp trigger log of one camera in a session folder.

    The video CSV is ``behavior-videos/<camera>/metadata.csv`` (AIND
    layout) or ``behavior-videos/<snake_case camera>.csv`` (older flat
    layout). The trigger log is
    ``behavior/raw.harp/BehaviorEvents/Event_94.bin``, or None.

    Raises
    ------
    FileNotFoundError
        If there is no video CSV.
    """
    session_dir = Path(session_dir)
    videos = None
    for root, dirs, _ in os.walk(session_dir):
        if "behavior-videos" in dirs:
            videos = Path(root) / "behavior-videos"
            break
    if videos is None:
        raise FileNotFoundError(
            f"no behavior-videos folder in {session_dir}")
    for csv in (videos / camera / "metadata.csv",
                videos / f"{_snake(camera)}.csv"):
        if csv.exists():
            break
    else:
        raise FileNotFoundError(f"no video CSV for {camera} in {videos}")
    logs = sorted(session_dir.rglob("raw.harp/BehaviorEvents/Event_94.bin"))
    return {"video_csv": csv, "trigger_log": logs[0] if logs else None}


def find_labeled_video(session_id, data_root):
    """Lightning Pose labeled video of a session.

    The first ``<data_root>/<session_id>*/pred_outputs/video_preds/
    labeled_videos/*_labeled.mp4``.

    Raises
    ------
    FileNotFoundError
        If there is none.
    """
    data_root = Path(data_root)
    for subdir in data_root.glob(f"{session_id}*"):
        labeled_dir = (subdir / "pred_outputs" / "video_preds"
                       / "labeled_videos")
        matches = list(labeled_dir.glob("*_labeled.mp4"))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"Labeled video not found for {session_id}")


# --- Timing ----------------------------------------------------------------
def frame_times(video_csv, trigger_log=None):
    """Corrected Harp time of every video frame.

    Returns
    -------
    harp_time : numpy.ndarray
        One time per frame (row of the video CSV).
    timing : dict
        ``verdict``, ``failed_checks``, ``harp_sources`` (rows per source)
        and ``trigger_log`` (path or None).

    Raises
    ------
    ValueError
        If the timing verdict is not ``use``.
    """
    timing = video_timing_qc.load_video_timing(video_csv)
    triggers = (None if trigger_log is None
                else video_timing_qc.read_harp_trigger_log(trigger_log))
    checks = video_timing_qc.check_video_timing(timing,
                                                trigger_times=triggers)
    fixed = video_timing_qc.correct_video_timing(timing,
                                                 trigger_times=triggers)
    failed = checks.loc[checks["passed"].eq(False), "check"].tolist()
    return fixed["harp_time"].to_numpy(float), {
        "verdict": video_timing_qc.timing_verdict(checks),
        "failed_checks": failed,
        "harp_sources": {str(k): int(v) for k, v in
                         fixed["harp_source"].value_counts().items()},
        "trigger_log": None if trigger_log is None else str(trigger_log),
    }


def session_time_zero(trials: pd.DataFrame) -> float:
    """Harp time of session time 0, from a trials table with both
    ``goCue_start_time`` and ``goCue_start_time_in_session``.

    The data-utils trials table has only the second; then pass
    ``time_zero`` from ``load_nwb_tables`` instead.
    """
    missing = {"goCue_start_time", "goCue_start_time_in_session"} - set(
        trials.columns)
    if missing:
        raise ValueError(
            f"trials lacks {sorted(missing)}: pass time_zero (the Harp time "
            "of the first go cue, e.g. from load_nwb_tables)")
    offset = (trials["goCue_start_time"]
              - trials["goCue_start_time_in_session"])
    return float(offset.dropna().median())


# --- Events ----------------------------------------------------------------
def _lick_side(event):
    """'left' / 'right' from the NWB lick ``event`` name."""
    s = event.astype("string")
    return s.str.extract(r"(left|right)", expand=False)


def annotate_movements(movements, trials, licks, lick_tolerance_s=0.001):
    """Trial and lick columns for each movement; movement for each lick.

    Parameters
    ----------
    movements : pandas.DataFrame
        From ``pipeline.process`` (``movement_id, start_time, end_time``),
        session time.
    trials : pandas.DataFrame
        ``trial, goCue_start_time_in_session``.
    licks : pandas.DataFrame
        ``timestamps`` (session time); ``cue_response``, ``rewarded`` and
        ``event`` are carried over when present.
    lick_tolerance_s : float
        A lick belongs to a movement if it is within this much of it
        (default 1 ms, half a frame at 500 Hz; see ``match_licks``).

    Returns
    -------
    movements : pandas.DataFrame
        Adds ``trial`` (last go cue at or before the start; NA before the
        first), ``go_cue_time``, ``latency_from_go``, ``spans_go_cue``
        (a go cue falls inside the movement), ``number_in_trial``,
        ``n_licks``, and from the first lick ``lick_time``,
        ``cue_response``, ``rewarded``, ``lick_side``.
    licks : pandas.DataFrame
        Adds ``movement_id``, ``closest_movement_id`` and
        ``closest_movement_offset_s`` (see ``match_licks``).
    """
    mv = movements.sort_values("start_time", kind="stable").copy()
    tr = trials.sort_values("goCue_start_time_in_session")
    go = tr["goCue_start_time_in_session"].to_numpy(float)
    trial_ids = tr["trial"].to_numpy()
    st = mv["start_time"].to_numpy(float)
    en = mv["end_time"].to_numpy(float)

    row = np.searchsorted(go, st, side="right") - 1
    has = row >= 0
    mv["trial"] = pd.array(
        np.where(has, trial_ids[np.maximum(row, 0)], 0), dtype="Int64")
    mv.loc[~has, "trial"] = pd.NA
    mv["go_cue_time"] = np.where(has, go[np.maximum(row, 0)], np.nan)
    mv["latency_from_go"] = st - mv["go_cue_time"]
    nxt = np.minimum(row + 1, len(go) - 1)
    mv["spans_go_cue"] = (row + 1 < len(go)) & (go[nxt] < en)
    mv["number_in_trial"] = pd.array(
        mv.groupby("trial").cumcount().to_numpy() + 1, dtype="Int64")
    mv.loc[~has, "number_in_trial"] = pd.NA

    mv, lk = match_licks(mv, licks, "movement_id", lick_tolerance_s)
    return mv.sort_index(), lk


def match_licks(segments, licks, id_col, lick_tolerance_s=0.001):
    """Match each lick to the segment it falls in, and record the closest.

    A lick is the tongue touching the spout, so it should fall inside a
    tracked segment. ``lick_tolerance_s`` allows only for timing
    uncertainty between the lickometer and the camera: half a frame by
    default, which reproduces the matching before 0.5.0 (nearest frame,
    then that frame's movement). Licks just outside a segment are not
    matched, so coverage counts them as missed, but how far they are is
    kept so a looser rule can be applied later.

    Parameters
    ----------
    segments : pandas.DataFrame
        ``id_col, start_time, end_time``, not overlapping (e.g. movements
        or submovements of one keypoint).
    licks : pandas.DataFrame
        ``timestamps``; ``cue_response``, ``rewarded`` and ``event`` are
        carried over when present.
    id_col : str
        Segment id column, ``<kind>_id``.
    lick_tolerance_s : float
        Largest distance from a segment for a match (default 1 ms).

    Returns
    -------
    segments : pandas.DataFrame
        Adds ``n_licks`` and, from the first matched lick, ``lick_time``,
        ``cue_response``, ``rewarded`` (False without a lick) and
        ``lick_side``.
    licks : pandas.DataFrame
        Adds ``<kind>_id`` (the matched segment, NA when none),
        ``closest_<kind>_id`` and ``closest_<kind>_offset_s`` (signed time
        from the lick to the closest segment: 0 inside, negative before
        its start, positive after its end).
    """
    kind = id_col[:-3] if id_col.endswith("_id") else id_col
    seg = segments.sort_values("start_time", kind="stable").copy()
    lk = licks.copy()
    row, offset = closest_interval(lk["timestamps"].to_numpy(float),
                                   seg["start_time"], seg["end_time"])
    ids = seg[id_col].to_numpy()
    closest = pd.array(np.where(row >= 0, ids[np.maximum(row, 0)], 0),
                       dtype="Int64")
    closest[row < 0] = pd.NA
    hit = (row >= 0) & (np.abs(offset) <= lick_tolerance_s)
    lk[id_col] = closest.copy()
    lk.loc[~hit, id_col] = pd.NA
    lk[f"closest_{kind}_id"] = closest
    lk[f"closest_{kind}_offset_s"] = offset

    matched = lk[hit]
    if "event" in matched:
        matched = matched.assign(lick_side=_lick_side(matched["event"]))
    first = matched.sort_values("timestamps").groupby(id_col).first()
    seg["n_licks"] = (seg[id_col].map(matched.groupby(id_col).size())
                      .fillna(0).astype(int))
    seg["lick_time"] = seg[id_col].map(first["timestamps"])
    for col in ("cue_response", "rewarded"):
        if col in first:
            seg[col] = seg[id_col].map(
                first[col].astype("boolean")).astype("boolean")
            seg.loc[seg["n_licks"] == 0, col] = False
    seg["lick_side"] = seg[id_col].map(
        first.get("lick_side", pd.Series(dtype="string"))).astype("string")
    return seg.sort_index(), lk


def summarize_trials(trials, movements, licks):
    """Per trial: movement and lick summaries, joined to ``trials``.

    Adds ``n_movements``, ``n_lick_movements``, ``first_movement_id``,
    ``first_movement_latency``, ``cue_response_movement_id``,
    ``cue_response_movement_number``, ``lick_latency`` (first cue-response
    lick after the go cue), ``n_licks`` and ``lick_coverage`` (fraction of
    the trial's licks matched to a movement). A trial runs from its go cue
    to the next one.
    """
    tr = trials.sort_values("goCue_start_time_in_session").reset_index(
        drop=True)
    go = tr["goCue_start_time_in_session"].to_numpy(float)
    win = pd.DataFrame({"trial": tr["trial"].to_numpy(), "start_time": go,
                        "end_time": np.r_[go[1:], np.inf]})
    mv = movements.assign(has_lick=movements["n_licks"] > 0)
    out = summarize_by_window(mv, win, "start_time", {
        "n_movements": (None, "count"),
        "n_lick_movements": ("has_lick", "sum"),
        "first_movement_id": ("movement_id", "first"),
        "first_movement_latency": ("latency_from_go", "first"),
    })
    if "cue_response" in mv:
        cue = mv[mv["cue_response"].fillna(False).astype(bool)]
        cue_sum = summarize_by_window(cue, win, "start_time", {
            "cue_response_movement_id": ("movement_id", "first"),
            "cue_response_movement_number": ("number_in_trial", "first"),
            "lick_latency": ("lick_time", "first"),
        })
        out = out.join(cue_sum[["cue_response_movement_id",
                                "cue_response_movement_number",
                                "lick_latency"]])
        out["lick_latency"] = out["lick_latency"] - out["start_time"]
    lk = licks.assign(matched=licks["movement_id"].notna())
    lick_sum = summarize_by_window(lk, win, "timestamps", {
        "n_licks": (None, "count"),
        "lick_coverage": ("matched", "mean"),
    })
    out = out.join(lick_sum[["n_licks", "lick_coverage"]])
    for col in ("n_lick_movements",):
        out[col] = out[col].fillna(0).astype(int)
    for col in ("first_movement_id", "cue_response_movement_id",
                "cue_response_movement_number"):
        if col in out:
            out[col] = out[col].astype("Int64")
    out = out.drop(columns=["start_time", "end_time"])
    return trials.merge(out, on="trial", how="left")


def quality(movements, licks) -> dict:
    """Session quality numbers used to include sessions.

    ``lick_coverage`` is the fraction of licks inside a movement (the
    ``match_licks`` rule). ``lick_coverage_within_10ms`` also counts licks
    up to 10 ms outside one; a large gap between the two means licks sit
    just outside tracked movements (tracking starting late or ending early,
    or a clock offset).
    """
    n = int(len(licks))
    matched = int(licks["movement_id"].notna().sum())
    near = int((licks["closest_movement_offset_s"].abs() <= 0.010).sum())
    return {
        "n_licks": n,
        "n_licks_in_movement": matched,
        "lick_coverage": matched / n if n else float("nan"),
        "lick_coverage_within_10ms": near / n if n else float("nan"),
        "movement_duration_p50": float(movements["duration"].median())
        if len(movements) else float("nan"),
    }


def quality_summary(session_dir) -> dict:
    """``{session, lick_coverage, movement_duration_p50}`` from a saved
    session; NaN for anything missing."""
    rec = pl.read_record(session_dir)
    q = rec.get("quality", {}) or {}
    return {"session": rec.get("session"),
            "lick_coverage": float(q.get("lick_coverage", np.nan)),
            "movement_duration_p50": float(
                q.get("movement_duration_p50", np.nan))}


# --- Clips -----------------------------------------------------------------
def cut_trial_clips(trials, kinematics, video_path, save_dir,
                    clip_duration_s=10.0, pad_s=0.5):
    """Cut, per trial, the frames from ``pad_s`` before its go cue to
    ``clip_duration_s + pad_s`` after it, frame-exact.

    Frame ``f`` of ``kinematics`` is video frame ``f`` and its ``time`` is
    that frame's corrected Harp time in session time, the clock of
    ``goCue_start_time_in_session``, so the go cues are placed on those
    times by ``video_clips.cut_event_clips``. Right also in sessions with
    dropped frames, where a clip has fewer frames than the window's
    duration at the nominal rate.

    Parameters
    ----------
    trials : pandas.DataFrame
        Trials with ``goCue_start_time_in_session``; the index numbers the
        clips.
    kinematics : pandas.DataFrame
        ``frame`` and ``time`` (session time), e.g. the ``kinematics``
        table of ``run_session``; several keypoints may share a frame.
    video_path : str or pathlib.Path
        The session's video (an MP4 frame-aligned with the predictions).
    save_dir : str or pathlib.Path
        Output folder (created if needed).
    clip_duration_s, pad_s : float
        Seconds after the go cue, and padding on both sides.

    Returns
    -------
    pandas.DataFrame
        One row per trial, as ``video_clips.cut_clips`` returns, plus
        ``trial``. Clips are ``trial_<n>_f<start_frame:07d>.mp4`` with a
        JSON sidecar; an existing clip is kept, and a window that runs past
        either end of the video (or of ``kinematics``) is skipped
        (``clip_path`` NaN).

    Raises
    ------
    ValueError
        If ``kinematics`` does not have every frame from 0.
    """
    per_frame = kinematics.drop_duplicates("frame").sort_values("frame")
    if not np.array_equal(per_frame["frame"].to_numpy(),
                          np.arange(len(per_frame))):
        raise ValueError("kinematics needs every frame from 0")
    frame_time = per_frame["time"].to_numpy(float)
    rows = []
    for trial_num, trial_row in trials.iterrows():
        out = video_clips.cut_event_clips(
            video_path, trial_row["goCue_start_time_in_session"], frame_time,
            before=pad_s, after=clip_duration_s + pad_s,
            out_dir=save_dir, prefix=f"trial_{trial_num}",
        )
        out.insert(0, "trial", trial_num)
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --- One session, many sessions ---------------------------------------------
def run_session(pose_path, video_csv, trials, licks, trigger_log=None,
                settings=TONGUE_SETTINGS, session=None, events=None,
                lick_tolerance_s=None, time_zero=None) -> dict:
    """Process one session; returns the tables and record (see module).

    Parameters
    ----------
    pose_path : str or pathlib.Path
        Tracker output (``pose_io.read_pose``).
    video_csv, trigger_log : path
        The camera's frame times (``frame_times``).
    trials, licks : pandas.DataFrame
        NWB trials and licks tables (``load_nwb_tables``).
    events : pandas.DataFrame, optional
        Saved alongside unchanged.
    lick_tolerance_s : float, optional
        See ``match_licks``; default half a frame (``0.5 / fs_hz``).
    time_zero : float, optional
        Harp time of the first go cue (session time 0), as returned by
        ``load_nwb_tables``. Default: from ``trials`` (``session_time_zero``).
    """
    session = session or session_name_from_path(pose_path)
    harp_time, timing = frame_times(video_csv, trigger_log)
    if time_zero is None:
        time_zero = session_time_zero(trials)
    if lick_tolerance_s is None:
        lick_tolerance_s = 0.5 / settings.fs_hz
    pose, frames = signal.add_time(pose_io.read_pose(pose_path),
                                   harp_time - time_zero)
    result = pl.process(pose, settings, session=session)
    result["movements"], result["licks"] = annotate_movements(
        result["movements"], trials, licks, lick_tolerance_s)
    if "submovements" in result:
        result["submovements"], result["licks"] = match_licks(
            result["submovements"], result["licks"], "submovement_id",
            lick_tolerance_s)
    result["trials"] = summarize_trials(trials, result["movements"],
                                        result["licks"])
    if events is not None:
        result["events"] = events
    for name in ("licks", "trials", "events"):
        if name in result and "session" not in result[name]:
            result[name].insert(0, "session", session)
    result["record"].update({
        "pose_path": str(pose_path),
        "video_csv": str(video_csv),
        "clock": {"name": "session time",
                  "description": "seconds from the first go cue",
                  "time_zero": time_zero},
        "frames": frames,
        "timing": timing,
        "lick_tolerance_s": lick_tolerance_s,
        "quality": quality(result["movements"], result["licks"]),
    })
    return result


def load_nwb_tables(session, nwb_folder=None):
    """Trials, licks and events tables of a session's behavior NWB, and
    the Harp time of its first go cue.

    Uses ``kinematics_nwb_utils.get_nwb_file`` (which builds the NWB from
    the session JSON if needed) and the AIND data-utils tables, whose
    ``*_in_session`` times start at the first go cue.

    Returns
    -------
    trials, licks, events : pandas.DataFrame
    time_zero : float
        ``nwb.trials["goCue_start_time"][0]``, as before 0.5.0.
    """
    import aind_dynamic_foraging_data_utils.nwb_utils as nwb_utils
    from aind_dynamic_foraging_basic_analysis.licks import annotation

    from aind_dynamic_foraging_behavior_video_analysis.kinematics.kinematics_nwb_utils import (  # noqa: E501
        get_nwb_file,
    )

    kwargs = {} if nwb_folder is None else {"nwb_folder": nwb_folder}
    nwb = get_nwb_file(session, **kwargs)
    nwb.df_events = nwb_utils.create_df_events(nwb)
    try:
        nwb.df_trials = nwb_utils.create_df_trials(nwb, cut_interruptions=True)
    except TypeError as e:
        if "cut_interruptions" not in str(e):
            raise
        nwb.df_trials = nwb_utils.create_df_trials(nwb)
    licks = annotation.annotate_licks(nwb)
    time_zero = float(nwb.trials["goCue_start_time"][0])
    return nwb.df_trials, licks, nwb.df_events, time_zero


def _snake(camera):
    """``BottomCamera`` -> ``bottom_camera`` (the screen's camera names)."""
    return re.sub(r"(?<!^)([A-Z])", r"_\1", camera).lower()


def screen_status(screen, session, camera):
    """None if ``screen`` allows the session's camera, else why not.

    ``screen`` is the ``video_screen`` table (``load_screen``): one row per
    session x camera with ``use`` and ``reason``.
    """
    rows = screen[(screen["session"] == session)
                  & (screen["camera"] == _snake(camera))]
    if rows.empty:
        return "excluded: not screened"
    if not bool(rows["use"].iloc[0]):
        return f"excluded: {rows['reason'].iloc[0]}"
    return None


def run_batch(pose_paths, data_root, out_root, settings=TONGUE_SETTINGS,
              camera="BottomCamera", use_trigger_log=None, force=False,
              nwb_folder=None, screen=None):
    """Process many sessions into ``<out_root>/<session>/``.

    A session already written is skipped unless ``force``. Errors are
    printed, collected in ``<out_root>/errors.txt`` and do not stop the
    batch. ``use_trigger_log``: None uses it when present, True requires
    it, False ignores it. ``screen`` (``video_screen.load_screen``), when
    given, excludes sessions whose camera is not in use or not screened.

    Returns
    -------
    pandas.DataFrame
        ``session, status`` ('done', 'skipped', 'excluded: ...' or
        'error: ...').
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for pose_path in pose_paths:
        session = session_name_from_path(pose_path)
        out_dir = out_root / session
        if pl.is_done(out_dir) and not force:
            rows.append({"session": session, "status": "skipped"})
            continue
        excluded = (None if screen is None
                    else screen_status(screen, session, camera))
        if excluded:
            rows.append({"session": session, "status": excluded})
            continue
        try:
            files = find_inputs(Path(data_root) / session, camera)
            log = files["trigger_log"] if use_trigger_log is not False \
                else None
            if use_trigger_log and log is None:
                raise FileNotFoundError("no camera trigger log (Event_94.bin)")
            trials, licks, events, t0 = load_nwb_tables(session, nwb_folder)
            result = run_session(pose_path, files["video_csv"], trials,
                                 licks, log, settings, session, events,
                                 time_zero=t0)
            pl.write(result, out_dir)
            rows.append({"session": session, "status": "done"})
        except Exception as e:  # noqa: BLE001 - one session must not stop all
            print(f"{session}: {e!r}")
            rows.append({"session": session, "status": f"error: {e!r}"})
    status = pd.DataFrame(rows, columns=["session", "status"])
    errors = status[status["status"].str.startswith("error")]
    if len(errors):
        (out_root / "errors.txt").write_text(
            "\n".join(f"{s}: {m}" for s, m in errors.itertuples(index=False)))
    return status
