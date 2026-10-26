"""Run the steps for every keypoint from one settings object, and save.

``process(pose, settings)`` turns a pose table with ``time`` into:

``kinematics``
    One row per frame per keypoint: ``frame, time, keypoint, valid, x, y,
    vx, vy, speed, radial_velocity``, then ``<kind>_id`` for each kind of
    segment (NA outside one). Positions are smoothed image pixels; NaN where
    not valid.
``<kind>s`` (e.g. ``movements``, ``submovements``, ``bouts``)
    One row per segment: ``<kind>_id, keypoint, start_frame, end_frame,
    start_time, end_time, duration, n_frames, n_valid``; ``<within>_id`` and
    ``number_in_<within>`` for segments inside other segments;
    ``direction`` for ``zero_crossings``; ``n_<of>s`` and ``rate_hz`` for
    bouts (and ``bout_id`` added to the grouped table); then the feature
    columns.
``landmarks``
    ``keypoint, x, y, n_valid, stat``.
``pose``
    The input, unchanged.
``record``
    Settings, versions and per-keypoint counts (a dict, saved as JSON).

Ids are 1-based, unique within a session, in time order per keypoint.
Every table gets a ``session`` column when ``session`` is given.
"""
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy

from aind_dynamic_foraging_behavior_video_analysis import __version__
from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
    features as feat,
    landmarks as lm,
    segment,
    signal,
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics.intervals import (  # noqa: E501
    label_frames,
)

RECORD_FILE = "record.json"
KINEMATICS_COLUMNS = ["frame", "time", "keypoint", "valid", "x", "y", "vx",
                      "vy", "speed", "radial_velocity"]
SEGMENT_COLUMNS = ["keypoint", "start_frame", "end_frame", "start_time",
                   "end_time", "duration", "n_frames", "n_valid"]


@dataclass(frozen=True)
class Smooth:
    """How positions are smoothed; see ``signal.smooth``."""

    cutoff_hz: Optional[float] = 50.0
    order: int = 4
    max_gap_s: float = 0.02
    kind: str = "cubic"


@dataclass(frozen=True)
class Segmentation:
    """One kind of segment for a keypoint.

    ``method`` and its ``params``:

    * ``tracked_runs``: ``max_gap_frames``.
    * ``above_threshold``: ``column`` (default 'speed'), ``on``, ``off``,
      ``min_frames``, ``max_gap_frames``.
    * ``zero_crossings``: ``column`` (default 'radial_velocity'),
      ``deadband``, ``max_gap_frames``, ``min_samples``, ``short``.
    * ``speed_minima``: ``column`` (default 'speed'), ``min_prominence``,
      ``max_minimum_speed``, ``max_gap_frames``, ``min_samples``,
      ``short``.
    * ``bouts``: ``of`` (kind of segment to group), ``max_gap_s``,
      ``min_count``.

    ``within`` names another kind to split (e.g. submovements within
    movements). ``features`` are group names from ``features``.
    """

    method: str
    params: dict = field(default_factory=dict)
    within: Optional[str] = None
    features: tuple = ()


@dataclass(frozen=True)
class KeypointSettings:
    """How one keypoint is processed.

    ``reference`` is a landmark keypoint: positions in features are
    relative to it and ``radial_velocity`` is computed from it. ``flip_x``
    / ``flip_y`` mirror relative positions. ``segments`` maps a kind name
    (e.g. 'movement') to its ``Segmentation``, in the order they run.
    ``feature_options`` are passed to feature groups (e.g.
    ``{'spectrum': {'band': (4, 12)}}``).
    """

    name: str
    min_likelihood: float = 0.9
    smooth: Smooth = Smooth()
    reference: Optional[str] = None
    flip_x: bool = False
    flip_y: bool = False
    segments: dict = field(default_factory=dict)
    feature_options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    """Everything ``process`` needs besides the data.

    ``landmarks`` are keypoints summarized as one position per session
    (references are added automatically), masked at
    ``landmark_min_likelihood`` unless they also have keypoint settings.
    """

    fs_hz: float
    keypoints: tuple = ()
    landmarks: tuple = ()
    landmark_min_likelihood: float = 0.9

    def to_dict(self) -> dict:
        """Plain dict (JSON-ready)."""
        return json.loads(json.dumps(dataclasses.asdict(self)))

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        """Inverse of ``to_dict``."""
        kps = []
        for k in d.get("keypoints", ()):
            k = dict(k)
            k["smooth"] = Smooth(**k.get("smooth", {}))
            k["segments"] = {
                name: Segmentation(**{**s, "features": tuple(
                    s.get("features", ()))})
                for name, s in k.get("segments", {}).items()
            }
            kps.append(KeypointSettings(**k))
        return cls(fs_hz=d["fs_hz"], keypoints=tuple(kps),
                   landmarks=tuple(d.get("landmarks", ())),
                   landmark_min_likelihood=d.get("landmark_min_likelihood",
                                                 0.9))


def _intervals(frames, seg: Segmentation, found):
    """Run one segmentation; ``found`` holds earlier kinds' intervals."""
    p = dict(seg.params)
    within = found[seg.within] if seg.within else None
    if seg.method == "tracked_runs":
        return segment.tracked_runs(frames["valid"].to_numpy(bool), **p)
    if seg.method == "above_threshold":
        col = p.pop("column", "speed")
        return segment.above_threshold(frames[col].to_numpy(float), **p)
    if seg.method in ("zero_crossings", "speed_minima"):
        default = ("radial_velocity" if seg.method == "zero_crossings"
                   else "speed")
        col = p.pop("column", default)
        if col not in frames:
            raise ValueError(f"{seg.method} needs column {col!r}; set a "
                             "reference point for radial_velocity")
        fn = getattr(segment, seg.method)
        return fn(frames[col].to_numpy(float), within=within, **p)
    raise ValueError(f"unknown segmentation method {seg.method!r}")


def _base_columns(frames, iv, kp_name):
    """Core segment columns from an intervals table."""
    fr = frames["frame"].to_numpy()
    t = frames["time"].to_numpy(float)
    nvalid = np.r_[0, np.cumsum(frames["valid"].to_numpy(bool))]
    s, e = iv["start"].to_numpy(), iv["end"].to_numpy()
    return pd.DataFrame({
        "keypoint": kp_name,
        "start_frame": fr[s], "end_frame": fr[e],
        "start_time": t[s], "end_time": t[e], "duration": t[e] - t[s],
        "n_frames": e - s + 1, "n_valid": nvalid[e + 1] - nvalid[s],
    })


def _process_keypoint(frames, kps: KeypointSettings, context, next_id):
    """Segments and labels for one keypoint; returns (frames, tables)."""
    found, tables = {}, {}
    for kind, seg in kps.segments.items():
        if seg.method == "bouts":
            p = dict(seg.params)
            of = p.pop("of")
            src = tables[of]
            bout_of, b = segment.group_into_bouts(
                src["start_time"], src["end_time"], **p)
            iv = pd.DataFrame({
                "start": found[of]["start"].to_numpy()[b["first"]],
                "end": found[of]["end"].to_numpy()[b["last"]],
            })
        else:
            iv = _intervals(frames, seg, found)
        ids = next_id(kind, len(iv))
        table = _base_columns(frames, iv, kps.name)
        table.insert(0, f"{kind}_id", pd.array(ids, dtype="Int64"))
        if seg.method == "bouts":
            table[f"n_{of}s"] = b["n_segments"].to_numpy()
            span = (src["start_time"].to_numpy()[b["last"]]
                    - src["start_time"].to_numpy()[b["first"]])
            with np.errstate(invalid="ignore", divide="ignore"):
                table["rate_hz"] = np.where(
                    span > 0, (b["n_segments"].to_numpy() - 1) / span, np.nan)
            col = pd.array([pd.NA] * len(src), dtype="Int64")
            grouped = bout_of >= 0
            col[grouped] = ids[bout_of[grouped]]
            src[f"{kind}_id"] = col
        if seg.within:
            parent_ids = tables[seg.within][f"{seg.within}_id"].to_numpy()
            table[f"{seg.within}_id"] = pd.array(
                parent_ids[iv["within"].to_numpy()], dtype="Int64")
            table[f"number_in_{seg.within}"] = pd.array(
                table.groupby(f"{seg.within}_id").cumcount().to_numpy() + 1,
                dtype="Int64")
        if seg.method == "zero_crossings":
            table["direction"] = pd.array(iv["direction"], dtype="Int8")
        if seg.features:
            table = pd.concat(
                [table, feat.describe(frames, iv, seg.features, context)],
                axis=1)
        frames[f"{kind}_id"] = label_frames(len(frames), iv, ids)
        found[kind], tables[kind] = iv, table
    return frames, tables


def _id_counter():
    """``next_id(kind, n)``: the next ``n`` ids (1-based) for ``kind``."""
    counter = {}

    def next_id(kind, n):
        """Ids for ``n`` new segments of ``kind``."""
        start = counter.get(kind, 0)
        counter[kind] = start + n
        return np.arange(start + 1, start + n + 1, dtype=np.int64)

    return next_id


def _prepare_keypoint(masked, kps, fs_hz, landmarks):
    """One keypoint's smoothed frames (with radial velocity when it has a
    reference point) and the context for its features."""
    frames = (masked[masked["keypoint"] == kps.name]
              .sort_values("frame").reset_index(drop=True))
    if frames.empty:
        raise ValueError(f"keypoint {kps.name!r} is not in the pose")
    frames = signal.smooth(frames, fs_hz, **dataclasses.asdict(kps.smooth))
    origin = lm.origin(landmarks, kps.reference) if kps.reference else None
    if origin is not None:
        frames["radial_velocity"] = signal.radial_velocity(frames, origin)
    context = {"fs_hz": fs_hz, "origin": origin, "flip_x": kps.flip_x,
               "flip_y": kps.flip_y, "options": kps.feature_options}
    return frames, context


def process(pose: pd.DataFrame, settings: Settings, session=None) -> dict:
    """Process every keypoint in ``settings``; see the module docstring.

    Parameters
    ----------
    pose : pandas.DataFrame
        Pose table with ``time`` (``signal.add_time``).
    settings : Settings
    session : str, optional
        Added as a ``session`` column to every table.
    """
    if "time" not in pose:
        raise ValueError("pose needs a time column (signal.add_time)")
    thresholds = {kp.name: kp.min_likelihood for kp in settings.keypoints}
    names = list(dict.fromkeys(
        list(settings.landmarks)
        + [kp.reference for kp in settings.keypoints if kp.reference]))
    for n in names:
        thresholds.setdefault(n, settings.landmark_min_likelihood)
    masked = signal.mask(pose, thresholds)
    landmarks = lm.landmarks(masked, names)

    next_id = _id_counter()
    kin_parts, seg_parts, kp_counts = [], {}, {}
    for kps in settings.keypoints:
        frames, context = _prepare_keypoint(masked, kps, settings.fs_hz,
                                            landmarks)
        frames, tables = _process_keypoint(frames, kps, context, next_id)
        ids = [c for c in frames.columns if c.endswith("_id")]
        kin_parts.append(frames.reindex(columns=KINEMATICS_COLUMNS + ids))
        for kind, t in tables.items():
            seg_parts.setdefault(kind, []).append(t)
        kp_counts[kps.name] = {
            "n_frames": int(len(frames)),
            "n_valid": int(frames["valid"].sum()),
            "tracked_fraction": float(frames["valid"].mean()),
        }

    result = {"pose": pose, "kinematics": pd.concat(kin_parts,
                                                    ignore_index=True),
              "landmarks": landmarks}
    for kind, parts in seg_parts.items():
        result[f"{kind}s"] = pd.concat(parts, ignore_index=True)
    result["record"] = {
        "session": session,
        "settings": settings.to_dict(),
        "keypoints": kp_counts,
        "versions": versions(),
    }
    if session is not None:
        for name, t in result.items():
            if isinstance(t, pd.DataFrame) and "session" not in t:
                t.insert(0, "session", session)
    return result


def versions() -> dict:
    """Versions of this library and the numeric stack."""
    return {"aind_dynamic_foraging_behavior_video_analysis": __version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__}


def _json_default(o):
    """JSON for numpy scalars and arrays."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write(result: dict, out_dir) -> Path:
    """Save every table as ``<name>.parquet`` and the record as
    ``record.json`` (written last, so its presence means done)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in result.items():
        if isinstance(t, pd.DataFrame):
            t.to_parquet(out_dir / f"{name}.parquet", index=False)
    with open(out_dir / RECORD_FILE, "w") as f:
        json.dump(result.get("record", {}), f, indent=2,
                  default=_json_default)
    return out_dir


def read(out_dir, tables=None) -> dict:
    """Read what ``write`` saved: ``{name: DataFrame, 'record': dict}``.

    ``tables`` limits which tables are read (the per-frame ones are large).
    """
    out_dir = Path(out_dir)
    names = (tables if tables is not None
             else sorted(p.stem for p in out_dir.glob("*.parquet")))
    out = {n: pd.read_parquet(out_dir / f"{n}.parquet") for n in names}
    out["record"] = read_record(out_dir)
    return out


def read_record(out_dir) -> dict:
    """The saved record; raises ``FileNotFoundError`` if not done."""
    with open(Path(out_dir) / RECORD_FILE) as f:
        return json.load(f)


def is_done(out_dir) -> bool:
    """True once ``write`` has finished for ``out_dir``."""
    return (Path(out_dir) / RECORD_FILE).exists()
