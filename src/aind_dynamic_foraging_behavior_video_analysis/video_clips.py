"""Cut frame-exact clips from a behavior video, and frames for labeling.

Everything here is keyed on *(source video, frame index)*. A clip is
frames ``[start_frame, start_frame + n_frames)`` of an MP4; a labeled image
is frame ``k`` of a clip, so source frame ``start_frame + k``. Row ``i`` of
the video CSV is frame ``i`` of the MP4, also when frames were dropped, so a
source frame is also a CSV row and a pose-prediction row. The module takes
no times and runs no QC:

- which cameras to use is the screen's (``video_screen``);
- event times -> frame ranges is ``video_alignment.event_frame_ranges``,
  with the corrected ``harp_time`` of ``video_timing_qc``;
- behavior time is joined on afterwards, from the same ``harp_time``.

Clips: :func:`cut_clip` cuts one; :func:`cut_clips` cuts a table of ranges,
names each ``<prefix>_f<start_frame:07d>.mp4``, writes a JSON sidecar
(:func:`read_clip_info`) and resumes. Seeks use the MP4's own sample
timestamps (``aind_video_utils.read_mp4_frame_index``), never a frame rate,
and the MP4 may be an HTTPS URL. A clip that spans dropped frames does not
play in real time: the missing frames are absent and the clip keeps even
timestamps.

Labeling (DeepLabCut, then Lightning Pose): :func:`select_frames` writes
PNGs named as DLC's own extraction names them; :func:`add_context_frames`
adds the neighbours Lightning Pose's context models read, after labeling;
:func:`labeled_frames_table` maps every labeled image back to its source
frame from the names alone.

Example::

    ranges = va.event_frame_ranges(go_cues, timing["harp_time"], 1.0, 1.0)
    clips = cut_clips(mp4, ranges, "clips/", f"{session}_{camera}")
    for clip in clips["clip_path"].dropna().unique():  # skipped: NaN
        select_frames(clip, 10, "dlc_project/labeled-data/")
    # ... label in DLC ...
    add_context_frames("dlc_project/labeled-data/", "clips/")
    labels = labeled_frames_table("dlc_project/labeled-data/")
    labels["behavior_time"] = timing["harp_time"].to_numpy()[
        labels["source_frame"]
    ]

Needs ``ffmpeg`` on ``PATH`` and the ``video-clips`` extra. Design and
evidence: ``VIDEO_CLIPS_PLAN.md``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from aind_video_utils import __version__ as aind_video_utils_version
from aind_video_utils import read_mp4_frame_index
from aind_video_utils.utils import http_input_flags

from aind_dynamic_foraging_behavior_video_analysis import __version__

# Sidecar next to each clip, written by cut_clips.
SIDECAR_FILE = "{stem}.json"
# Labels written by DeepLabCut (one per labeled-data folder) and Lightning
# Pose (one per project); searched for under labeled_data_dir.
LABELS_GLOB = "CollectedData*.csv"

# Clip encoder: near-lossless H.264, playable everywhere.
ENCODE_ARGS = [
    "-c:v",
    "libx264",
    "-crf",
    "18",
    "-preset",
    "fast",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
]
# Size of the gray thumbnails k-means clusters (as DLC's resizewidth=30).
KMEANS_SIZE = (40, 30)

_STEM = re.compile(r"^(?P<prefix>.+)_f(?P<start>\d+)$")
_IMAGE = re.compile(r"^(?P<head>.*?)(?P<digits>\d+)\.png$")


def _ffmpeg():
    """Return the start of every ffmpeg command."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]


def _cut_command(mp4, seek_seconds, n_frames, out_path):
    """Return the ffmpeg command for one clip (see :func:`cut_clip`)."""
    return [
        *_ffmpeg(),
        "-accurate_seek",
        # 9 decimals resolve the media tick; -accurate_seek then decodes
        # forward to exactly this presentation time.
        "-ss",
        f"{seek_seconds:.9f}",
        *http_input_flags(mp4),
        "-i",
        str(mp4),
        "-map",
        "0:v:0",
        "-frames:v",
        str(int(n_frames)),
        # Even timestamps at the source's nominal rate, for playback only:
        # a variable-rate source would otherwise give repeated timestamps.
        "-vf",
        "setpts=N/FRAME_RATE/TB",
        # Neither duplicate nor drop frames to fit a frame rate.
        "-fps_mode",
        "passthrough",
        *ENCODE_ARGS,
        str(out_path),
    ]


def _png_command(clip, clip_frame, out_path):
    """Return the ffmpeg command writing frame ``clip_frame`` as a PNG."""
    return [
        *_ffmpeg(),
        "-i",
        str(clip),
        "-vf",
        f"select=eq(n\\,{int(clip_frame)})",
        "-frames:v",
        "1",
        "-fps_mode",
        "passthrough",
        str(out_path),
    ]


def _gray_command(clip):
    """Return the ffmpeg command decoding a clip to gray thumbnails."""
    width, height = KMEANS_SIZE
    return [
        *_ffmpeg(),
        "-i",
        str(clip),
        "-vf",
        f"scale={width}:{height},format=gray",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _run(command, stdout=None):
    """Run an ffmpeg command; return its stdout.

    Raises
    ------
    RuntimeError
        If ffmpeg fails, with its error output.
    """
    result = subprocess.run(command, stdout=stdout, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed: {message}")
    return result.stdout


def _temp_path(path):
    """Return a sibling path to write before renaming onto ``path``."""
    path = Path(path)
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def _png_name(clip_frame, n_frames):
    """Return DeepLabCut's file name for frame ``clip_frame`` of a clip.

    DLC (``frame_extraction.py``): ``"img" + str(k).zfill(indexlength)``,
    ``indexlength = ceil(log10(n_frames))``.
    """
    width = int(math.ceil(math.log10(n_frames))) if n_frames > 1 else 0
    return "img" + str(int(clip_frame)).zfill(width) + ".png"


def _parse_stem(stem):
    """Return ``(prefix, start_frame)`` from a clip stem.

    Raises
    ------
    ValueError
        If the stem is not ``<prefix>_f<start_frame>``.
    """
    match = _STEM.match(stem)
    if match is None:
        raise ValueError(f"Not a clip name (<prefix>_f<frame>): {stem!r}")
    return match["prefix"], int(match["start"])


def _clip_frame_count(clip):
    """Return the number of frames in a clip, from its MP4 index."""
    return read_mp4_frame_index(clip).n_samples


def cut_clip(mp4, start_frame, n_frames, out_path, index=None):
    """Cut frames ``[start_frame, start_frame + n_frames)`` into a new MP4.

    The seek is the source frame's own presentation time, read from the
    MP4's sample tables (``index.presentation_seconds``), so it lands on
    that frame without assuming a frame rate. ``-frames:v`` counts frames
    exactly and ``-fps_mode passthrough`` neither duplicates nor drops any.
    The clip is re-encoded (H.264) and written atomically; no sidecar.

    Parameters
    ----------
    mp4 : str or pathlib.Path
        Source MP4, a local path or an http(s) URL (only the bytes needed
        are read).
    start_frame : int
        First source frame (0-based, presentation order = CSV row).
    n_frames : int
        Number of frames.
    out_path : str or pathlib.Path
        Clip to write; its folder must exist.
    index : aind_video_utils.Mp4FrameIndex, optional
        The source's frame index, when cutting many clips from one MP4.

    Returns
    -------
    pathlib.Path
        ``out_path``.

    Raises
    ------
    ValueError
        If the range is empty or past the end of the video, the MP4's edit
        list does not allow addressing frames by index, or the container
        timestamps are not increasing at ``start_frame``.
    RuntimeError
        If ffmpeg fails.
    """
    if index is None:
        index = read_mp4_frame_index(mp4)
    start_frame, n_frames = int(start_frame), int(n_frames)
    if n_frames < 1 or start_frame < 0:
        raise ValueError(
            f"Empty or negative range: start {start_frame}, {n_frames} frames"
        )
    if start_frame + n_frames > index.n_samples:
        raise ValueError(
            f"Frames [{start_frame}, {start_frame + n_frames}) run past the "
            f"end of the video ({index.n_samples} frames)"
        )
    if not index.is_frame_addressing_safe():
        raise ValueError(
            f"{mp4}: edit list is not safe for addressing frames by index"
        )
    seek = index.presentation_seconds(start_frame)
    out_path = Path(out_path)
    temp = _temp_path(out_path)
    try:
        _run(_cut_command(mp4, seek, n_frames, temp))
        os.replace(temp, out_path)
    finally:
        temp.unlink(missing_ok=True)
    return out_path


def read_clip_info(clip):
    """Return the sidecar of a clip written by :func:`cut_clips`.

    Parameters
    ----------
    clip : str or pathlib.Path
        The clip (``.mp4``) or its sidecar (``.json``).

    Returns
    -------
    dict
        ``source_video`` (as given to :func:`cut_clips`), ``start_frame``,
        ``n_frames``, ``versions``.
    """
    clip = Path(clip)
    sidecar = clip.with_name(SIDECAR_FILE.format(stem=clip.stem))
    with open(sidecar) as f:
        return json.load(f)


def _write_sidecar(clip, info):
    """Write a clip's sidecar atomically."""
    sidecar = clip.with_name(SIDECAR_FILE.format(stem=clip.stem))
    temp = _temp_path(sidecar)
    temp.write_text(json.dumps(info, indent=2) + "\n")
    os.replace(temp, sidecar)


def _clip_exists(clip, source_video, n_frames):
    """Whether a finished clip of the same frames is already there."""
    try:
        info = read_clip_info(clip)
    except (OSError, ValueError):
        return False
    return (
        clip.exists()
        and info.get("source_video") == source_video
        and info.get("n_frames") == n_frames
    )


def cut_clips(mp4, ranges, out_dir, prefix):
    """Cut one clip per row of ``ranges``, with sidecars, resuming.

    Each clip is ``<out_dir>/<prefix>_f<start_frame:07d>.mp4``, so re-running
    never renumbers or overwrites another clip, and rows with the same
    window give one clip. Its sidecar (:data:`SIDECAR_FILE`) is written
    last, after ffmpeg succeeds. A clip whose sidecar names the same
    ``source_video`` and ``n_frames`` is not cut again.

    Parameters
    ----------
    mp4 : str or pathlib.Path
        Source MP4, a local path or an http(s) URL.
    ranges : pandas.DataFrame
        ``start_frame`` and ``n_frames`` per row, e.g. from
        ``video_alignment.event_frame_ranges``. Rows with ``in_video``
        False are skipped.
    out_dir : str or pathlib.Path
        Folder for clips and sidecars (created if needed).
    prefix : str
        Start of every clip name, e.g. ``f"{session}_{camera}"``.

    Returns
    -------
    pandas.DataFrame
        ``ranges`` plus ``clip_path`` (NaN if skipped) and ``status``:
        ``cut``, ``exists`` or ``skipped: <why>``. A row that cannot be cut
        is skipped; the others still are.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = read_mp4_frame_index(mp4)
    source_video = str(mp4)
    versions = {
        "aind_dynamic_foraging_behavior_video_analysis": __version__,
        "aind_video_utils": aind_video_utils_version,
    }
    clips, statuses = [], []
    for row in ranges.itertuples(index=False):
        if not getattr(row, "in_video", True):
            clips.append(None)
            statuses.append("skipped: window runs past the video")
            continue
        start, n = int(row.start_frame), int(row.n_frames)
        clip = out_dir / f"{prefix}_f{start:07d}.mp4"
        if _clip_exists(clip, source_video, n):
            clips.append(str(clip))
            statuses.append("exists")
            continue
        try:
            cut_clip(mp4, start, n, clip, index=index)
        except (ValueError, RuntimeError) as error:
            clips.append(None)
            statuses.append(f"skipped: {error}")
            continue
        _write_sidecar(
            clip,
            {
                "source_video": source_video,
                "start_frame": start,
                "n_frames": n,
                "versions": versions,
            },
        )
        clips.append(str(clip))
        statuses.append("cut")
    out = ranges.copy()
    out["clip_path"] = clips
    out["status"] = statuses
    return out


def _write_png(clip, clip_frame, out_path):
    """Write frame ``clip_frame`` of ``clip`` to ``out_path`` atomically."""
    temp = _temp_path(out_path)
    try:
        _run(_png_command(clip, clip_frame, temp))
        if not temp.exists():
            raise RuntimeError(f"{clip}: no frame {clip_frame}")
        os.replace(temp, out_path)
    finally:
        temp.unlink(missing_ok=True)


def _kmeans_pick(features, num_frames, seed):
    """Return, per k-means cluster, the row nearest its centroid (sorted)."""
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=num_frames, random_state=seed, n_init=3
    ).fit(features)
    picks = []
    for cluster, centre in enumerate(model.cluster_centers_):
        members = np.flatnonzero(model.labels_ == cluster)
        if len(members):
            distance = np.linalg.norm(features[members] - centre, axis=1)
            picks.append(int(members[np.argmin(distance)]))
    return np.unique(picks)


def _gray_thumbnails(clip):
    """Decode every frame of a clip to a flat gray thumbnail."""
    width, height = KMEANS_SIZE
    raw = _run(_gray_command(clip), stdout=subprocess.PIPE)
    frames = np.frombuffer(raw, dtype=np.uint8)
    return frames.reshape(-1, width * height).astype(np.float32)


def select_frames(
    clip,
    num_frames,
    labeled_data_dir,
    algorithm="uniform",
    seed=0,
    margin=2,
):
    """Write frames of a clip as PNGs for labeling in DeepLabCut.

    PNGs go to ``<labeled_data_dir>/<clip stem>/`` (DLC's layout when the
    clip is one of the project's videos) and are named as DLC's own frame
    extraction names them, ``img`` + the clip frame zero-padded to
    ``ceil(log10(n_frames))`` digits, so extracting the same clip in DLC
    gives the same files. The first and last ``margin`` frames are never
    picked, so each labeled frame has the neighbours Lightning Pose's
    context models read (:func:`add_context_frames`).

    Parameters
    ----------
    clip : str or pathlib.Path
        A clip from :func:`cut_clips`.
    num_frames : int
        Frames to pick (fewer if the clip is too short).
    labeled_data_dir : str or pathlib.Path
        The DLC project's ``labeled-data`` folder.
    algorithm : {"uniform", "random", "kmeans"}
        Evenly spaced; uniformly random (``seed``); or k-means on gray
        thumbnails of every frame, one frame per cluster, the one nearest
        its centroid (``seed``).
    seed : int
        Seed for ``random`` and ``kmeans``.
    margin : int
        Frames at each end never picked.

    Returns
    -------
    numpy.ndarray
        The clip frames written (or already there), increasing.

    Raises
    ------
    ValueError
        For an unknown algorithm.
    """
    clip = Path(clip)
    n_frames = _clip_frame_count(clip)
    candidates = np.arange(margin, n_frames - margin)
    num_frames = min(int(num_frames), len(candidates))
    if algorithm == "uniform":
        picks = candidates[
            np.round(np.linspace(0, len(candidates) - 1, num_frames)).astype(
                int
            )
        ]
    elif algorithm == "random":
        rng = np.random.default_rng(seed)
        picks = rng.choice(candidates, size=num_frames, replace=False)
    elif algorithm == "kmeans":
        if num_frames:
            features = _gray_thumbnails(clip)[candidates]
            picks = candidates[_kmeans_pick(features, num_frames, seed)]
        else:
            picks = candidates[:0]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
    picks = np.unique(picks)
    folder = Path(labeled_data_dir) / clip.stem
    folder.mkdir(parents=True, exist_ok=True)
    for k in picks:
        png = folder / _png_name(k, n_frames)
        if not png.exists():
            _write_png(clip, k, png)
    return picks


def _labeled_images(labeled_data_dir):
    """Return the images in every labels CSV under ``labeled_data_dir``.

    Works on both CSV layouts: DLC 2.3+ indexes rows by three columns
    (``labeled-data``, clip, image), older DLC and Lightning Pose by one
    path. Header rows (scorer, bodyparts, coords) hold no ``.png``.
    """
    images = []
    for path in sorted(Path(labeled_data_dir).rglob(LABELS_GLOB)):
        with open(path, newline="") as f:
            for row in csv.reader(f):
                for i, cell in enumerate(row):
                    if cell.endswith(".png"):
                        parts = [c for c in row[: i + 1] if c]
                        images.append("/".join(parts).replace("\\", "/"))
                        break
    return list(dict.fromkeys(images))


def labeled_frames_table(labeled_data_dir):
    """Map every labeled image to its clip and source frame.

    Reads every ``CollectedData*.csv`` under ``labeled_data_dir`` (DLC's
    per-folder files or Lightning Pose's) and parses names only: the folder
    is the clip stem (``<prefix>_f<start_frame>``) and the digits of the
    PNG name are the clip frame. No clip or sidecar is read.

    Parameters
    ----------
    labeled_data_dir : str or pathlib.Path
        A ``labeled-data`` folder, or any folder above the labels CSVs.

    Returns
    -------
    pandas.DataFrame
        One row per labeled image: ``image`` (as in the CSV), ``clip_stem``,
        ``prefix``, ``clip_frame``, ``start_frame``, ``source_frame``
        (``start_frame + clip_frame``; a row of the camera's CSV and of its
        pose predictions).

    Raises
    ------
    ValueError
        If a folder or image name does not follow the conventions.
    """
    rows = []
    for image in _labeled_images(labeled_data_dir):
        parts = image.split("/")
        clip, name = parts[-2], parts[-1]
        prefix, start = _parse_stem(clip)
        match = _IMAGE.match(name)
        if match is None:
            raise ValueError(f"No frame number in image name: {image!r}")
        clip_frame = int(match["digits"])
        rows.append(
            {
                "image": image,
                "clip_stem": clip,
                "prefix": prefix,
                "clip_frame": clip_frame,
                "start_frame": start,
                "source_frame": start + clip_frame,
            }
        )
    columns = [
        "image",
        "clip_stem",
        "prefix",
        "clip_frame",
        "start_frame",
        "source_frame",
    ]
    return pd.DataFrame(rows, columns=columns)


def add_context_frames(labeled_data_dir, clips_dir, offsets=(-2, -1, 1, 2)):
    """Write the neighbours of every labeled frame, for Lightning Pose.

    Lightning Pose's context models read frames ``t-2 ... t+2`` as PNGs in
    the same folder as each labeled frame ``t``, named by the same digits
    and width, and silently use the centre frame for any that are missing.
    Run this after labeling (DLC's labeling GUI shows every PNG in a
    folder), on the DLC project's or LP's ``labeled-data`` folder, as long
    as the folder names are the clip stems. Existing files are kept; the
    labels CSVs are never touched.

    Across a dropped frame the neighbours are the adjacent *saved* frames,
    so further apart in time than elsewhere; not corrected.

    Parameters
    ----------
    labeled_data_dir : str or pathlib.Path
        ``labeled-data`` folder holding the labels CSVs and PNGs.
    clips_dir : str or pathlib.Path
        Folder with the clips, ``<clip stem>.mp4``.
    offsets : sequence of int
        Neighbours to write, relative to each labeled frame.

    Returns
    -------
    pandas.DataFrame
        One row per PNG written: ``clip_stem``, ``labeled_frame``,
        ``clip_frame``, ``path``.
    """
    labeled_data_dir, clips_dir = Path(labeled_data_dir), Path(clips_dir)
    written = []
    n_frames = {}
    for image in _labeled_images(labeled_data_dir):
        parts = image.split("/")
        clip, name = parts[-2], parts[-1]
        match = _IMAGE.match(name)
        if match is None:
            raise ValueError(f"No frame number in image name: {image!r}")
        labeled, width = int(match["digits"]), len(match["digits"])
        clip_path = clips_dir / f"{clip}.mp4"
        if clip not in n_frames:
            n_frames[clip] = _clip_frame_count(clip_path)
        for offset in offsets:
            k = labeled + int(offset)
            if not 0 <= k < n_frames[clip]:
                continue
            png = (
                labeled_data_dir
                / clip
                / f"{match['head']}{str(k).zfill(width)}.png"
            )
            if png.exists():
                continue
            _write_png(clip_path, k, png)
            written.append(
                {
                    "clip_stem": clip,
                    "labeled_frame": labeled,
                    "clip_frame": k,
                    "path": str(png),
                }
            )
    return pd.DataFrame(
        written, columns=["clip_stem", "labeled_frame", "clip_frame", "path"]
    )
