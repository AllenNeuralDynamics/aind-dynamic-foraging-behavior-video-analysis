"""Measure and check the image quality of a behavior video MP4.

Pipeline, one function per step (:func:`video_quality` chains them):

1. :func:`sample_window`: the frames to sample, the task (first trial
   start to last trial end, :func:`video_alignment.task_frame_window`) or,
   failing that, the middle ``FALLBACK_FRACTION`` of the file. Recordings
   often run past the session, and an empty rig fails every check.
2. :func:`sample_keyframes`: up to ``N_SAMPLES`` keyframes spread evenly
   over the window, as a ``(n, h, w)`` uint8 luma array and a samples
   table (``frame_index``, ``video_time``). A seek lands on a keyframe and
   one decode returns it, so no other frame is decoded, and every sample is
   the same frame type (a keyframe's Laplacian variance is 35% higher than a
   B-frame's on these files).
3. :func:`measure`: adds every metric to the samples table, one row per
   keyframe (luma only; the cameras are monochrome IR).
4. :func:`run_checks`: applies ``CHECKS``, one row per check.
5. :func:`quality_action`: ``use``, or ``exclude: <check>`` for the first
   failed check. The module never removes data.

:func:`write_video_quality` saves the samples (parquet) and the checks
(JSON). The evidence for every value here is in
``VIDEO_QUALITY_QC_PLAN.md`` ("Findings", "Decisions: calibration").

Example::

    frames, samples, checks, note = video_quality(
        "behavior-videos/bottom_camera.mp4", "bottom_camera",
        "behavior/<subject>_<datetime>.json",
        "behavior-videos/bottom_camera.csv",
    )
    quality_action(checks)  # "use" or "exclude: <check>"
    write_video_quality(samples, checks, "results/", "bottom_camera", note)
"""

from __future__ import annotations

import json
import operator
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
from aind_video_utils import __version__ as aind_video_utils_version
from aind_video_utils import (
    get_video_range_info,
    luma_range,
    probe,
    read_mp4_frame_index,
)

from aind_dynamic_foraging_behavior_video_analysis import __version__
from aind_dynamic_foraging_behavior_video_analysis.video_alignment import (
    task_frame_window,
)

N_SAMPLES = 100
# Never sample frame 0 (embedded metadata) or this fraction at each end.
EDGE_FRACTION = 0.01
# Without a task window, sample this middle fraction of the file.
FALLBACK_FRACTION = 0.5
# FFmpeg options for URLs: give up on a stalled read after 60 s
# (microseconds) instead of hanging, and reconnect on drops.
HTTP_OPTIONS = {
    "rw_timeout": "60000000",
    "reconnect": "1",
    "reconnect_on_network_error": "1",
}
SAMPLES_FILE = "video_quality_{camera}.parquet"
RECORD_FILE = "video_quality_{camera}.json"

# (metric, over, op, value, cameras). A sample passes when ``metric op
# value``. ``over="samples"`` fails on two or more consecutive failing
# samples (one alone is a paw in front of the lens); "median" or "p5" tests
# that statistic of the samples. ``cameras`` is "all" or a view, "bottom"
# or "side". Evidence: "Findings: full survey" and "Decisions: calibration"
# (301 FIP sessions, 602 cameras).
CHECKS = [
    ("sharpness_dev", "samples", "<=", 0.45, "all"),  # was sharpness_stable
    ("mean_dev", "samples", "<=", 0.15, "all"),  # was brightness_stable
    ("similarity", "samples", ">=", 0.7, "all"),  # was scene_stable
    ("similarity", "p5", "<", 0.998, "all"),  # was scene_moves
    # Too dark or too bright. Survey medians 54.7-102.4 (bottom), 71.4-104.8
    # (side): no camera outside these limits (2026-10-01).
    ("mean", "median", ">=", 50, "all"),
    ("mean", "median", "<=", 150, "all"),
    ("pct_clipped_high", "median", "<=", 3.75, "side"),  # was exposure_ok
]
OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt}
STATISTICS = {"median": np.median, "p5": lambda v: np.percentile(v, 5)}

# 8-bit pixel formats whose first plane is luma, one byte per pixel.
LUMA_FORMATS = {"yuv420p", "yuvj420p", "yuv422p", "yuv444p", "nv12"}


# --- Per-frame metrics ---------------------------------------------------


def downsample2(luma):
    """Return ``luma`` 2x downsampled by 2x2 mean (odd edges dropped)."""
    y = np.asarray(luma, dtype=np.float32)
    y = y[: y.shape[0] // 2 * 2, : y.shape[1] // 2 * 2]
    return (y[0::2, 0::2] + y[1::2, 0::2] + y[0::2, 1::2] + y[1::2, 1::2]) / 4


def laplacian_variance(image):
    """Variance of the 4-neighbour Laplacian over the image interior."""
    d = np.asarray(image, dtype=np.float32)
    lap = (
        4 * d[1:-1, 1:-1]
        - d[:-2, 1:-1]
        - d[2:, 1:-1]
        - d[1:-1, :-2]
        - d[1:-1, 2:]
    )
    return float(lap.var())


def noise_sigma(luma):
    """Gaussian noise sigma (Immerkaer 1996): the difference of two
    Laplacians cancels most image structure; scale its mean response."""
    y = np.asarray(luma, dtype=np.float64)
    r = (
        y[:-2, :-2]
        - 2 * y[:-2, 1:-1]
        + y[:-2, 2:]
        - 2 * y[1:-1, :-2]
        + 4 * y[1:-1, 1:-1]
        - 2 * y[1:-1, 2:]
        + y[2:, :-2]
        - 2 * y[2:, 1:-1]
        + y[2:, 2:]
    )
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def similarity(reference, image):
    """Pearson correlation of two images; 0 if either is flat."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(image, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _parabolic_offset(left, centre, right):
    """Sub-sample offset of a peak from three samples around it."""
    denom = left - 2 * centre + right
    return 0.0 if denom == 0 else 0.5 * (left - right) / denom


def phase_shift(reference, image):
    """``(dx, dy)`` translation of ``image`` from ``reference``.

    Phase correlation with a Hann window and parabolic sub-pixel peak.
    Positive ``dx`` means the content moved right.
    """
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    window = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    cross = np.fft.fft2((img - img.mean()) * window) * np.conj(
        np.fft.fft2((ref - ref.mean()) * window)
    )
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-12)).real
    h, w = corr.shape
    y, x = np.unravel_index(int(corr.argmax()), corr.shape)
    dy = y + _parabolic_offset(
        corr[(y - 1) % h, x], corr[y, x], corr[(y + 1) % h, x]
    )
    dx = x + _parabolic_offset(
        corr[y, (x - 1) % w], corr[y, x], corr[y, (x + 1) % w]
    )
    # Peaks past the midpoint are negative shifts (the FFT wraps around).
    return float(dx - w if dx > w / 2 else dx), float(
        dy - h if dy > h / 2 else dy
    )


def reference_frame(frames):
    """Pixel-wise median of every sampled frame: the session's typical
    view. It ignores a moving mouse, and anything present in less than half
    the session (a dark start, an empty rig at the end), so those samples
    read as dissimilar rather than the rest of the session."""
    return np.round(np.median(frames, axis=0)).astype(np.uint8)


# --- Pipeline ------------------------------------------------------------


def sample_window(behavior_json, video_csv, trigger_log=None):
    """``(window, note)``: the task frames and ``"task"``, or ``None`` (the
    middle ``FALLBACK_FRACTION`` of the file) and ``"middle 50%: <reason>"``
    when an input is missing or :func:`task_frame_window` raises."""
    fallback = f"middle {FALLBACK_FRACTION:.0%}"
    if behavior_json is None or video_csv is None:
        return None, f"{fallback}: no behavior JSON or video CSV"
    try:
        return task_frame_window(behavior_json, video_csv, trigger_log), "task"
    except (ValueError, OSError) as e:
        return None, f"{fallback}: {e}"


def luma_plane(frame):
    """The frame's luma plane as coded.

    Not ``frame.to_ndarray(format="gray")``: swscale stretches TV-range
    (16-235) luma to 0-255, which puts every metric on the wrong scale.
    """
    if frame.format.name not in LUMA_FORMATS:
        raise ValueError(f"Unsupported pixel format {frame.format.name!r}")
    plane = frame.planes[0]
    rows = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane.line_size)
    return rows[: frame.height, : frame.width].copy()


def sample_keyframes(path, window=None):
    """Decode up to ``N_SAMPLES`` keyframes spread evenly over ``window``.

    Keyframes come from the MP4 index, never frame 0 nor the
    ``EDGE_FRACTION`` at each end; ``window=None`` means the middle
    ``FALLBACK_FRACTION`` of the file. Each is sought in one open container
    and must decode as that keyframe at the expected PTS.

    Parameters
    ----------
    path : str or pathlib.Path
        Local path or URL of an MP4.
    window : (int, int), optional
        ``[start, end)`` frame indices (= video CSV rows).

    Returns
    -------
    frames : numpy.ndarray
        ``(n, h, w)`` uint8 coded luma.
    samples : pandas.DataFrame
        ``frame_index`` (presentation order) and ``video_time`` (s).
    color_range : str
        ``"tv"``, ``"pc"`` or ``"unknown"``.

    Raises
    ------
    ValueError
        Not an MP4, an edit list unsafe for frame addressing, no keyframe
        in the window, or a seek that lands elsewhere.
    """
    info = probe(path)
    if "mp4" not in info.get("format", {}).get("format_name", "").split(","):
        raise ValueError(f"Not an MP4: {path}")
    color_range, _ = get_video_range_info(info)
    index = read_mp4_frame_index(path)
    if not index.is_frame_addressing_safe():
        raise ValueError(f"Edit list is not frame-addressing-safe: {path}")
    n = index.n_samples
    display = np.empty(n, dtype=np.int64)
    display[index.display_order] = np.arange(n)
    keys = np.sort(display[index.keyframe_decode_indices])
    if window is None:
        margin = int(n * (1 - FALLBACK_FRACTION) / 2)
        window = (margin, n - margin)
    edge = int(np.ceil(n * EDGE_FRACTION))
    keys = keys[(keys >= max(1, edge, window[0]))]
    keys = keys[keys < min(n - edge, window[1])]
    if keys.size == 0:
        raise ValueError(f"No keyframe in frames {window}: {path}")
    if keys.size > N_SAMPLES:
        targets = np.linspace(keys[0], keys[-1], N_SAMPLES)
        keys = np.unique(keys[np.abs(keys[:, None] - targets).argmin(axis=0)])
    # FFmpeg's MP4 demuxer subtracts the edit list's media_time from PTS.
    media_time = index.edits[0].media_time if index.edits else 0
    pts = index.pts[index.display_order] - media_time
    is_url = str(path).startswith(("http://", "https://"))
    frames = []
    with av.open(str(path), options=HTTP_OPTIONS if is_url else {}) as f:
        stream = f.streams.video[0]
        to_stream = Fraction(1, index.media_timescale) / stream.time_base
        for key in keys:
            target = int(round(pts[key] * to_stream))
            f.seek(target, stream=stream, backward=True)
            frame = next(f.decode(stream))
            if frame.pts != target or not frame.key_frame:
                raise ValueError(
                    f"Seek to frame {key} (pts {target}) returned pts "
                    f"{frame.pts}, key_frame={frame.key_frame}: {path}"
                )
            frames.append(luma_plane(frame))
    samples = pd.DataFrame(
        {"frame_index": keys, "video_time": pts[keys] / index.media_timescale}
    )
    return np.stack(frames), samples, color_range


def measure(frames, samples, color_range):
    """Return ``samples`` with every metric added, one row per frame.

    Intensity: ``mean``, ``std``, ``p1``, ``p99``, ``dynamic_range`` (p99 -
    p1), ``contrast_rms`` (std / mean), ``entropy_bits``,
    ``pct_clipped_low`` and ``pct_clipped_high`` (% of pixels at or beyond
    the tagged floor / ceiling: TV range clips at 235, not 255),
    ``histogram`` (luma counts). Sharpness and noise: ``sharpness``
    (Laplacian variance, 2x downsampled), ``noise_sigma``. Against the
    reference frame: ``similarity`` (correlation, 2x downsampled),
    ``shift_x``, ``shift_y``, ``shift`` (phase correlation; reported only,
    since a lick-spout move reads as a camera shift). Stability:
    ``sharpness_dev`` and ``mean_dev``, ``|x / median - 1|``. Also
    ``color_range``, for the report's display scale.
    """
    floor, ceiling = luma_range(8, color_range == "pc")
    reference = reference_frame(frames)
    reference_small = downsample2(reference)
    rows = []
    for luma in frames:
        small = downsample2(luma)
        counts = np.bincount(luma.ravel(), minlength=256)
        p = counts[counts > 0] / luma.size
        p1, p99 = np.percentile(luma, [1, 99])
        mean, std = luma.mean(), luma.std()
        dx, dy = phase_shift(reference, luma)
        rows.append(
            {
                "mean": mean,
                "std": std,
                "p1": p1,
                "p99": p99,
                "dynamic_range": p99 - p1,
                "contrast_rms": std / mean if mean else 0.0,
                "entropy_bits": -(p * np.log2(p)).sum(),
                "pct_clipped_low": 100 * (luma <= floor).mean(),
                "pct_clipped_high": 100 * (luma >= ceiling).mean(),
                "sharpness": laplacian_variance(small),
                "noise_sigma": noise_sigma(luma),
                "similarity": similarity(reference_small, small),
                "shift_x": dx,
                "shift_y": dy,
                "shift": np.hypot(dx, dy),
                "histogram": counts,
            }
        )
    out = pd.concat(
        [samples.reset_index(drop=True), pd.DataFrame(rows)], axis=1
    )
    for metric in ("sharpness", "mean"):
        out[f"{metric}_dev"] = (out[metric] / out[metric].median() - 1).abs()
    out["color_range"] = color_range
    return out


def camera_view(camera):
    """``"bottom"`` or ``"side"`` from a camera name in either folder layout
    (``bottom_camera``, ``BottomCamera``, ``SideCameraRight``); else None."""
    name = str(camera).lower()
    return next((v for v in ("bottom", "side") if name.startswith(v)), None)


def run_checks(samples, camera):
    """Apply every check in ``CHECKS`` that covers ``camera``.

    Returns
    -------
    pandas.DataFrame
        One row per check: ``check`` (name, e.g. ``similarity >= 0.7``),
        ``metric``, ``over``, ``op``, ``value``, ``observed`` (the
        statistic, or for ``over="samples"`` the number of failing samples
        in runs of two or more), ``passed``, ``samples`` (every failing
        sample number, isolated ones included).
    """
    rows = []
    for metric, over, op, value, cameras in CHECKS:
        if cameras not in ("all", camera_view(camera)):
            continue
        values = samples[metric].to_numpy(dtype=np.float64)
        bad = []
        if over == "samples":
            bad = np.flatnonzero(~OPS[op](values, value))
            in_runs = np.isin(bad + 1, bad) | np.isin(bad - 1, bad)
            observed, passed = int(in_runs.sum()), not in_runs.any()
            name = f"{metric} {op} {value:g}"
        else:
            observed = float(STATISTICS[over](values))
            passed = bool(OPS[op](observed, value))
            name = f"{metric} {over} {op} {value:g}"
        rows.append(
            {
                "check": name,
                "metric": metric,
                "over": over,
                "op": op,
                "value": value,
                "observed": observed,
                "passed": passed,
                "samples": [int(i) for i in bad],
            }
        )
    return pd.DataFrame(rows)


def quality_action(checks):
    """``"exclude: <check>"`` for the first failed check, otherwise
    ``"use"``."""
    failed = checks.loc[~checks["passed"].astype(bool), "check"]
    return f"exclude: {failed.iloc[0]}" if len(failed) else "use"


def write_video_quality(samples, checks, out_dir, camera, note):
    """Write ``SAMPLES_FILE`` (the samples, with histograms) and
    ``RECORD_FILE`` (camera, window note, versions, checks, action) into
    ``out_dir``; return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / SAMPLES_FILE.format(camera=camera)
    record_path = out / RECORD_FILE.format(camera=camera)
    samples.to_parquet(samples_path, index=False)
    record = {
        "camera": camera,
        "window": note,
        "versions": {
            "aind_dynamic_foraging_behavior_video_analysis": __version__,
            "aind_video_utils": aind_video_utils_version,
        },
        "checks": json.loads(checks.to_json(orient="records")),
        "action": quality_action(checks),
    }
    record_path.write_text(json.dumps(record, indent=2))
    return record_path, samples_path


def video_quality(
    path, camera, behavior_json=None, video_csv=None, trigger_log=None
):
    """Window, sample, measure and check one camera's MP4.

    Returns
    -------
    (frames, samples, checks, note)
        See :func:`sample_keyframes`, :func:`measure`, :func:`run_checks`
        and :func:`sample_window`.
    """
    window, note = sample_window(behavior_json, video_csv, trigger_log)
    frames, samples, color_range = sample_keyframes(path, window)
    samples = measure(frames, samples, color_range)
    return frames, samples, run_checks(samples, camera), note
