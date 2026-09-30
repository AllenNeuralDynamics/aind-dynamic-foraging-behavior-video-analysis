"""Measure and check the image quality of a behavior video MP4.

Quality is measured on keyframes sampled evenly across the file: a seek
lands on a keyframe and one decode returns it, so no other frame is ever
decoded. Sampling only keyframes also keeps every sample the same frame
type; on the AIND MP4s a keyframe's Laplacian variance is 35% higher than a
B-frame's at full resolution (see ``VIDEO_QUALITY_QC_PLAN.md``).

Metrics per sampled keyframe (luma only; the cameras are monochrome):

- exposure statistics from ``aind_video_utils.qc_batch.compute_frame_stats``
  (mean, std, percentiles, clipping, entropy);
- ``contrast_rms`` (std / mean) and ``dynamic_range`` (p99 - p1);
- ``pct_clipped_low``, ``pct_clipped_high``: % of pixels at or beyond the
  tagged floor / ceiling. A limited-range (TV) file saturates at the
  ceiling (235), not at 255, so the imported ``pct_at_max`` misses it;
- ``sharpness``: variance of the Laplacian of the 2x downsampled frame;
- ``noise_sigma``: Immerkaer's fast noise estimate;
- ``similarity``: correlation with the reference frame (the pixel-wise
  median of the first samples);
- ``shift_x``, ``shift_y``, ``shift``: phase-correlation shift from the
  reference, in pixels. Reported only: on the bottom camera the motorized
  lick spouts dominate it, so a spout move reads as a 14 px shift while
  the camera stays put (``VIDEO_QUALITY_QC_PLAN.md``, "Findings").

Checks, one question each (:func:`check_video_quality` runs them all):

- ``sharpness_stable``: is sharpness within ``SHARPNESS_TOLERANCE`` of the
  session median throughout?
- ``brightness_stable``: is mean luma within ``BRIGHTNESS_TOLERANCE`` of
  the session median throughout?
- ``scene_stable``: does every sample correlate with the reference by at
  least ``MIN_SIMILARITY``?
- ``sharp_enough``, ``exposure_ok``, ``contrast_ok``: are the session
  medians within the level thresholds for this camera? Skipped when no
  threshold exists (see ``LEVEL_THRESHOLDS``).

A single sample out of tolerance is counted but does not fail a stability
check (a paw in front of the lens); two consecutive samples do.

The recording often runs past the session (on
``behavior_816212_2025-12-05_13-47-41`` the last 8 minutes show an empty
rig), which fails every stability check; pass ``time_window`` to measure
the task only.

Decision (:func:`quality_action`): ``exclude: <check>`` for the first
failed check, otherwise ``use``. The module never removes data.

Example::

    qc = measure_video_quality("behavior-videos/BottomCamera/video.mp4")
    checks = check_video_quality(qc, camera="BottomCamera")
    quality_action(checks)   # "use" or "exclude: <check>"
    write_video_quality(qc, checks, "results/", camera="BottomCamera")
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import pandas as pd
from aind_video_utils import __version__ as aind_video_utils_version
from aind_video_utils import (
    get_frame_dimensions,
    get_video_range_info,
    luma_range,
    probe,
    read_mp4_frame_index,
)
from aind_video_utils.mp4_index import Mp4FrameIndex
from aind_video_utils.probe import ProbeDict, get_r_frame_rate
from aind_video_utils.qc_batch import FrameExposureStats, compute_frame_stats

from aind_dynamic_foraging_behavior_video_analysis import __version__

# Stability tolerances, relative to the session's own median. First guesses
# (2026-09-30); to be set from the population survey (plan, Phase 2).
SHARPNESS_TOLERANCE = 0.30
BRIGHTNESS_TOLERANCE = 0.15
MIN_SIMILARITY = 0.8

# Level thresholds per camera: min_sharpness, min_mean, max_mean,
# max_pct_clipped (median % of pixels at or above the tagged ceiling),
# min_dynamic_range. Empty until calibrated on the
# population (plan, Phase 2); level checks are skipped without them.
LEVEL_THRESHOLDS: dict[str, dict[str, float]] = {}

# The reference frame is the pixel-wise median of this many first samples.
REFERENCE_SAMPLES = 10

# Fraction of the file skipped at each end (aind-video-utils default).
DEFAULT_EDGE_FRACTION = 0.01

# Files written by :func:`write_video_quality`, per camera.
VIDEO_QUALITY_JSON = "video_quality_{camera}.json"
VIDEO_QUALITY_SAMPLES = "video_quality_{camera}_samples.parquet"

STABILITY_CHECKS = [
    "sharpness_stable",
    "brightness_stable",
    "scene_stable",
]
LEVEL_CHECKS = ["sharp_enough", "exposure_ok", "contrast_ok"]

# Session summaries (median, p5, p95, spread) are kept for these columns.
SUMMARY_METRICS = [
    "sharpness",
    "noise_sigma",
    "mean",
    "contrast_rms",
    "dynamic_range",
    "pct_clipped_low",
    "pct_clipped_high",
    "pct_below_floor",
    "pct_above_ceiling",
    "shift",
    "similarity",
]

LumaFrame = npt.NDArray[np.uint8]


# --- Results -------------------------------------------------------------


@dataclass(frozen=True)
class FrameQualityStats:
    """Quality statistics for one sampled keyframe.

    ``exposure`` is ``aind_video_utils``' ``FrameExposureStats``; the other
    fields are added here. Shift and similarity are relative to the
    session's reference frame, so they are filled in after sampling.
    """

    exposure: FrameExposureStats
    contrast_rms: float
    dynamic_range: float
    pct_clipped_low: float
    pct_clipped_high: float
    sharpness: float
    noise_sigma: float


@dataclass(frozen=True)
class VideoQualityQc:
    """Per-video quality summary: file format, sampling, metric summaries.

    ``<metric>_med``, ``_p5``, ``_p95`` summarize each metric in
    ``SUMMARY_METRICS`` over the samples; ``_spread`` is
    ``(p95 - p5) / median``.
    """

    codec: str
    pix_fmt: str
    color_range: str
    color_transfer: str | None
    bit_depth: int
    width: int
    height: int
    fps: float | None
    n_frames: int
    n_keyframes: int
    gop: float
    n_samples: int
    edge_fraction: float
    window_start: float | None
    window_end: float | None
    sharpness_med: float
    sharpness_p5: float
    sharpness_p95: float
    sharpness_spread: float
    noise_sigma_med: float
    noise_sigma_p5: float
    noise_sigma_p95: float
    noise_sigma_spread: float
    mean_med: float
    mean_p5: float
    mean_p95: float
    mean_spread: float
    contrast_rms_med: float
    contrast_rms_p5: float
    contrast_rms_p95: float
    contrast_rms_spread: float
    dynamic_range_med: float
    dynamic_range_p5: float
    dynamic_range_p95: float
    dynamic_range_spread: float
    pct_clipped_low_med: float
    pct_clipped_low_p5: float
    pct_clipped_low_p95: float
    pct_clipped_low_spread: float
    pct_clipped_high_med: float
    pct_clipped_high_p5: float
    pct_clipped_high_p95: float
    pct_clipped_high_spread: float
    pct_below_floor_med: float
    pct_below_floor_p5: float
    pct_below_floor_p95: float
    pct_below_floor_spread: float
    pct_above_ceiling_med: float
    pct_above_ceiling_p5: float
    pct_above_ceiling_p95: float
    pct_above_ceiling_spread: float
    shift_med: float
    shift_p5: float
    shift_p95: float
    shift_spread: float
    similarity_med: float
    similarity_p5: float
    similarity_p95: float
    similarity_spread: float


@dataclass(frozen=True, eq=False)
class VideoQualityResult:
    """Everything :func:`measure_video_quality` measured on one video.

    Attributes
    ----------
    video_path : str
        The video measured.
    summary : VideoQualityQc
        Format info and per-metric session summaries.
    samples : pandas.DataFrame
        One row per sampled keyframe, in time order: ``sample``,
        ``frame_index`` (presentation order; equals the video CSV row),
        ``video_time`` (s), ``pts``, the exposure statistics, and the
        metrics listed in the module docstring.
    histograms : numpy.ndarray
        ``(n_samples, 2**bit_depth)`` luma counts per sample.
    reference : numpy.ndarray
        Full-resolution uint8 reference frame.
    thumbnails : numpy.ndarray
        ``(n_samples, h // 2, w // 2)`` uint8 frames for the report.
    """

    video_path: str
    summary: VideoQualityQc
    samples: pd.DataFrame
    histograms: npt.NDArray[np.int64]
    reference: LumaFrame
    thumbnails: LumaFrame


def qc_result_fieldnames() -> list[str]:
    """Return the ordered field names of ``VideoQualityQc`` for CSV output."""
    return [f.name for f in fields(VideoQualityQc)]


# --- Per-frame metrics ---------------------------------------------------


def downsample2(luma: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Return ``luma`` 2x downsampled by 2x2 mean (odd edges dropped)."""
    y = np.asarray(luma, dtype=np.float32)
    h, w = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2
    y = y[:h, :w]
    return (y[0::2, 0::2] + y[1::2, 0::2] + y[0::2, 1::2] + y[1::2, 1::2]) / 4


def laplacian_variance(image: npt.ArrayLike) -> float:
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


def noise_sigma(luma: npt.ArrayLike) -> float:
    """Estimate Gaussian noise sigma (Immerkaer 1996).

    Convolves with the difference of two Laplacians, which cancels most
    image structure, and scales the mean absolute response.
    """
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


def luma_histogram(luma: LumaFrame, bit_depth: int) -> npt.NDArray[np.int64]:
    """Return the count of each luma value, ``2**bit_depth`` bins."""
    return np.bincount(luma.ravel(), minlength=1 << bit_depth)


def frame_quality_stats(
    luma: LumaFrame, color_range: str, bit_depth: int
) -> FrameQualityStats:
    """Compute the reference-free metrics for one luma frame.

    Parameters
    ----------
    luma : numpy.ndarray
        Luma plane, shape ``(h, w)``.
    color_range : str
        ``"pc"`` (full) or ``"tv"`` (limited).
    bit_depth : int
        Bits per sample.

    Returns
    -------
    FrameQualityStats
    """
    exposure = compute_frame_stats(luma, color_range, bit_depth)
    floor, ceiling = luma_range(bit_depth, color_range == "pc")
    return FrameQualityStats(
        exposure=exposure,
        contrast_rms=exposure.std / exposure.mean if exposure.mean else 0.0,
        dynamic_range=exposure.p99 - exposure.p1,
        pct_clipped_low=100.0 * float((luma <= floor).mean()),
        pct_clipped_high=100.0 * float((luma >= ceiling).mean()),
        sharpness=laplacian_variance(downsample2(luma)),
        noise_sigma=noise_sigma(luma),
    )


def _hann2d(shape: tuple[int, int]) -> npt.NDArray[np.float64]:
    """2D Hann window, which suppresses edge effects in the FFT."""
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def _parabolic_offset(left: float, centre: float, right: float) -> float:
    """Sub-sample offset of a peak from three samples around it."""
    denom = left - 2 * centre + right
    return 0.0 if denom == 0 else 0.5 * (left - right) / denom


def phase_shift(
    reference: npt.ArrayLike, image: npt.ArrayLike
) -> tuple[float, float]:
    """Return the ``(dx, dy)`` translation of ``image`` from ``reference``.

    Phase correlation with a Hann window and parabolic sub-pixel peak
    interpolation. Positive ``dx`` means the content moved right.
    """
    ref = np.asarray(reference, dtype=np.float64)
    img = np.asarray(image, dtype=np.float64)
    window = _hann2d(ref.shape)
    f_ref = np.fft.fft2((ref - ref.mean()) * window)
    f_img = np.fft.fft2((img - img.mean()) * window)
    cross = f_img * np.conj(f_ref)
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-12)).real
    peak_y, peak_x = np.unravel_index(int(corr.argmax()), corr.shape)
    h, w = corr.shape
    dy = peak_y + _parabolic_offset(
        corr[(peak_y - 1) % h, peak_x],
        corr[peak_y, peak_x],
        corr[(peak_y + 1) % h, peak_x],
    )
    dx = peak_x + _parabolic_offset(
        corr[peak_y, (peak_x - 1) % w],
        corr[peak_y, peak_x],
        corr[peak_y, (peak_x + 1) % w],
    )
    # Peaks past the midpoint are negative shifts (the FFT wraps around).
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(dx), float(dy)


def similarity(reference: npt.ArrayLike, image: npt.ArrayLike) -> float:
    """Pearson correlation of two images; 0 if either is flat."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(image, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# --- Sampling ------------------------------------------------------------


def keyframe_display_indices(index: Mp4FrameIndex) -> npt.NDArray[np.int64]:
    """Presentation-order frame index of every keyframe, ascending."""
    rank = np.empty(index.n_samples, dtype=np.int64)
    rank[index.display_order] = np.arange(index.n_samples)
    return np.sort(rank[index.keyframe_decode_indices])


def choose_keyframes(
    index: Mp4FrameIndex,
    n_samples: int,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
    time_window: tuple[float, float] | None = None,
) -> npt.NDArray[np.int64]:
    """Pick up to ``n_samples`` keyframes evenly across the file.

    Frame 0 (which carries embedded metadata) and the first and last
    ``edge_fraction`` of the file are never picked. ``time_window``
    (``(start, end)`` in video seconds) further limits the choice. With
    fewer eligible keyframes than ``n_samples``, each is used once.

    Returns
    -------
    numpy.ndarray
        Presentation-order frame indices, ascending, unique.

    Raises
    ------
    ValueError
        If no keyframe is eligible.
    """
    n = index.n_samples
    edge = int(np.ceil(n * edge_fraction))
    keys = keyframe_display_indices(index)
    keys = keys[(keys >= max(1, edge)) & (keys < n - edge)]
    if time_window is not None:
        seconds = _presentation_pts(index)[keys] / index.media_timescale
        start, end = time_window
        keys = keys[(seconds >= start) & (seconds <= end)]
    if keys.size == 0:
        raise ValueError("No keyframe outside the skipped edges and window")
    if keys.size <= n_samples:
        return keys
    targets = np.linspace(keys[0], keys[-1], n_samples)
    pos = np.searchsorted(keys, targets)
    pos = np.clip(pos, 1, keys.size - 1)
    nearer_left = targets - keys[pos - 1] <= keys[pos] - targets
    return np.unique(keys[np.where(nearer_left, pos - 1, pos)])


def _presentation_pts(index: Mp4FrameIndex) -> npt.NDArray[np.int64]:
    """PTS of each frame in presentation order, as the demuxer reports it.

    FFmpeg's MP4 demuxer subtracts the edit list's ``media_time``.
    """
    media_time = index.edits[0].media_time if index.edits else 0
    return index.pts[index.display_order] - media_time


# 8-bit formats whose first plane is luma, one byte per pixel.
LUMA_PLANE_FORMATS = {"yuv420p", "yuvj420p", "yuv422p", "yuv444p", "nv12"}


def luma_plane(frame: av.VideoFrame) -> LumaFrame:
    """Return the frame's luma plane as coded, without range conversion.

    ``frame.to_ndarray(format="gray")`` goes through swscale, which
    stretches limited-range (16-235) luma to 0-255; every exposure metric
    would then be on the wrong scale.

    Raises
    ------
    ValueError
        If the pixel format is not an 8-bit format with a luma plane.
    """
    if frame.format.name not in LUMA_PLANE_FORMATS:
        raise ValueError(f"Unsupported pixel format {frame.format.name!r}")
    plane = frame.planes[0]
    rows = np.frombuffer(plane, dtype=np.uint8).reshape(-1, plane.line_size)
    return rows[: frame.height, : frame.width].copy()


def read_keyframes(
    video_path: str | Path,
    index: Mp4FrameIndex,
    frame_indices: npt.ArrayLike,
) -> list[tuple[int, LumaFrame]]:
    """Decode the luma plane of each requested keyframe.

    Seeks to each keyframe's PTS in one open container and decodes one
    frame, which must be the keyframe asked for.

    Returns
    -------
    list of (pts, luma)
        ``pts`` in the stream time base, ``luma`` uint8 ``(h, w)``.

    Raises
    ------
    ValueError
        If the index is not safe for frame addressing, or a seek does not
        land on the requested keyframe.
    """
    if not index.is_frame_addressing_safe():
        raise ValueError(
            f"Edit list is not frame-addressing-safe: {video_path}"
        )
    media_pts = _presentation_pts(index)
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        to_stream = Fraction(1, index.media_timescale) / stream.time_base
        for frame_index in np.asarray(frame_indices, dtype=np.int64):
            target = int(round(media_pts[frame_index] * to_stream))
            container.seek(target, stream=stream, backward=True)
            frame = next(container.decode(stream))
            if frame.pts != target or not frame.key_frame:
                raise ValueError(
                    f"Seek to frame {frame_index} (pts {target}) returned "
                    f"pts {frame.pts}, key_frame={frame.key_frame}: "
                    f"{video_path}"
                )
            frames.append((target, luma_plane(frame)))
    return frames


# --- Measuring -----------------------------------------------------------


def _summarize(samples: pd.DataFrame) -> dict[str, float]:
    """Median, p5, p95 and relative spread of each summary metric."""
    out = {}
    for metric in SUMMARY_METRICS:
        values = samples[metric].to_numpy(dtype=np.float64)
        p5, med, p95 = np.percentile(values, [5, 50, 95])
        out[f"{metric}_med"] = float(med)
        out[f"{metric}_p5"] = float(p5)
        out[f"{metric}_p95"] = float(p95)
        out[f"{metric}_spread"] = float((p95 - p5) / med) if med else np.nan
    return out


def _format_info(
    probe_json: ProbeDict, index: Mp4FrameIndex
) -> dict[str, object]:
    """File format fields of ``VideoQualityQc``."""
    stream = probe_json["streams"][0]
    color_range, bit_depth = get_video_range_info(probe_json)
    width, height = get_frame_dimensions(probe_json)
    rate = stream.get("avg_frame_rate", "0/0")
    num, den = (int(x) for x in rate.split("/"))
    if not den:
        rate = get_r_frame_rate(probe_json)
        num, den = rate if rate else (0, 0)
    n_keys = int(index.is_keyframe.sum())
    return {
        "codec": stream.get("codec_name", ""),
        "pix_fmt": stream.get("pix_fmt", ""),
        "color_range": color_range,
        "color_transfer": stream.get("color_transfer"),
        "bit_depth": bit_depth,
        "width": width,
        "height": height,
        "fps": num / den if den else None,
        "n_frames": index.n_samples,
        "n_keyframes": n_keys,
        "gop": index.n_samples / n_keys,
    }


def _require_mp4(probe_json: ProbeDict, video_path) -> None:
    """Raise ValueError unless the file is an MP4."""
    format_name = probe_json.get("format", {}).get("format_name", "")
    if "mp4" not in format_name.split(","):
        raise ValueError(f"Not an MP4 (format {format_name!r}): {video_path}")


def measure_video_quality(
    video_path: str | Path,
    n_samples: int = 100,
    edge_fraction: float = DEFAULT_EDGE_FRACTION,
    time_window: tuple[float, float] | None = None,
    probe_json: ProbeDict | None = None,
) -> VideoQualityResult:
    """Sample keyframes across an MP4 and measure image quality.

    Parameters
    ----------
    video_path : str or pathlib.Path
        Local path or HTTPS URL of an MP4.
    n_samples : int, optional
        Keyframes to sample (default 100).
    edge_fraction : float, optional
        Fraction of the file skipped at each end (default 0.01).
    time_window : (float, float), optional
        Sample only between these video times (s), e.g. the task.
    probe_json : dict, optional
        ``aind_video_utils.probe`` output, if already probed.

    Returns
    -------
    VideoQualityResult

    Raises
    ------
    ValueError
        If the file is not an MP4 or has no eligible keyframe.
    """
    pj = probe_json if probe_json is not None else probe(video_path)
    _require_mp4(pj, video_path)
    index = read_mp4_frame_index(video_path)
    info = _format_info(pj, index)
    chosen = choose_keyframes(index, n_samples, edge_fraction, time_window)
    decoded = read_keyframes(video_path, index, chosen)
    color_range, bit_depth = info["color_range"], info["bit_depth"]
    media_pts = _presentation_pts(index)

    lumas = [luma for _, luma in decoded]
    reference = np.median(np.stack(lumas[:REFERENCE_SAMPLES]), axis=0).astype(
        np.uint8
    )
    reference_small = downsample2(reference)
    rows = []
    for i, (frame_index, (pts, luma)) in enumerate(zip(chosen, decoded)):
        stats = frame_quality_stats(luma, color_range, bit_depth)
        dx, dy = phase_shift(reference, luma)
        rows.append(
            {
                "sample": i,
                "frame_index": int(frame_index),
                "video_time": float(
                    media_pts[frame_index] / index.media_timescale
                ),
                "pts": pts,
                **asdict(stats.exposure),
                "contrast_rms": stats.contrast_rms,
                "dynamic_range": stats.dynamic_range,
                "pct_clipped_low": stats.pct_clipped_low,
                "pct_clipped_high": stats.pct_clipped_high,
                "sharpness": stats.sharpness,
                "noise_sigma": stats.noise_sigma,
                "shift_x": dx,
                "shift_y": dy,
                "shift": float(np.hypot(dx, dy)),
                "similarity": similarity(reference_small, downsample2(luma)),
            }
        )
    samples = pd.DataFrame(rows)
    summary = VideoQualityQc(
        **info,
        n_samples=len(samples),
        edge_fraction=edge_fraction,
        window_start=time_window[0] if time_window else None,
        window_end=time_window[1] if time_window else None,
        **_summarize(samples),
    )
    return VideoQualityResult(
        video_path=str(video_path),
        summary=summary,
        samples=samples,
        histograms=np.stack([luma_histogram(y, bit_depth) for y in lumas]),
        reference=reference,
        thumbnails=np.stack(
            [np.round(downsample2(y)).astype(np.uint8) for y in lumas]
        ),
    )


# --- Checks --------------------------------------------------------------


def _result(check, passed, message, samples=(), count=None) -> dict:
    """One check's outcome. ``passed`` is None when the check was skipped."""
    samples = np.asarray(samples, dtype=int)
    return {
        "check": check,
        "passed": passed,
        "message": message,
        "count": int(len(samples) if count is None else count),
        "samples": samples.tolist(),
    }


def _stability_result(check, out_of_tolerance, what) -> dict:
    """Fail on two or more consecutive samples out of tolerance.

    A single sample out of line is counted but passes: something brief
    (a paw, a reflection) crossed the view.
    """
    bad = np.flatnonzero(np.asarray(out_of_tolerance, dtype=bool))
    in_runs = np.isin(bad + 1, bad) | np.isin(bad - 1, bad)
    if in_runs.any():
        return _result(
            check,
            False,
            f"{in_runs.sum()} samples in runs {what}",
            bad,
        )
    if bad.size:
        return _result(check, True, f"{bad.size} isolated samples {what}", bad)
    return _result(check, True, f"no sample {what}")


def check_sharpness_stable(sharpness, tolerance=SHARPNESS_TOLERANCE) -> dict:
    """Is sharpness within ``tolerance`` of the session median throughout?"""
    s = np.asarray(sharpness, dtype=np.float64)
    off = np.abs(s / np.median(s) - 1) > tolerance
    return _stability_result(
        "sharpness_stable", off, f"off the median by > {tolerance:.0%}"
    )


def check_brightness_stable(mean, tolerance=BRIGHTNESS_TOLERANCE) -> dict:
    """Is mean luma within ``tolerance`` of the session median throughout?"""
    m = np.asarray(mean, dtype=np.float64)
    off = np.abs(m / np.median(m) - 1) > tolerance
    return _stability_result(
        "brightness_stable", off, f"off the median by > {tolerance:.0%}"
    )


def check_scene_stable(similarity, min_similarity=MIN_SIMILARITY) -> dict:
    """Does every sample correlate with the reference by ``min_similarity``?"""
    off = np.asarray(similarity, dtype=np.float64) < min_similarity
    return _stability_result(
        "scene_stable",
        off,
        f"correlating < {min_similarity:g} with the reference",
    )


def _skipped(check, camera) -> dict:
    """A level check with no threshold for this camera."""
    return _result(check, None, f"no threshold for camera {camera!r}")


def check_sharp_enough(summary, thresholds, camera=None) -> dict:
    """Is the median sharpness at least ``min_sharpness``?"""
    if "min_sharpness" not in thresholds:
        return _skipped("sharp_enough", camera)
    limit = thresholds["min_sharpness"]
    value = summary.sharpness_med
    return _result(
        "sharp_enough",
        bool(value >= limit),
        f"median sharpness {value:.1f} (min {limit:g})",
    )


def check_exposure_ok(summary, thresholds, camera=None) -> dict:
    """Is the median mean luma within [min_mean, max_mean], and the
    median % of clipped pixels at most ``max_pct_clipped``?"""
    keys = ["min_mean", "max_mean", "max_pct_clipped"]
    present = [k for k in keys if k in thresholds]
    if not present:
        return _skipped("exposure_ok", camera)
    mean, clipped = summary.mean_med, summary.pct_clipped_high_med
    problems = []
    if mean < thresholds.get("min_mean", -np.inf):
        problems.append(f"too dark (mean {mean:.1f})")
    if mean > thresholds.get("max_mean", np.inf):
        problems.append(f"too bright (mean {mean:.1f})")
    if clipped > thresholds.get("max_pct_clipped", np.inf):
        problems.append(f"{clipped:.2f}% clipped")
    if problems:
        return _result("exposure_ok", False, "; ".join(problems))
    return _result(
        "exposure_ok",
        True,
        f"median mean {mean:.1f}, {clipped:.2f}% clipped",
    )


def check_contrast_ok(summary, thresholds, camera=None) -> dict:
    """Is the median dynamic range (p99 - p1) at least
    ``min_dynamic_range``?"""
    if "min_dynamic_range" not in thresholds:
        return _skipped("contrast_ok", camera)
    limit = thresholds["min_dynamic_range"]
    value = summary.dynamic_range_med
    return _result(
        "contrast_ok",
        bool(value >= limit),
        f"median dynamic range {value:.1f} (min {limit:g})",
    )


def check_video_quality(
    qc: VideoQualityResult,
    camera: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run every check on one measured video.

    Parameters
    ----------
    qc : VideoQualityResult
        From :func:`measure_video_quality`.
    camera : str, optional
        Looks up level thresholds in ``LEVEL_THRESHOLDS``.
    thresholds : dict, optional
        Level thresholds to use instead of the camera's.

    Returns
    -------
    pandas.DataFrame
        One row per check: ``check``, ``passed`` (True, False, or None if
        skipped), ``message``, ``count``, ``samples`` (offending sample
        numbers).
    """
    if thresholds is None:
        thresholds = LEVEL_THRESHOLDS.get(camera, {})
    s = qc.samples
    results = [
        check_sharpness_stable(s["sharpness"]),
        check_brightness_stable(s["mean"]),
        check_scene_stable(s["similarity"]),
        check_sharp_enough(qc.summary, thresholds, camera),
        check_exposure_ok(qc.summary, thresholds, camera),
        check_contrast_ok(qc.summary, thresholds, camera),
    ]
    return pd.DataFrame(results)


def quality_action(checks: pd.DataFrame) -> str:
    """``exclude: <check>`` for the first failed check, otherwise ``use``.

    A ``use`` with skipped level checks is not an absolute verdict; the
    skipped rows say which thresholds are missing.
    """
    failed = checks.loc[checks["passed"].eq(False), "check"]
    return f"exclude: {failed.iloc[0]}" if len(failed) else "use"


# --- Output --------------------------------------------------------------


def _json_ready(value):
    """Convert numpy scalars and NaN for JSON."""
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def quality_record(
    qc: VideoQualityResult, checks: pd.DataFrame, camera: str | None = None
) -> dict:
    """The standardized per-camera record written by
    :func:`write_video_quality`."""
    return {
        "camera": camera,
        "video_path": qc.video_path,
        "action": quality_action(checks),
        "versions": {
            "aind_dynamic_foraging_behavior_video_analysis": __version__,
            "aind_video_utils": aind_video_utils_version,
        },
        "summary": {k: _json_ready(v) for k, v in asdict(qc.summary).items()},
        "sample_frame_indices": qc.samples["frame_index"].tolist(),
        "checks": [
            {k: _json_ready(v) for k, v in row.items()}
            for row in checks.to_dict("records")
        ],
    }


def write_video_quality(
    qc: VideoQualityResult,
    checks: pd.DataFrame,
    output_dir: str | Path,
    camera: str,
) -> tuple[Path, Path]:
    """Write the per-camera JSON and the per-sample parquet.

    File names are ``VIDEO_QUALITY_JSON`` and ``VIDEO_QUALITY_SAMPLES``
    formatted with ``camera``. The parquet holds ``qc.samples`` plus a
    ``histogram`` column (luma counts per sample).

    Returns
    -------
    (json_path, parquet_path)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / VIDEO_QUALITY_JSON.format(camera=camera)
    parquet_path = out / VIDEO_QUALITY_SAMPLES.format(camera=camera)
    with open(json_path, "w") as f:
        json.dump(quality_record(qc, checks, camera), f, indent=2)
    samples = qc.samples.copy()
    samples["histogram"] = [h.tolist() for h in qc.histograms]
    samples.to_parquet(parquet_path, index=False)
    return json_path, parquet_path


# --- Whole session -------------------------------------------------------


def find_session_videos(behavior_videos_path) -> list[tuple[str, Path]]:
    """Return ``(camera, path)`` for every MP4 in a ``behavior-videos``
    folder: ``<Camera>/video.mp4`` (New/AIND) or ``<camera>.mp4``
    (Old/flat)."""
    folder = Path(behavior_videos_path)
    new = [(p.parent.name, p) for p in sorted(folder.glob("*/video.mp4"))]
    old = [(p.stem, p) for p in sorted(folder.glob("*.mp4"))]
    return new + old


def check_session(
    behavior_videos_path, n_samples=100, time_window=None
) -> pd.DataFrame:
    """Measure and check every camera MP4 in a ``behavior-videos`` folder.

    ``time_window`` (video seconds) applies to every camera.

    Returns
    -------
    pandas.DataFrame
        One row per camera: ``camera``, ``video_path``, ``action``,
        ``failed_checks``, ``skipped_checks``, then the ``VideoQualityQc``
        fields. Unreadable videos get ``action`` ``exclude: unreadable``
        and the error text in ``error``.
    """
    rows = []
    for camera, path in find_session_videos(behavior_videos_path):
        info = {"camera": camera, "video_path": str(path)}
        try:
            qc = measure_video_quality(
                path, n_samples=n_samples, time_window=time_window
            )
        except (ValueError, OSError, subprocess.CalledProcessError) as e:
            info.update({"action": "exclude: unreadable", "error": str(e)})
        else:
            checks = check_video_quality(qc, camera=camera)
            info.update(
                {
                    "action": quality_action(checks),
                    "failed_checks": checks.loc[
                        checks["passed"].eq(False), "check"
                    ].tolist(),
                    "skipped_checks": checks.loc[
                        checks["passed"].isna(), "check"
                    ].tolist(),
                    **asdict(qc.summary),
                }
            )
        rows.append(info)
    return pd.DataFrame(rows)
