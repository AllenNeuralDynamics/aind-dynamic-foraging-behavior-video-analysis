"""Session card for :mod:`video_quality_qc`: one fixed-layout page per camera.

Kept apart from ``video_quality_qc`` so measuring never imports
matplotlib. Drawn from the sampled frames, the samples table and the
checks, so no video is decoded here. Every frame is shown on the tagged
luma range (no per-frame auto-contrast), so brightness changes stay
visible.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from aind_video_utils import luma_range

from aind_dynamic_foraging_behavior_video_analysis.video_quality_qc import (
    quality_action,
    reference_frame,
)

PAGE_SIZE = (11, 8.5)  # inches, landscape letter
INK = "#52514e"
MUTED = "#898781"
BAND = "#e1e0d9"
FAIL = "#d03b3b"
PASS = "#006300"
N_THUMBNAILS = 8


def _mmss(seconds):
    """Format seconds as m:ss."""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _show(ax, image, display_range, title=""):
    """Draw a frame on the display range (RGB as is), without axes."""
    lo, hi = display_range
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
    ax.set_axis_off()
    ax.set_title(title, fontsize=7, color=INK, pad=2)


def _clipping(reference, display_range):
    """RGB reference: blue at or below the floor, red at or above the
    ceiling (the pixels ``pct_clipped_low`` / ``pct_clipped_high`` count)."""
    lo, hi = display_range
    gray = np.clip((reference.astype(float) - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[reference <= lo] = (0.13, 0.4, 0.67)
    rgb[reference >= hi] = (0.82, 0.23, 0.23)
    return rgb


def _style(ax, title):
    """Recessive axes with a small left-aligned title."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    ax.set_title(title, fontsize=7, color=INK, loc="left", pad=1)


def _time_course(fig, spec, samples, checks):
    """One panel per per-sample check, threshold dashed, failing samples
    red; then the shift, reported only."""
    minutes = samples["video_time"] / 60
    panels = [
        (row.metric, row.value, row.samples, row.check)
        for row in checks.itertuples()
        if row.over == "samples"
    ] + [("shift", None, [], "shift from reference, px (reported only)")]
    grid = spec.subgridspec(len(panels), 1, hspace=0.4)
    for i, (metric, value, bad, title) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        values = samples[metric]
        ax.plot(minutes, values, color=INK, lw=1.2)
        if value is not None:
            ax.axhline(value, color=MUTED, lw=0.8, ls="--")
        ax.plot(minutes.iloc[bad], values.iloc[bad], "o", ms=4, color=FAIL)
        _style(ax, title)
        ax.tick_params(labelbottom=i == len(panels) - 1)
    ax.set_xlabel("video time (min)", fontsize=7, color=INK)


def _histogram(ax, samples, display_range, bin_width=4):
    """Median luma distribution over samples, p5-p95 band. Summed into
    ``bin_width`` bins, which removes the comb left by range conversion."""
    counts = np.stack(samples["histogram"].to_numpy())
    counts = counts.reshape(len(counts), -1, bin_width).sum(axis=2)
    frac = np.maximum(counts / counts.sum(axis=1, keepdims=True), 1e-6)
    x = np.arange(frac.shape[1]) * bin_width + bin_width / 2
    lo, med, hi = np.percentile(frac, [5, 50, 95], axis=0)
    ax.fill_between(x, lo, hi, color=BAND, lw=0, step="mid")
    ax.step(x, med, where="mid", color=INK, lw=1)
    for v in display_range:
        ax.axvline(v, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("luma (dashed: tagged range)", fontsize=7, color=INK)
    _style(ax, "fraction of pixels, median and p5–p95 over samples")


def _outliers(fig, spec, frames, samples, display_range):
    """The frames most likely to show a problem, plus |last - first|."""
    picks = [
        ("blurriest", samples["sharpness"].idxmin(), "sharpness"),
        ("darkest", samples["mean"].idxmin(), "mean"),
        ("brightest", samples["mean"].idxmax(), "mean"),
        ("least similar", samples["similarity"].idxmin(), "similarity"),
        ("most shifted", samples["shift"].idxmax(), "shift"),
    ]
    grid = spec.subgridspec(2, 3, wspace=0.05, hspace=0.25)
    for k, (name, i, metric) in enumerate(picks):
        title = (
            f"{name} {_mmss(samples['video_time'][i])} "
            f"({metric} {samples[metric][i]:.3g})"
        )
        _show(
            fig.add_subplot(grid[k // 3, k % 3]),
            frames[i],
            display_range,
            title,
        )
    diff = np.abs(frames[-1].astype(int) - frames[0].astype(int))
    ax = fig.add_subplot(grid[1, 2])
    ax.imshow(diff, cmap="gray_r", vmin=0, vmax=64)
    ax.set_axis_off()
    ax.set_title("|last − first| sample", fontsize=7, color=INK, pad=2)


def session_card(frames, samples, checks, title):
    """One page summarizing a camera's quality, built for scanning by eye.

    Header (action, headline medians); the reference frame with clipped
    pixels marked and 8 evenly spaced frames; a time course per per-sample
    check and the shift; the luma histogram; the outlier frames.

    Parameters
    ----------
    frames : numpy.ndarray
        ``(n, h, w)`` uint8, from ``video_quality_qc.sample_keyframes``.
    samples : pandas.DataFrame
        From ``video_quality_qc.measure``.
    checks : pandas.DataFrame
        From ``video_quality_qc.run_checks``.
    title : str
        First line of the page, e.g. ``"<session>  <camera>  (task)"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    display_range = luma_range(8, samples["color_range"].iloc[0] == "pc")
    action = quality_action(checks)
    bold = {"weight": "bold", "color": PASS if action == "use" else FAIL}
    med = samples.median(numeric_only=True)
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    for y, text, style in [
        (0.975, title, {"fontsize": 10, "color": "#0b0b0b"}),
        (0.945, f"ACTION: {action}", {"fontsize": 11, **bold}),
        (
            0.918,
            f"{len(samples)} keyframes · sharpness {med['sharpness']:.0f} · "
            f"mean {med['mean']:.1f} · dynamic range "
            f"{med['dynamic_range']:.0f} · clipped "
            f"{med['pct_clipped_high']:.2f}% · noise σ "
            f"{med['noise_sigma']:.2f} · similarity p5 "
            f"{samples['similarity'].quantile(0.05):.3f} · shift p95 "
            f"{samples['shift'].quantile(0.95):.1f} px",
            {"fontsize": 8, "color": INK},
        ),
    ]:
        fig.text(0.01, y, text, va="top", **style)
    outer = fig.add_gridspec(
        3,
        1,
        left=0.05,
        right=0.99,
        top=0.88,
        bottom=0.05,
        height_ratios=[1.25, 1.1, 1.0],
        hspace=0.28,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 2.1], wspace=0.04)
    _show(
        fig.add_subplot(top[0]),
        _clipping(reference_frame(frames), display_range),
        display_range,
        "reference (median of all samples); "
        "clipped: blue ≤ floor, red ≥ ceiling",
    )
    thumbs = top[1].subgridspec(2, 4, wspace=0.03, hspace=0.18)
    n = len(frames)
    picks = np.unique(np.linspace(0, n - 1, min(N_THUMBNAILS, n)).round())
    for k, i in enumerate(picks.astype(int)):
        ax = fig.add_subplot(thumbs[k // 4, k % 4])
        _show(ax, frames[i], display_range, _mmss(samples["video_time"][i]))
    _time_course(fig, outer[1], samples, checks)
    bottom = outer[2].subgridspec(1, 2, width_ratios=[1, 1.6], wspace=0.15)
    _histogram(fig.add_subplot(bottom[0]), samples, display_range)
    _outliers(fig, bottom[1], frames, samples, display_range)
    return fig
