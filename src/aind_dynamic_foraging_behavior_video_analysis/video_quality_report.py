"""Figures for :mod:`video_quality_qc`: session cards and batch PDFs.

Kept apart from ``video_quality_qc`` so measuring never imports
matplotlib. Everything is drawn from a ``VideoQualityResult`` (the sampled
keyframes are kept as thumbnails), so no video is decoded here.

- :func:`session_card`: one fixed-layout page per camera: reference frame
  with clipping highlighted, evenly spaced thumbnails, a time course of
  each metric, the luma histogram, and the outlier frames.
- :func:`contact_sheet`: one row per video (action and 8 thumbnails),
  for scanning many sessions quickly.
- :func:`batch_pdf`: an index page, then contact sheets, then one card per
  video, sorted by action and a chosen metric.

The reference marks clipped pixels (at or beyond the tagged floor and
ceiling). Thumbnails share one display scale (the tagged luma range), so a
change in brightness between frames or sessions stays visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aind_video_utils import luma_range
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from aind_dynamic_foraging_behavior_video_analysis.video_quality_qc import (
    BRIGHTNESS_TOLERANCE,
    MIN_SIMILARITY,
    SHARPNESS_TOLERANCE,
    VideoQualityResult,
    quality_action,
)

PAGE_SIZE = (11, 8.5)  # inches, landscape letter
INK = "#52514e"
MUTED = "#898781"
BAND = "#e1e0d9"
FAIL = "#d03b3b"
PASS = "#006300"
N_THUMBNAILS = 8
CONTACT_ROWS = 10


def _display_range(qc: VideoQualityResult) -> tuple[int, int]:
    """The tagged luma range, used as the display scale for every frame."""
    s = qc.summary
    return luma_range(s.bit_depth, s.color_range == "pc")


def _mmss(seconds: float) -> str:
    """Format seconds as m:ss."""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _failed_samples(checks: pd.DataFrame, check: str) -> list[int]:
    """Offending samples of a failed check (none if it passed)."""
    row = checks.loc[checks["check"] == check]
    if row.empty or row["passed"].iloc[0] is not False:
        return []
    return list(row["samples"].iloc[0])


def _even_picks(n: int, k: int) -> list[int]:
    """``k`` evenly spaced indices into ``n`` items (fewer if ``n < k``)."""
    return sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int)))


def _show(ax, image, qc, title=None) -> None:
    """Draw a frame on the shared display scale, no axes."""
    lo, hi = _display_range(qc)
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, color=INK, pad=2)


def _show_clipping(ax, qc) -> None:
    """The reference on the shared scale, with pixels at or below the
    tagged floor in blue and at or above the ceiling in red (clipped, as
    counted by ``pct_clipped_low`` / ``pct_clipped_high``)."""
    lo, hi = _display_range(qc)
    ref = qc.reference
    gray = np.clip((ref.astype(float) - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[ref <= lo] = (0.13, 0.4, 0.67)
    rgb[ref >= hi] = (0.82, 0.23, 0.23)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _style(ax) -> None:
    """Recessive axes: no top/right spines, muted ticks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)


def _header(fig, qc, checks, label) -> None:
    """Title line, action, and headline medians."""
    s = qc.summary
    action = quality_action(checks)
    skipped = checks.loc[checks["passed"].isna(), "check"].tolist()
    window = (
        f"window {_mmss(s.window_start)}–{_mmss(s.window_end)}"
        if s.window_start is not None
        else "whole file"
    )
    fig.text(
        0.01,
        0.975,
        f"{label}    {s.codec} {s.pix_fmt} {s.color_transfer or ''} "
        f"{s.color_range}   {s.width}×{s.height}   "
        f"{(s.fps or 0):.0f} Hz   {s.n_samples} keyframes, {window}",
        fontsize=10,
        color="#0b0b0b",
        va="top",
    )
    fig.text(
        0.01,
        0.945,
        f"ACTION: {action}",
        fontsize=11,
        weight="bold",
        color=PASS if action == "use" else FAIL,
        va="top",
    )
    note = f"   (not checked: {', '.join(skipped)})" if skipped else ""
    fig.text(
        0.01,
        0.918,
        f"sharpness {s.sharpness_med:.0f} · mean {s.mean_med:.1f} · "
        f"dynamic range {s.dynamic_range_med:.0f} · "
        f"clipped {s.pct_clipped_high_med:.2f}% · "
        f"noise σ {s.noise_sigma_med:.2f} · "
        f"similarity p5 {s.similarity_p5:.2f} · "
        f"shift p95 {s.shift_p95:.1f} px{note}",
        fontsize=8,
        color=INK,
        va="top",
    )


def _time_course(fig, spec, qc, checks) -> None:
    """One small panel per metric (no shared y-axis), failed samples red."""
    s = qc.samples
    minutes = s["video_time"] / 60
    panels = [
        (
            "sharpness",
            "sharpness, % of median",
            "sharpness_stable",
            SHARPNESS_TOLERANCE,
        ),
        (
            "mean",
            "mean luma, % of median",
            "brightness_stable",
            BRIGHTNESS_TOLERANCE,
        ),
        ("similarity", "similarity to reference", "scene_stable", None),
        ("shift", "shift from reference, px (reported only)", None, None),
    ]
    grid = spec.subgridspec(len(panels), 1, hspace=0.35)
    for i, (metric, ylabel, check, tol) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        values = s[metric].to_numpy(dtype=float)
        if tol is not None:
            values = 100 * values / np.median(values)
            ax.axhspan(100 * (1 - tol), 100 * (1 + tol), color=BAND, lw=0)
        if metric == "similarity":
            ax.axhspan(MIN_SIMILARITY, 1, color=BAND, lw=0)
        ax.plot(minutes, values, color=INK, lw=1.2)
        bad = _failed_samples(checks, check) if check else []
        if bad:
            ax.plot(
                minutes.iloc[bad],
                values[bad],
                "o",
                ms=4,
                color=FAIL,
                label="failed",
            )
            ax.legend(fontsize=6, frameon=False, loc="upper right")
        ax.set_title(ylabel, fontsize=7, color=INK, loc="left", pad=1)
        _style(ax)
        if i < len(panels) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("video time (min)", fontsize=7, color=INK)


def _histogram(ax, qc, bin_width=4) -> None:
    """Median luma distribution over samples with a p5–p95 band.

    Counts are summed into ``bin_width``-value bins for display, which
    removes the comb left by the encoder's range conversion.
    """
    counts = qc.histograms
    counts = counts[:, : counts.shape[1] // bin_width * bin_width]
    counts = counts.reshape(len(counts), -1, bin_width).sum(axis=2)
    frac = counts / counts.sum(axis=1, keepdims=True)
    x = np.arange(frac.shape[1]) * bin_width + bin_width / 2
    lo_band, med, hi_band = np.percentile(frac, [5, 50, 95], axis=0)
    floor_y = 1e-6
    ax.fill_between(
        x,
        np.maximum(lo_band, floor_y),
        np.maximum(hi_band, floor_y),
        color=BAND,
        lw=0,
        step="mid",
    )
    ax.step(x, np.maximum(med, floor_y), where="mid", color=INK, lw=1)
    floor, ceiling = _display_range(qc)
    for v in (floor, ceiling):
        ax.axvline(v, color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor_y)
    ax.set_xlim(0, frac.shape[1] * bin_width)
    ax.set_xlabel("luma (dashed: tagged range)", fontsize=7, color=INK)
    ax.set_ylabel("fraction of pixels", fontsize=7, color=INK)
    ax.set_title(
        "luma histogram, median and p5–p95 over samples",
        fontsize=7,
        color=INK,
    )
    _style(ax)


def _outliers(fig, spec, qc) -> None:
    """The frames most likely to show a problem, plus first − last."""
    s = qc.samples
    picks = [
        ("blurriest", int(s["sharpness"].idxmin()), "sharpness"),
        ("darkest", int(s["mean"].idxmin()), "mean"),
        ("brightest", int(s["mean"].idxmax()), "mean"),
        ("least similar", int(s["similarity"].idxmin()), "similarity"),
        ("most shifted", int(s["shift"].idxmax()), "shift"),
    ]
    grid = spec.subgridspec(2, 3, wspace=0.05, hspace=0.25)
    for k, (name, i, metric) in enumerate(picks):
        ax = fig.add_subplot(grid[k // 3, k % 3])
        _show(
            ax,
            qc.thumbnails[i],
            qc,
            f"{name} {_mmss(s['video_time'][i])} "
            f"({metric} {s[metric][i]:.3g})",
        )
    ax = fig.add_subplot(grid[1, 2])
    diff = np.abs(qc.thumbnails[-1].astype(int) - qc.thumbnails[0].astype(int))
    ax.imshow(diff, cmap="gray_r", vmin=0, vmax=64)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("|last − first| sample", fontsize=7, color=INK, pad=2)


def session_card(
    qc: VideoQualityResult,
    checks: pd.DataFrame,
    label: str | None = None,
) -> Figure:
    """One page summarizing a camera's quality, built for scanning by eye.

    Parameters
    ----------
    qc : VideoQualityResult
        From ``measure_video_quality``.
    checks : pandas.DataFrame
        From ``check_video_quality``.
    label : str, optional
        Title, e.g. ``"<session> BottomCamera"``; defaults to the path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    _header(fig, qc, checks, label or qc.video_path)
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
    ax = fig.add_subplot(top[0])
    _show_clipping(ax, qc)
    ax.set_title(
        "reference (median of first samples); "
        "clipped: blue ≤ floor, red ≥ ceiling",
        fontsize=7,
        color=INK,
    )
    thumbs = top[1].subgridspec(2, 4, wspace=0.03, hspace=0.18)
    times = qc.samples["video_time"]
    for k, i in enumerate(_even_picks(len(qc.samples), N_THUMBNAILS)):
        _show(
            fig.add_subplot(thumbs[k // 4, k % 4]),
            qc.thumbnails[i],
            qc,
            _mmss(times[i]),
        )

    _time_course(fig, outer[1], qc, checks)

    bottom = outer[2].subgridspec(1, 2, width_ratios=[1, 1.6], wspace=0.15)
    _histogram(fig.add_subplot(bottom[0]), qc)
    _outliers(fig, bottom[1], qc)
    return fig


def contact_sheet(
    items: list[tuple[str, VideoQualityResult, pd.DataFrame]],
    n_thumbnails: int = 8,
) -> Figure:
    """One row per video: label and action, then evenly spaced frames.

    Parameters
    ----------
    items : list of (label, VideoQualityResult, checks)
        At most ``CONTACT_ROWS`` fit on a page.
    n_thumbnails : int, optional
        Frames per row (default 8).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(
        CONTACT_ROWS,
        n_thumbnails + 1,
        left=0.01,
        right=0.99,
        top=0.98,
        bottom=0.02,
        width_ratios=[1.3] + [1] * n_thumbnails,
        wspace=0.03,
        hspace=0.08,
    )
    for row, (label, qc, checks) in enumerate(items[:CONTACT_ROWS]):
        action = quality_action(checks)
        text_ax = fig.add_subplot(grid[row, 0])
        text_ax.axis("off")
        text_ax.text(0, 0.65, label, fontsize=7, color=INK, va="center")
        text_ax.text(
            0,
            0.3,
            action,
            fontsize=8,
            weight="bold",
            color=PASS if action == "use" else FAIL,
            va="center",
        )
        picks = _even_picks(len(qc.samples), n_thumbnails)
        for col, i in enumerate(picks):
            _show(
                fig.add_subplot(grid[row, col + 1]),
                qc.thumbnails[i],
                qc,
            )
    return fig


def _index_page(table: pd.DataFrame) -> Figure:
    """A table of every video in the PDF, in PDF order."""
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    fig.text(0.01, 0.98, "Video quality: index", fontsize=12, va="top")
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.9])
    ax.axis("off")
    cells = table.astype(str).values.tolist()
    t = ax.table(
        cellText=cells, colLabels=list(table.columns), loc="upper left"
    )
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    return fig


def batch_pdf(
    items: list[tuple[str, VideoQualityResult, pd.DataFrame]],
    path: str | Path,
    sort_by: str = "sharpness_med",
) -> Path:
    """Write many videos to one PDF for scrolling.

    Order: excluded videos first, then by ``sort_by`` (a ``VideoQualityQc``
    field) ascending. Pages: an index (40 rows per page), contact sheets
    (``CONTACT_ROWS`` videos per page), then one session card per video.

    Parameters
    ----------
    items : list of (label, VideoQualityResult, checks)
    path : str or pathlib.Path
        The PDF to write.
    sort_by : str, optional
        Summary field to sort by (default ``"sharpness_med"``).

    Returns
    -------
    pathlib.Path
    """
    keyed = [
        (quality_action(checks) == "use", getattr(qc.summary, sort_by), i)
        for i, (_, qc, checks) in enumerate(items)
    ]
    ordered = [items[i] for _, _, i in sorted(keyed)]
    table = pd.DataFrame(
        {
            "video": [label for label, _, _ in ordered],
            "action": [quality_action(c) for _, _, c in ordered],
            sort_by: [
                f"{getattr(qc.summary, sort_by):.3g}" for _, qc, _ in ordered
            ],
        }
    )
    path = Path(path)
    with PdfPages(path) as pdf:
        for start in range(0, len(table), 40):
            fig = _index_page(table.iloc[start : start + 40])
            pdf.savefig(fig)
            plt.close(fig)
        for start in range(0, len(ordered), CONTACT_ROWS):
            fig = contact_sheet(ordered[start : start + CONTACT_ROWS])
            pdf.savefig(fig)
            plt.close(fig)
        for label, qc, checks in ordered:
            fig = session_card(qc, checks, label)
            pdf.savefig(fig)
            plt.close(fig)
    return path
