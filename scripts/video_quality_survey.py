"""Screen many sessions from s3://aind-open-data with ``video_screen``.

Builds the inputs table from the public bucket (per raw session asset: each
camera's MP4 and video CSV in either ``behavior-videos`` layout, the
session JSON and the camera trigger log, all as HTTPS URLs) and passes it
to ``video_screen.screen_sessions`` with session cards, writing into
``<out>/``: ``video_screen.jsonl`` and ``video_screen.csv`` (one row per
camera), and per camera into ``<out>/<session>/`` the timing and quality
records, the quality samples and the session card. An interrupted run
resumes where it stopped. Sessions with no MP4 are listed and skipped.

``--report`` then builds ``<out>/video_quality_survey.pdf``: an index and
every session card, cameras not in use first. ``--thresholds`` builds
``<out>/video_quality_thresholds.pdf``, one page per level metric and
camera: the distribution across sessions with 12 sessions marked from one
extreme to the other, and a full-resolution frame from each, for picking a
cutoff by eye.

Usage::

    python scripts/video_quality_survey.py sessions.csv out/ --workers 8
    python scripts/video_quality_survey.py sessions.csv out/ --report
    python scripts/video_quality_survey.py sessions.csv out/ --thresholds

``sessions.csv`` needs a ``raw_session`` column (e.g.
``behavior_816212_2025-12-05_13-47-41``); ``--filter-column`` keeps rows
where that column is true.
"""

import argparse
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from aind_dynamic_foraging_behavior_video_analysis import (
    video_quality_qc as vqq,
)
from aind_dynamic_foraging_behavior_video_analysis import (
    video_quality_report as vqr,
)
from aind_dynamic_foraging_behavior_video_analysis import video_screen as vs

BUCKET = "https://aind-open-data.s3.us-west-2.amazonaws.com"


def list_keys(prefix):
    """All object keys under ``prefix`` in the public bucket."""
    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = f"{BUCKET}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            xml = response.read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        match = re.search(
            r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml
        )
        if not match:
            return keys
        token = match.group(1)


def session_files(session):
    """Cameras (name, mp4 key, csv key), the behavior JSON key and the
    trigger log key of one raw session (None where absent)."""
    keys = set(list_keys(f"{session}/"))
    cameras = []
    for key in sorted(keys):
        rel = key.removeprefix(f"{session}/")
        new = re.fullmatch(r"behavior-videos/([^/]+)/video\.mp4", rel)
        old = re.fullmatch(r"behavior-videos/([^/]+)\.mp4", rel)
        if new:
            csv = f"{session}/behavior-videos/{new.group(1)}/metadata.csv"
            cameras.append((new.group(1), key, csv if csv in keys else None))
        elif old:
            csv = f"{session}/behavior-videos/{old.group(1)}.csv"
            cameras.append((old.group(1), key, csv if csv in keys else None))
    jsons = sorted(
        k
        for k in keys
        if re.fullmatch(rf"{re.escape(session)}/behavior/\d[^/]*\.json", k)
    )
    log = f"{session}/behavior/raw.harp/BehaviorEvents/Event_94.bin"
    return (
        cameras,
        (jsons[0] if jsons else None),
        (log if log in keys else None),
    )


def url(key):
    """HTTPS URL of an object key; None for None."""
    return None if key is None else f"{BUCKET}/{urllib.parse.quote(key)}"


def session_inputs(session):
    """The ``video_screen`` inputs rows of one raw session, all URLs."""
    cameras, behavior_json, log = session_files(session)
    return [
        {
            "session": session,
            "camera": camera,
            "mp4": url(mp4),
            "video_csv": url(csv),
            "behavior_json": url(behavior_json),
            "trigger_log": url(log),
        }
        for camera, mp4, csv in cameras
    ]


def build_inputs(sessions, workers):
    """Inputs rows for every session, listed in parallel threads.
    Sessions with no MP4, or whose listing failed, are printed."""
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(session_inputs, s): s for s in sessions}
        for future in as_completed(futures):
            try:
                found = future.result()
            except OSError as e:
                print(f"{futures[future]}: listing failed: {e!r}")
                continue
            if not found:
                print(f"{futures[future]}: no mp4")
            rows += found
    order = {s: i for i, s in enumerate(sessions)}
    rows.sort(key=lambda r: (order[r["session"]], r["camera"]))
    return pd.DataFrame(rows)


def survey(sessions, out_dir, workers):
    """Screen every camera of ``sessions`` (with cards) into ``out_dir``."""
    inputs = build_inputs(sessions, workers)
    screen = vs.screen_sessions(inputs, out_dir, cards=True, workers=workers)
    print(screen["reason"].replace("", "use").value_counts().to_string())


def report(out_dir, sort_by="sharpness"):
    """Index and session cards for every camera screened."""
    out_dir = Path(out_dir)
    table = vs.load_screen(out_dir / vs.SCREEN_FILE)
    table["status"] = table["reason"].replace("", "use")
    table = table.sort_values(["use", "camera", sort_by], na_position="first")
    path = out_dir / "video_quality_survey.pdf"
    with PdfPages(path) as pdf:
        cols = ["session", "camera", "status", "window", sort_by]
        for start in range(0, len(table), 45):
            fig = plt.figure(figsize=vqr.PAGE_SIZE)
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.95])
            ax.axis("off")
            part = table[cols].iloc[start:][:45].copy()
            for col in ("status", "window"):
                part[col] = part[col].fillna("").str.slice(0, 45)
            part[sort_by] = part[sort_by].map(
                lambda v: "" if pd.isna(v) else f"{v:.3g}"
            )
            t = ax.table(
                cellText=part.astype(str).values,
                colLabels=cols,
                loc="upper left",
            )
            t.auto_set_font_size(False)
            t.set_fontsize(6)
            fig.suptitle("Video screen: index", fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)
        for r in table.to_dict("records"):
            card = (
                out_dir
                / r["session"]
                / vs.CARD_FILE.format(camera=r["camera"])
            )
            if not card.exists():
                continue
            fig = plt.figure(figsize=vqr.PAGE_SIZE)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(plt.imread(card))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {path}")


# Level metrics for the threshold pages: (session column, per-sample
# column, label). Session values are medians over the session's samples.
THRESHOLD_METRICS = [
    ("contrast_rms", "RMS contrast (std / mean)"),
    ("p1", "black level (p1 luma)"),
    ("mean", "mean luma"),
    ("sharpness", "sharpness (Laplacian variance, 2x down)"),
    ("noise_sigma", "noise sigma"),
    ("pct_clipped_high", "% of pixels clipped at the ceiling"),
    ("dynamic_range", "dynamic range (p99 - p1)"),
    ("entropy_bits", "entropy (bits)"),
]
THRESHOLD_PERCENTILES = [0, 1, 3, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def _session_table(out_dir):
    """One row per camera with quality measured: session medians of every
    threshold metric and the per-sample table kept for frame choice."""
    screen = vs.load_screen(out_dir / vs.SCREEN_FILE)
    table = []
    for r in screen[screen["quality"].notna()].to_dict("records"):
        path = (
            out_dir
            / r["session"]
            / vqq.SAMPLES_FILE.format(camera=r["camera"])
        )
        if not path.exists():
            continue
        samples = pd.read_parquet(
            path, columns=["frame_index"] + [m for m, _ in THRESHOLD_METRICS]
        )
        table.append(
            {
                "session": r["session"],
                "camera": r["camera"],
                "view": r["view"],
                "mp4": r["mp4"],
                "samples": samples,
                **{m: samples[m].median() for m, _ in THRESHOLD_METRICS},
            }
        )
    return pd.DataFrame(table)


def _fetch_frame(mp4, frame_index, cache):
    """Full-resolution luma of one keyframe, cached on disk."""
    name = urllib.parse.urlparse(mp4).path.strip("/").replace("/", "__")
    path = cache / f"{name}__{frame_index}.npy"
    if path.exists():
        return np.load(path)
    window = (frame_index, frame_index + 1)
    luma = vqq.sample_keyframes(mp4, window)[0][0]
    np.save(path, luma)
    return luma


def _picks(group, metric):
    """Sessions at THRESHOLD_PERCENTILES of ``metric``, unique, sorted,
    each with the keyframe closest to its own median."""
    ordered = group.sort_values(metric).reset_index(drop=True)
    ranks = sorted(
        {
            int(round(q / 100 * (len(ordered) - 1)))
            for q in THRESHOLD_PERCENTILES
        }
    )
    picks = []
    for rank in ranks:
        r = ordered.iloc[rank]
        s = r["samples"]
        i = int((s[metric] - r[metric]).abs().idxmin())
        picks.append((rank, r, int(s["frame_index"][i])))
    return picks


def _threshold_page(group, metric, label, view, frames):
    """Distribution strip plus the picked frames, numbered in order."""
    picks = _picks(group, metric)
    fig = plt.figure(figsize=vqr.PAGE_SIZE, facecolor="white")
    fig.text(
        0.01,
        0.985,
        f"{view} camera: {label}   (n = {len(group)} sessions; "
        "frames are each marked session's most typical keyframe)",
        fontsize=11,
        va="top",
    )
    ax = fig.add_axes([0.05, 0.80, 0.92, 0.13])
    values = group[metric].to_numpy()
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(values))
    ax.scatter(values, jitter, s=8, color="#898781", alpha=0.5, lw=0)
    for k, (_, r, _) in enumerate(picks, 1):
        ax.scatter([r[metric]], [0], s=60, color="#2a78d6", zorder=3, lw=0)
        ax.annotate(
            str(k),
            (r[metric], 0),
            xytext=(0, 9 if k % 2 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#0b0b0b",
        )
    ax.set_yticks([])
    ax.set_ylim(-0.7, 0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=8, colors="#52514e")
    grid = fig.add_gridspec(
        3,
        4,
        left=0.01,
        right=0.99,
        top=0.74,
        bottom=0.01,
        wspace=0.02,
        hspace=0.14,
    )
    for k, (rank, r, frame_index) in enumerate(picks):
        ax = fig.add_subplot(grid[k // 4, k % 4])
        luma = frames[(r["mp4"], frame_index)]
        if metric == "pct_clipped_high":
            gray = np.clip((luma - 16) / 219, 0, 1)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            rgb[luma >= 235] = (0.82, 0.23, 0.23)
            ax.imshow(rgb)
        else:
            ax.imshow(luma, cmap="gray", vmin=16, vmax=235)
        ax.set_xticks([])
        ax.set_yticks([])
        pct = 100 * rank / max(len(group) - 1, 1)
        ax.set_title(
            f"#{k + 1}  {r[metric]:.3g}  (p{pct:.0f})   "
            f"{r['session'][9:26]}",
            fontsize=8,
            pad=2,
        )
    return fig


def threshold_pages(out_dir, workers=8):
    """Write ``video_quality_thresholds.pdf`` (see the module docstring)."""
    out_dir = Path(out_dir)
    table = _session_table(out_dir)
    cache = out_dir / "threshold_frames"
    cache.mkdir(exist_ok=True)
    needed = {
        (r["mp4"], frame_index)
        for view, group in table.groupby("view")
        for metric, _ in THRESHOLD_METRICS
        for _, r, frame_index in _picks(group, metric)
    }
    print(f"{len(table)} cameras; fetching {len(needed)} frames")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_frame, mp4, fi, cache): (mp4, fi)
            for mp4, fi in needed
        }
        frames = {futures[f]: f.result() for f in as_completed(futures)}
    path = out_dir / "video_quality_thresholds.pdf"
    with PdfPages(path) as pdf:
        for metric, label in THRESHOLD_METRICS:
            for view, group in table.groupby("view"):
                fig = _threshold_page(group, metric, label, view, frames)
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
    print(f"wrote {path}")


def main():
    """Command line entry point."""
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", help="CSV with a raw_session column")
    parser.add_argument("out", help="output folder")
    parser.add_argument("--filter-column", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--thresholds", action="store_true")
    args = parser.parse_args()
    if args.report:
        report(args.out)
        return
    if args.thresholds:
        threshold_pages(args.out, args.workers)
        return
    table = pd.read_csv(args.sessions)
    if args.filter_column:
        table = table[table[args.filter_column].astype(str).eq("True")]
    sessions = list(dict.fromkeys(table["raw_session"]))[: args.limit]
    survey(sessions, args.out, args.workers)
    print(json.dumps({"screen": str(Path(args.out) / vs.SCREEN_FILE)}))


if __name__ == "__main__":
    main()
