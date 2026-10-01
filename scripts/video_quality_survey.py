"""Run video quality QC on many sessions from s3://aind-open-data.

Phase 2 of ``VIDEO_QUALITY_QC_PLAN.md``. For each raw session asset:
list its ``behavior-videos`` MP4s (either layout), download each camera's
video CSV and the camera trigger log to a temporary folder (the CSV reader
needs local files), find the task window from the session JSON, measure
the MP4 over HTTPS, and write, per camera, into ``<out>/<session>/``:

- ``video_quality_<camera>.json`` and ``..._samples.parquet``
  (``write_video_quality``);
- ``session_card_<camera>.png``;
- ``thumbnails_<camera>.npz``: 8 evenly spaced frames, for contact sheets.

One row per camera is appended to ``<out>/summary.jsonl`` as each session
finishes, so an interrupted run resumes where it stopped; ``summary.csv``
is rebuilt from it at the end. ``--report``
then builds ``<out>/video_quality_survey.pdf``: an index, contact sheets,
and every session card, excluded cameras first. ``--thresholds`` builds
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
import shutil
import socket
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from aind_dynamic_foraging_behavior_video_analysis import (  # noqa: E402
    video_quality_qc as vqq,
)
from aind_dynamic_foraging_behavior_video_analysis import (  # noqa: E402
    video_quality_report as vqr,
)

BUCKET = "https://aind-open-data.s3.us-west-2.amazonaws.com"
# Downloads (CSV, trigger log, listings) give up on a stalled read instead
# of hanging a worker; the camera then records an error and can be re-run.
socket.setdefaulttimeout(120)
N_THUMBNAILS = 8


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
        rel = key[len(session) + 1 :]
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


def download(key, folder):
    """Download one object into ``folder``; return the local path."""
    path = Path(folder) / Path(key).name
    urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", path)
    return path


def run_session(session, out_dir, n_samples):
    """Measure every camera of one session; return one row per camera."""
    rows = []
    session_out = Path(out_dir) / session
    session_out.mkdir(parents=True, exist_ok=True)
    try:
        cameras, behavior_json, log_key = session_files(session)
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        return [{"session": session, "action": "error", "error": repr(e)}]
    if not cameras:
        return [{"session": session, "action": "no mp4", "error": ""}]
    tmp = Path(tempfile.mkdtemp(prefix="vqq_"))
    try:
        trigger_log = download(log_key, tmp) if log_key else None
        for camera, mp4_key, csv_key in cameras:
            rows.append(
                run_camera(
                    session,
                    camera,
                    mp4_key,
                    csv_key,
                    behavior_json,
                    trigger_log,
                    tmp,
                    session_out,
                    n_samples,
                )
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def _task_window(behavior_json, csv_key, trigger_log, tmp):
    """(frame_window, note), falling back to the whole file."""
    if behavior_json is None or csv_key is None:
        return vqq.task_window_or_whole_file(behavior_json, None)
    csv = download(csv_key, tmp)
    try:
        return vqq.task_window_or_whole_file(
            f"{BUCKET}/{urllib.parse.quote(behavior_json)}", csv, trigger_log
        )
    finally:
        csv.unlink()


def run_camera(
    session,
    camera,
    mp4_key,
    csv_key,
    behavior_json,
    trigger_log,
    tmp,
    session_out,
    n_samples,
):
    """Measure, check and write one camera; return its summary row."""
    row = {"session": session, "camera": camera, "mp4": mp4_key}
    start = time.perf_counter()
    try:
        window, note = _task_window(behavior_json, csv_key, trigger_log, tmp)
        row["window"] = note
        qc = vqq.measure_video_quality(
            f"{BUCKET}/{urllib.parse.quote(mp4_key)}",
            n_samples=n_samples,
            frame_window=window,
        )
        checks = vqq.check_video_quality(qc, camera=camera)
        vqq.write_video_quality(qc, checks, session_out, camera=camera)
        fig = vqr.session_card(qc, checks, f"{session}  {camera}")
        fig.savefig(session_out / f"session_card_{camera}.png", dpi=90)
        plt.close(fig)
        picks = vqr._even_picks(len(qc.samples), N_THUMBNAILS)
        np.savez_compressed(
            session_out / f"thumbnails_{camera}.npz",
            thumbnails=qc.thumbnails[picks][:, ::2, ::2],
            video_time=qc.samples["video_time"].to_numpy()[picks],
        )
        s = qc.samples
        row.update(
            {
                "action": vqq.quality_action(checks),
                "failed_checks": ";".join(
                    checks.loc[checks["passed"].eq(False), "check"]
                ),
                "max_shift": s["shift"].max(),
                "max_edges_shifted": int(s["edges_shifted"].max()),
                "samples_3plus_edges": int((s["edges_shifted"] >= 3).sum()),
                **{f"edge_{e}_max": s[f"edge_{e}"].max() for e in vqq.EDGES},
                **{
                    f"edge_{e}_peak_med": s[f"edge_{e}_peak"].median()
                    for e in vqq.EDGES
                },
                **asdict(qc.summary),
                "error": "",
            }
        )
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        row.update(
            {
                "action": "error",
                "error": f"{e!r}\n{traceback.format_exc(limit=3)}",
            }
        )
    row["seconds"] = round(time.perf_counter() - start, 1)
    return row


def survey(sessions, out_dir, workers, n_samples):
    """Run every session not already in ``summary.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.jsonl"
    done = set()
    if summary.exists():
        done = {json.loads(line)["session"] for line in summary.open()}
    todo = [s for s in sessions if s not in done]
    print(f"{len(sessions)} sessions, {len(done)} done, {len(todo)} to run")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_session, s, out_dir, n_samples): s for s in todo
        }
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            with summary.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r, default=_json_default) + "\n")
            actions = ", ".join(
                f"{r.get('camera', '-')}: {r['action']}" for r in rows
            )
            elapsed = time.perf_counter() - start
            print(
                f"[{i}/{len(todo)} {elapsed / 60:.1f} min] "
                f"{futures[future]}: {actions}",
                flush=True,
            )
    rows = [json.loads(line) for line in summary.open()]
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


def _json_default(value):
    """numpy scalars to Python."""
    return value.item() if hasattr(value, "item") else str(value)


def _thumb_row(fig, grid, row, label, action, npz):
    """One contact-sheet row from saved thumbnails."""
    ax = fig.add_subplot(grid[row, 0])
    ax.axis("off")
    ax.text(0, 0.65, label, fontsize=6, color=vqr.INK, va="center")
    ax.text(
        0,
        0.3,
        action,
        fontsize=7,
        weight="bold",
        color=vqr.PASS if action == "use" else vqr.FAIL,
        va="center",
    )
    if npz is None:
        return
    data = np.load(npz)
    for col, (thumb, t) in enumerate(
        zip(data["thumbnails"], data["video_time"])
    ):
        ax = fig.add_subplot(grid[row, col + 1])
        ax.imshow(thumb, cmap="gray", vmin=16, vmax=235)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(vqr._mmss(t), fontsize=5, pad=1)


def report(out_dir, sort_by="sharpness_med"):
    """Index, contact sheets and cards for every camera in the survey."""
    out_dir = Path(out_dir)
    table = pd.read_csv(out_dir / "summary.csv")
    table["use"] = table["action"].eq("use")
    table = table.sort_values(["use", "camera", sort_by], na_position="first")
    path = out_dir / "video_quality_survey.pdf"
    with PdfPages(path) as pdf:
        cols = ["session", "camera", "action", "window", sort_by]
        for start in range(0, len(table), 45):
            fig = plt.figure(figsize=vqr.PAGE_SIZE)
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.95])
            ax.axis("off")
            part = table.iloc[start : start + 45][cols].copy()
            part["window"] = part["window"].fillna("").str.slice(0, 40)
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
            fig.suptitle("Video quality survey: index", fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)
        rows = table.to_dict("records")
        per_page = vqr.CONTACT_ROWS
        for start in range(0, len(rows), per_page):
            fig = plt.figure(figsize=vqr.PAGE_SIZE)
            grid = fig.add_gridspec(
                per_page,
                N_THUMBNAILS + 1,
                left=0.01,
                right=0.99,
                top=0.97,
                bottom=0.02,
                width_ratios=[1.6] + [1] * N_THUMBNAILS,
                wspace=0.03,
                hspace=0.25,
            )
            for i, r in enumerate(rows[start : start + per_page]):
                npz = out_dir / r["session"] / f"thumbnails_{r['camera']}.npz"
                _thumb_row(
                    fig,
                    grid,
                    i,
                    f"{r['session']}\n{r['camera']}",
                    r["action"],
                    npz if npz.exists() else None,
                )
            pdf.savefig(fig)
            plt.close(fig)
        for r in rows:
            card = out_dir / r["session"] / f"session_card_{r['camera']}.png"
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


def _view(camera):
    """``bottom`` or ``side`` for either folder layout's camera name."""
    return "bottom" if camera.lower().startswith("bottom") else "side"


def _session_table(out_dir):
    """One row per measured camera with session medians of every
    threshold metric and the per-sample table kept for frame choice."""
    rows = [json.loads(line) for line in (out_dir / "summary.jsonl").open()]
    table = []
    for r in rows:
        if r["action"] in ("error", "no mp4"):
            continue
        p = (
            out_dir
            / r["session"]
            / f"video_quality_{r['camera']}_samples.parquet"
        )
        if not p.exists():
            continue
        samples = pd.read_parquet(
            p, columns=["frame_index"] + [m for m, _ in THRESHOLD_METRICS]
        )
        table.append(
            {
                "session": r["session"],
                "camera": r["camera"],
                "view": _view(r["camera"]),
                "mp4": r["mp4"],
                "samples": samples,
                **{m: samples[m].median() for m, _ in THRESHOLD_METRICS},
            }
        )
    return pd.DataFrame(table)


def _fetch_frame(mp4_key, frame_index, cache):
    """Full-resolution luma of one keyframe, cached on disk."""
    path = cache / f"{mp4_key.replace('/', '__')}__{frame_index}.npy"
    if path.exists():
        return np.load(path)
    url = f"{BUCKET}/{urllib.parse.quote(mp4_key)}"
    index = vqq.read_mp4_frame_index(url)
    luma = vqq.read_keyframes(url, index, [frame_index])[0][1]
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", help="CSV with a raw_session column")
    parser.add_argument("out", help="output folder")
    parser.add_argument("--filter-column", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
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
    survey(sessions, args.out, args.workers, args.samples)
    print(json.dumps({"summary": str(Path(args.out) / "summary.csv")}))


if __name__ == "__main__":
    main()
