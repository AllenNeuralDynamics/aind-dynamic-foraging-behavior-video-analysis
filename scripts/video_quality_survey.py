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
and every session card, excluded cameras first.

Usage::

    python scripts/video_quality_survey.py sessions.csv out/ --workers 8
    python scripts/video_quality_survey.py sessions.csv out/ --report

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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    args = parser.parse_args()
    if args.report:
        report(args.out)
        return
    table = pd.read_csv(args.sessions)
    if args.filter_column:
        table = table[table[args.filter_column].astype(str).eq("True")]
    sessions = list(dict.fromkeys(table["raw_session"]))[: args.limit]
    survey(sessions, args.out, args.workers, args.samples)
    print(json.dumps({"summary": str(Path(args.out) / "summary.csv")}))


if __name__ == "__main__":
    main()
