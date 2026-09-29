"""Compare two run_batch_analysis output folders, file by file.

Use this to see what a library change does to the tongue-kinematics
pipeline. Run the batch twice on the same sessions, once with each version
of the library, into two folders:

    # env A: pip install "git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis@main"      # noqa: E501
    # env B: pip install -e .   (this branch)
    from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import run_batch_analysis  # noqa: E501
    run_batch_analysis(pred_csvs, data_root, "out_old", extract_clips=False)
    run_batch_analysis(pred_csvs, data_root, "out_new", extract_clips=False)

then:

    python scripts/compare_batch_outputs.py out_old out_new

It prints one line per session and parquet file in
``<session>/intermediate_data/``: ``identical``, or what differs (row
count, columns, and the numeric columns with the largest differences).
Sessions that ran in only one folder, and each folder's
``batch_error_log.txt``, are listed at the end. ``--csv PATH`` also saves
the table.

For the video timing QC change, expect ``identical`` for ok sessions, small
differences in time columns for Harp-glitch sessions, and large ones
(time and everything matched by time: trials, licks) for dropped-frame
sessions.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare_frames(old, new, top=3):
    """Return a short description of how ``new`` differs from ``old``."""
    notes = []
    if len(old) != len(new):
        notes.append(f"rows {len(old)} -> {len(new)}")
    added = sorted(set(new.columns) - set(old.columns))
    removed = sorted(set(old.columns) - set(new.columns))
    if added:
        notes.append(f"added {added}")
    if removed:
        notes.append(f"removed {removed}")
    n = min(len(old), len(new))
    diffs = {}
    for column in old.columns.intersection(new.columns):
        a = old[column].iloc[:n].reset_index(drop=True)
        b = new[column].iloc[:n].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(
            b
        ):
            a = a.to_numpy(dtype="float64")
            b = b.to_numpy(dtype="float64")
            nan_mismatch = np.isnan(a) != np.isnan(b)
            delta = np.abs(a - b)
            delta[np.isnan(delta)] = 0
            changed = int((delta > 0).sum() + nan_mismatch.sum())
            if changed:
                diffs[column] = (float(delta.max()), changed)
        elif not a.astype(str).equals(b.astype(str)):
            diffs[column] = (
                np.nan,
                int((a.astype(str) != b.astype(str)).sum()),
            )
    if diffs:
        ranked = sorted(diffs.items(), key=lambda kv: -kv[1][1])[:top]
        notes.append(
            f"{len(diffs)} columns differ, e.g. "
            + ", ".join(
                f"{c} ({n_changed} rows"
                + ("" if np.isnan(d) else f", max {d:.4g}")
                + ")"
                for c, (d, n_changed) in ranked
            )
        )
    return "; ".join(notes) or "identical"


def compare_folders(old_root, new_root):
    """Compare every session's intermediate parquet files."""
    old_root, new_root = Path(old_root), Path(new_root)
    rows = []
    for old_file in sorted(old_root.glob("*/intermediate_data/*.parquet")):
        relative = old_file.relative_to(old_root)
        new_file = new_root / relative
        if not new_file.exists():
            result = "missing in new"
        else:
            result = compare_frames(
                pd.read_parquet(old_file), pd.read_parquet(new_file)
            )
        rows.append(
            {
                "session": relative.parts[0],
                "file": old_file.name,
                "result": result,
            }
        )
    for new_file in sorted(new_root.glob("*/intermediate_data/*.parquet")):
        if not (old_root / new_file.relative_to(new_root)).exists():
            rows.append(
                {
                    "session": new_file.relative_to(new_root).parts[0],
                    "file": new_file.name,
                    "result": "missing in old",
                }
            )
    return pd.DataFrame(rows, columns=["session", "file", "result"])


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("old_root")
    parser.add_argument("new_root")
    parser.add_argument("--csv", help="also save the table here")
    args = parser.parse_args()

    table = compare_folders(args.old_root, args.new_root)
    with pd.option_context("display.max_colwidth", None):
        for session, group in table.groupby("session"):
            print(f"\n{session}")
            for _, row in group.iterrows():
                print(f"  {row['file']:<28} {row['result']}")
    for root in (args.old_root, args.new_root):
        log = Path(root) / "batch_error_log.txt"
        if log.exists():
            print(f"\nErrors in {root}:\n{log.read_text()}")
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
