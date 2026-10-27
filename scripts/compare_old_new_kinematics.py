"""Compare pre-0.5.0 tongue outputs with 0.5.0 outputs, session by session.

Run the old batch (``run_batch_analysis``, library <= 0.4.0) and the new one
(``tasks.dynamic_foraging_tongue.run_batch``) on the same sessions, then::

    python scripts/compare_old_new_kinematics.py <old_root> <new_root> \
        [--csv PATH]

``old_root/<session>/intermediate_data/`` holds ``tongue_movs.parquet``,
``nwb_df_licks.parquet``; ``old_root/<session>/tongue_quality_stats.json``.
``new_root/<session>/`` holds ``movements.parquet``, ``licks.parquet``,
``landmarks.parquet``, ``record.json``.

Per session it prints: movements in each and how many match by start and end
time; for each mapped feature, the median and 99th percentile relative
difference (pairs where the old value is 0 are counted separately); the
angle difference in degrees; the jaw reference in each; per-lick agreement
of the matched movement; and lick coverage in each.

Expect identical movements and lick matches, and feature differences from
the new smoothing (per tracked run: interior frames unchanged, edge frames
differ) and reference point (median of confident jaw points, was the mean
of all jaw points: a few px).
Old absolute positions are compared with new relative ones plus the new jaw.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# old column -> new column, or a function of (new movements, jaw (x, y)).
FEATURES = {
    "duration": "duration",
    "n_datapoints": "n_valid",
    "total_distance": "path_length",
    "peak_velocity": "peak_speed",
    "mean_velocity": "mean_speed",
    "out_duration": "out_duration",
    "out_total_distance": "out_path_length",
    "out_peak_velocity": "out_peak_speed",
    "out_mean_velocity": "out_mean_speed",
    "endpoint_x": lambda m, jaw: m["end_x"] + jaw[0],
    "endpoint_y": lambda m, jaw: m["end_y"] + jaw[1],
    "lick_time": "lick_time",
}


def relative_difference(old, new):
    """``|new - old| / |old|`` where both are finite and ``old != 0``, and
    how many pairs differ although ``old`` is 0."""
    old = pd.Series(old, dtype=float).to_numpy()
    new = pd.Series(new, dtype=float).to_numpy()
    ok = np.isfinite(old) & np.isfinite(new)
    zero = ok & (old == 0)
    n_zero_differ = int((zero & (new != 0)).sum())
    keep = ok & ~zero
    return np.abs(new[keep] - old[keep]) / np.abs(old[keep]), n_zero_differ


def _note(rel, n_zero_differ):
    """Median and 99th percentile of relative differences."""
    if not len(rel):
        return "no values"
    note = f"median {np.median(rel):.2%}, p99 {np.percentile(rel, 99):.2%}"
    if n_zero_differ:
        note += f"; {n_zero_differ} differ where old is 0"
    return note


def compare_session(old_dir, new_dir, tol_s=1e-6):
    """One session's comparison as a list of ``{item, old, new, note}``."""
    old_dir, new_dir = Path(old_dir), Path(new_dir)
    inter = old_dir / "intermediate_data"
    old = (pd.read_parquet(inter / "tongue_movs.parquet")
           .sort_values("start_time").reset_index(drop=True))
    new = pd.read_parquet(new_dir / "movements.parquet")
    new = new[new["keypoint"] == new["keypoint"].iloc[0]] if len(new) else new
    new = new.sort_values("start_time").reset_index(drop=True)
    rows = [{"item": "movements", "old": len(old), "new": len(new),
             "note": ""}]

    # Match movements by start and end time.
    pos = np.searchsorted(new["start_time"].to_numpy(),
                          old["start_time"].to_numpy() - tol_s)
    pos = np.minimum(pos, max(len(new) - 1, 0))
    same = (len(new) > 0) & (
        np.abs(new["start_time"].to_numpy()[pos] - old["start_time"]) <= tol_s
    ) & (np.abs(new["end_time"].to_numpy()[pos] - old["end_time"]) <= tol_s)
    same = np.asarray(same, bool)
    rows.append({"item": "movements matched", "old": int(same.sum()),
                 "new": int(same.sum()),
                 "note": f"{same.mean():.2%} of old" if len(old) else ""})
    o, n = old[same].reset_index(drop=True), new.iloc[pos[same]].reset_index(
        drop=True)

    lm = pd.read_parquet(new_dir / "landmarks.parquet").set_index("keypoint")
    jaw = (float(lm.loc["jaw", "x"]), float(lm.loc["jaw", "y"]))
    for old_col, new_col in FEATURES.items():
        if old_col not in o:
            continue
        new_vals = new_col(n, jaw) if callable(new_col) else n.get(new_col)
        if new_vals is None:
            continue
        rows.append({"item": old_col, "old": old_col,
                     "new": new_col if isinstance(new_col, str) else "derived",
                     "note": _note(*relative_difference(o[old_col],
                                                        new_vals))})
    if "excursion_angle_deg" in o and "angle_deg" in n:
        d = np.abs((n["angle_deg"] - o["excursion_angle_deg"] + 180) % 360
                   - 180).dropna()
        rows.append({"item": "angle |new - old| (deg)",
                     "old": "excursion_angle_deg", "new": "angle_deg",
                     "note": f"median {d.median():.1f}, p90 "
                             f"{d.quantile(0.9):.1f}" if len(d) else ""})
    old_jaw = inter / "kps_raw_jaw.parquet"
    if old_jaw.exists():
        oj = pd.read_parquet(old_jaw)[["x", "y"]].mean()
        rows.append({"item": "jaw (old mean -> new median)",
                     "old": f"({oj['x']:.1f}, {oj['y']:.1f})",
                     "new": f"({jaw[0]:.1f}, {jaw[1]:.1f})",
                     "note": "shift {:.1f} px".format(
                         np.hypot(oj["x"] - jaw[0], oj["y"] - jaw[1]))})
    if "has_lick" in o:
        agree = (o["has_lick"].astype(bool).to_numpy()
                 == (n["n_licks"] > 0).to_numpy())
        rows.append({"item": "has_lick", "old": int(o["has_lick"].sum()),
                     "new": int((n["n_licks"] > 0).sum()),
                     "note": f"{agree.mean():.2%} agree"})

    old_licks = pd.read_parquet(inter / "nwb_df_licks.parquet")
    new_licks = pd.read_parquet(new_dir / "licks.parquet")
    if len(old_licks) == len(new_licks):
        # Movement ids are 1-based in time order in both, so equal
        # movements have equal ids.
        same = (old_licks["nearest_movement_id"].astype("Float64")
                .fillna(-1).to_numpy()
                == new_licks["movement_id"].astype("Float64").fillna(-1)
                .to_numpy())
        rows.append({"item": "lick -> movement", "old": "nearest_movement_id",
                     "new": "movement_id",
                     "note": f"{same.mean():.2%} of licks agree"})
    rows.append({
        "item": "lick coverage",
        "old": round(float(old_licks["nearest_movement_id"].notna().mean()),
                     4),
        "new": round(float(new_licks["movement_id"].notna().mean()), 4),
        "note": "",
    })
    stats = old_dir / "tongue_quality_stats.json"
    if stats.exists():
        old_cov = json.loads(stats.read_text()).get("coverage_pct", np.nan)
        rec = json.loads((new_dir / "record.json").read_text())
        rows.append({"item": "quality lick_coverage",
                     "old": round(old_cov / 100, 4),
                     "new": round(rec["quality"]["lick_coverage"], 4),
                     "note": ""})
    return rows


def compare_roots(old_root, new_root):
    """Every session present in both roots, as one table."""
    old_root, new_root = Path(old_root), Path(new_root)
    out = []
    old_sessions = {p.parent.parent.name for p in
                    old_root.glob("*/intermediate_data/tongue_movs.parquet")}
    new_sessions = {p.parent.name for p in new_root.glob("*/record.json")}
    for s in sorted(old_sessions | new_sessions):
        if s not in old_sessions or s not in new_sessions:
            where = "old" if s in old_sessions else "new"
            out.append({"session": s, "item": f"only in {where}", "old": "",
                        "new": "", "note": ""})
            continue
        for row in compare_session(old_root / s, new_root / s):
            out.append({"session": s, **row})
    return pd.DataFrame(out, columns=["session", "item", "old", "new",
                                      "note"])


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("old_root")
    parser.add_argument("new_root")
    parser.add_argument("--csv", help="also save the table here")
    args = parser.parse_args()
    table = compare_roots(args.old_root, args.new_root)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        for session, group in table.groupby("session", sort=False):
            print(f"\n{session}")
            print(group.drop(columns="session").to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
