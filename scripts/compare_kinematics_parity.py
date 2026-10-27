"""Check the new kinematics core against the pre-0.5.0 outputs of a session.

Usage::

    python scripts/compare_kinematics_parity.py <tongue_kins.parquet> \
        <all_tongue_movements.parquet> [session]

``tongue_kins.parquet`` is one session's old frame table (filtered ``x, y,
v, xv, yv``, ``time_in_session``, ``movement_id``); the pooled movements
table holds the same session's old per-movement rows. The session is found
by matching movement start times when not given.

Checks:

1. ``segment.tracked_runs(valid, 10)`` gives the old movements exactly.
2. ``features.describe`` on the old filtered frames gives the old feature
   values, with the jaw position recovered from the old columns.

Smoothing is not compared: the old frame table holds only filtered
positions, and the new smoothing deliberately differs at run edges.
"""
import sys

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics import (
    features,
    segment,
)


def find_session(kins, movs):
    """Session whose movement start times match the frame table."""
    st = np.sort(kins.groupby("movement_id")["time_in_session"].min()
                 .to_numpy())
    for s, g in movs.groupby("session"):
        g = np.sort(g["start_time"].to_numpy())
        if len(g) == len(st) and np.allclose(g, st, atol=1e-9):
            return s
    raise SystemExit("no session in the movements table matches")


def recover_jaw(old):
    """Session jaw position implied by the old ``max_*`` columns.

    ``max_x_distance = |max_x_from_jaw - jaw_x|`` (same for y), so each
    movement gives two candidates per axis; the jaw is the value they share.
    """
    out = []
    for at, dist in (("max_x_from_jaw", "max_x_distance"),
                     ("max_y_from_jaw", "max_y_distance")):
        cand = np.r_[old[at] - old[dist], old[at] + old[dist]]
        vals, counts = np.unique(np.round(cand, 6), return_counts=True)
        out.append(float(vals[np.argmax(counts)]))
    return tuple(out)


def compare(name, old, new, atol=1e-9):
    """Print and return whether two columns agree."""
    old = pd.Series(old, dtype=float).to_numpy()
    new = pd.Series(new, dtype=float).to_numpy()
    both_nan = np.isnan(old) & np.isnan(new)
    close = np.isclose(old, new, atol=atol, rtol=1e-9) | both_nan
    worst = np.nanmax(np.abs(old - new)) if (~both_nan).any() else 0.0
    print(f"  {name:<28} {close.mean():8.4%} agree   max |diff| {worst:.3g}")
    return bool(close.all())


def main(kins_path, movs_path, session=None):
    """Run both checks; exit 1 if any fails."""
    kins = pd.read_parquet(kins_path).reset_index(drop=True)
    movs = pd.read_parquet(movs_path)
    session = session or find_session(kins, movs)
    old = (movs[movs["session"] == session].sort_values("start_time")
           .reset_index(drop=True))
    print(f"session {session}: {len(kins)} frames, {len(old)} movements")

    valid = kins["x"].notna().to_numpy()
    iv = segment.tracked_runs(valid, max_gap_frames=10)
    rows = np.arange(len(kins))
    mid = kins["movement_id"]
    old_iv = (pd.DataFrame({"m": mid, "r": rows}).dropna()
              .groupby("m")["r"].agg(["min", "max"]).sort_values("min"))
    same = (len(iv) == len(old_iv)
            and (iv["start"].to_numpy() == old_iv["min"].to_numpy()).all()
            and (iv["end"].to_numpy() == old_iv["max"].to_numpy()).all())
    print(f"1. segmentation: {len(iv)} new vs {len(old_iv)} old movements, "
          f"{'identical' if same else 'DIFFERENT'}")

    jaw = recover_jaw(old)
    print(f"2. features (jaw recovered at {jaw[0]:.3f}, {jaw[1]:.3f})")
    frames = pd.DataFrame({
        "time": kins["time_in_session"], "x": kins["x"], "y": kins["y"],
        "vx": kins["xv"], "vy": kins["yv"], "speed": kins["v"],
        "valid": valid,
    })
    new = features.describe(frames, iv, ("path", "excursion", "outbound"),
                            {"fs_hz": 500.0, "origin": jaw})
    t = frames["time"].to_numpy()
    s, e = iv["start"].to_numpy(), iv["end"].to_numpy()
    nvalid = np.r_[0, np.cumsum(valid)]
    n_valid = nvalid[e + 1] - nvalid[s]
    ok = [
        compare("duration", old["duration"], t[e] - t[s]),
        compare("n_datapoints / n_valid", old["n_datapoints"], n_valid),
        compare("dropped_frames_n", old["dropped_frames_n"],
                (e - s + 1) - n_valid),
        compare("total_distance / path_length", old["total_distance"],
                new["path_length"]),
        compare("peak_velocity / peak_speed", old["peak_velocity"],
                new["peak_speed"]),
        compare("mean_velocity / mean_speed", old["mean_velocity"],
                new["mean_speed"]),
        compare("startpoint_x / start_x", old["startpoint_x"] - jaw[0],
                new["start_x"], 1e-6),
        compare("startpoint_y / start_y", old["startpoint_y"] - jaw[1],
                new["start_y"], 1e-6),
        compare("endpoint_x / end_x", old["endpoint_x"] - jaw[0],
                new["end_x"], 1e-6),
        compare("endpoint_y / end_y", old["endpoint_y"] - jaw[1],
                new["end_y"], 1e-6),
        # The jaw is recovered to 1e-6 px, which moves angles by ~1e-6 deg.
        compare("excursion_angle_deg", old["excursion_angle_deg"],
                new["angle_deg"], 1e-4),
        compare("max_y_distance / |extent_y|", old["max_y_distance"],
                new["extent_y"].abs(), 1e-6),
        compare("time_to_endpoint / out_dur.", old["time_to_endpoint"],
                new["out_duration"]),
        compare("out_duration", old["out_duration"], new["out_duration"]),
        compare("out_total_distance", old["out_total_distance"],
                new["out_path_length"], 1e-6),
        compare("out_peak_velocity", old["out_peak_velocity"],
                new["out_peak_speed"]),
        compare("out_mean_velocity", old["out_mean_velocity"],
                new["out_mean_speed"]),
    ]
    # Old max_x_distance is |largest signed x - jaw|; equal to |extent_x|
    # when the farthest x is on the positive side.
    pos = new["extent_x"] >= 0
    ok.append(compare("max_x_distance / |extent_x|*",
                      old.loc[pos, "max_x_distance"],
                      new.loc[pos, "extent_x"], 1e-6))
    print(f"  * {pos.mean():.2%} of movements reach farthest on +x")
    if not (same and all(ok)):
        sys.exit(1)
    print("all checks agree")


if __name__ == "__main__":
    main(*sys.argv[1:])
