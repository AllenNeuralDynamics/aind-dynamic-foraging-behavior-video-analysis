"""Smoothing before and after 0.5.0 on simulated licks with known velocity.

Before 0.5.0 (``kinematics_filter``): cubic interpolation across every
gap in the session, then one Butterworth pass. From 0.5.0
(``signal.smooth``): each tracked run filtered on its own. Prints the
median speed error on the first and last 3 frames of each lick and on the
rest, and the bias of each lick's peak speed, at three noise levels.

Usage, with the old module from a checkout of tag ``v0.4.0``::

    git worktree add ../old v0.4.0
    python scripts/simulate_smoothing_edges.py ../old/src/\
aind_dynamic_foraging_behavior_video_analysis/kinematics/\
tongue_kinematics_utils.py

Simulated licks: 60-100 ms out and back, 40-80 px, 40-120 ms apart, visible
only beyond 12 px from the jaw (so tracking starts with the tongue already
moving), 3% of frames dropped, 500 Hz, five 20 s sessions per noise level.
"""

import importlib.util
import sys

import numpy as np
import pandas as pd

from aind_dynamic_foraging_behavior_video_analysis.kinematics import signal

spec = importlib.util.spec_from_file_location("old", sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
FS = 500.0


def simulate(seed, noise):
    rng = np.random.default_rng(seed)
    n = int(20 * FS)
    t = np.arange(n) / FS
    x = np.full(n, 300.0)
    y = np.full(n, 300.0)
    vx = np.zeros(n)
    vy = np.zeros(n)
    vis = np.zeros(n, bool)
    lick = np.full(n, -1)
    t0, k = 0.1, 0
    while t0 < 19.8:
        T = rng.uniform(0.06, 0.10)
        A = rng.uniform(40, 80)
        th = rng.normal(0, 0.3)
        m = (t >= t0) & (t < t0 + T)
        u = (t[m] - t0) / T
        r = A * np.sin(np.pi * u)
        dr = A * np.pi / T * np.cos(np.pi * u)
        x[m] = 300 + r * np.cos(th)
        y[m] = 300 + r * np.sin(th)
        vx[m] = dr * np.cos(th)
        vy[m] = dr * np.sin(th)
        v = m.copy()
        v[m] = r > 12  # tongue visible once out of the mouth
        vis |= v
        lick[v] = k
        t0 += T + rng.uniform(0.04, 0.12)
        k += 1
    drop = rng.random(n) < 0.03  # occasional low-confidence frames
    valid = vis & ~drop
    xo = np.where(valid, x + rng.normal(0, noise, n), np.nan)
    yo = np.where(valid, y + rng.normal(0, noise, n), np.nan)
    return t, xo, yo, valid, np.hypot(vx, vy), lick


def edge_mask(lick, valid):
    """First/last 3 valid frames of each lick."""
    e = np.zeros(len(lick), bool)
    for k in np.unique(lick[lick >= 0]):
        idx = np.flatnonzero((lick == k) & valid)
        e[idx[:3]] = True
        e[idx[-3:]] = True
    return e


rows = []
for noise in (0.0, 1.0, 2.0):
    for seed in range(5):
        t, x, y, valid, vtrue, lick = simulate(seed, noise)
        df = pd.DataFrame({"time": t, "x": x, "y": y})
        o = old.kinematics_filter(
            df.copy(),
            frame_rate=500,
            cutoff_freq=50,
            filter_order=4,
            filter_kind="cubic",
        )
        nw = signal.smooth(
            df.assign(valid=valid), FS, cutoff_hz=50, order=4, max_gap_s=0.02
        )
        e = edge_mask(lick, valid)
        for name, v in (
            ("old", o["v"].to_numpy()),
            ("new", nw["speed"].to_numpy()),
        ):
            err = np.abs(v - vtrue)
            pk_err = []
            for k in np.unique(lick[lick >= 0]):
                m = (lick == k) & valid & np.isfinite(v)
                if m.sum() >= 3:
                    pk_err.append(v[m].max() / vtrue[m].max() - 1)
            rows.append(
                {
                    "noise_px": noise,
                    "method": name,
                    "edge_err": np.nanmedian(err[valid & e]),
                    "edge_err_p90": np.nanpercentile(err[valid & e], 90),
                    "interior_err": np.nanmedian(err[valid & ~e]),
                    "peak_bias": np.median(pk_err),
                    "peak_abs_err": np.median(np.abs(pk_err)),
                }
            )
r = pd.DataFrame(rows).groupby(["noise_px", "method"]).mean().round(3)
print(
    "speed error in px/s (true speeds ~1500-4000 px/s); "
    "peak errors as fractions of the true peak"
)
print(r.to_string())
