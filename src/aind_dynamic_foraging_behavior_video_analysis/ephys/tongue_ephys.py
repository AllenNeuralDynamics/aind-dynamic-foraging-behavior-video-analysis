import os
import re
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from typing import Sequence, Optional

from matplotlib import colormaps

from scipy.stats import linregress, mannwhitneyu

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_analysis_arch_result_access.han_pipeline import get_mle_model_fitting
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import get_session_name_from_path
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (
    annotate_movement_timing,
    add_lick_metadata_to_movements,
)


# Core Utilities For Rasters
# =========================

def get_events_dict(df_trials, df_licks, tongue_kinematics):
    return {
        'goCue': (
            df_trials,
            'goCue_start_time_in_session',
            'trial'
        ),
        'firstLick': (
            df_licks,
            'timestamps',
            'trial',
            {'cue_response': True}  # Only include cue-response licks
        ),
        'firstMove': (
            tongue_kinematics,
            'time_in_session',
            'trial',
            {'min_after_cue': True}  # Only movements after go cue
        ),
        'reward': (
            df_trials,
            'reward_time_in_session',
            'trial'
            # No filters needed unless you want to restrict rewarded trials
        )
    }

def generate_event_times(df, time_col, trial_col, filters=None, cue_times=None):
    """
    Extracts a single event time per trial based on filtering and trial structure.

    Parameters:
        df (DataFrame): Source dataframe containing time data.
        time_col (str): Name of the column with event timestamps.
        trial_col (str): Column indicating trial identity.
        filters (dict): Optional filters. 
                        Special keys include 'cue_response' and 'min_after_cue'.
        cue_times (dict): Dict of trial -> go cue time, used for filtering.

    Returns:
        dict: trial_id -> event_time
    """
    times = {}
    for tr in df[trial_col].unique():
        sub = df[df[trial_col] == tr].copy()
        if filters:
            for fcol, fval in filters.items():
                if fcol == 'cue_response':
                    sub = sub[sub[fcol] == fval]
                elif fcol == 'min_after_cue':
                    t0 = cue_times.get(tr, np.inf)
                    sub = sub[sub[time_col] > t0]
        if sub.empty:
            continue
        if 'movement_id' in sub.columns:
            mid = sub['movement_id'].min()
            t_event = sub[sub['movement_id'] == mid][time_col].min()
        else:
            t_event = sub[time_col].min()
        times[tr] = t_event
    return times


def build_event_df(events_dict):
    """
    Constructs a DataFrame of event times for each trial.

    Parameters:
        events_dict (dict): event_name -> (df, time_col, trial_col[, filters])

    Returns:
        DataFrame: index = trials, columns = event times
    """
    if 'goCue' not in events_dict:
        raise ValueError("events_dict must contain a 'goCue' entry")
    
    cue_times = generate_event_times(*events_dict['goCue'])
    trials = sorted(cue_times.keys())
    E = pd.DataFrame(index=trials)

    for name, params in events_dict.items():
        df, time_col, trial_col, *rest = params
        filt = rest[0] if rest else {}
        times = generate_event_times(df, time_col, trial_col, filt, cue_times)
        E[name] = pd.Series(times)

    return E.astype(float)


def sort_trials_by_latency(trials, event_times, event1, event2):
    """
    Sorts trials by the latency between two events.

    Parameters:
        trials (list): Trial IDs.
        event_times (dict): event_name -> {trial_id -> time}.
        event1, event2 (str): Event names to compute latency.

    Returns:
        list: Sorted trial IDs.
    """
    return sorted(
        [
            tr for tr in trials
            if np.isfinite(event_times[event1].get(tr, np.nan))
            and np.isfinite(event_times[event2].get(tr, np.nan))
        ],
        key=lambda tr: event_times[event1][tr] - event_times[event2][tr]
    )


# =========================
# Spike Raster Matrix
# =========================

def compute_raster_matrix(spikes, trial_ids, event_times, pre=1.0, post=2.0, bin_size=0.001):
    """
    Converts spike times to a trial-by-time bin raster matrix.

    Parameters:
        spikes (1D array): All spike times.
        trial_ids (list): List of trial IDs.
        event_times (dict): trial_id -> alignment time.
        pre, post (float): Time before/after event to include (seconds).
        bin_size (float): Time bin width in seconds.

    Returns:
        tuple: (raster matrix, bin centers, trial_ids)
    """
    n_trials = len(trial_ids)
    n_bins = int(np.round((pre + post) / bin_size))
    raster = np.zeros((n_trials, n_bins), dtype=np.uint8)
    bins = np.linspace(-pre, post, n_bins, endpoint=False) + bin_size / 2

    for i, tr in enumerate(trial_ids):
        t0 = event_times.get(tr, np.nan)
        if np.isnan(t0):
            continue
        rel = spikes - t0
        sel = (rel >= -pre) & (rel < post)
        rel = rel[sel]
        idx = ((rel + pre) / bin_size).astype(int)
        for j in idx:
            if 0 <= j < n_bins:
                raster[i, j] += 1
    return raster, bins, trial_ids


# =========================
# Plotting Utilities
# =========================

def plot_spikes(raster, bins, ax=None, color='black'):
    """
    Plot spike raster given binary/count matrix.

    Parameters:
        raster (2D array): Trials x time bins.
        bins (1D array): Time bin centers.
        ax (Axes): Optional matplotlib axis.
        color (str): Line color.

    Returns:
        Axes: Modified axis.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for i, row in enumerate(raster):
        spk_idx = np.nonzero(row)[0]
        ax.vlines(bins[spk_idx], i + 0.5, i + 1.5, color=color)
    return ax


def plot_event_lines(ax, ordered_trials, events):
    """
    Overlay event-aligned lines on spike raster plot.

    Parameters:
        ax (Axes): Matplotlib axis.
        ordered_trials (list): List of trial indices in display order.
        events (dict): event_name -> {'times': {trial_idx: t_rel}, 'style': dict}

    Returns:
        Axes: Modified axis.
    """
    n = len(ordered_trials)
    for name, info in events.items():
        times_dict = info['times']
        style = info.get('style', {})
        for i in range(n):
            t_rel = times_dict.get(i)
            if t_rel is not None:
                style = info.get('style', {}).copy()
                if i == 0:
                    style['label'] = name
                ax.axvline(t_rel, ymin=i/n, ymax=(i+1)/n, **style)
    return ax


# =========================
# RasterPlotter Class
# =========================

class RasterPlotter:
    """
    Encapsulates raster generation and plotting routines.
    """

    def __init__(self, spikes, trial_ids, event_times, pre=1.0, post=2.0, bin_size=0.001):
        self.spikes = spikes
        self.trial_ids = trial_ids
        self.event_times = event_times
        self.pre = pre
        self.post = post
        self.bin_size = bin_size
        self.raster, self.bins, self.ordered = compute_raster_matrix(
            spikes, trial_ids, event_times, pre, post, bin_size
        )

    def plot_raster(self, ax=None, spike_color='black'):
        """
        Plot spike raster only.
        """
        ax = plot_spikes(self.raster, self.bins, ax=ax, color=spike_color)
        ax.set_xlim(-self.pre, self.post)
        ax.set_ylim(0.5, len(self.ordered) + 0.5)
        ax.set_ylabel('Trial')
        ax.set_xlabel('Time (s)')
        return ax

    def add_events(self, ax, events):
        """
        Add vertical event lines to an existing raster.
        """
        return plot_event_lines(ax, self.ordered, events)

    def plot_with_events(self, events, ax=None, spike_color='black'):
        """
        Plot raster with overlaid event markers.
        """
        ax = self.plot_raster(ax=ax, spike_color=spike_color)
        ax = self.add_events(ax, events)
        return ax


# =========================
# PSTH Utilities
# =========================

def compute_psth(raster, bin_size=0.001, trials=None, as_rate=True):
    """
    Compute a Peri-Stimulus Time Histogram (PSTH) from spike raster.

    Parameters:
        raster (2D array): Trials x time bins.
        bin_size (float): Width of time bin in seconds.
        trials (list): Optional list of trial indices to include.
        as_rate (bool): If True, normalize to spikes/sec.

    Returns:
        tuple: (psth, trial_count)
    """
    if trials is not None:
        data = raster[trials, :]
        count = len(trials)
    else:
        data = raster
        count = raster.shape[0]

    psth = data.sum(axis=0)
    if as_rate:
        psth = psth / (bin_size * count)
    return psth, count


def smooth_vector(vec, bin_size, sigma=0.025):
    """
    Apply causal Gaussian smoothing to a vector.

    Parameters:
        vec (1D array): Input signal.
        bin_size (float): Bin size in seconds.
        sigma (float): Std dev of Gaussian in seconds.

    Returns:
        1D array: Smoothed signal.
    """
    L = int(np.ceil(3 * sigma / bin_size))
    lags = np.arange(L+1) * bin_size
    gauss = np.exp(-0.5 * (lags / sigma) ** 2)
    gauss /= gauss.sum()
    smooth = np.convolve(vec, gauss, mode='full')[:len(vec)]
    return smooth


def plot_psth(
    bins,
    psth,
    psth_smooth=None,
    ax=None,
    label='PSTH',
    plot_raw=True,
    color=None,
):
    """
    Plot raw and/or smoothed PSTH.

    Parameters
    ----------
    bins : 1D array
        Time bin centers.
    psth : 1D array
        Raw PSTH.
    psth_smooth : 1D array, optional
        Smoothed PSTH.
    ax : matplotlib Axes, optional
        Axis to plot into. If None, a new figure/axis is created.
    label : str
        Legend label.
    plot_raw : bool, default=True
        Whether to plot the raw PSTH trace.
    color : str, optional
        Matplotlib color for the trace(s). If None, uses default cycle.

    Returns
    -------
    ax : matplotlib Axes
        The axis containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if plot_raw and psth is not None:
        ax.plot(bins, psth, label=label, color=color)

    if psth_smooth is not None:
        # Use same label if raw is hidden, otherwise append
        lbl = label if not plot_raw else f"{label} (smoothed)"
        ax.plot(bins, psth_smooth, label=lbl, color=color)

    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (spk/s)')
    ax.legend()
    return ax

#wrappers

def make_rp_and_events(
    spikes,
    trials: list,
    event_dicts: dict,
    align_by: str,
    sort_by: str,
    events_to_plot: Sequence[str] = None,
    pre: float = 1.0,
    post: float = 2.0,
    bin_size: float = 0.001,
):
    """
    spikes           : 1D array of spike times
    trials           : list of trial IDs (Ev.index)
    event_dicts      : {event_name -> {trial_id -> absolute_time}}
    align_by         : which event to align each trial on
    sort_by          : which event to sort trials by (latency from align_by)
    events_to_plot   : subset of event_names to overlay (defaults to all but align_by)
    pre, post, bin_size : raster params
    """
    # 1) sort trials
    ordered = sort_trials_by_latency(
        trials=trials,
        event_times=event_dicts,    # can look up both sort_by & align_by
        event1=sort_by,
        event2=align_by,
    )

    # 2) make raster aligned to align_by
    align_times = event_dicts[align_by]
    rp = RasterPlotter(spikes, ordered, align_times,
                       pre=pre, post=post, bin_size=bin_size)

    # 3) pick events
    if events_to_plot is None:
        events_to_plot = [e for e in event_dicts if e != align_by]

    # 4) build relative‐time dicts
    events = {}
    for ix, name in enumerate(events_to_plot):
        times_rel = {}
        for i, tr in enumerate(ordered):
            t_abs = event_dicts[name].get(tr)
            t0    = align_times.get(tr)
            if t_abs is not None and t0 is not None:
                times_rel[i] = t_abs - t0
        events[name] = {
            'times': times_rel,
            'style': {'color': f"C{ix}", 'linestyle':'--', 'linewidth':2}
        }

    return rp, events


def plot_unit_panels(spikes, unit_n,
                     trials,
                     event_dicts,
                     combos,
                     events_to_plot=None,
                     pre=1.0,
                     post=2.0,
                     bin_size=0.001,
                     sigma=0.025):
    """
    spikes         : 1D array of spike times
    trials         : list of trial IDs
    event_dicts    : {event_name -> {trial_id -> event_time}}
    combos         : list of (align_by, sort_by) pairs
    events_to_plot : list of event_names to overlay (defaults to all but align_by)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col')

    for col, (align_by, sort_by) in enumerate(combos):
        # build the RasterPlotter + event overlays
        rp, events = make_rp_and_events(
            spikes,
            trials,
            event_dicts,
            align_by=align_by,
            sort_by=sort_by,
            events_to_plot=events_to_plot,  # or None to use all but align_by
            pre=pre,
            post=post,
            bin_size=bin_size
        )

        # top row: raster + event lines
        ax_r = axes[0, col]
        rp.plot_raster(ax=ax_r, spike_color='black')
        rp.add_events(ax_r, events)
        ax_r.set_title(f'align: {align_by}   sort: {sort_by}')
        if col == 0:
            ax_r.set_ylabel('Trial')
            ax_r.legend(loc='upper left', title='Events')
        ax_r.set_xlabel(None)

        # bottom row: PSTH
        ax_p = axes[1, col]
        psth, _    = compute_psth(rp.raster, bin_size=rp.bin_size)
        psth_sm    = smooth_vector(psth, bin_size=rp.bin_size, sigma=sigma)
        plot_psth(rp.bins, psth, psth_sm, ax=ax_p, label='PETH')
        if col == 0:
            ax_p.set_ylabel('Firing rate (spk/s)')
        ax_p.set_xlabel(f'Time from {align_by} (s)')
    plt.suptitle(f'Unit {unit_n}', fontsize=16)
    plt.tight_layout()
    return fig