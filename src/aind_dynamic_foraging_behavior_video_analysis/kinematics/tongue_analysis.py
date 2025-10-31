
# === Standard Library ===
import os
import json
from pathlib import Path

# === Third-Party Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Project-Specific Libraries ===
from aind_dynamic_foraging_behavior_video_analysis.kinematics.video_clip_utils import (
    extract_clips_ffmpeg_after_reencode,
    find_labeled_video,
    get_video_time,
    extract_trial_clip
)
from aind_dynamic_foraging_behavior_video_analysis.kinematics.kinematics_nwb_utils import get_nwb_file
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_kinematics_utils import (
    load_keypoints_from_csv,
    find_behavior_videos_folder,
    find_video_csv_path,
    integrate_keypoints_with_video_time,
    mask_keypoint_data,
    kinematics_filter,
    segment_movements_trimnans,
    annotate_trials_in_kinematics,
    annotate_licks_in_kinematics,
    assign_movements_to_licks,
    aggregate_tongue_movements,
    add_lick_metadata_to_movements,
    get_session_name_from_path,
    get_trial_level_df,
    select_percentile_movements,
    plot_movement_tiles_scatter,
    plot_keypoint_confidence_analysis,
    add_time_in_session_from_nwb,
    annotate_trials_by_gocue
)
import aind_dynamic_foraging_data_utils.nwb_utils as nwb_utils
from aind_dynamic_foraging_basic_analysis.licks import annotation


# main functions
def session_already_done(session_save_dir: Path) -> bool:
    """Check if final analysis output exists for this session."""
    return (session_save_dir / "tongue_quality_stats.json").exists()

def run_batch_analysis(
    pred_csv_list, 
    data_root, 
    save_root, 
    percentiles=None, 
    extract_clips=True,
    force_rerun=False  
):
    """
    Run analysis for multiple sessions in batch.

    Parameters
    ----------
    pred_csv_list : list of str or Path
        List of prediction CSV paths (one per session).
    data_root : str or Path
        Root folder where behavior_<...> session folders live.
    save_root : str or Path
        Root folder to save all analysis outputs.
    percentiles : list, optional
        Percentiles for movement quality plots (default: [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).
    extract_clips : bool, optional
        Whether to extract example video clips for each session (default: True).
    """
    percentiles = percentiles or [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    error_log = []

    for pred_csv in pred_csv_list:
        pred_csv = Path(pred_csv)
        session_id = get_session_name_from_path(str(pred_csv))
        session_save_dir = save_root / session_id
        session_save_dir.mkdir(parents=True, exist_ok=True)

        # ---- Skip check ----
        if not force_rerun and session_already_done(session_save_dir):
            print(f"\nSkipping {session_id} (analysis already complete)")
            continue

        
        print(f"\n🔹 Starting analysis for: {session_id}")
        try:
            # ---- 1) Generate DFs ----
            nwb, tongue_kins, tongue_movs, kps_raw, tongue_trials = generate_tongue_dfs(pred_csv, data_root)

            # ---- 1a) Save intermediate data ----
            intermediate_folder = session_save_dir / "intermediate_data"
            intermediate_folder.mkdir(exist_ok=True)

            tongue_kins.to_parquet(intermediate_folder / "tongue_kins.parquet")
            tongue_movs.to_parquet(intermediate_folder / "tongue_movs.parquet")
            tongue_trials.to_parquet(intermediate_folder / "tongue_trials.parquet")

            for key, df in kps_raw.items():
                df.to_parquet(intermediate_folder / f"kps_raw_{key}.parquet")

            nwb.df_licks.to_parquet(intermediate_folder / "nwb_df_licks.parquet")
            nwb.df_trials.to_parquet(intermediate_folder / "nwb_df_trials.parquet")
            nwb.df_events.to_parquet(intermediate_folder / "nwb_df_events.parquet")

            # ---- 2) Run analysis ----
            analyze_tongue_movement_quality(
                kps_raw=kps_raw,
                tongue_kins=tongue_kins,
                tongue_movs=tongue_movs,
                tongue_trials=tongue_trials,
                nwb=nwb,
                save_dir=str(session_save_dir),
                percentiles=percentiles,
                pred_csv=pred_csv
            )

            # ---- 3) Optionally extract example clips ----
            if extract_clips:
                try:
                    extract_example_clips_for_session(
                        session_id, 
                        save_root,  # analysis_root
                        data_root
                    )
                except Exception as e:
                    print(f"Warning: Could not extract clips for {session_id}: {e}")

        except Exception as e:
            error_msg = f"❌ Error in {session_id}: {repr(e)}"
            print(error_msg)
            error_log.append(error_msg)
            continue  # Move to the next session

    # ---- Print & Save Error Log ----
    if error_log:
        log_file = save_root / "batch_error_log.txt"
        with open(log_file, "w") as f:
            f.write("\n".join(error_log))
        print(f"\n⚠️ Completed with errors. See log: {log_file}")
    else:
        print("\n✅ Batch analysis completed successfully for all sessions!")


def analyze_tongue_movement_quality(
    kps_raw: dict,
    tongue_kins: pd.DataFrame,
    tongue_movs: pd.DataFrame,
    tongue_trials: pd.DataFrame,  
    nwb,
    save_dir: str,
    percentiles: list = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    pred_csv=None 
):

    """
    Analyze and visualize tongue movement quality for a single session.

    Saves figures and key summary stats in a session-specific folder.

    Parameters
    ----------
    tongue_kins : pd.DataFrame
        Frame-level kinematics data.
    tongue_movs : pd.DataFrame
        Movement-level kinematics data (one row per movement).
    nwb : NWB object with df_licks.
    pred_csv : str
        Path to the prediction CSV (used to infer session name).
    save_dir : str
        Directory where results will be saved.
    percentiles : list
        Percentiles to sample for movement quality plots.
    """

    # ----------------
    # Setup & Folders
    # ----------------
    os.makedirs(save_dir, exist_ok=True)
    session_id = os.path.basename(save_dir)

    print(f"Analyzing session: {session_id}")
    
    # ----------------
    # Confidence figure
    # ----------------
    keypt = 'tongue_tip_center'  # Example, can be parameterized in wrapper
    plot_keypoint_confidence_analysis(
        keypoint_dfs=kps_raw,
        keypt=keypt,
        save_dir=save_dir,
        save_figures=True
        )
        
    # ----------------
    # Lick Coverage from trial-level DF
    # ----------------
    total_licks = tongue_trials['lick_count'].sum()
    with_mov = (tongue_trials['coverage_pct'] * tongue_trials['lick_count'] / 100).sum()
    coverage_pct = 100 * with_mov / total_licks if total_licks else np.nan

    lick_movs = tongue_movs[tongue_movs['has_lick']]
    lick_times = nwb.df_licks['timestamps']
    has_mov = nwb.df_licks['nearest_movement_id'].notna()
    covered_times = lick_times[has_mov]
    missed_times = lick_times[~has_mov]

    # ----------------
    # Lick Coverage Figure
    # ----------------
    fig = plt.figure(constrained_layout=True, figsize=(14, 8))
    parent_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    gs_top = parent_gs[0].subgridspec(1, 3, width_ratios=[0.5, 6, 3])
    gs_bottom = parent_gs[1].subgridspec(1, 3)

    ax_cov = fig.add_subplot(gs_top[0, 0])
    ax_raster = fig.add_subplot(gs_top[0, 1])
    ax_scat = fig.add_subplot(gs_top[0, 2])
    ax_h0 = fig.add_subplot(gs_bottom[0, 0])
    ax_h1 = fig.add_subplot(gs_bottom[0, 1])
    ax_h2 = fig.add_subplot(gs_bottom[0, 2])

    # --- Coverage Bar ---
    n_missed = total_licks - with_mov
    ax_cov.bar(0, coverage_pct, color='green', label=f'Covered (n={with_mov})')
    ax_cov.bar(0, 100 - coverage_pct, bottom=coverage_pct,
               color='red', label=f'Missed (n={n_missed})')
    ax_cov.set_ylim(0, 100)
    ax_cov.set_xticks([])
    ax_cov.set_title("Lick Coverage (%)", fontsize=10)
    ax_cov.legend(fontsize=7, loc='lower center')

    # --- Raster ---
    ax_raster.eventplot(
        [covered_times, missed_times],
        lineoffsets=[1, 0], linelengths=0.8,
        colors=['green', 'red']
    )
    ax_raster.set_yticks([1, 0])
    ax_raster.set_yticklabels(['Covered', 'Missed'])
    ax_raster.set_xlabel('Time in session (s)')
    ax_raster.set_title('Lick coverage over session')

    # --- Scatter ---
    ax_scat.scatter(lick_movs['duration'], lick_movs['dropped_frames_pct'],
                    alpha=0.05, edgecolor='k')
    ax_scat.set_xlabel('Duration (s)')
    ax_scat.set_ylabel('Dropped Frame %')
    ax_scat.set_title('Duration vs Drop%')

    # --- Histograms ---
    ax_h0.hist(lick_movs['n_datapoints'], bins=30)
    ax_h0.set(title='Datapoints')
    ax_h1.hist(lick_movs['duration'], bins=30)
    ax_h1.set(title='Duration')
    ax_h2.hist(lick_movs['dropped_frames_pct'], bins=30)
    ax_h2.set(title='Dropped %')

    plt.suptitle(f'{session_id}', y=1.02)
    fig.savefig(os.path.join(save_dir, "lick_coverage_summary.png"), dpi=150)
    plt.close(fig)

    # ----------------
    # Movement Percentile Plots
    # ----------------
    tongue_kins['time_in_movement'] = (
        tongue_kins['time'] -
        tongue_kins.groupby('movement_id')['time'].transform('first')
    )

    percentile_results = {}
    for metric_col in ['dropped_frames_n', 'duration']:
        sel = select_percentile_movements(tongue_movs, metric_col=metric_col, percentiles=percentiles)
        labels = [f"{int(p*100)}%ile: {val:.2f}" 
                  for p, val in zip(sel['percentile'], sel[metric_col])]
        percentile_results[metric_col] = dict(zip(sel['percentile'], sel[metric_col]))


        fig = plot_movement_tiles_scatter(
            tongue_segmented=tongue_kins,
            movement_ids=sel['movement_id'].tolist(),
            x_col='time_in_movement',
            y_col='x',
            labels=labels,
            color='gray',
            title=metric_col,
            return_fig=True
        )
        fig.savefig(os.path.join(save_dir, f"{metric_col}_tiles.png"), dpi=150)
        plt.close(fig)

    # ----------------
    # Save Everything to JSON
    # ----------------
    results_dict = {
        "session_id": os.path.basename(save_dir),
        "pred_csv": str(pred_csv) if pred_csv else None,
        "total_licks": int(total_licks),
        "licks_with_movement": int(with_mov),
        "coverage_pct": float(coverage_pct),
        "percentiles": percentile_results
    }

    with open(os.path.join(save_dir, "tongue_quality_stats.json"), "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"✅ Finished analysis for {session_id}. Results saved to {save_dir}")


def generate_tongue_dfs(predictions_csv_path: Path, data_root: Path, tolerance=0.01):
    """
    Run the end-to-end pipeline for a single session and return:
      - the NWB object (with licks/trials annotated),
      - frame-level annotated tongue kinematics,
      - movement-level aggregated tongue movements,
      - trimmed/synced keypoints,
      - trial-level aggregates.

    Parameters
    ----------
    predictions_csv_path : Path
        Path to the Lightning Pose predictions CSV (LP_csv).
    data_root : Path
        Root directory containing session subfolders named like 'behavior_<...>'.
    tolerance : float, optional
        Max absolute time difference (seconds) when matching licks to kinematics (default 0.01).

    Returns
    -------
    tuple
        (nwb, tongue_kin, tongue_movs, kps_trim, tongue_trials)
            nwb : NWBFile
                NWB object with derived tables (events, trials, licks) populated/annotated.
            tongue_kin : pandas.DataFrame
                Frame-level, trial/lick-annotated tongue kinematics.
            tongue_movs : pandas.DataFrame
                Movement-level aggregates derived from the annotated kinematics.
            kps_trim : dict or pandas-like
                Keypoints synchronized to video time and trimmed for analysis.
            tongue_trials : pandas.DataFrame
                Trial-level aggregates derived from licks and trials.
    """
    # --- 1) Resolve session from predictions path and log context ---
    lp_csv = predictions_csv_path
    session_id = get_session_name_from_path(str(lp_csv))
    print(f"\n=== Generating tongue data for session: {session_id} ===")
    print(f"Predictions CSV: {lp_csv}")

    # --- 2) Load raw keypoints from predictions CSV ---
    kps = load_keypoints_from_csv(str(lp_csv))
    print(f"Loaded keypoints: {len(kps)} raw dataframes")

    # --- 3) Find the synchronized video CSV for this session ---
    videos_folder = find_behavior_videos_folder(str(data_root / session_id))
    if videos_folder is None:
        raise FileNotFoundError(f"Videos folder not found for session {session_id}")
    video_csv = find_video_csv_path(videos_folder)
    if not video_csv.exists():
        raise FileNotFoundError(f"Expected video CSV at {video_csv}")
    print(f"Found video CSV: {video_csv}")

    # --- 4) Synchronize keypoints to video timestamps ---
    kps_trim, _ = integrate_keypoints_with_video_time(str(video_csv), kps)
    print(f"Synced keypoints")

    # --- 5) Mask, filter, and segment tongue movements ---
    tongue_masked = mask_keypoint_data(kps_trim, 'tongue_tip_center', confidence_threshold=0.90)
    tongue_filtered = kinematics_filter(tongue_masked, cutoff_freq=50, filter_order=4, filter_kind='cubic')
    tongue_seg = segment_movements_trimnans(tongue_filtered, max_dropped_frames=10)
    print(f"Segmented {tongue_seg['movement_id'].nunique()} unique movements")

    # --- 6) Load NWB and build derived tables/annotations ---
    nwb = get_nwb_file(session_id)
    nwb.df_events = nwb_utils.create_df_events(nwb)
    nwb.df_trials = nwb_utils.create_df_trials(nwb)
    nwb.df_licks = annotation.annotate_licks(nwb)
    print(f"NWB load: {len(nwb.df_trials)} trials, {len(nwb.df_licks)} licks")

    # Align to session time and annotate trials/licks in kinematics
    ts = add_time_in_session_from_nwb(tongue_seg, nwb)
    tongue_annot = annotate_trials_by_gocue(ts, nwb.df_trials)
    tongue_kin = annotate_licks_in_kinematics(tongue_annot, nwb.df_licks, tolerance=tolerance)
    nwb.df_licks = assign_movements_to_licks(tongue_kin, nwb.df_licks)
    print("Annotated kinematics with trials & licks")

    # --- 7) Aggregate movement-level features and attach lick metadata ---
    tongue_movs = aggregate_tongue_movements(tongue_kin, kps_trim)
    tongue_movs = add_lick_metadata_to_movements(
        tongue_movs, nwb.df_licks, fields=['cue_response','rewarded','event']
    )
    print(f"Aggregated movements DF shape: {tongue_movs.shape}")

    # --- 8) Build trial-level aggregates ---
    tongue_trials = get_trial_level_df(nwb.df_licks, nwb.df_trials)
    print(f"Aggregated trial-level DF shape: {tongue_trials.shape}")

    return nwb, tongue_kin, tongue_movs, kps_trim, tongue_trials


def extract_example_clips_for_session(session_id, analysis_root, data_root):
    # Load data
    inter_dir = Path(analysis_root) / session_id / "intermediate_data"
    tongue_movs = pd.read_parquet(inter_dir / "tongue_movs.parquet")
    tongue_kins = pd.read_parquet(inter_dir / "tongue_kins.parquet")
    nwb_df_licks = pd.read_parquet(inter_dir / "nwb_df_licks.parquet")
    nwb_df_trials = pd.read_parquet(inter_dir / "nwb_df_trials.parquet")
    # Get trial-level stats
    trial_df = get_trial_level_df(nwb_df_licks, nwb_df_trials)
    
    # Only consider trials with at least 5 licks
    trial_df = trial_df[trial_df['first10s_lick_count'] >= 5]

    # # Find trial with highest and lowest coverage (among those with high lick count)
    # Top 3 trials by coverage
    top3 = trial_df.sort_values(['coverage_pct', 'lick_count'], ascending=[False, False]).head(3)
    # Bottom 3 trials by coverage
    bottom3 = trial_df.sort_values(['coverage_pct', 'lick_count'], ascending=[True, False]).head(3)

    # Output dirs
    good_dir = Path(analysis_root) / session_id / "example_clips" / "good"
    bad_dir = Path(analysis_root) / session_id / "example_clips" / "bad"
    good_dir.mkdir(exist_ok=True, parents=True)
    bad_dir.mkdir(exist_ok=True, parents=True)

    # Try to find video, but allow plotting if not found
    try:
        video_path = find_labeled_video(session_id, data_root)
        video_found = True
    except FileNotFoundError:
        print(f"Warning: No labeled video found for {session_id}. Skipping video extraction, will plot only.")
        video_found = False

    for _, trial_row in top3.iterrows():
        if video_found:
            extract_trial_clip(session_id, trial_row, tongue_kins, video_path, good_dir,
                               clip_duration_s=10, pad_s=0.5)
        fig_path = good_dir / f"Trial{trial_row.name}_xy_vs_time.png"
        plot_kinematic_vs_time(tongue_kins, trial_row, time_col="time_in_session", value_cols=['x', 'y'],
                               save_path=fig_path, licks_df=nwb_df_licks, covered_col="nearest_movement_id", pad_s=0.5)

    for _, trial_row in bottom3.iterrows():
        if video_found:
            extract_trial_clip(session_id, trial_row, tongue_kins, video_path, bad_dir,
                               clip_duration_s=10, pad_s=0.5)
        fig_path = bad_dir / f"Trial{trial_row.name}_xy_vs_time.png"
        plot_kinematic_vs_time(tongue_kins, trial_row, time_col="time_in_session", value_cols=['x', 'y'],
                               save_path=fig_path, licks_df=nwb_df_licks, covered_col="nearest_movement_id", pad_s=0.5)


def plot_kinematic_vs_time(
    tongue_kins, trial_row, time_col, value_cols, save_path,
    clip_duration_s=None, pad_s=0.5, axes=None, title_prefix="",
    licks_df=None,
    covered_col='nearest_movement_id'
):
    """
    Plots one or more kinematic parameters vs. time in stacked subplots.
    Optionally overlays vertical lines for lick times (green=covered, red=missed).

    Parameters:
        tongue_kins (pd.DataFrame): Kinematics dataframe.
        trial_row (pd.Series): Row from trials dataframe.
        time_col (str): Name of the time column (e.g., 'session_time').
        value_cols (list of str): List of kinematic columns to plot.
        save_path (str or Path): Where to save the figure.
        clip_duration_s (float or None): If set, use start+clip_duration_s for end, else use bonsai_stop_time_in_session.
        pad_s (float): Padding before/after clip window.
        axes (list of plt.Axes): Optional list of externally created axes.
        title_prefix (str): Optional string prefix for title.
        licks_df (pd.DataFrame): DataFrame of licks (must have 'timestamps', 'trial', and coverage col).
        covered_col (str): Column in licks_df indicating coverage (notna means covered).
    """
    start = trial_row['goCue_start_time_in_session']
    if clip_duration_s is not None:
        end = start + clip_duration_s
    else:
        end = trial_row['bonsai_stop_time_in_session']

    window_start = start - pad_s
    window_end = end + pad_s

    df = tongue_kins[
        (tongue_kins[time_col] >= window_start) &
        (tongue_kins[time_col] <= window_end)
    ]

    if axes is None:
        fig, axes = plt.subplots(len(value_cols), 1, figsize=(8, 3 * len(value_cols)), sharex=True)
        if len(value_cols) == 1:
            axes = [axes]  # make iterable if only one plot
    else:
        fig = axes[0].figure

    # For legend handles
    covered_handle = None
    missed_handle = None

    for ax, val_col in zip(axes, value_cols):
        ax.scatter(df[time_col], df[val_col], s=5, alpha=0.5, color='gray')
        ax.set_ylabel(val_col)
        ax.set_xlim(window_start, window_end)

    # --- Add vertical lines for licks ---
    if licks_df is not None:
        trial_num = trial_row.name if 'name' in dir(trial_row) else trial_row['trial']
        licks_in_trial = licks_df[licks_df['trial'] == trial_num]
        # Filter to window
        licks_in_window = licks_in_trial[
            (licks_in_trial['timestamps'] >= window_start) &
            (licks_in_trial['timestamps'] <= window_end)
        ]
        # Covered: where covered_col is notna
        covered = licks_in_window[licks_in_window[covered_col].notna()]
        missed = licks_in_window[licks_in_window[covered_col].isna()]
        for t in covered['timestamps']:
            for ax in axes:
                covered_handle = ax.axvline(
                    t, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Covered lick'
                )
        for t in missed['timestamps']:
            for ax in axes:
                missed_handle = ax.axvline(
                    t, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Missed lick'
                )
        # Add legend to the top axis only, avoiding duplicate labels
        handles = []
        labels = []
        if covered_handle is not None:
            handles.append(covered_handle)
            labels.append('Covered lick')
        if missed_handle is not None:
            handles.append(missed_handle)
            labels.append('Missed lick')
        if handles:
            axes[0].legend(handles, labels, loc='upper left', fontsize=9)

    axes[-1].set_xlabel(f"{time_col} (s)")
    axes[0].set_title(f"{title_prefix} Trial {trial_row.name}: {', '.join(value_cols)} vs {time_col}")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

