import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps  
import scipy
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import os
import re


### ANALYSIS ###
def select_percentile_movements(
    df: pd.DataFrame,
    metric_col: str,
    percentiles: list = [0, 0.25, 0.5, 0.75, 1.0]
) -> pd.DataFrame:
    """
    Return movement_ids and corresponding metric values at specified percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'movement_id' and the metric column.
    metric_col : str
        Name of the numeric column to sort and index into.
    percentiles : list of float
        Values between 0 and 1 for desired percentiles.

    Returns
    -------
    pd.DataFrame
        Columns: ['movement_id', metric_col, 'percentile']
    """
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found")
    series = df[['movement_id', metric_col]].dropna()
    if not pd.api.types.is_numeric_dtype(series[metric_col]):
        raise ValueError(f"Column '{metric_col}' must be numeric")
    if any(p < 0 or p > 1 for p in percentiles):
        raise ValueError("Percentiles must be in [0, 1]")

    sorted_df = series.sort_values(metric_col).reset_index(drop=True)
    N = len(sorted_df)
    if N == 0:
        return pd.DataFrame(columns=['movement_id', metric_col, 'percentile'])

    rows = []
    for p in percentiles:
        idx = int(round(p * (N - 1)))
        rows.append({
            'movement_id': int(sorted_df.loc[idx, 'movement_id']),
            metric_col: sorted_df.loc[idx, metric_col],
            'percentile': p
        })

    return pd.DataFrame(rows)

def filter_timestamps_refractory(timestamps, t_refractory):
    
    # Sort the timestamps
    timestamps.sort()
    
    filtered_timestamps = []
    last_timestamp = None
    
    for ts in timestamps:
        if last_timestamp is None or (ts - last_timestamp) > t_refractory:
            filtered_timestamps.append(ts)
            last_timestamp = ts
    
    print(f"Filtered {len(timestamps)-len(filtered_timestamps)} events!")

    return filtered_timestamps


def calculate_metrics_witheventkeys(ground_truth, detected_events, time_window=0.05):
    # calculate metrics, output include eventkeys for plotting
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    gt_events = np.array(ground_truth)
    detected = np.array(detected_events)
    
    # Sort events for easier comparison
    gt_events = np.sort(gt_events)
    detected = np.sort(detected)
    
    gt_index = 0
    det_index = 0
    
    # Dictionaries to store event keys
    gt_keys = {event: 'Unclassified' for event in gt_events}
    det_keys = {event: 'Unclassified' for event in detected}
    
    while gt_index < len(gt_events) and det_index < len(detected):
        if abs(detected[det_index] - gt_events[gt_index]) <= time_window:
            tp += 1
            gt_keys[gt_events[gt_index]] = 'True Positive'
            det_keys[detected[det_index]] = 'True Positive'
            gt_index += 1
            det_index += 1
        elif detected[det_index] < gt_events[gt_index]:
            fp += 1
            det_keys[detected[det_index]] = 'False Positive'
            det_index += 1
        else:
            fn += 1
            gt_keys[gt_events[gt_index]] = 'False Negative'
            gt_index += 1
    
    # Remaining false positives
    while det_index < len(detected):
        fp += 1
        det_keys[detected[det_index]] = 'False Positive'
        det_index += 1
    
    # Remaining false negatives
    while gt_index < len(gt_events):
        fn += 1
        gt_keys[gt_events[gt_index]] = 'False Negative'
        gt_index += 1
    
    # Assuming we have a defined observation period
    total_observations = max(gt_events[-1] if gt_events.size else 0,
                             detected[-1] if detected.size else 0)
    tn = total_observations - (tp + fp + fn)

    gt_df = pd.DataFrame(list(gt_keys.items()), columns=['Time', 'Status'])
    det_df = pd.DataFrame(list(det_keys.items()), columns=['Time', 'Status'])

    
    return tp, fp, fn, tn, gt_df, det_df


def calculate_metrics(ground_truth, detected_events, time_window=0.05):
    # calculate sensitivity / specificity
    # detect concurrent licks with 50 msec shoulders

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    gt_events = np.array(ground_truth)
    detected = np.array(detected_events)
    
    # Sort events (likely already sorted)
    gt_events = np.sort(gt_events)
    detected = np.sort(detected)
    
    gt_index = 0
    det_index = 0
    
    while gt_index < len(gt_events) and det_index < len(detected):
        if abs(detected[det_index] - gt_events[gt_index]) <= time_window:
            tp += 1
            gt_index += 1
            det_index += 1
        elif detected[det_index] < gt_events[gt_index]:
            fp += 1
            det_index += 1
        else:
            fn += 1
            gt_index += 1
    
    # Count remaining false positives
    fp += len(detected) - det_index
    
    # Count remaining false negatives
    fn += len(gt_events) - gt_index
    
    # Assuming we have a defined observation period
    total_observations = max(gt_events[-1] if gt_events.size else 0,
                             detected[-1] if detected.size else 0)
    tn = total_observations - (tp + fp + fn)
    
    return tp, fp, fn, tn


def detect_licks(tongue_df, timestamps, spoutL, spoutR, threshold):
    """
    Detect the timestamps of licks based on proximity to spouts.

    Parameters:
    - tongue_df: Pandas DataFrame with columns 'x' and 'y' for tongue positions
    - timestamps: Pandas Series with timestamps corresponding to tongue_df
    - spoutL: Pandas Series with x and y coordinates of the left spout
    - spoutR: Pandas Series with x and y coordinates of the right spout
    - threshold: Distance threshold for detecting a lick

    Returns:
    - List of timestamps for detected licks
    """

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    detected_licks = []
    is_licking = False

    # Convert spout positions to tuples
    spoutL_pos = (spoutL['x'], spoutL['y'])
    spoutR_pos = (spoutR['x'], spoutR['y'])

    for i in range(len(tongue_df)):
        # Extract tongue position
        tongue_pos = tongue_df.iloc[i]
        
        # Skip rows where tongue position is NaN
        if pd.isna(tongue_pos['x']) or pd.isna(tongue_pos['y']):
            continue
        
        tongue_pos = (tongue_pos['x'], tongue_pos['y'])
        
        dist_to_spoutL = distance(tongue_pos, spoutL_pos)
        dist_to_spoutR = distance(tongue_pos, spoutR_pos)

        if (dist_to_spoutL <= threshold or dist_to_spoutR <= threshold):
            if not is_licking:
                # Start of a lick
                detected_licks.append(timestamps.iloc[i])
                is_licking = True
        else:
            is_licking = False

    return detected_licks

### PROCESSING / ANNOTATING ###

import pandas as pd

def annotate_movement_timing(tongue_movements: pd.DataFrame,
                             df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate each movement with trial-relative metrics and
    invalidate any movement that spans a go-cue boundary.

    New columns (nullable):
      • movement_number_in_trial       Int64  
      • cue_response_movement_number   Int64  
      • movement_before_cue_response   boolean  
      • movement_latency_from_go       Float64  
      • lick_latency                   Float64 (lick_time - go cue time)

    Pre-trial or spanning movements keep all-new as NA.
    """

    # --- Sanity checks ---
    required_cols = {'trial', 'start_time', 'end_time', 'cue_response', 'lick_time'}
    missing = required_cols - set(tongue_movements.columns)
    if missing:
        raise ValueError(f"Missing columns in tongue_movements: {missing}")
    if not {'trial', 'goCue_start_time_in_session'}.issubset(df_trials):
        raise ValueError("df_trials must contain 'trial' and 'goCue_start_time_in_session'")

    df = tongue_movements.copy()

    # --- Null-out any movement spanning a go-cue --- (animal shouldn't initiate tongue movement before go cue)
    go_times = df_trials['goCue_start_time_in_session']
    uniq = df[['movement_id', 'start_time', 'end_time']].drop_duplicates()
    bad_ids = {
        m for m, s, e in uniq.itertuples(index=False)
        if ((go_times > s) & (go_times < e)).any()
    }
    df.loc[df['movement_id'].isin(bad_ids), 'trial'] = pd.NA
    df.loc[df['movement_id'].isin(bad_ids), 'cue_response'] = pd.NA


    # --- Map go-cue times onto each movement row ---
    go_map = df_trials.set_index('trial')['goCue_start_time_in_session']
    df['goCue_start_time_in_session'] = df['trial'].map(go_map)

    # --- Validate ordering by (trial, start_time) ---
    valid_trials = df['trial'].dropna()
    if not valid_trials.is_monotonic_increasing:
        raise ValueError("Trials not in ascending order")
    within_trial_sorted = (
        df.dropna(subset=['trial'])
          .groupby('trial', sort=False)['start_time']
          .apply(lambda x: x.dropna().is_monotonic_increasing)
          .all()
    )
    if not within_trial_sorted:
        raise ValueError("start_time not monotonic within trial")

    # --- Annotate movement timing ---
    df['movement_number_in_trial'] = df.groupby('trial').cumcount() + 1
    df['cue_response_movement_number'] = (
        df.groupby('trial')['movement_number_in_trial']
          .transform(lambda m: m.where(df.loc[m.index, 'cue_response']).max())
    )
    df['movement_before_cue_response'] = (
        df['movement_number_in_trial'] < df['cue_response_movement_number']
    )
    df['movement_latency_from_go'] = (
        df['start_time'] - df['goCue_start_time_in_session']
    ).where(df['start_time'] >= df['goCue_start_time_in_session'])

    # --- Compute lick latency (lick_time - go cue time, only for cue_response trials that are valid) ---
    lick_latency_map = (
        df.loc[df['cue_response'].fillna(False)]        
        .dropna(subset=['trial'])                     
        .set_index('trial')
        .eval('lick_time - goCue_start_time_in_session')
        .rename('lick_latency')
    )
    df['lick_latency'] = df['trial'].map(lick_latency_map)

    # --- Cast to nullable dtypes ---
    df = df.astype({
        'movement_number_in_trial':       'Int64',
        'cue_response_movement_number':   'Int64',
        'movement_before_cue_response':   'boolean',
        'movement_latency_from_go':       'Float64',
        'lick_latency':                   'Float64'
    })

    # --- Blank out all new annotations for pre-trial or invalid rows ---
    is_pre = df['trial'].isna()
    df.loc[is_pre, [
        'movement_number_in_trial',
        'cue_response_movement_number',
        'movement_before_cue_response',
        'movement_latency_from_go',
        'lick_latency'
    ]] = pd.NA

    return df


def get_trial_level_df(nwb_df_licks, nwb_df_trials):
    # Aggregate licks per trial
    licks_per_trial = nwb_df_licks.groupby('trial').size().rename('lick_count')
    # Aggregate coverage per trial: percent of licks with a movement
    covered = nwb_df_licks['nearest_movement_id'].notna().groupby(nwb_df_licks['trial']).mean().rename('coverage_pct')
    # Lick count in first 10s of each trial
    first_licks = []
    for trial, row in nwb_df_trials.set_index('trial').iterrows():
        start = row['goCue_start_time_in_session']
        end = start + 10
        licks_in_window = nwb_df_licks[
            (nwb_df_licks['trial'] == trial) &
            (nwb_df_licks['timestamps'] >= start) &
            (nwb_df_licks['timestamps'] < end)
        ]
        first_licks.append(len(licks_in_window))
    first10s_lick_count = pd.Series(first_licks, index=nwb_df_trials['trial'], name='first10s_lick_count')
    # Merge with trial info
    trial_df = nwb_df_trials.set_index('trial').join([licks_per_trial, covered, first10s_lick_count])
    trial_df['lick_count'] = trial_df['lick_count'].fillna(0).astype(int)
    trial_df['coverage_pct'] = trial_df['coverage_pct'] * 100
    trial_df['first10s_lick_count'] = trial_df['first10s_lick_count'].fillna(0).astype(int)
    return trial_df


def add_lick_metadata_to_movements(tongue_movements, licks_df,
                                   fields=None, lick_index_col='first_lick_index'):
    """
    Adds lick-level metadata (e.g., cue_response) from licks_df to tongue_movements
    using the lick index specified in `lick_index_col`.

    Parameters:
    ----------
    tongue_movements : pd.DataFrame
        DataFrame of segmented tongue movements.
    
    licks_df : pd.DataFrame
        Lick events dataframe (e.g., nwb.df_licks).
    
    fields : list of str, optional
        Lick metadata fields to merge into tongue_movements.
        Defaults to ['cue_response'].
    
    lick_index_col : str, optional
        Column in tongue_movements that contains indices into licks_df.
        Defaults to 'first_lick_index'.
    
    Returns:
    -------
    pd.DataFrame
        A copy of tongue_movements with requested lick metadata merged in.
    """

    if fields is None:
        fields = ['cue_response']

    # Safety checks
    if not isinstance(tongue_movements, pd.DataFrame):
        raise TypeError("tongue_movements must be a pandas DataFrame.")
    if not isinstance(licks_df, pd.DataFrame):
        raise TypeError("licks_df must be a pandas DataFrame.")
    if lick_index_col not in tongue_movements.columns:
        raise ValueError(
            f"'{lick_index_col}' not found in tongue_movements. "
            "Ensure you've run earlier steps to annotate lick indices."
        )
    for field in fields:
        if field not in licks_df.columns:
            raise ValueError(f"'{field}' not found in licks_df. Available: {list(licks_df.columns)}")

    out = tongue_movements.copy()

    # Map each requested field by lick index.
    # This overwrites with the same values if already present (idempotent).
    for field in fields:
        mapped = out[lick_index_col].map(licks_df[field])
        out[field] = mapped
        # If boolean-like, fill missing with False
        if pd.api.types.is_bool_dtype(out[field]) or str(out[field].dtype) == 'boolean':
            out[field] = out[field].fillna(False)

    return out



def aggregate_tongue_movements(tongue_segmented: pd.DataFrame,
                               keypoint_dfs_trimmed: dict) -> pd.DataFrame:
    """
    Aggregate kinematic, lick, and frame-level features of tongue movements.

    Parameters:
        tongue_segmented (pd.DataFrame): Frame-level data with columns:
            'movement_id', 'time_in_session', 'x', 'y', 'xv', 'yv', 'v',
            'lick', 'lick_index', 'trial'.
        keypoint_dfs_trimmed (dict): Dictionary of keypoint DataFrames. Must include 'jaw'.

    Returns:
        pd.DataFrame: One row per movement_id with summary statistics.
    """
    # -- ensure required columns
    required = {"movement_id", "time_in_session", "x", "y", "v",
                "xv", "yv", "lick", "lick_index", "trial"}
    if not required.issubset(tongue_segmented.columns):
        missing = required - set(tongue_segmented.columns)
        raise ValueError(f"Missing columns in tongue_segmented: {missing}")

    # 1) Core kinematic metrics per movement
    movement_metrics = tongue_segmented.groupby("movement_id").agg(
        start_time=("time_in_session", "min"),
        end_time=("time_in_session", "max"),
        duration=("time_in_session", lambda x: x.max() - x.min()),
        min_x=("x", "min"), max_x=("x", "max"),
        min_y=("y", "min"), max_y=("y", "max"),
        min_xv=("xv", "min"), max_xv=("xv", "max"),
        min_yv=("yv", "min"), max_yv=("yv", "max"),
        peak_velocity=("v", "max"), mean_velocity=("v", "mean")
    )

    # 2) Total distance traveled
    tongue_sorted = tongue_segmented.sort_values(["movement_id", "time_in_session"])
    distances = []
    for mid, grp in tongue_sorted.groupby("movement_id"):
        grp = grp.dropna(subset=["x", "y"]).reset_index(drop=True)
        if len(grp) < 2:
            distances.append((mid, np.nan))
        else:
            d = np.sqrt(np.diff(grp["x"])**2 + np.diff(grp["y"])**2).sum()
            distances.append((mid, d))
    movement_distances = pd.DataFrame(distances, columns=["movement_id", "total_distance"]).set_index("movement_id")

    # 3) Excursion endpoints from jaw
    jaw_mean = keypoint_dfs_trimmed['jaw'][['x', 'y']].mean()
    jx, jy = jaw_mean['x'], jaw_mean['y']
    excursions = []
    for mid, grp in tongue_sorted.groupby("movement_id"):
        grp = grp.dropna(subset=["x", "y"]).reset_index(drop=True)
        if grp.empty:
            continue
        # start
        sx, sy = grp.loc[0, ["x", "y"]]
        # furthest Euclidean
        ed = np.sqrt((grp["x"] - jx)**2 + (grp["y"] - jy)**2)
        idx = ed.idxmax(); row = grp.loc[idx]
        ex, ey = row["x"], row["y"]
        tt = row["time_in_session"] - grp.loc[0, "time_in_session"]
        angle = np.degrees(np.arctan2(ey - jy, ex - jx))
        # max x and y distances separately
        dx = (grp["x"] - jx); idx_x = dx.idxmax();
        dy = (grp["y"] - jy).abs(); idx_y = dy.idxmax()
        excursions.append({
            "movement_id": mid,
            "startpoint_x": sx, "startpoint_y": sy,
            "endpoint_x": ex, "endpoint_y": ey,
            "time_to_endpoint": tt, "excursion_angle_deg": angle,
            "max_x_from_jaw": grp.loc[idx_x, "x"], "max_x_from_jaw_y": grp.loc[idx_x, "y"],
            "max_y_from_jaw": grp.loc[idx_y, "y"], "max_y_from_jaw_x": grp.loc[idx_y, "x"],
            "max_x_distance": abs(grp.loc[idx_x, "x"] - jx),
            "max_y_distance": abs(grp.loc[idx_y, "y"] - jy)
        })
    excursions = pd.DataFrame(excursions).set_index("movement_id")

    # 4) Frame-level stats: datapoints & dropped frames
    frame_stats = tongue_segmented.groupby("movement_id").agg(
        n_datapoints=("x", lambda s: s.notna().sum()),
        dropped_frames_n=("x", lambda s: s.isna().sum())
    )
    frame_stats['dropped_frames_pct'] = (
        100 * frame_stats['dropped_frames_n'] /
        (frame_stats['n_datapoints'] + frame_stats['dropped_frames_n'])
    )

    # 5) Lick info per movement
    lick_info = tongue_segmented.groupby("movement_id").agg(
        has_lick=("lick", "max"),
        first_lick_index=("lick_index", lambda s: s.dropna().min()),
        lick_count=("lick_index", lambda s: s.dropna().nunique())
    )

    # 6) Trial mapping
    trial_info = tongue_segmented.groupby("movement_id")["trial"].first()

    # 7) Combine all
    movements = pd.concat([
        movement_metrics,
        movement_distances,
        excursions,
        frame_stats,
        lick_info,
        trial_info.rename("trial")
    ], axis=1).reset_index()

    return movements


def annotate_trials_in_kinematics(tongue_segmented, df_trials):
    """
    Adds 'time_in_session' and 'trial' columns to tongue_segmented dataframe
    by linking to df_trials using goCue timing.
    """
    # Normalize time
    tongue_segmented = tongue_segmented.copy()
    tongue_segmented['time_in_session'] = tongue_segmented['time_raw'] - df_trials['goCue_start_time_raw'].iloc[0]

    # Merge to get trial ID
    merged_df = pd.merge_asof(
        tongue_segmented,
        df_trials,
        left_on='time_in_session',
        right_on='goCue_start_time_in_session',
        direction='backward'
    )
    tongue_segmented['trial'] = merged_df['trial']
    
    return tongue_segmented

def annotate_licks_in_kinematics(tongue_segmented, df_licks, tolerance=0.01):
    """
    Marks frames in tongue_segmented that occur near lick timestamps in licks_df.
    
    Adds:
      - 'lick': boolean, True if within tolerance of any lick
      - 'lick_index': index from licks_df of the closest lick (or pd.NA)
    """
    tongue_segmented = tongue_segmented.copy()
    tongue_segmented['lick'] = False
    tongue_segmented['lick_index'] = pd.NA

    frame_times = tongue_segmented['time_in_session'].to_numpy()
    lick_times = df_licks['timestamps'].to_numpy()

    # todo: consider whether we want a lick to be able to attached to multiple frames
    for i, lick_time in enumerate(lick_times):
        # Find frames within the time window
        diffs = np.abs(frame_times - lick_time)
        within_tolerance = diffs <= tolerance

        if within_tolerance.any():
            closest_frame = np.argmin(np.where(within_tolerance, diffs, np.inf))
            tongue_segmented.at[closest_frame, 'lick'] = True
            tongue_segmented.at[closest_frame, 'lick_index'] = i  # link back to licks_df

    return tongue_segmented


def assign_movements_to_licks(tongue_segmented, df_licks):
    """
    Maps movement_ids back to licks dataframe
    """
    if not hasattr(tongue_segmented, "lick_index"):
        print("You need to annotate licks into kinematics: annotate_licks_in_kinematics")
        return

    df_licks = df_licks.copy()

    # Map lick_index → movement_id
    #    We drop any frames that weren't matched to a lick (lick_index is NA)
    mapping = (
        tongue_segmented
        .dropna(subset=['lick_index'])
        .groupby('lick_index')['movement_id']
        .first()              # if multiple frames map to the same lick, just take the first
    )
    df_licks['nearest_movement_id'] = df_licks.index.map(mapping).astype('Int64')

    return df_licks


def segment_movements(df, max_dropped_frames=3):
    df = df.copy()
    df['movement_id'] = np.nan  # Initialize movement ID column
    movement_id = 0
    nan_counter = 0
    in_movement = False
    
    for i, row in df.iterrows():
        if pd.isna(row['x']) or pd.isna(row['y']):  # Object not detected
            nan_counter += 1
        else:  # Object detected
            nan_counter = 0
            if not in_movement:
                movement_id += 1  # Start a new movement
                in_movement = True
        
        if in_movement:
            if nan_counter <= max_dropped_frames:
                df.at[i, 'movement_id'] = movement_id
            else:
                in_movement = False  # End current movement

    df['movement_id'] = df['movement_id'].astype("Int64")  # Pandas nullable integer type
    return df

def segment_movements_trimnans(df, max_dropped_frames=3):
    # Work on a copy so that the original DataFrame remains unchanged.
    df = df.copy()
    df['movement_id'] = pd.NA  # Initialize movement ID column
    
    movement_id = 0
    nan_counter = 0
    in_movement = False

    # First pass: assign movement IDs using the nan threshold.
    for i, row in df.iterrows():
        if pd.isna(row['x']) or pd.isna(row['y']):
            nan_counter += 1
        else:
            nan_counter = 0
            if not in_movement:
                movement_id += 1
                in_movement = True
        
        if in_movement:
            if nan_counter <= max_dropped_frames:
                df.at[i, 'movement_id'] = movement_id
            else:
                in_movement = False

    # Convert movement_id column to Pandas nullable integer type.
    df['movement_id'] = df['movement_id'].astype("Int64")

    # Second pass: trim trailing rows that are NaN in 'x' from each movement
    # Create a numeric index column for vectorized operations.
    df = df.reset_index(drop=True)
    df['row_idx'] = np.arange(len(df))
    
    # For each movement, get the maximum row index that has a non-NaN in 'x'
    valid_mapping = (
        df[df['movement_id'].notna() & df['x'].notna()]
        .groupby('movement_id')['row_idx']
        .max()
    )
    
    # Map the last valid row index for each movement back onto the DataFrame.
    df['last_valid'] = df['movement_id'].map(valid_mapping)
    
    # If the current row index is greater than the last valid index for its movement,
    # then clear the movement_id.
    mask = df['row_idx'] > df['last_valid']
    df.loc[mask & df['movement_id'].notna(), 'movement_id'] = pd.NA
    
    # Clean up temporary columns.
    df.drop(columns=['row_idx', 'last_valid'], inplace=True)
    
    # Ensure movement_id is the correct nullable integer type.
    df['movement_id'] = df['movement_id'].astype("Int64")
    
    return df


def mask_keypoint_data(keypoint_dfs,keypoint, confidence_threshold=0.9, mask_value=np.nan):
    """
    Mask the 'x' or 'y' data for a specific keypoint based on a confidence threshold.

    Parameters:
    - keypoint_dfs: keypoint dataframe from 'load_keypoints_from_csv'
    - keypoint: str, name of the keypoint to process
    - confidence_threshold: float, the confidence value threshold for masking
    - mask_value: value to use for masking (default is np.nan)

    Returns:
    - masked_df: DataFrame with masked 'x' and 'y' values
    """
    if keypoint in keypoint_dfs:
        kp_df = keypoint_dfs[keypoint].copy()  # Copy to avoid modifying original DataFrame
        
        # Apply the mask based on the confidence threshold
        kp_df.loc[kp_df['confidence'] < confidence_threshold, ['x', 'y']] = mask_value
        
        return kp_df
    else:
        print(f"Keypoint {keypoint} not found")
        return None

def kinematics_filter(df, frame_rate=500, cutoff_freq=20, filter_order=8, filter_kind='cubic'):
    """
    Applies interpolation and low-pass filtering to kinematic data to smooth trajectories and velocities.
    
    Parameters:
    df : pandas.DataFrame
        Input DataFrame containing time-series kinematic data with required columns: ['time', 'x', 'y'].
    frame_rate : int, optional
        Sampling frequency of the data in Hz (default is 500 Hz).
    cutoff_freq : float, optional
        Cutoff frequency for the low-pass Butterworth filter in Hz (default is 20 Hz).
    filter_order : int, optional
        Order of the Butterworth filter (default is 8).

    Returns:
    pandas.DataFrame
        A DataFrame with interpolated and filtered kinematic data, including time, position (x, y),
        velocity components (xv, yv), and speed (v).
    
    Notes:
    - Interpolates missing data points to ensure evenly spaced timestamps.
    - Computes centered velocity from positional changes over time.
    - Applies a zero-phase low-pass Butterworth filter to smooth kinematic signals.
    - Retains only the originally available (non-NaN) time points in the final output.
    """
    # Ensure the DataFrame has the required columns
    required_columns = ['time', 'x', 'y']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain the columns: {required_columns}")
    
    # Identify extra columns to preserve
    extra_columns = [col for col in df.columns if col not in ['time', 'x', 'y']]

    # Generate new timestamps for interpolation
    t = df['time'].values
    dt = np.diff(t)
    new_ts = []
    for num in range(len(dt)):
        tspace = dt[num] / (1 / frame_rate)
        intgr = int(np.floor(tspace))
        if intgr >= 2:
            new_t = np.linspace(t[num], t[num + 1], intgr)
            new_ts.extend(new_t)
    new_t = np.unique(np.concatenate((t, new_ts)))
    
    # Interpolate missing data points
    x_nonan = df['x'][df['x'].notna()].values
    y_nonan = df['y'][df['y'].notna()].values
    t_nonan = df['time'][df['x'].notna()].values
    
    # x = np.interp(new_t, t_nonan, x_nonan)
    # y = np.interp(new_t, t_nonan, y_nonan)
    # intrp = pd.DataFrame({'time': new_t, 'x': x, 'y': y})

    # Restrict to the “safe” interval    
    t_min, t_max = t_nonan.min(), t_nonan.max()
    mask         = (new_t >= t_min) & (new_t <= t_max)
    new_t        = new_t[mask]

    f_x = interp1d(t_nonan, x_nonan, kind=filter_kind)
    f_y = interp1d(t_nonan, y_nonan, kind=filter_kind)

    x_interp = f_x(new_t)
    y_interp = f_y(new_t)
    intrp = pd.DataFrame({'time': new_t, 'x': x_interp, 'y': y_interp})

    
    
    # Apply low-pass Butterworth filter to position data
    cutoff = cutoff_freq / (frame_rate / 2)              
    b, a = butter(int(filter_order / 2), cutoff)          
    filtered_xy = filtfilt(b, a, intrp[['x', 'y']].values, axis=0)  
    intrp['x'] = filtered_xy[:, 0]                        
    intrp['y'] = filtered_xy[:, 1]                        
    
    # Compute velocity from filtered positions
    times = intrp['time'].values
    t_diff = np.gradient(times)
    xv = np.gradient(intrp['x'].values) / t_diff          
    yv = np.gradient(intrp['y'].values) / t_diff          
    v = np.sqrt(xv**2 + yv**2)                            
    
    intrp['v'] = v
    intrp['xv'] = xv
    intrp['yv'] = yv
    
    # Keep data only at original (non-NaN) timestamps
    df_temp = df.reindex(columns=list(intrp.columns.tolist()))
    df_nonan_index = df['time'][df['x'].notna()].index.tolist()
    filtered_df_nonan_index = intrp[intrp['time'].isin(t_nonan)].index.tolist()
    df_temp.iloc[df_nonan_index] = intrp.iloc[filtered_df_nonan_index]
    
    result = df_temp.reset_index(drop=True)
    
    # Add back any extra columns from original df (by position)
    extra_cols = [col for col in df.columns if col not in result.columns]
    for col in extra_cols:
        result[col] = df[col].values

    # ——— sanity check: ensure we didn’t shift the time base ———
    assert np.all(result['time'].values == df['time'].values), \
        "‼️ Time base was altered during filtering!"

    return result


### PLOTTING ###
def plot_movement_tiles_scatter(
    tongue_segmented: pd.DataFrame,
    movement_ids: list,
    x_col: str,
    y_col: str,
    labels: list = None,
    color: str = 'gray',
    s: int = 5,
    title: str = None,
    return_fig=False
):
    """
    Plot scatter of any two kinematic columns for a given list of movement_ids.

    Parameters
    ----------
    tongue_segmented : pd.DataFrame
        Frame-level data with at least 'movement_id', x_col, y_col.
    movement_ids : list
        Movement IDs to plot (one subplot per movement).
    x_col : str
        Column in tongue_segmented to plot on x-axis.
    y_col : str
        Column in tongue_segmented to plot on y-axis.
    labels : list, optional
        List of strings or values to annotate each subplot (same length as movement_ids).
    color : str
        Point color for scatter.
    s : int
        Point size for scatter.
    title : str
        Title of figure
    """
    n = len(movement_ids)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # Global axis limits
    all_x, all_y = [], []
    for mid in movement_ids:
        df = tongue_segmented[tongue_segmented['movement_id'] == mid]
        df = df[[x_col, y_col]].dropna()
        all_x.extend(df[x_col])
        all_y.extend(df[y_col])
    if not all_x:
        raise RuntimeError("No valid movements found to plot.")
    xlim = (min(all_x), max(all_x))
    ylim = (min(all_y), max(all_y))

    for i, (ax, mid) in enumerate(zip(axes, movement_ids)):
        df = tongue_segmented[tongue_segmented['movement_id'] == mid]
        df = df[[x_col, y_col]].dropna()

        if len(df) < 1:
            ax.scatter([0], [0], s=10, color='black')
        else:
            ax.scatter(df[x_col], df[y_col], s=s, color=color)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(x_col, fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel(y_col, fontsize=7)

        if labels is not None:
            ax.set_title(str(labels[i]), fontsize=8)
    
    if title:
        plt.suptitle(title, fontsize=10)
    else:
        plt.suptitle(f"{y_col} vs {x_col}", fontsize=10)
    
    # plt.suptitle(f"{y_col} vs {x_col}", fontsize=10)
    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.show()

def plot_basic_kinematics_movement_segment(tongue_segmented, movement_ids=None):
    """
    Plot kinematic data for specified movement segments from the tongue_segmented DataFrame (output of segment_movements annotation)
    
    Parameters:
        tongue_segmented (DataFrame): DataFrame containing kinematics data with at least
                                      columns: 'time', 'x', 'y', 'v', 'xv', 'yv', 'movement_id'
        movement_ids (int, list, or range, optional): A single movement id or a sequence of movement ids.
                                                     If None, the function plots the first 10 unique movements.
    """
    # Determine which movement ids to plot (filtering out np.nan values)
    if movement_ids is None:
        unique_ids = sorted(tongue_segmented['movement_id'].dropna().unique())
        movement_ids = unique_ids[:10]
    elif isinstance(movement_ids, int):
        movement_ids = [movement_ids]
    elif isinstance(movement_ids, range):
        movement_ids = list(movement_ids)
    
    # Loop through each specified movement id
    for movement_id in movement_ids:
        filtered_df = tongue_segmented[tongue_segmented['movement_id'] == movement_id]
        if filtered_df.empty:
            print(f"No data found for movement_id {movement_id}")
            continue

        # Create a figure with two subplots (position and velocity)
        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

        # Plot Position: offset the x and y coordinates relative to their initial value
        ax[0].plot(filtered_df['time'], filtered_df['x'] - filtered_df['x'].iloc[0],
                   label='X Position', color='b', linestyle='-', marker='o',markersize=2,
                   linewidth=0.5)
        ax[0].plot(filtered_df['time'], filtered_df['y'] - filtered_df['y'].iloc[0],
                   label='Y Position', color='r',linestyle='-', marker='o',markersize=2,
                   linewidth=0.5)
        ax[0].set_ylabel('Change in Pos. (pixel)')
        ax[0].set_title('Position')
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot Velocity: plot speed and the x and y components of velocity
        ax[1].plot(filtered_df['time'], filtered_df['v'], label='Speed', color='g')
        ax[1].plot(filtered_df['time'], filtered_df['xv'], label='X Velocity',
                   color='b', linestyle='--')
        ax[1].plot(filtered_df['time'], filtered_df['yv'], label='Y Velocity',
                   color='r', linestyle='--')
        ax[1].axhline(0, color='k', linestyle='--')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Velocity (pixel/s)')
        ax[1].set_title('Velocity')
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle(f'Movement ID: {int(movement_id)}')
        plt.tight_layout()
        plt.show()

def plot_keypoint_confidence_analysis(keypoint_dfs, keypt, save_dir=None, save_figures=False):
    """Generates a 2x2 grid of plots analyzing confidence data for a specified keypoint.

    Parameters:
    keypoint_dfs (dict): Dictionary containing DataFrames of keypoint data.
    keypt (str): The name of the keypoint to analyze.
    """
    if save_figures:
        os.makedirs(save_dir, exist_ok=True)
    
    df = keypoint_dfs[keypt].copy()  # Copy DataFrame for safety
    conf_values = df['confidence']
    
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 6), gridspec_kw={'width_ratios': [1, 1]}, sharex='col')
    bins = 100  

    # --- Top Left (0,0): Histogram (Linear Scale) ---
    sns.histplot(conf_values, bins=bins, edgecolor='black', ax=axs[0, 0])
    axs[0, 0].set_xlabel('')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Confidence Distribution (Linear)')
    axs[0, 0].grid(True)

    avg_conf = conf_values.mean()
    axs[0, 0].text(
        0.5, 0.95, f'Avg Conf: {avg_conf:.2f}', transform=axs[0, 0].transAxes,
        fontsize=10, color='black', ha='center', va='top',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
    )

    # --- Bottom Left (1,0): Histogram (Log-Scaled Y-axis + KDE) ---
    sns.histplot(conf_values, bins=bins, kde=True, edgecolor='black', ax=axs[1, 0])
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel('Confidence')
    axs[1, 0].set_ylabel('Frequency (Log Scale)')
    axs[1, 0].set_title('Confidence Distribution (Log) & KDE')
    axs[1, 0].grid(True, which='both', axis='y')

    # --- Top Right (0,1): Confidence Over Time with Moving Averages ---
    window_sizes_s = [0.1, 3, 100]  
    frame_rate = 500  
    window_sizes = [int(w * frame_rate) for w in window_sizes_s]  
    cmap = colormaps['jet']
    colors = [cmap(i / (len(window_sizes) - 1)) for i in range(len(window_sizes))]

    axs[0, 1].plot(df['time'], df['confidence'], alpha=0.2, label='Raw', color='gray')

    for i, (window, color) in enumerate(zip(window_sizes, colors)):
        smoothed = df['confidence'].rolling(window=window, center=True).mean()
        axs[0, 1].plot(df['time'], smoothed, label=f'{window_sizes_s[i]} sec', 
                       linewidth=2 if window > 1 else 0.4, alpha=0.6 if window > 1 else 0.3, color=color)

    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Confidence')
    axs[0, 1].set_title('Confidence Over Time')
    axs[0, 1].legend(title="Moving Avg.", fontsize=8, title_fontsize=10, loc="lower left",
                     labelspacing=0.4, handlelength=1.5)
    axs[0, 1].grid(True)

    # --- Bottom Right (1,1): Spatial Heatmap of Confidence ---
    hb = axs[1, 1].hexbin(df['x'], df['y'], C=df['confidence'], gridsize=30,
                          reduce_C_function=np.mean, cmap='viridis')
    axs[1, 1].set_xlim(0, 720)
    axs[1, 1].set_ylim(0, 540)
    axs[1, 1].set_xlabel("X Pos (pixels)")
    axs[1, 1].set_ylabel("Y Pos (pixels)")
    axs[1, 1].set_title("Avg Confidence over Space")

    # Add colorbar
    cbar = fig.colorbar(hb, ax=axs[1, 1], aspect=20)
    cbar.set_label("Average Confidence")

    # Overall figure title
    fig.suptitle(f"{keypt.capitalize()} Confidence Analysis", fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to fit the title

    if save_figures:
        fig.savefig(f"{save_dir}/confidence_analysis.png", dpi=300)
        fig.savefig(f"{save_dir}/confidence_analysis.svg", format="svg", dpi=300)
    else:
        plt.show()

def plot_processing_steps(keypoint_dfs_trimmed, tongue_masked, tongue_filtered, tongue_segmented,
                          start_time=48.5, end_time=54, save_dir=None, save_figures=False):
    if save_figures:
        os.makedirs(save_dir, exist_ok=True)

    # === Subset data ===
    original_data = keypoint_dfs_trimmed['tongue_tip_center']
    original_data = original_data[(original_data['time'] >= start_time) & (original_data['time'] <= end_time)]
    masked_data = tongue_masked[(tongue_masked['time'] >= start_time) & (tongue_masked['time'] <= end_time)]
    filtered_data = tongue_filtered[(tongue_filtered['time'] >= start_time) & (tongue_filtered['time'] <= end_time)]
    segmented_data = tongue_segmented[(tongue_segmented['time'] >= start_time) & (tongue_segmented['time'] <= end_time)]

    # === Setup Figure with 5 vertical subplots ===
    fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

    # === 1. Original X ===
    axs[0].scatter(original_data['time'], original_data['x'], color='blue', s=5)
    axs[0].set_title('Original X Position')
    axs[0].set_ylabel('X (px)')

    # === 2. Original X colored by Confidence ===
    sc = axs[1].scatter(original_data['time'], original_data['x'], c=original_data['confidence'],
                        cmap='viridis', s=5, alpha=0.7)
    axs[1].set_title('Original X (Color = Confidence)')
    axs[1].set_ylabel('X (px)')
    cbar = fig.colorbar(sc, ax=axs[1])
    cbar.set_label('Confidence')

    # === 3. Masked X ===
    axs[2].scatter(masked_data['time'], masked_data['x'], color='grey', s=5)
    axs[2].set_title('Masked X (Confidence > 0.9)')
    axs[2].set_ylabel('X (px)')

    # === 4. Filtered (50 Hz) ===
    axs[3].scatter(masked_data['time'], masked_data['x'], color='gray', s=20, alpha=0.3, label='Masked')
    axs[3].scatter(filtered_data['time'], filtered_data['x'], color='green', s=5, label='Filtered (50 Hz)')
    axs[3].set_title('Filtered X (50 Hz)')
    axs[3].set_ylabel('X (px)')
    axs[3].legend()

    # === 5. Segmented Movements ===
    cmap = plt.get_cmap('tab10')
    for i, movement_id in enumerate(segmented_data['movement_id'].dropna().unique()):
        data = segmented_data[segmented_data['movement_id'] == movement_id]
        axs[4].scatter(data['time'], data['x'], color=cmap(i % 10), s=5, label=f'Move {movement_id}')
    axs[4].set_title('Segmented Movements')
    axs[4].set_ylabel('X (px)')
    axs[4].set_xlabel('Time (s)')

    plt.suptitle('Processing Steps')

    if save_figures:
        fig.savefig(f"{save_dir}/processing_steps.png", dpi=300)
        fig.savefig(f"{save_dir}/processing_steps.svg", format="svg", dpi=300)

    plt.show()

def plot_filtering_steps(tongue_masked, kinematics_filter,
                         cutoff_freqs=[28, 40, 50],
                         start_time=49.83, end_time=50.1,
                         save_dir=None, save_figures=False):
    if save_figures:
        os.makedirs(save_dir, exist_ok=True)

    # Assign colors from matplotlib colormap for flexibility
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(cutoff_freqs))]

    filtered_datasets = {
        freq: kinematics_filter(tongue_masked, cutoff_freq=freq, filter_order=4)
        for freq in cutoff_freqs
    }

    # Subset original for zoomed view
    orig_zoomed = tongue_masked[(tongue_masked['time'] >= start_time) & (tongue_masked['time'] <= end_time)]

    # Create subplots: 1 for original + 1 per cutoff freq
    n_subplots = 1 + len(cutoff_freqs)
    fig, axs = plt.subplots(n_subplots, 1, figsize=(10, 2 * n_subplots), sharex=True, constrained_layout=True)

    # 1. Original
    axs[0].scatter(orig_zoomed['time'], orig_zoomed['x'], color='blue', s=30, alpha=0.6)
    axs[0].set_title('Original X Position')
    axs[0].set_ylabel('X (px)')

    # 2+. Filtered traces
    for ax, freq, color in zip(axs[1:], cutoff_freqs, colors):
        filt_zoomed = filtered_datasets[freq]
        filt_zoomed = filt_zoomed[(filt_zoomed['time'] >= start_time) & (filt_zoomed['time'] <= end_time)]

        ax.scatter(orig_zoomed['time'], orig_zoomed['x'], color='gray', s=30, alpha=0.3, label='Original')
        ax.scatter(filt_zoomed['time'], filt_zoomed['x'], color=color, s=10, label=f'Filtered @ {freq} Hz')
        ax.set_ylabel('X (px)')
        ax.set_title(f'Filtered X Position ({freq} Hz)')
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Time (s)')

    if save_figures:
        fig.savefig(f"{save_dir}/filtering_steps.png", dpi=300)
        fig.savefig(f"{save_dir}/filtering_steps.svg", format="svg", dpi=300)

    plt.show()


### LOADING DATA ###
def integrate_keypoints_with_video_time(video_csv_path, keypoint_dfs):
    """
    Imports, checks, and preprocesses video CSV, then trims keypoint data to match video length.

    Parameters:
    - video_csv_path: Path to the original bonsai video acquisition CSV
    - keypoint_dfs: Dictionary of dataframes from load_keypoints_from_csv

    Returns:
    - keypoint_dfs_trimmed: Trimmed keypoint dataframes
    - video_csv_trimmed: Processed and trimmed video CSV dataframe
    - keypoint_timebase: Timebase for kinematics data, in time aligned to NWB time.
    """

    # Step 1: Load video CSV
    video_csv = pd.read_csv(video_csv_path, names=['Behav_Time', 'Frame', 'Camera_Time', 'Gain', 'Exposure'])
    
    # Step 2: Convert Camera_Time to seconds
    video_csv['Camera_Time'] = video_csv['Camera_Time'] / 1e9

    # Step 3: Quality control checks
    def check_frame_monotonicity(df):
        """Ensure frame numbers increase strictly by 1."""
        frame_diff = df['Frame'].diff().dropna()
        if not (frame_diff == 1).all():
            print("Warning: Non-monotonic frame numbering detected.")
            print(df.loc[frame_diff[frame_diff != 1].index])
        else:
            print("Video QC: Frame numbers are sequential with no gaps.")

    check_frame_monotonicity(video_csv)

    def qc_and_fix_timing(df,
                      time_col='Behav_Time',
                      camera_col='Camera_Time',
                      expected_interval=1/500,
                      tol_multiplier=2,
                      bracket_tol=0.1,
                      auto_fix=True):
        """
        QC and fix timing inconsistencies, including:
        - Large timing differences
        - Backward-in-time frames

        If auto_fix=True, any isolated bad frame is fixed by interpolation if
        bracketed by good frames.
        """
        behav_diff = df[time_col].diff()
        cam_diff   = df[camera_col].diff()
        delta      = (behav_diff - cam_diff).abs()

        thresh = tol_multiplier * expected_interval
        flagged_large = set(delta[delta > thresh].index)
        flagged_backwards = set(behav_diff[behav_diff < 0].index) | set(cam_diff[cam_diff < 0].index)
        flagged = sorted(flagged_large | flagged_backwards)

        report = pd.DataFrame({
            'Behav_Time':       df.loc[flagged, time_col],
            'Camera_Time':      df.loc[flagged, camera_col],
            'Behav_Time_Diff':  behav_diff.loc[flagged],
            'Camera_Time_Diff': cam_diff.loc[flagged],
            'Time_Diff':        delta.loc[flagged],
            'Backward_Flag':    [idx in flagged_backwards for idx in flagged]
        })

        if report.empty:
            print("Video QC: Timing differences are within expected range.")
        else:
            print("Warning: Timing differences or backward frames detected.")
            print(report.to_string())

        if auto_fix:
            for idx in flagged:
                if idx <= 0 or idx >= len(df) - 1:
                    continue  # skip edges

                behav_err = (abs(behav_diff.loc[idx] - expected_interval) > thresh) or (behav_diff.loc[idx] < 0)
                cam_err   = (abs(cam_diff.loc[idx]   - expected_interval) > thresh) or (cam_diff.loc[idx] < 0)
                if behav_err ^ cam_err:  # only one column is wrong
                    bad_col = time_col if behav_err else camera_col
                    t_prev  = df.at[idx-1, bad_col]
                    t_next  = df.at[idx+1, bad_col]
                    if abs((t_next - t_prev) - 2*expected_interval) < bracket_tol:
                        df.at[idx, bad_col] = 0.5 * (t_prev + t_next)
                        print(f"  Fixed idx={idx} in '{bad_col}' by interpolation")
                    else:
                        print(f"  Skipped idx={idx}: bracket check failed")
                else:
                    print(f"  Ambiguous error at idx={idx}, skipping fix")

        return df


    qc_and_fix_timing(video_csv,
                  time_col='Behav_Time',
                  camera_col='Camera_Time',
                  expected_interval=1/500,
                  tol_multiplier=2,
                  bracket_tol=0.1,
                  auto_fix=True)


    # Step 4: Trim kinematics timebase to match video
    def trim_kinematics_timebase_to_match(keypoint_dfs, video_csv):
        LP_samples = len(keypoint_dfs[list(keypoint_dfs.keys())[0]])
        video_samples = len(video_csv)

        if LP_samples > video_samples:
            print(f"keypoint_df trimmed from {LP_samples} to {video_samples}")
        elif LP_samples < video_samples:
            print(f"video_csv trimmed from {video_samples} to {LP_samples}")
        else:
            print("no change")

        min_samples = min(LP_samples, video_samples)
        video_csv_trimmed = video_csv.iloc[:min_samples]

        keypoint_dfs_trimmed = keypoint_dfs.copy()
        for key in keypoint_dfs.keys():
            keypoint_dfs_trimmed[key] = keypoint_dfs[key].iloc[:min_samples]

        return keypoint_dfs_trimmed, video_csv_trimmed

    keypoint_dfs_trimmed, video_csv_trimmed = trim_kinematics_timebase_to_match(keypoint_dfs, video_csv)
    keypoint_timebase = video_csv_trimmed['Behav_Time']

    # Step 5: Add 'time' column to each keypoint dataframe
    for key in keypoint_dfs_trimmed.keys():
        keypoint_dfs_trimmed[key].insert(0, 'time', keypoint_timebase - keypoint_timebase.iloc[0])
        keypoint_dfs_trimmed[key].insert(1, 'time_raw', keypoint_timebase)

    return keypoint_dfs_trimmed, video_csv_trimmed

def find_behavior_videos_folder(top_level_folder):
    """
    Searches for the 'behavior-videos' folder within a given top-level folder.

    Args:
        top_level_folder (str): Path to the top-level behavior folder.

    Returns:
        str: Path to the 'behavior-videos' folder if found, otherwise None.
    """
    for root, dirs, files in os.walk(top_level_folder):
        if 'behavior-videos' in dirs:
            return os.path.join(root, 'behavior-videos')
    return None

def get_session_name_from_path(lp_csv_path: str) -> str:
    """
    Extracts the session name (behavior_<animalID>_<YYYY-MM-DD>_<HH-MM-SS>)
    from any LP_csv_path format by searching for 'behavior_' and validating.
    """

    # Regex for expected session format
    session_pattern = re.compile(r"behavior_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

    match = session_pattern.search(lp_csv_path)
    if not match:
        raise ValueError(f"Could not find valid session name in path: {lp_csv_path}")

    return match.group(0)

def load_keypoints_from_csv(path_to_csv):
    """
    Load keypoints from Lightning Pose csv into data frame

    Parameters:
    - path_to_csv: path to your csv file from LP. Assumes follow format:
        -first column is extraneous
        -first row is keypoint labels (e.g. 'tongue_tip')
        -second row is data content labels (e.g. 'x' position)

    Returns:
    - keypoint_dfs: DataFrame with 'x', 'y', and 'confidence' values, organized by keypoint
    """
    
    df = pd.read_csv(path_to_csv, dtype=str, low_memory=False)

    #remove first column
    df = df.iloc[:, 1:]

    # Extract header information
    header_labels = df.iloc[0]  # First row: keypoint labels
    types = df.iloc[1]          # Second row: 'x', 'y', or 'confidence'

    # Drop the first two rows and reset the index
    df = df[2:].reset_index(drop=True)

    # Convert data to numeric, replacing errors with NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Create a dictionary to store DataFrames for each keypoint
    keypoint_dfs = {}

    # Loop over the columns in the DataFrame
    for i in range(0, len(df.columns), 3):
        # Extract keypoint name from the header_labels
        keypoint = header_labels.iloc[i]
        
        # Check if the columns exist in the DataFrame
        if i + 2 < len(df.columns):
            keypoint_df = pd.DataFrame({
                'x': df.iloc[:, i].astype('float'),
                'y': df.iloc[:, i + 1].astype('float'),
                'confidence': df.iloc[:, i + 2].astype('float')
            })
            keypoint_dfs[keypoint] = keypoint_df
        else:
            print(f"Warning: Missing columns for keypoint {keypoint}")
    
    print(f'keypoints extracted: {list(keypoint_dfs.keys())}')

    return keypoint_dfs