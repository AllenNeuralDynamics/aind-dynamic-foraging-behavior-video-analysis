import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps  
import scipy
from scipy.signal import butter, filtfilt

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

def kinematics_filter(df, frame_rate=500, cutoff_freq=20, filter_order=8):
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
    - Computes velocity from positional changes over time.
    - Applies a zero-phase low-pass Butterworth filter to smooth kinematic signals.
    - Retains only the originally available (non-NaN) time points in the final output.
    """
    # Ensure the DataFrame has the required columns
    required_columns = ['time', 'x', 'y']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain the columns: {required_columns}")
    
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
    
    x = np.interp(new_t, t_nonan, x_nonan)
    y = np.interp(new_t, t_nonan, y_nonan)
    intrp = pd.DataFrame({'time': new_t, 'x': x, 'y': y})
    
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
    
    return df_temp.reset_index(drop=True)


### PLOTTING ###
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
        fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

        # Plot Position: offset the x and y coordinates relative to their initial value
        ax[0].plot(filtered_df['time'], filtered_df['x'] - filtered_df['x'].iloc[0],
                   label='X Position', color='b')
        ax[0].plot(filtered_df['time'], filtered_df['y'] - filtered_df['y'].iloc[0],
                   label='Y Position', color='r')
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

def plot_keypoint_confidence_analysis(keypoint_dfs, keypt):
    """Generates a 2x2 grid of plots analyzing confidence data for a specified keypoint.

    Parameters:
    keypoint_dfs (dict): Dictionary containing DataFrames of keypoint data.
    keypt (str): The name of the keypoint to analyze.
    """
    
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

    def check_timing_consistency(df, expected_interval=1/500):
        """Check consistency between Behav_Time and Camera_Time."""
        behav_diffs = df['Behav_Time'].diff().dropna()
        camera_diffs = df['Camera_Time'].diff().dropna()
        time_diff = (behav_diffs - camera_diffs).abs()
        flagged_indices = time_diff[time_diff > expected_interval * 2].index

        if not flagged_indices.empty:
            print("Warning: Timing differences exceed expected variation.")
            print(df.loc[flagged_indices, ['Behav_Time', 'Camera_Time']])
        else:
            print("Video QC: Timing differences are within expected range.")

    check_frame_monotonicity(video_csv)
    check_timing_consistency(video_csv)

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

    return keypoint_dfs_trimmed, video_csv_trimmed, keypoint_timebase

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