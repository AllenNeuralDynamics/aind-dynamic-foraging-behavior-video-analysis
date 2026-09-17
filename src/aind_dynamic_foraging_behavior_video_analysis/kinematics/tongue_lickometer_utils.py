"""Spout-contact lick detection and scoring against the lickometer.

Layering
--------
Sibling of ``tongue_kinematics_utils``; see the ownership split in that
module's docstring. This module owns the *contact* definition of a lick -
the tracked tongue within ``threshold`` pixels of a spout - and the
event-level scoring of detected licks against lickometer times. It does not
load or filter keypoints (``tongue_kinematics_utils.load_keypoints_from_csv``
/ ``mask_keypoint_data`` / ``filter_timestamps_refractory``) and does not
cut video clips (``video_clip_utils.extract_clips_ffmpeg_encode``).
"""
import numpy as np
import pandas as pd

def calculate_metrics_witheventkeys(ground_truth, detected_events, time_window=0.05):
    # calculate metrics, output include eventkeys for plotting
    tp = 0
    fp = 0
    fn = 0
    
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
    

    gt_df = pd.DataFrame(list(gt_keys.items()), columns=['Time', 'Status'])
    det_df = pd.DataFrame(list(det_keys.items()), columns=['Time', 'Status'])

    
    return tp, fp, fn, gt_df, det_df


def calculate_metrics(ground_truth, detected_events, time_window=0.05):
    # calculate sensitivity / specificity
    # detect concurrent licks with 50 msec shoulders

    tp = 0
    fp = 0
    fn = 0
    
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
    
    
    return tp, fp, fn




def detect_licks(tongue_df, spoutL, spoutR, threshold):
    """
    Detect the timestamps of licks based on proximity to spouts.

    Parameters:
    - tongue_df: Pandas DataFrame with columns 'x' and 'y' for tongue positions over 'time'
    - spoutL: Pandas Series with x and y coordinates of the left spout
    - spoutR: Pandas Series with x and y coordinates of the right spout
    - threshold: Distance threshold for detecting a lick

    Returns:
    - List of timestamps for detected licks
    """

    # Convert spout positions to numpy arrays
    spoutL_pos = np.array([spoutL['x'], spoutL['y']])
    spoutR_pos = np.array([spoutR['x'], spoutR['y']])
    
    # Extract tongue positions and keep only relevant columns
    tongue_positions = tongue_df[['time', 'x', 'y']].dropna()

    # Compute distances to both spouts
    dist_to_spoutL = np.linalg.norm(tongue_positions[['x', 'y']].to_numpy() - spoutL_pos, axis=1)
    dist_to_spoutR = np.linalg.norm(tongue_positions[['x', 'y']].to_numpy() - spoutR_pos, axis=1)

    # Create a boolean array indicating where licks are detected
    licking = (dist_to_spoutL <= threshold) | (dist_to_spoutR <= threshold)

    # Get the timestamps where licking starts
    detected_licks = []
    is_licking = False

    for i in range(len(licking)):
        if licking[i]:
            if not is_licking:
                detected_licks.append(tongue_positions.iloc[i]['time'])
                is_licking = True
        else:
            is_licking = False

    return detected_licks
