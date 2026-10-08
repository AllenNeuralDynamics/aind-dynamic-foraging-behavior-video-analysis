import os
import re
import glob
import json
import cv2
import subprocess
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# from datetime import datetime
from matplotlib import colormaps  
from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import Dict, List, Optional, Tuple, Union
import warnings

from aind_dynamic_foraging_behavior_video_analysis.video_alignment import (
    event_frame_ranges,
)
from aind_dynamic_foraging_behavior_video_analysis.video_clips import cut_clip


def extract_clips_ffmpeg_encode(input_video_path, timestamps, clip_length, output_dir):
    """
    Cut clips with a libx264 re-encode so the seek is frame-accurate.

    Use this when clip boundaries matter (single-event review clips).
    ``extract_clips_ffmpeg_after_reencode`` below is the ``-c copy`` variant:
    much faster, but the start snaps to the previous keyframe.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, start_time in enumerate(timestamps):
        # Calculate end time
        end_time = start_time + clip_length
        
        # Define the output filename
        input_basename_ext = os.path.basename(input_video_path)
        input_basename, _ = os.path.splitext(input_basename_ext)
        output_filename = input_basename + f"_clip_{idx+1}_{start_time:.2f}s_to_{end_time:.2f}s.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Skip if file already exists
        if os.path.isfile(output_path):
            continue


        # FFmpeg command to extract the clip
        command = [
            'ffmpeg',
            '-ss', str(start_time),  # Start time
            '-i', input_video_path,  # Input file
            '-t', str(clip_length),  # Duration of the clip
            '-c:v', 'libx264',       # Video codec: H.264
            '-pix_fmt', 'yuv420p',   # pixel format yuv420p for compatibility
            output_path               # Output file
        ]
        
        # Execute the command
        subprocess.run(command, check=True)
        
        print(f"Clip saved to {output_path}")


def extract_clips_ffmpeg_after_reencode(input_video_path, timestamps, clip_length, output_dir, filename_stems=None):
    input_video_path = str(input_video_path)
    output_dir       = str(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, start_time in enumerate(timestamps):
        end_time = start_time + clip_length
        input_basename_ext = os.path.basename(input_video_path)
        input_basename, _ = os.path.splitext(input_basename_ext)
        # Use custom stem if provided, else default
        if filename_stems is not None and idx < len(filename_stems):
            output_filename = f"{filename_stems[idx]}_{start_time:.3f}s_to_{end_time:.3f}s.mp4"
        else:
            output_filename = input_basename + f"_clip_{idx+1}_{start_time:.3f}s_to_{end_time:.3f}s.mp4"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.isfile(output_path):
            continue

        command = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', input_video_path,
            '-t', str(clip_length),
            '-c', 'copy',
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Clip saved to {output_path}")


def create_labeled_video(
    clip: VideoFileClip,
    xs_arr: np.ndarray,
    ys_arr: np.ndarray,
    mask_array: Optional[np.ndarray] = None,
    dotsize: int = 4,
    colormap: str = "cool",
    fps: Optional[float] = None,
    filename: str = "movie.mp4",
    start_time: float = 0.0,
) -> None:
    """Helper function for creating annotated videos.

    Args
        clip
        xs_arr: shape T x n_joints
        ys_arr: shape T x n_joints
        mask_array: shape T x n_joints; timepoints/joints with a False entry will not be plotted
        dotsize: size of marker dot on labeled video
        colormap: matplotlib color map for markers
        fps: None to default to fps of original video
        filename: video file name
        start_time: time (in seconds) of video start

    """

    if mask_array is None:
        mask_array = ~np.isnan(xs_arr)

    n_frames, n_keypoints = xs_arr.shape

    # set colormap for each color
    colors = make_cmap(n_keypoints, cmap=colormap)

    # extract info from clip
    nx, ny = clip.size
    dur = int(clip.duration - clip.start)
    fps_og = clip.fps

    # add marker to each frame t, where t is in sec
    def add_marker(get_frame, t):
        image = get_frame(t * 1.0)
        # frame [ny x ny x 3]
        frame = image.copy()
        # convert from sec to indices
        index = int(np.round(t * 1.0 * fps_og))
        # ----------------
        # markers
        # ----------------
        for bpindex in range(n_keypoints):
            if index >= n_frames:
                print("Skipped frame {}, marker {}".format(index, bpindex))
                continue
            if mask_array[index, bpindex]:
                xc = min(int(xs_arr[index, bpindex]), nx - 1)
                yc = min(int(ys_arr[index, bpindex]), ny - 1)
                frame = cv2.circle(
                    frame,
                    center=(xc, yc),
                    radius=dotsize,
                    color=colors[bpindex].tolist(),
                    thickness=-1,
                )
        return frame

    clip_marked = clip.fl(add_marker)
    clip_marked.write_videofile(filename, fps=fps or fps_og or 20.0)
    clip_marked.close()


def make_cmap(number_colors: int, cmap: str = "cool"):
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors


def process_and_label_clips(input_video_path, timestamps, clip_length, clip_output_dir, label_output_dir, keypoint_dataframes, confidence_level = 0.8, fps=None):
    # Step 1: Extract clips
    extract_clips_ffmpeg_after_reencode(input_video_path, timestamps, clip_length, clip_output_dir)
    
    # For each timestamp/clip
    for idx, start_time in enumerate(timestamps):
        # Construct expected clip filename (should match the naming scheme in your extract function)
        input_basename_ext = os.path.basename(input_video_path)
        input_basename, _ = os.path.splitext(input_basename_ext)
        clip_filename = f"{input_basename}_clip_{idx+1}_{start_time:.3f}s_to_{start_time+clip_length:.3f}s.mp4"
        clip_path = os.path.join(clip_output_dir, clip_filename)
        
        # Load the clip
        clip = VideoFileClip(clip_path)
        
        # Step 2 & 3: Build xs_arr and ys_arr for the clip
        # We assume each dataframe's 'time' column is in seconds relative to the original video.
        xs_list = []
        ys_list = []
        conf_list = []
        for key, df in keypoint_dataframes.items():
            # Filter the dataframe for the clip’s time window.
            # You might need to adjust tolerance if your times are not perfectly aligned.
            clip_df = df[(df['time'] >= start_time) & (df['time'] <= start_time + clip_length)]
            
            # Here, we assume one row per frame. 
            # If the number of rows doesn't match the number of frames in the clip,
            # you could resample or interpolate the keypoint positions.
            xs_list.append(clip_df['x'].to_numpy())
            ys_list.append(clip_df['y'].to_numpy())
            conf_list.append(clip_df['confidence'].to_numpy())
        
        # Convert lists to 2D arrays: each column corresponds to a keypoint.
        # (This requires that all keypoint arrays have the same length.)
        xs_arr = np.column_stack(xs_list)
        ys_arr = np.column_stack(ys_list)
        conf_arr = np.column_stack(conf_list)
        
        # Optional: Verify that xs_arr.shape[0] (number of timepoints) matches expected frame count.
        expected_frames = clip.reader.nframes
        if xs_arr.shape[0] != expected_frames:
            print(f"Warning: Number of keypoint frames ({xs_arr.shape[0]}) does not match video frames ({expected_frames}).")
            # You could add interpolation or padding here if needed.
        
        # Step 4: Create labeled video for this clip
        labeled_clip_filename = f"{input_basename}_clip_{idx+1}_{start_time:.3f}s_to_{start_time+clip_length:.3f}s_labeled.mp4"
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir)
        labeled_clip_path = os.path.join(label_output_dir, labeled_clip_filename)

        mask_array = conf_arr > confidence_level

        create_labeled_video(clip, xs_arr, ys_arr, mask_array=mask_array, filename=labeled_clip_path)
        clip.close()

def find_labeled_video(session_id, data_root):
    # Find the labeled video file for a session_id, searching for any folder that starts with session_id
    data_root = Path(data_root)
    for subdir in data_root.glob(f"{session_id}*"):
        labeled_dir = subdir / "pred_outputs" / "video_preds" / "labeled_videos"
        matches = list(labeled_dir.glob("*_labeled.mp4"))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"Labeled video not found for {session_id}")

def get_video_time(session_time, tongue_kins):
    """Deprecated: session time plus a constant offset, which is not video time.

    The offset is the kinematics ``time`` (Harp seconds since the first
    frame) of the first row, so the result is wrong by the time lost to
    dropped frames before ``session_time``. :func:`extract_trial_clip` no
    longer uses it; to find frames, use
    ``video_alignment.event_frame_ranges`` on the corrected Harp time.
    """
    warnings.warn(
        "get_video_time is deprecated and wrong in sessions with dropped "
        "frames; use video_alignment.event_frame_ranges on the corrected "
        "Harp time (tongue_kins['time_raw']).",
        DeprecationWarning,
        stacklevel=2,
    )
    offset = tongue_kins.iloc[0]['time'] - tongue_kins.iloc[0]['time_in_session']
    return session_time + offset


def extract_trial_clip(
    session_id, trial_row, tongue_kins, video_path, save_dir,
    clip_duration_s=10.0, pad_s=0.5
):
    """Cut the frames from ``pad_s`` before a trial's go cue to
    ``clip_duration_s + pad_s`` after it, frame-exact.

    Row ``i`` of ``tongue_kins`` is video frame ``i`` and its ``time_raw`` is
    that frame's corrected Harp time (``integrate_keypoints_with_video_time``
    keeps every frame), so the window is found on those times
    (``video_alignment.event_frame_ranges``) and cut by frame index
    (``video_clips.cut_clip``). Right also in sessions with dropped frames,
    where the clip has fewer frames than the window's duration at the
    nominal rate. The window is clipped to the video.

    Parameters
    ----------
    session_id : str
        Unused; kept for callers.
    trial_row : pandas.Series
        A trial with ``goCue_start_time_in_session``; its name (or
        ``trial``) numbers the clip.
    tongue_kins : pandas.DataFrame
        Frame-level kinematics with ``time_raw`` and ``time_in_session``.
    video_path : str or pathlib.Path
        The session's video (an MP4 frame-aligned with the predictions).
    save_dir : str or pathlib.Path
        Output folder.
    clip_duration_s, pad_s : float
        Seconds after the go cue, and padding on both sides.

    Returns
    -------
    pathlib.Path or None
        The clip (``trial_<n>_f<start_frame>.mp4``; an existing one is
        kept), or None if the window has no frames in the video.
    """
    harp_time = tongue_kins['time_raw'].to_numpy()
    # time_in_session = time_raw - first go cue, the same constant per row.
    first_go_cue = float(
        tongue_kins['time_raw'].iloc[0] - tongue_kins['time_in_session'].iloc[0]
    )
    go_cue = trial_row['goCue_start_time_in_session'] + first_go_cue
    window = event_frame_ranges(
        go_cue, harp_time, before=pad_s, after=clip_duration_s + pad_s
    ).iloc[0]
    start = int(window['start_frame'])
    n_frames = min(int(window['n_frames']), len(harp_time) - start)

    trial_num = trial_row.name if hasattr(trial_row, 'name') else trial_row['trial']
    if n_frames < 1:
        print(f"Trial {trial_num}: window not in the video, no clip")
        return None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"trial_{trial_num}_f{start:07d}.mp4"
    if not out_path.exists():
        cut_clip(video_path, start, n_frames, out_path)
    print(f"Saved clip for trial {trial_num} to {save_dir}")
    return out_path
