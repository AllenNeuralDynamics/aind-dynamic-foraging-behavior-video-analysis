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


def extract_clips_ffmpeg_after_reencode(input_video_path, timestamps, clip_length, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, start_time in enumerate(timestamps):
        end_time = start_time + clip_length
        input_basename_ext = os.path.basename(input_video_path)
        input_basename, _ = os.path.splitext(input_basename_ext)
        output_filename = input_basename + f"_clip_{idx+1}_{start_time:.3f}s_to_{end_time:.3f}s.mp4"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.isfile(output_path):
            continue

        command = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', input_video_path,
            '-t', str(clip_length),
            '-c', 'copy',             # Copy codec (no re-encoding)
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