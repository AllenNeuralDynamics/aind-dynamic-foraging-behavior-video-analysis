import os
import re
import glob
import pandas as pd
from datetime import datetime
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from TransferToNWB import bonsai_to_nwb

def parse_session_id(file_name):
    """
    Extracts animal ID and session date from filenames.
    Handles cases where the filename starts with 'behavior_'.
    """
    parts = re.split('[_.]', file_name)
    if parts[0] == "behavior":
        parts = parts[1:]
    
    if len(parts[0]) == 6 and parts[0].isdigit():
        ani_id = parts[0]
        try:
            date_obj = datetime.strptime(parts[1], "%Y-%m-%d")
        except ValueError:
            return None, None
    else:
        ani_id, date_obj = None, None
    
    return ani_id, date_obj

def find_json_in_behavior_folder(top_level_folder):
    """
    Finds a JSON file in the 'behavior/' subdirectory matching the expected naming pattern.
    """
    match = re.search(r'^[^_]+_(\d+)_', os.path.basename(top_level_folder))
    if not match:
        return None
    
    six_digit_code = match.group(1)
    behavior_subdir = os.path.join(top_level_folder, 'behavior')
    if not os.path.isdir(behavior_subdir):
        return None
    
    for root, _, files in os.walk(behavior_subdir):
        for file in files:
            if re.match(rf'^(?:behavior_)?{six_digit_code}.*\.json$', file):
                return os.path.join(root, file)
    return None

def get_nwb_file(session_name, nwb_folder="/root/capsule/data/foraging_nwb_bonsai"):
    """
    Retrieves or generates an NWB file based on the given session name.
    """
    file_list = glob.glob(os.path.join(nwb_folder, "*.nwb"))
    file_names = [os.path.basename(file) for file in file_list]
    results = [parse_session_id(file_name) for file_name in file_names]
    ani_ids, dates = zip(*results)
    session_info = pd.DataFrame({'sessionID': file_names, 'aniID': ani_ids, 'date': dates})
    
    match = re.search(r'^[^_]+_(\d+)_([\d-]+)', session_name)
    if not match:
        raise ValueError("Unexpected folder naming format")
    
    anim_name, session_date = match.groups()
    session_date_obj = datetime.strptime(session_date, "%Y-%m-%d")
    session_info_filtered = session_info[(session_info['aniID'] == anim_name) & (session_info['date'] == session_date_obj)]
    
    if not session_info_filtered.empty:
        session_id = session_info_filtered['sessionID'].values[0]
        nwb_file = os.path.join(nwb_folder, session_id)
        print(f'Loading NWB from {nwb_file}')
        return load_nwb(nwb_file)
    
    print(f"NWB file not found for {anim_name} on {session_date}. Generating it now...")
    json_top_folder = f'/root/capsule/data/{session_name}'
    json_file = find_json_in_behavior_folder(json_top_folder)
    if not json_file or not os.path.exists(json_file):
        raise FileNotFoundError(f"No valid JSON file found in {json_top_folder}")
    
    save_folder = json_top_folder.replace('/data/', '/scratch/')
    os.makedirs(save_folder, exist_ok=True)
    bonsai_to_nwb(json_file, save_folder=save_folder)
    
    nwb_files = glob.glob(os.path.join(save_folder, "*.nwb"))
    if not nwb_files:
        raise FileNotFoundError(f"No NWB file was generated in {save_folder}")
    
    nwb_file = max(nwb_files, key=os.path.getctime)
    print(f'Generated NWB file: {nwb_file}')
    return load_nwb(nwb_file)
