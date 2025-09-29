import os
import torch
import pandas as pd
from tqdm import tqdm

# Adjust this import path if needed
from argoverse.map_representation.map_api import ArgoverseMap
from datasets.argoverse_v1_dataset import process_argoverse, TemporalData 

ROOT_DIR = os.path.expanduser("~/hivt-porsche/datasets")
SPLIT = "val"  # or "train"/"val"
SAVE_DIR = os.path.join(ROOT_DIR, SPLIT, "processed_lanes")
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize map
am = ArgoverseMap()

# List all CSVs
raw_dir = os.path.join(ROOT_DIR, SPLIT, "data")
raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

for raw_file in tqdm(raw_files[:5]):  # small number of instances for testing
    raw_path = os.path.join(raw_dir, raw_file)
    lane_kwargs = process_argoverse(SPLIT, raw_path, am, radius=50)
    
    # Only save lane-related info + scene metadata
    lane_data = {
        "lane_vectors": lane_kwargs["lane_vectors"],
        "is_intersections": lane_kwargs["is_intersections"],
        "turn_directions": lane_kwargs["turn_directions"],
        "traffic_controls": lane_kwargs["traffic_controls"],
        "lane_actor_index": lane_kwargs["lane_actor_index"],
        "lane_actor_vectors": lane_kwargs["lane_actor_vectors"],
        "seq_id": lane_kwargs["seq_id"],
        "origin": lane_kwargs["origin"],
        "theta": lane_kwargs["theta"],
        "city": lane_kwargs["city"]
    }
    
    torch.save(lane_data, os.path.join(SAVE_DIR, f"{lane_kwargs['seq_id']}_lanes.pt"))
