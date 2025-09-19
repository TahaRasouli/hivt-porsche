"""
Complete NuScenes preprocessing script.
Processes raw NuScenes data and saves preprocessed .pt files ready for training.
"""

import os
import shutil
from itertools import permutations
from typing import Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# Local imports
from utils import TemporalData


def _get_sample_chain(nusc: NuScenes, first_sample_token: str) -> List[Dict]:
    """Collect all sample dicts in the scene (ordered)."""
    samples = []
    token = first_sample_token
    while token:
        s = nusc.get("sample", token)
        samples.append(s)
        token = s["next"]
    return samples

def process_nuscenes(nusc: NuScenes, scene: Dict, radius: float) -> Optional[Dict]:
    """Process a single NuScenes scene into HiVT format with fixed coordinate system."""
    try:
        samples = _get_sample_chain(nusc, scene["first_sample_token"])
        
        # Fixed sequence length to avoid dimension mismatch
        history_steps = 10  # Fixed history
        future_steps = 20   # Fixed future  
        total_steps = history_steps + 1 + future_steps  # 31 total (10 + 1 + 20)
        
        if len(samples) < total_steps:  # Skip scenes that are too short
            print(f"Skipping scene {scene['name']}: too short ({len(samples)} < {total_steps})")
            return None
        
        # Use middle frame as reference to ensure we have enough history and future
        ref_idx = history_steps
        ref_sample = samples[ref_idx]
        ref_time = ref_sample["timestamp"]

        # Build target times around reference
        dt_s = 0.5  # NuScenes samples at 2Hz, so 0.5s between samples
        offsets = np.arange(-history_steps, future_steps + 1) * dt_s
        target_times = ref_time + (offsets * 1e6).astype(np.int64)

        # Get ego pose for reference frame FIRST (before processing positions)
        lidar_key = "LIDAR_TOP"
        sd_token = ref_sample["data"].get(lidar_key, None) or next(iter(ref_sample["data"].values()))
        sample_data = nusc.get("sample_data", sd_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        ego_origin = np.array(ego_pose["translation"][:2])  # Reference ego position
        
        # Get ego rotation
        w, x, y, z = ego_pose["rotation"]
        ego_yaw = 2.0 * np.arctan2(z, w)  # Convert quaternion to yaw
        cos_yaw, sin_yaw = np.cos(ego_yaw), np.sin(ego_yaw)
        ego_rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        instance_positions = {}
        instance_types = {}

        # Collect all positions in GLOBAL coordinates first
        for s in samples:
            timestamp = s["timestamp"]
            for ann_token in s["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                inst = ann["instance_token"]
                global_pos = np.array(ann["translation"][:2])  # Global NuScenes coordinates
                obj_type = ann.get("category_name", ann.get("category", None))
                if inst not in instance_positions:
                    instance_positions[inst] = []
                    instance_types[inst] = obj_type
                instance_positions[inst].append((timestamp, global_pos[0], global_pos[1]))

        instance_tokens = sorted(list(instance_positions.keys()))
        num_nodes = len(instance_tokens)
        if num_nodes == 0:
            print(f"Warning: No annotated instances in scene {scene['name']}")
            return None

        positions = torch.zeros(num_nodes, total_steps, 2, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, total_steps, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

        # Process each instance
        for i, inst in enumerate(instance_tokens):
            times_xy = sorted(instance_positions[inst], key=lambda x: x[0])
            times = np.array([t for t, _, _ in times_xy], dtype=np.int64)
            global_xs = np.array([x for _, x, _ in times_xy], dtype=np.float64)
            global_ys = np.array([y for _, _, y in times_xy], dtype=np.float64)

            if len(times) < 2:
                continue

            # Convert times to seconds for interpolation
            times_s = times.astype(np.float64) / 1e6
            target_times_s = target_times.astype(np.float64) / 1e6

            # Interpolate in global coordinates
            valid_mask = (target_times_s >= times_s[0]) & (target_times_s <= times_s[-1])
            
            if valid_mask.sum() < 2:  # Need at least 2 valid points
                continue
                
            # Interpolate global positions
            interp_global_x = np.interp(target_times_s, times_s, global_xs)
            interp_global_y = np.interp(target_times_s, times_s, global_ys)
            
            # Convert to ego-centric coordinates (relative to reference frame ego pose)
            for t in range(total_steps):
                if valid_mask[t]:
                    global_pos = np.array([interp_global_x[t], interp_global_y[t]])
                    # Transform: (global - ego_origin) rotated by ego_rotation
                    ego_relative = global_pos - ego_origin
                    ego_pos = ego_rotation.T @ ego_relative  # Transpose for inverse rotation
                    
                    positions[i, t, 0] = ego_pos[0]
                    positions[i, t, 1] = ego_pos[1]
                    padding_mask[i, t] = False

            # Calculate heading from the last two valid positions
            valid_indices = torch.where(~padding_mask[i])[0]
            if len(valid_indices) >= 2:
                last_two = valid_indices[-2:]
                heading_vec = positions[i, last_two[1]] - positions[i, last_two[0]]
                rotate_angles[i] = torch.atan2(heading_vec[1], heading_vec[0])

        # Create model inputs
        x = positions.clone()
        
        # History: relative displacements between consecutive frames
        for i in range(num_nodes):
            x[i, 0] = torch.zeros(2)  # First frame is always zero
            for t in range(1, history_steps + 1):
                if not (padding_mask[i, t-1] or padding_mask[i, t]):
                    x[i, t] = positions[i, t] - positions[i, t-1]  # Delta from previous
                else:
                    x[i, t] = torch.zeros(2)

        # Future: relative to current frame (index = history_steps)
        current_idx = history_steps
        for i in range(num_nodes):
            if padding_mask[i, current_idx]:  # No valid current position
                x[i, current_idx+1:] = torch.zeros(future_steps, 2)
            else:
                for t in range(current_idx + 1, total_steps):
                    if not padding_mask[i, t]:
                        x[i, t] = positions[i, t] - positions[i, current_idx]  # Delta from current
                    else:
                        x[i, t] = torch.zeros(2)

        # BOS mask for temporal encoder
        bos_mask = torch.zeros(num_nodes, history_steps + 1, dtype=torch.bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]  # First frame
        for t in range(1, history_steps + 1):
            bos_mask[:, t] = padding_mask[:, t-1] & (~padding_mask[:, t])  # Start of sequence

        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        y = positions[:, current_idx + 1:].clone()  # Future positions (absolute, ego-centric)

        # Lane processing (simplified - empty for now)
        log_rec = nusc.get("log", scene["log_token"])
        map_name = log_rec["location"]
        
        lane_vectors = torch.zeros(0, 2)
        is_intersections = torch.zeros(0, dtype=torch.uint8)
        turn_directions = torch.zeros(0, dtype=torch.uint8)
        traffic_controls = torch.zeros(0, dtype=torch.uint8)
        lane_actor_index = torch.zeros(2, 0, dtype=torch.long)
        lane_actor_vectors = torch.zeros(0, 2, dtype=torch.float)

        av_index, agent_index = 0, 0
        seq_id = scene["name"]

        return {
            'x': x[:, :history_steps + 1],  # History + current (11 frames)
            'positions': positions,         # All positions (31 frames)
            'edge_index': edge_index,
            'y': y,                        # Future positions (20 frames)
            'num_nodes': num_nodes,
            'padding_mask': padding_mask,
            'bos_mask': bos_mask,
            'rotate_angles': rotate_angles,
            'lane_vectors': lane_vectors,
            'is_intersections': is_intersections,
            'turn_directions': turn_directions,
            'traffic_controls': traffic_controls,
            'lane_actor_index': lane_actor_index,
            'lane_actor_vectors': lane_actor_vectors,
            'seq_id': seq_id,
            'av_index': av_index,
            'agent_index': agent_index,
            'city': map_name,
            'origin': torch.tensor([[0.0, 0.0]]),  # Origin is now (0,0) in ego coordinates
            'theta': 0.0,  # No rotation needed, already in ego frame
        }

    except Exception as e:
        print(f"Error processing scene {scene['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_split(nusc: NuScenes, scenes: List[Dict], split_name: str, output_dir: str, local_radius: float):
    """Process all scenes for a given split."""
    split_dir = os.path.join(output_dir, split_name, "processed")
    os.makedirs(split_dir, exist_ok=True)
    
    successful_count = 0
    failed_count = 0
    
    print(f"Processing {len(scenes)} scenes for {split_name} split...")
    
    for scene in tqdm(scenes, desc=f"Processing {split_name}"):
        try:
            # Process scene
            data_dict = process_nuscenes(nusc, scene, local_radius)
            
            if data_dict is None:
                failed_count += 1
                continue
            
            # Create TemporalData object
            data = TemporalData(**data_dict)
            
            # Save to file
            output_path = os.path.join(split_dir, f"{scene['name']}.pt")
            torch.save(data, output_path)
            successful_count += 1
            
        except Exception as e:
            print(f"Failed to process scene {scene['name']}: {e}")
            failed_count += 1
            continue
    
    print(f"{split_name} processing complete: {successful_count} successful, {failed_count} failed")
    return successful_count, failed_count


def main():
    """Main preprocessing function."""
    # Configuration
    nuscenes_root = "/mnt/d/projects/datasets/NuScenes/v1.0-trainval"
    output_root = "/mnt/d/projects/HiVT-Nu/datasets"
    version = "v1.0-trainval"
    local_radius = 50.0
    train_split_ratio = 0.7
    
    print("Starting NuScenes preprocessing...")
    print(f"NuScenes root: {nuscenes_root}")
    print(f"Output root: {output_root}")
    print(f"Local radius: {local_radius}")
    
    # Verify NuScenes dataset exists
    if not os.path.exists(nuscenes_root):
        raise FileNotFoundError(f"NuScenes dataset not found at {nuscenes_root}")
    
    # Initialize NuScenes
    print("Loading NuScenes dataset...")
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=True)
    
    # Split scenes into train/val
    num_scenes = len(nusc.scene)
    split_idx = int(num_scenes * train_split_ratio)
    train_scenes = nusc.scene[:split_idx]
    val_scenes = nusc.scene[split_idx:]
    
    print(f"Total scenes: {num_scenes}")
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Validation scenes: {len(val_scenes)}")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Process training data
    print("\n" + "="*50)
    print("PROCESSING TRAINING DATA")
    print("="*50)
    train_success, train_failed = process_split(nusc, train_scenes, "train", output_root, local_radius)
    
    # Process validation data
    print("\n" + "="*50)
    print("PROCESSING VALIDATION DATA")
    print("="*50)
    val_success, val_failed = process_split(nusc, val_scenes, "val", output_root, local_radius)
    
    # Summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Training data: {train_success} successful, {train_failed} failed")
    print(f"Validation data: {val_success} successful, {val_failed} failed")
    print(f"Total successful: {train_success + val_success}")
    print(f"Total failed: {train_failed + val_failed}")
    print(f"Output directory: {output_root}")
    
    # Verify output
    train_files = len([f for f in os.listdir(os.path.join(output_root, "train", "processed")) if f.endswith('.pt')])
    val_files = len([f for f in os.listdir(os.path.join(output_root, "val", "processed")) if f.endswith('.pt')])
    
    print(f"Files created - Train: {train_files}, Val: {val_files}")
    
    if train_files > 0 and val_files > 0:
        print("Preprocessing completed successfully!")
        print("You can now use the preprocessed data for training.")
    else:
        print("Warning: No files were created. Check for errors above.")


if __name__ == "__main__":
    main()
