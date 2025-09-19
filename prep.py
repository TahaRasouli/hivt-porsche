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
    """Process a single NuScenes scene into HiVT format with adaptive sequence length."""
    try:
        samples = _get_sample_chain(nusc, scene["first_sample_token"])
        
        # Adaptive sequence length based on available frames
        max_available = len(samples)
        if max_available < 15:  # Skip very short scenes
            print(f"Skipping scene {scene['name']}: too short ({max_available} frames)")
            return None
        
        # Adaptive time windows
        history_steps = min(10, max_available - 5)  # Use up to 10 history steps
        future_steps = min(20, max_available - history_steps - 1)  # Use up to 20 future steps
        total_steps = history_steps + future_steps + 1  # +1 for current frame
        
        # Use the last possible reference frame that allows full future prediction
        ref_idx = min(history_steps, max_available - future_steps - 1)
        ref_sample = samples[ref_idx]
        ref_time = ref_sample["timestamp"]

        # Build adaptive target times
        dt_s = 0.1
        hist_offsets = np.arange(-history_steps, 0) * dt_s
        future_offsets = np.arange(1, future_steps + 1) * dt_s
        current_offset = np.array([0.0])
        
        all_offsets = np.concatenate([hist_offsets, current_offset, future_offsets])
        target_times = ref_time + (all_offsets * 1e6).astype(np.int64)

        instance_positions = {}
        instance_types = {}

        for s in samples:
            timestamp = s["timestamp"]
            for ann_token in s["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                inst = ann["instance_token"]
                x, y, _ = ann["translation"]
                obj_type = ann.get("category_name", ann.get("category", None))
                if inst not in instance_positions:
                    instance_positions[inst] = []
                    instance_types[inst] = obj_type
                instance_positions[inst].append((timestamp, x, y))

        instance_tokens = sorted(list(instance_positions.keys()))
        num_nodes = len(instance_tokens)
        if num_nodes == 0:
            print(f"Warning: No annotated instances in scene {scene['name']}")
            return None

        # Use adaptive total_steps instead of fixed 50
        positions = torch.zeros(num_nodes, total_steps, 2, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, total_steps, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

        # Fill with interpolation where possible, rest stays zero (padding)
        for i, inst in enumerate(instance_tokens):
            times_xy = sorted(instance_positions[inst], key=lambda x: x[0])
            times = np.array([t for t, _, _ in times_xy], dtype=np.int64)
            xs = np.array([x for _, x, _ in times_xy], dtype=np.float64)
            ys = np.array([y for _, _, y in times_xy], dtype=np.float64)

            if len(times) == 0:
                continue

            times_s = times.astype(np.float64) / 1e6
            xs_s = xs
            ys_s = ys
            tt_s = target_times.astype(np.float64) / 1e6

            inside_mask = (tt_s >= times_s[0]) & (tt_s <= times_s[-1])
            if inside_mask.any() and len(times_s) >= 2:
                interp_x = np.interp(tt_s[inside_mask], times_s, xs_s)
                interp_y = np.interp(tt_s[inside_mask], times_s, ys_s)
                positions[i, inside_mask, 0] = torch.from_numpy(interp_x).float()
                positions[i, inside_mask, 1] = torch.from_numpy(interp_y).float()
                padding_mask[i, inside_mask] = False

                # Calculate rotation angles from history
                hist_indices = np.where(inside_mask & (np.arange(total_steps) < history_steps))[0]
                if len(hist_indices) >= 2:
                    a = positions[i, hist_indices[-2]]
                    b = positions[i, hist_indices[-1]]
                    heading_vec = b - a
                    rotate_angles[i] = torch.atan2(heading_vec[1], heading_vec[0])

        # Ego-centric transform
        lidar_key = "LIDAR_TOP"
        sd_token = ref_sample["data"].get(lidar_key, None) or next(iter(ref_sample["data"].values()))
        sample_data = nusc.get("sample_data", sd_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        origin = torch.tensor(ego_pose["translation"][:2], dtype=torch.float)

        w, xq, yq, z = ego_pose["rotation"]
        siny_cosp = 2.0 * (w * z + xq * yq)
        cosy_cosp = 1.0 - 2.0 * (yq * yq + z * z)
        theta = float(np.arctan2(siny_cosp, cosy_cosp))
        c = np.cos(theta)
        s = np.sin(theta)
        rotate_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float)

        positions_rel = positions.clone()
        for i in range(num_nodes):
            for t in range(total_steps):
                if not padding_mask[i, t]:
                    positions_rel[i, t] = torch.matmul(positions_rel[i, t] - origin, rotate_mat)

        # Build model inputs with adaptive dimensions
        x = positions_rel.clone()
        current_frame_idx = history_steps  # Index of current frame
        mask_current = padding_mask[:, current_frame_idx]

        # History deltas (relative to previous timestep)
        for i in range(num_nodes):
            for t in range(1, history_steps + 1):  # Include current frame
                if not (padding_mask[i, t - 1] or padding_mask[i, t]):
                    x[i, t] = positions_rel[i, t] - positions_rel[i, t - 1]
                else:
                    x[i, t] = torch.zeros(2)
            x[i, 0] = torch.zeros(2)  # First timestep is always zero

        # Future deltas (relative to current frame)
        for i in range(num_nodes):
            if mask_current[i]:
                x[i, current_frame_idx + 1:] = torch.zeros(future_steps, 2)
            else:
                for t in range(current_frame_idx + 1, total_steps):
                    if not padding_mask[i, t]:
                        x[i, t] = positions_rel[i, t] - positions_rel[i, current_frame_idx]
                    else:
                        x[i, t] = torch.zeros(2)

        # Adaptive BOS mask
        bos_mask = torch.zeros(num_nodes, history_steps + 1, dtype=torch.bool)  # +1 for current
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:history_steps + 1] = padding_mask[:, :history_steps] & (~padding_mask[:, 1:history_steps + 1])
        
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        y = positions_rel[:, current_frame_idx + 1:].clone()  # Future positions

        # Lane processing (simplified)
        log_rec = nusc.get("log", scene["log_token"])
        map_name = log_rec["location"]
        
        # Create empty lane data (can be enhanced later)
        lane_vectors = torch.zeros(0, 2)
        is_intersections = torch.zeros(0, dtype=torch.uint8)
        turn_directions = torch.zeros(0, dtype=torch.uint8)
        traffic_controls = torch.zeros(0, dtype=torch.uint8)
        lane_actor_index = torch.zeros(2, 0, dtype=torch.long)
        lane_actor_vectors = torch.zeros(0, 2, dtype=torch.float)

        av_index, agent_index = 0, 0
        seq_id = scene["name"]

        return {
            'x': x[:, :history_steps + 1],  # History + current frame
            'positions': positions_rel,
            'edge_index': edge_index,
            'y': y,  # Future positions only
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
            'origin': origin.unsqueeze(0),
            'theta': theta,
            # Store adaptive dimensions for model
            'historical_steps': history_steps + 1,  # +1 for current
            'future_steps': future_steps,
        }

    except Exception as e:
        print(f"Error processing scene {scene['name']}: {e}")
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
