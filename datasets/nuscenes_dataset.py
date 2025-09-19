import os
from typing import Callable, Dict, List, Optional
import torch
from torch_geometric.data import Data, Dataset
from utils import TemporalData


class NuScenesDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Dataset for loading preprocessed NuScenes data.
        
        Args:
            root: path to ./datasets (base directory)
            split: "train" or "val" 
            transform: optional data transform
        """
        self.root = root
        self.split = split
        self.transform = transform

        # FIXED: Correct directory structure
        self._processed_dir = os.path.join(root, split, "processed")
        
        if not os.path.exists(self._processed_dir):
            raise FileNotFoundError(
                f"No processed directory found at {self._processed_dir}. "
                f"Expected structure: {root}/{split}/processed/"
                f"Make sure you've run preprocessing and have the correct directory structure."
            )

        # Get all .pt files
        self.pt_files = sorted([
            os.path.join(self._processed_dir, f)
            for f in os.listdir(self._processed_dir)
            if f.endswith(".pt")
        ])
        
        if len(self.pt_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {self._processed_dir}")

        print(f"[NuScenesDataset] Found {len(self.pt_files)} preprocessed samples in {self._processed_dir}")

        # Load and validate data
        self.data_list: List[TemporalData] = []
        valid_count = 0
        
        for f in self.pt_files:
            try:
                # Load with safe globals
                with torch.serialization.safe_globals([TemporalData, Data]):
                    loaded_data = torch.load(f, weights_only=False, map_location='cpu')
                
                # Skip invalid data
                if loaded_data is None or isinstance(loaded_data, str):
                    print(f"Warning: Skipping invalid file {f} (contains: {type(loaded_data)})")
                    continue
                
                # Convert dict to TemporalData if needed (for backward compatibility)
                if isinstance(loaded_data, dict):
                    loaded_data = self._dict_to_temporal_data(loaded_data)
                elif not isinstance(loaded_data, (TemporalData, Data)):
                    print(f"Warning: Unexpected data type {type(loaded_data)} in {f}")
                    continue
                
                # Validate data has required attributes
                if not hasattr(loaded_data, 'num_nodes') or loaded_data.num_nodes == 0:
                    print(f"Warning: Skipping file {f} - no valid nodes")
                    continue
                
                self.data_list.append(loaded_data)
                valid_count += 1
                
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

        if len(self.data_list) == 0:
            raise ValueError(
                f"No valid data found in {self._processed_dir}! "
                f"Your preprocessed files may be corrupted or incompatible. "
                f"Please run preprocessing again with the corrected script."
            )

        print(f"[NuScenesDataset] Successfully loaded {valid_count} valid samples")
        super().__init__(root, transform)

    def _dict_to_temporal_data(self, data_dict: Dict) -> TemporalData:
        """Convert dictionary to TemporalData object (backward compatibility)."""
        # Create TemporalData with required parameters
        temporal_data = TemporalData(
            x=data_dict.get('x'),
            positions=data_dict.get('positions'),
            edge_index=data_dict.get('edge_index'),
            y=data_dict.get('y'),
            num_nodes=data_dict.get('num_nodes'),
            padding_mask=data_dict.get('padding_mask'),
            bos_mask=data_dict.get('bos_mask'),
            rotate_angles=data_dict.get('rotate_angles'),
            lane_vectors=data_dict.get('lane_vectors'),
            is_intersections=data_dict.get('is_intersections'),
            turn_directions=data_dict.get('turn_directions'),
            traffic_controls=data_dict.get('traffic_controls'),
            lane_actor_index=data_dict.get('lane_actor_index'),
            lane_actor_vectors=data_dict.get('lane_actor_vectors'),
            seq_id=data_dict.get('seq_id')
        )
        
        # Add any additional attributes that aren't in the constructor
        for key, value in data_dict.items():
            if not hasattr(temporal_data, key):
                setattr(temporal_data, key, value)
        
        return temporal_data

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx) -> TemporalData:
        data = self.data_list[idx]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data

    @property
    def processed_dir(self):
        return self._processed_dir

def _get_sample_chain(nusc: NuScenes, first_sample_token: str) -> List[Dict]:
    """Collect all sample dicts in the scene (ordered)."""
    samples = []
    token = first_sample_token
    while token:
        s = nusc.get("sample", token)
        samples.append(s)
        token = s["next"]
    return samples

def process_nuscenes(nusc: NuScenes,
                     scene: Dict,
                     radius: float) -> Optional[TemporalData]:  # Return TemporalData instead of Dict
    try:
        samples = _get_sample_chain(nusc, scene["first_sample_token"])
        num_frames = len(samples)
        max_len = 50  # 20 history + 30 future
        ref_idx = min(19, num_frames - 1)  # choose valid reference

        ref_sample = samples[ref_idx]
        ref_time = ref_sample["timestamp"]

        dt_s = 0.1
        offsets = np.arange(-19, 31) * dt_s
        target_times = ref_time + (offsets * 1e6).astype(np.int64)

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
            return None

        positions = torch.zeros(num_nodes, max_len, 2, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, max_len, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

        for i, inst in enumerate(instance_tokens):
            times_xy = sorted(instance_positions[inst], key=lambda x: x[0])
            times = np.array([t for t, _, _ in times_xy], dtype=np.int64)
            xs = np.array([x for _, x, _ in times_xy], dtype=np.float64)
            ys = np.array([y for _, _, y in times_xy], dtype=np.float64)

            if len(times) == 0:
                continue

            times_s = times.astype(np.float64) / 1e6
            xs_s, ys_s = xs, ys
            tt_s = target_times.astype(np.float64) / 1e6

            inside_mask = (tt_s >= times_s[0]) & (tt_s <= times_s[-1])
            if inside_mask.any() and len(times_s) >= 2:
                interp_x = np.interp(tt_s[inside_mask], times_s, xs_s)
                interp_y = np.interp(tt_s[inside_mask], times_s, ys_s)
                positions[i, inside_mask, 0] = torch.from_numpy(interp_x).float()
                positions[i, inside_mask, 1] = torch.from_numpy(interp_y).float()
                padding_mask[i, inside_mask] = False

                hist_indices = np.where(inside_mask & (np.arange(max_len) < 20))[0]
                if len(hist_indices) >= 2:
                    a = positions[i, hist_indices[-2]]
                    b = positions[i, hist_indices[-1]]
                    heading_vec = b - a
                    rotate_angles[i] = torch.atan2(heading_vec[1], heading_vec[0])
            else:
                for ti_idx, tt in enumerate(tt_s):
                    mask_exact = np.isclose(times_s, tt)
                    if mask_exact.any():
                        j = np.where(mask_exact)[0][0]
                        positions[i, ti_idx, 0] = float(xs_s[j])
                        positions[i, ti_idx, 1] = float(ys_s[j])
                        padding_mask[i, ti_idx] = False

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
        c, s = np.cos(theta), np.sin(theta)
        rotate_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float)

        positions_rel = positions.clone()
        for i in range(num_nodes):
            for t in range(max_len):
                if not padding_mask[i, t]:
                    positions_rel[i, t] = torch.matmul(positions_rel[i, t] - origin, rotate_mat)

        x = positions_rel.clone()
        mask_19 = padding_mask[:, ref_idx]

        # history delta
        for i in range(num_nodes):
            for t in range(1, 20):
                if not (padding_mask[i, t - 1] or padding_mask[i, t]):
                    x[i, t] = positions_rel[i, t] - positions_rel[i, t - 1]
                else:
                    x[i, t] = torch.zeros(2)
            x[i, 0] = torch.zeros(2)

        # future delta
        for i in range(num_nodes):
            if mask_19[i]:
                x[i, 20:] = torch.zeros(max_len - 20, 2)
            else:
                for t in range(20, max_len):
                    if not padding_mask[i, t]:
                        x[i, t] = positions_rel[i, t] - positions_rel[i, ref_idx]
                    else:
                        x[i, t] = torch.zeros(2)

        bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:20] = padding_mask[:, :19] & (~padding_mask[:, 1:20])
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        y = positions_rel[:, 20:].clone()

        # Lane processing (simplified)
        log_rec = nusc.get("log", scene["log_token"])
        map_name = log_rec["location"]
        nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)

        lane_vectors = torch.zeros(0, 2)
        is_intersections = torch.zeros(0, dtype=torch.uint8)
        turn_directions = torch.zeros(0, dtype=torch.uint8)
        traffic_controls = torch.zeros(0, dtype=torch.uint8)
        lane_actor_index = torch.zeros(2, 0, dtype=torch.long)
        lane_actor_vectors = torch.zeros(0, 2, dtype=torch.float)

        av_index, agent_index = 0, 0
        seq_id = scene["name"]

        # Return TemporalData object instead of dictionary
        return TemporalData(
            x=x[:, :20],
            positions=positions_rel,
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes,
            padding_mask=padding_mask,
            bos_mask=bos_mask,
            rotate_angles=rotate_angles,
            lane_vectors=lane_vectors,
            is_intersections=is_intersections,
            turn_directions=turn_directions,
            traffic_controls=traffic_controls,
            lane_actor_index=lane_actor_index,
            lane_actor_vectors=lane_actor_vectors,
            seq_id=seq_id,
            # Additional attributes
            av_index=av_index,
            agent_index=agent_index,
            city=map_name,
            origin=origin.unsqueeze(0),
            theta=theta,
        )
        
    except Exception as e:
        print(f"[process_nuscenes] Failed for scene {scene['name']}: {e}")
        return None