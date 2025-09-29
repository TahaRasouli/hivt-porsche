# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
from itertools import permutations, product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

from utils import TemporalData

# Make Argoverse optional
try:
    from argoverse.map_representation.map_api import ArgoverseMap
except ImportError:
    ArgoverseMap = None


class ArgoverseV1Dataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 lane_only: bool = False):
        self._split = split
        self._local_radius = local_radius
        self.lane_only = lane_only
        self.root = root
        self._directory = split if split != "sample" else "forecasting_sample"

        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self._raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]

        # Path to precomputed lanes
        self._lane_dir = os.path.join(root, self._directory, "processed_lanes")
        super().__init__(root, transform=transform)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self._directory, 'processed')

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        # Load trajectories
        with torch.serialization.safe_globals([TemporalData, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            data = torch.load(self._processed_paths[idx])

        # Load precomputed lane data if available
        lane_file = os.path.join(self._lane_dir, f"{data.seq_id}_lanes.pt")
        if os.path.exists(lane_file):
            lane_data = torch.load(lane_file)
            data.lane_vectors = lane_data["lane_vectors"]
            data.is_intersections = lane_data["is_intersections"]
            data.turn_directions = lane_data["turn_directions"]
            data.traffic_controls = lane_data["traffic_controls"]
            data.lane_actor_index = lane_data["lane_actor_index"]
            data.lane_actor_vectors = lane_data["lane_actor_vectors"]
            data.origin = lane_data["origin"]
            data.theta = lane_data["theta"]
            data.city = lane_data["city"]

        return data


def process_argoverse(split: str,
                      raw_path: str,
                      am: Optional[ArgoverseMap],
                      radius: float) -> Dict:
    """
    Original HiVT preprocessing, returns full scene + lane info.
    """
    df = pd.read_csv(raw_path)

    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[:20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # Scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(ts) for ts in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda s: s < 20, node_steps))
        if len(node_historical_steps) > 1:
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:
            padding_mask[node_idx, 20:] = True

    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:20] = padding_mask[:, :19] & ~padding_mask[:, 1:20]

    positions = x.clone()
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    x[:, 1:20] = torch.where((padding_mask[:, :19] | padding_mask[:, 1:20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1:20] - x[:, :19])
    x[:, 0] = torch.zeros(num_nodes, 2)

    # Lane features (requires ArgoverseMap)
    lane_vectors = is_intersections = turn_directions = traffic_controls = lane_actor_index = lane_actor_vectors = None
    if am is not None:
        df_19 = df[df['TIMESTAMP'] == timestamps[19]]
        node_inds_19 = [actor_ids.index(aid) for aid in df_19['TRACK_ID']]
        node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()
        (lane_vectors, is_intersections, turn_directions, traffic_controls,
         lane_actor_index, lane_actor_vectors) = get_lane_features(am, node_inds_19, node_positions_19,
                                                                   origin, rotate_mat, city, radius)

    y = None if split == 'test' else x[:, 20:]
    seq_id = int(os.path.splitext(os.path.basename(raw_path))[0])

    return {
        'x': x[:, :20],
        'positions': positions,
        'edge_index': edge_index,
        'y': y,
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
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_pos in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_pos[0], node_pos[1], city, radius))

    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, :2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)

        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        turn_map = {'NONE': 0, 'LEFT': 1, 'RIGHT': 2}
        turn_directions.append(turn_map.get(turn_direction, 0) * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))

    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
