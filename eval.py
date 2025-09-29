import os
import torch
from models.hivt import HiVT
from utils import TemporalData
import json
from torch.serialization import safe_globals


# -------------------------
# Settings
# -------------------------
scene_file = './datasets/val/processed/2645.pt'
ckpt_path = './checkpoints/epoch=63-step=411903.ckpt'
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------
# Load model checkpoint
# -------------------------
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.serialization import safe_globals

with safe_globals([ModelCheckpoint]):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

model = HiVT(
    historical_steps=20,
    future_steps=30,
    num_modes=6,
    rotate=True,
    node_dim=2,
    edge_dim=2,
    embed_dim=64,        # must match trained model
    num_heads=8,
    dropout=0.1,
    num_temporal_layers=4,
    num_global_layers=3,
    local_radius=50,     # must match trained model
    parallel=False,
    lr=5e-4,
    weight_decay=1e-4,
    T_max=64
)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()
model.to(device)

# -------------------------
# Load single scene
# -------------------------
with safe_globals([TemporalData]):
    data: TemporalData = torch.load(scene_file)
scene_id = os.path.splitext(os.path.basename(scene_file))[0]

# Load corresponding lane data if available
lane_file = os.path.join(os.path.dirname(scene_file), "processed_lanes", f"{scene_id}_lanes.pt")
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

data = data.to(device)

# -------------------------
# Run inference
# -------------------------
with torch.no_grad():
    y_hat, pi = model(data)

# -------------------------
# Save results
# -------------------------
output_file = os.path.join(results_dir, f'scene_{scene_id}.json')

sample_result = {
    'y_hat': y_hat.cpu().tolist(),
    'pi': pi.cpu().tolist(),
    'lane_vectors': data.lane_vectors.cpu().tolist() if data.lane_vectors is not None else None,
    'lane_actor_index': data.lane_actor_index.cpu().tolist() if data.lane_actor_index is not None else None,
    'lane_actor_vectors': data.lane_actor_vectors.cpu().tolist() if data.lane_actor_vectors is not None else None,
    'is_intersections': data.is_intersections.cpu().tolist() if data.is_intersections is not None else None,
    'turn_directions': data.turn_directions.cpu().tolist() if data.turn_directions is not None else None,
    'traffic_controls': data.traffic_controls.cpu().tolist() if data.traffic_controls is not None else None
}

with open(output_file, 'w') as f:
    json.dump(sample_result, f, indent=2)

print(f"Results saved to {output_file}")
