import torch
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from datasets import NuScenesDataModule
from models.hivt import HiVT
import os
import json

if __name__ == '__main__':
    pl.seed_everything(2022)

    root = './datasets'
    batch_size = 4
    num_workers = 8
    pin_memory = True
    persistent_workers = True
    gpus = 1
    ckpt_path = './checkpoints/epoch=63-step=411903.ckpt'

    # -------------------------
    # Load full Lightning checkpoint safely
    # -------------------------
    # Allowlist the ModelCheckpoint class for unpickling
    from pytorch_lightning.callbacks import ModelCheckpoint

    from torch.serialization import safe_globals
    with safe_globals([ModelCheckpoint]):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # -------------------------
    # Initialize model with matching hyperparameters
    # -------------------------
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

    # -------------------------
    # Load state_dict with strict=False to ignore missing/unexpected keys
    # -------------------------
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    if gpus > 0:
        model = model.cuda()

    # -------------------------
    # Validation dataloader
    # -------------------------
    val_dataset = NuScenesDataModule(root=root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # -------------------------
    # Run evaluation
    # -------------------------
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if gpus > 0:
                data = data.to('cuda')
            y_hat, pi = model(data)

            batch_results = []
            for i in range(y_hat.size(0)):
                sample_result = {
                    'y_hat': y_hat[i].cpu().tolist(),  # convert to list for JSON
                    'pi': pi[i].cpu().tolist()
                }
                batch_results.append(sample_result)

                # Print each sample's results
                print(f'Batch {batch_idx}, Sample {i}:')
                print(f'  y_hat: {sample_result["y_hat"]}')
                print(f'  pi: {sample_result["pi"]}')

            # Save batch results as JSON
            batch_file = os.path.join(results_dir, f'batch_{batch_idx}.json')
            with open(batch_file, 'w') as f:
                json.dump(batch_results, f, indent=2)