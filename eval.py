import torch
from torch_geometric.data import DataLoader
import pytorch_lightning as pl

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT

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
    val_dataset = ArgoverseV1Dataset(root=root, split='val', local_radius=model.hparams.local_radius)
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
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if gpus > 0:
                data = data.to('cuda')
            y_hat, pi = model(data)

            # Print batch-level info
            print(f'\nBatch {batch_idx} results:')
            for i in range(y_hat.size(0)):
                print(f'  Sample {i}:')
                print(f'    y_hat: {y_hat[i].cpu().numpy()}')
                print(f'    pi: {pi[i].cpu().numpy()}')
