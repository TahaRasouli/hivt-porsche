import torch
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from datasets import ArgoverseV1Dataset
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    # --- CONFIG ---
    root = './datasets'
    batch_size = 4
    num_workers = 8
    pin_memory = True
    persistent_workers = True
    ckpt_path = './checkpoints/epoch=63-step=411903.ckpt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- LOAD CHECKPOINT HYPERPARAMETERS ---
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt['hyper_parameters'] if 'hyper_parameters' in ckpt else ckpt['hyper_params']

    # --- INIT MODEL ---
    model = HiVT(
        historical_steps=hparams['historical_steps'],
        future_steps=hparams['future_steps'],
        num_modes=hparams['num_modes'],
        rotate=hparams['rotate'],
        node_dim=hparams['node_dim'],
        edge_dim=hparams['edge_dim'],
        embed_dim=hparams['embed_dim'],
        num_heads=hparams['num_heads'],
        dropout=hparams['dropout'],
        num_temporal_layers=hparams['num_temporal_layers'],
        num_global_layers=hparams['num_global_layers'],
        local_radius=hparams['local_radius'],
        parallel=False,
        lr=hparams.get('lr', 5e-4),
        weight_decay=hparams.get('weight_decay', 1e-4),
        T_max=hparams.get('T_max', 64)
    )

    # --- LOAD CHECKPOINT ---
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    # --- DATASET & DATALOADER ---
    val_dataset = ArgoverseV1Dataset(root=root, split='val', local_radius=model.hparams.local_radius)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # --- VALIDATION ---
    trainer = pl.Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1 if device == 'cuda' else None,
        logger=False
    )
    trainer.validate(model, val_loader)
