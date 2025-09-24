from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from datasets.argoverse_v1_dataset import ArgoverseV1Dataset
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    # Lightning 2.x: manually configure trainer
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        num_sanity_val_steps=0
    )

    # Load model from checkpoint
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)

    # Load preprocessed validation dataset (skip any raw data processing)
    val_dataset = ArgoverseV1Dataset(
        root=args.root,
        split='val',
        local_radius=model.hparams.local_radius
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )

    # Run evaluation
    trainer.validate(model, dataloader)
