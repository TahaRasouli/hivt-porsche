from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import nuscenes_datamodule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE',
                        choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)

    # HiVT model-specific args
    parser = HiVT.add_model_specific_args(parser)

    args = parser.parse_args()

    # Create model
    model = HiVT(**vars(args))

    # Create datamodule
    datamodule = nuscenes_datamodule.NuScenesDataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers
    )

    # Checkpoint callback
    model_checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        callbacks=[model_checkpoint]
    )

    # Train
    trainer.fit(model, datamodule=datamodule)
