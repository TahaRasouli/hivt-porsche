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
    parser.add_argument('--max_epochs', type=int, default=100)  # Increased for better convergence
    parser.add_argument('--monitor', type=str, default='val_minFDE',
                        choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)

    # HiVT model-specific args
    parser = HiVT.add_model_specific_args(parser)

    args = parser.parse_args()

    # Print configuration for verification
    print("Training Configuration:")
    print(f"  Historical steps: {args.historical_steps}")
    print(f"  Future steps: {args.future_steps}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Batch size: {args.train_batch_size}")
    print()

    # Create model
    model = HiVT(**vars(args))

    # Create datamodule
    datamodule = nuscenes_datamodule.NuScenesDataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers
    )

    # Enhanced diagnostics
    print("=== DATA VALIDATION ===")
    datamodule.setup('fit')
    sample = datamodule.train_dataset[0]
    print(f"Sample type: {type(sample)}")
    print(f"Has num_nodes: {hasattr(sample, 'num_nodes')}")
    
    if hasattr(sample, 'num_nodes'):
        print(f"Num nodes: {sample.num_nodes}")
        print(f"X shape: {sample.x.shape}")
        print(f"Y shape: {sample.y.shape}")
        print(f"Expected X shape: [N, {args.historical_steps}, 2]")
        print(f"Expected Y shape: [N, {args.future_steps}, 2]")
        
        # Check if shapes match expectations
        expected_x_shape = (sample.num_nodes, args.historical_steps, 2)
        expected_y_shape = (sample.num_nodes, args.future_steps, 2)
        
        x_shape_correct = sample.x.shape == expected_x_shape
        y_shape_correct = sample.y.shape == expected_y_shape
        
        print(f"X shape correct: {x_shape_correct}")
        print(f"Y shape correct: {y_shape_correct}")
        
        if not (x_shape_correct and y_shape_correct):
            print("WARNING: Data shapes don't match model expectations!")
            print("You may need to reprocess data or adjust model parameters.")
        
        # Check coordinate ranges
        if hasattr(sample, 'positions'):
            pos_min = sample.positions.min().item()
            pos_max = sample.positions.max().item()
            print(f"Position range: {pos_min:.2f} to {pos_max:.2f}")
            if abs(pos_max) > 200:
                print("WARNING: Positions seem to be in wrong coordinate system!")
        
        print(f"Padding ratio: {sample.padding_mask.float().mean():.1%}")
    print("=" * 25)

    # Checkpoint callback with more frequent saving
    model_checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode='min',
        save_last=True,  # Always save the last checkpoint
        every_n_epochs=5,  # Save every 5 epochs
        verbose=True
    )

    # Trainer with better logging
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        callbacks=[model_checkpoint],
        log_every_n_steps=10,  # Log more frequently
        enable_progress_bar=True,
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )

    # Train
    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)