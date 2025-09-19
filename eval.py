import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets.nuscenes_dataset import NuScenesDataset
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--persistent_workers', action='store_true')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    args = parser.parse_args()

    # Load model from checkpoint
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    
    # Print model configuration
    print("Model Configuration:")
    print(f"  Historical steps: {model.hparams.historical_steps}")
    print(f"  Future steps: {model.hparams.future_steps}")
    print(f"  Embed dim: {model.hparams.embed_dim}")
    print(f"  Local radius: {model.hparams.local_radius}")
    print()

    # Create dataset
    dataset = NuScenesDataset(
        root=args.root,
        split=args.split
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0
    )

    # Create trainer
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator=args.accelerator,
        logger=False,  # Disable logging for evaluation
        enable_checkpointing=False,  # Disable checkpointing for evaluation
    )

    # Run evaluation
    print(f"Running evaluation on {args.split} split...")
    if args.split == 'val':
        results = trainer.validate(model, dataloader, verbose=True)
    else:
        results = trainer.test(model, dataloader, verbose=True)
    
    # Print results
    print("\nEvaluation Results:")
    if results:
        for key, value in results[0].items():
            if 'val_' in key or 'test_' in key:
                metric_name = key.replace('val_', '').replace('test_', '')
                print(f"  {metric_name}: {value:.4f}")
    
    print("\nEvaluation completed!")