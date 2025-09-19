import os
from typing import Optional, Callable

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets.nuscenes_dataset import NuScenesDataset


class NuScenesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "./datasets",
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """Check that processed data exists."""
        train_path = os.path.join(self.root, "train", "processed")
        val_path = os.path.join(self.root, "val", "processed")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"No train processed directory found at {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"No val processed directory found at {val_path}")
            
        train_files = [f for f in os.listdir(train_path) if f.endswith('.pt')]
        val_files = [f for f in os.listdir(val_path) if f.endswith('.pt')]
        
        if len(train_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {train_path}")
        if len(val_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {val_path}")
            
        print(f"Found {len(train_files)} training files and {len(val_files)} validation files")

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training and validation."""
        if stage in (None, "fit"):
            # FIXED: Pass root as base directory, let dataset handle the split
            self.train_dataset = NuScenesDataset(
                root=self.root,        # Just "./datasets"
                split="train",         # Dataset will create "./datasets/train/processed"
                transform=self.train_transform
            )
            self.val_dataset = NuScenesDataset(
                root=self.root,        # Just "./datasets" 
                split="val",           # Dataset will create "./datasets/val/processed"
                transform=self.val_transform
            )

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,  # Ensures consistent batch sizes
        )

    def val_dataloader(self):
        """Create validation dataloader.""" 
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,  # Keep all validation samples
        )