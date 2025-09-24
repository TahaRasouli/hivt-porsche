import os
from typing import Optional, Callable

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import NuScenesDataset


class NuScenesDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        super(NuScenesDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        """
        Prepare the datasets. This is called only once on a single GPU.
        It's responsible for downloading/processing data if needed.
        """
        # Initialize datasets to trigger processing if needed
        train_dataset = NuScenesDataset(self.root, 'train', self.train_transform, self.local_radius)
        val_dataset = NuScenesDataset(self.root, 'val', self.val_transform, self.local_radius)
        
        # Check if processing is needed and trigger it
        if not os.path.exists(train_dataset.processed_dir) or len(os.listdir(train_dataset.processed_dir)) == 0:
            print("Processing training data...")
            train_dataset.process()
            
        if not os.path.exists(val_dataset.processed_dir) or len(os.listdir(val_dataset.processed_dir)) == 0:
            print("Processing validation data...")
            val_dataset.process()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for training and validation.
        This is called on every GPU in distributed training.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = NuScenesDataset(self.root, 'train', self.train_transform, self.local_radius)
            self.val_dataset = NuScenesDataset(self.root, 'val', self.val_transform, self.local_radius)
            
        if stage == 'test' or stage is None:
            # Add test dataset if needed
            self.test_dataset = NuScenesDataset(self.root, 'test', self.val_transform, self.local_radius)

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=self.shuffle,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers
        )
        
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training/testing."""
        # Optional cleanup code
        pass
