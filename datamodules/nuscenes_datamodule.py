import os
from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader as GeoDataLoader

from datasets.nuscenes_dataset import NuScenesDataset


class NuScenesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "./datasets",
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """Check that processed data exists."""
        train_path = os.path.join(self.root, "train", "processed")
        val_path = os.path.join(self.root, "val", "processed")
        if not os.path.exists(train_path) or not os.listdir(train_path):
            raise FileNotFoundError(f"No preprocessed train files found at {train_path}")
        if not os.path.exists(val_path) or not os.listdir(val_path):
            raise FileNotFoundError(f"No preprocessed val files found at {val_path}")

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            # Only pass the split folder (train/val), dataset handles 'processed' internally
            self.train_dataset = NuScenesDataset(os.path.join(self.root, "train"))
            self.val_dataset = NuScenesDataset(os.path.join(self.root, "val"))

    def train_dataloader(self):
        return GeoDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return GeoDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )