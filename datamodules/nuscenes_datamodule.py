import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.nuscenes_dataset import NuScenesDataset


class NuScenesDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str = "/mnt/d/projects/datasets/NuScenes/v1.0-trainval",
                 train_batch_size: int = 4,
                 val_batch_size: int = 4,
                 num_workers: int = 4,
                 local_radius: float = 50.0,
                 shuffle: bool = True):
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.local_radius = local_radius
        self.shuffle = shuffle

        self.train_dataset = None
        self.val_dataset = None

        # Placeholders for optional transforms (identity by default)
        self.train_transform = None
        self.val_transform = None

    def prepare_data(self):
        """Called only on 1 GPU to download/process data."""
        # Trigger processing of datasets
        _ = NuScenesDataset(root=self.root, split="train", transform=self.train_transform,
                            local_radius=self.local_radius)
        _ = NuScenesDataset(root=self.root, split="val", transform=self.val_transform,
                            local_radius=self.local_radius)

    def setup(self, stage: Optional[str] = None):
        """Called on every GPU. Instantiate datasets here."""
        if stage in (None, "fit"):
            self.train_dataset = NuScenesDataset(
                root=self.root,
                split="train",
                transform=self.train_transform,
                local_radius=self.local_radius
            )
            self.val_dataset = NuScenesDataset(
                root=self.root,
                split="val",
                transform=self.val_transform,
                local_radius=self.local_radius
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )