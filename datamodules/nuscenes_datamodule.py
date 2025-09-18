import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.nuscenes_dataset import NuScenesDataset


class NuScenesDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading preprocessed NuScenes data stored as .pt files.
    Assumes:
        ./datasets/train/data.pt
        ./datasets/val/data.pt
    """
    def __init__(self,
                 root: str = "./datasets",
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 num_workers: int = 4,
                 shuffle: bool = True):
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """No raw dataset preprocessing required, just check files exist."""
        train_path = os.path.join(self.root, "train", "data.pt")
        val_path = os.path.join(self.root, "val", "data.pt")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"No preprocessed train file found at {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"No preprocessed val file found at {val_path}")
        print(f"✅ Found preprocessed data at {train_path} and {val_path}")

    def setup(self, stage: Optional[str] = None):
        """Called on every GPU. Instantiate datasets here."""
        if stage in (None, "fit"):
            self.train_dataset = NuScenesDataset(root=self.root, split="train")
            self.val_dataset = NuScenesDataset(root=self.root, split="val")

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
