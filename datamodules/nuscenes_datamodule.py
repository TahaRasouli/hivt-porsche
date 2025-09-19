import os
from typing import Optional, Callable

import pytorch_lightning as pl
from torch_geometric.data import DataLoader, Batch

from datasets.nuscenes_dataset import NuScenesDataset


def pyg_collate_fn(batch):
    """
    Custom collate function for PyG datasets.
    Converts a list of Data objects into a Batch object.
    """
    if isinstance(batch, Batch):
        return batch
    if isinstance(batch, list) and all(hasattr(x, "num_nodes") for x in batch):
        return Batch.from_data_list(batch)
    raise TypeError(f"Unexpected batch type: {type(batch)}")


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
        train_path = os.path.join(self.root, "train", "processed")
        val_path = os.path.join(self.root, "val", "processed")
        if not os.path.exists(train_path) or not os.listdir(train_path):
            raise FileNotFoundError(f"No preprocessed train files found at {train_path}")
        if not os.path.exists(val_path) or not os.listdir(val_path):
            raise FileNotFoundError(f"No preprocessed val files found at {val_path}")

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = NuScenesDataset(
                os.path.join(self.root, "train"), transform=self.train_transform
            )
            self.val_dataset = NuScenesDataset(
                os.path.join(self.root, "val"), transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=pyg_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=pyg_collate_fn,
        )
