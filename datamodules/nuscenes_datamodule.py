from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
import pytorch_lightning as pl
import os
from typing import Optional
from datasets.nuscenes_dataset import NuScenesDataset

def pyg_collate_fn(batch):
    # If batch is already a Batch, just return it
    if isinstance(batch, Batch):
        return batch
    # If it's a list of Data objects, turn into Batch
    if isinstance(batch, list) and all(hasattr(x, 'num_nodes') for x in batch):
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
    ):
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def prepare_data(self):
        train_path = os.path.join(self.root, "train", "processed")
        val_path = os.path.join(self.root, "val", "processed")
        if not os.path.exists(train_path) or not os.listdir(train_path):
            raise FileNotFoundError(f"No preprocessed train files found at {train_path}")
        if not os.path.exists(val_path) or not os.listdir(val_path):
            raise FileNotFoundError(f"No preprocessed val files found at {val_path}")

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = NuScenesDataset(os.path.join(self.root, "train"))
            self.val_dataset = NuScenesDataset(os.path.join(self.root, "val"))

    def train_dataloader(self):
        return GeoDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=pyg_collate_fn
        )

    def val_dataloader(self):
        return GeoDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=pyg_collate_fn
        )