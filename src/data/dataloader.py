import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from config import ReProSegConfig
from data import SupportedSplit
from data.dataset import Dataset


class DataLoader(TorchDataLoader):
    """Custom wrapper for `torch.utils.data.DataLoader`."""

    dataset: TorchDataset
    to_shuffle: bool = False
    to_drop_last: bool = False

    def __init__(self, dataset: TorchDataset, cfg: ReProSegConfig):
        self.dataset = dataset
        super().__init__(
            self.dataset,
            batch_size=cfg.data.batch_size,
            shuffle=self.to_shuffle,
            sampler=None,
            pin_memory=torch.cuda.is_available(),
            num_workers=cfg.data.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(cfg.env.seed + worker_id),
            drop_last=self.to_drop_last,
        )

    @classmethod
    def from_split(cls, split: SupportedSplit, cfg: ReProSegConfig) -> "DataLoader":
        dataset = Dataset(cfg.data, split)
        return cls(dataset, cfg)
