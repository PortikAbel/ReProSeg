import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from config import ReProSegConfig
from config.schema.data import DatasetType
from data import SupportedSplit
from data.dataset import Dataset


class DataLoader(TorchDataLoader):
    """Base class for dataloaders"""

    split: SupportedSplit
    dataset: Dataset
    to_shuffle: bool = False
    to_drop_last: bool = False

    def __init__(self, split: SupportedSplit, cfg: ReProSegConfig):
        self.split = split
        self.dataset = self._create_dataset(cfg.data.dataset)
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

    def _create_dataset(self, dataset_name: DatasetType) -> Dataset:
        return Dataset(dataset_name, self.split)
