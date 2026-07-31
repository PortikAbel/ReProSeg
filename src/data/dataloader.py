import random

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from config import DataConfig

NUMPY_SEED_MODULUS = 2**32  # numpy seeds must be in range [0, 2^32)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % NUMPY_SEED_MODULUS
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataLoader(TorchDataLoader):
    """Custom wrapper for `torch.utils.data.DataLoader`."""

    dataset: TorchDataset
    to_shuffle: bool = False
    to_drop_last: bool = False

    def __init__(self, dataset: TorchDataset, cfg: DataConfig):
        self.dataset = dataset
        super().__init__(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=self.to_shuffle,
            sampler=None,
            pin_memory=torch.cuda.is_available(),
            num_workers=cfg.num_workers,
            worker_init_fn=seed_worker,
            drop_last=self.to_drop_last,
        )
