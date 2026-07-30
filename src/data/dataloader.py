import numpy as np
import random
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from config import ReProSegConfig


class DataLoader(TorchDataLoader):
    """Custom wrapper for `torch.utils.data.DataLoader`."""

    dataset: TorchDataset
    to_shuffle: bool = False
    to_drop_last: bool = False
    cfg: ReProSegConfig

    def __init__(self, dataset: TorchDataset, cfg: ReProSegConfig):
        self.dataset = dataset
        self.cfg = cfg
        super().__init__(
            self.dataset,
            batch_size=cfg.data.batch_size,
            shuffle=self.to_shuffle,
            sampler=None,
            pin_memory=torch.cuda.is_available(),
            num_workers=cfg.data.num_workers,
            # worker_init_fn=lambda worker_id: np.random.seed(cfg.env.seed + worker_id),
            worker_init_fn=self._seed_worker,
            drop_last=self.to_drop_last,
        )

    def _seed_worker(self, worker_id):
        worker_seed = self.cfg.env.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    