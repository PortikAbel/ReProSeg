from argparse import Namespace

import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader

from data import SupportedDataset, SupportedSplit
from data.dataset import Dataset


class DataLoader(TorchDataLoader):
    """Base class for dataloaders"""

    split: SupportedSplit
    dataset: Dataset
    to_shuffle: bool = False
    to_drop_last: bool = False

    def __init__(self, split: SupportedSplit, args: Namespace):
        self.split = split
        self.dataset = self._create_dataset(args.dataset)
        super().__init__(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=self.to_shuffle,
            sampler=None,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.num_workers,
            worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
            drop_last=self.to_drop_last,
        )

    def _create_dataset(self, dataset_name: SupportedDataset) -> Dataset:
        return Dataset(dataset_name, self.split)
