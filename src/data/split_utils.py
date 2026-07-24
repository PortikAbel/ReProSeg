from typing import Tuple

from torch import Generator
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

from config import ReProSegConfig
from data.data_split import DataSplit
from data.dataset.factory import DatasetFactory


def get_train_val_split(cfg: ReProSegConfig) -> Tuple[Subset[TorchDataset], Subset[TorchDataset]]:
    train_set = DatasetFactory.create(cfg.data, split=DataSplit.TRAIN)
    if cfg.data.validation_size is not None:
        train_subset: Subset[TorchDataset]
        valid_subset: Subset[TorchDataset]
        train_subset, valid_subset = random_split(
            train_set,
            [1 - cfg.data.validation_size, cfg.data.validation_size],
            generator=Generator().manual_seed(cfg.env.seed),
        )
    else:
        train_subset = Subset(train_set, range(len(train_set)))
        valid_set = DatasetFactory.create(cfg.data, split=DataSplit.VAL)
        valid_subset = Subset(valid_set, range(len(valid_set)))
    return train_subset, valid_subset
