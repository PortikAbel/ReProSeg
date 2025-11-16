from typing import Tuple

from torch import Generator
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

from config import ReProSegConfig
from data.data_split import DataSplit
from data.dataset import Dataset


def get_train_val_split(cfg: ReProSegConfig) -> Tuple[Subset[Dataset], Subset[Dataset]]:
    train_set = Dataset(cfg.data, DataSplit.TRAIN)
    if cfg.data.validation_size is not None:
        train_subset: Subset[Dataset]
        valid_subset: Subset[Dataset]
        train_subset, valid_subset = random_split(
            train_set,
            [1 - cfg.data.validation_size, cfg.data.validation_size],
            generator=Generator().manual_seed(cfg.env.seed),
        )
    else:
        train_subset = Subset(train_set, range(len(train_set)))
        valid_set = Dataset(cfg.data, DataSplit.VAL)
        valid_subset = Subset(valid_set, range(len(valid_set)))
    return train_subset, valid_subset
