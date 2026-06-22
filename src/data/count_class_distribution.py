from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from config import DataConfig
from data import DataLoader
from data.dataset.base import Dataset
from utils.log import Log


def get_class_weights(
    data: TorchDataset, cfg: DataConfig, log: Log
) -> torch.Tensor:
    cache_path = cfg.class_distribution_cache_path
    if cache_path.is_file():
        class_counts = np.load(cache_path)
        log.info(f"Loaded class counts from {cache_path}")
    else:
        ds = Dataset(cfg, data)
        dl = DataLoader(ds, cfg)
        class_counts = _count_class_distribution(dl, cfg.num_classes, cache_path)
        log.info("Calculated class counts.")
    class_weights = 1 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights


def _count_class_distribution(dl: DataLoader, num_classes: int, save_path: Path) -> np.ndarray:
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dl:
        for c in range(num_classes):
            class_counts[c] += (label == c).sum()

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, class_counts)

    return class_counts
