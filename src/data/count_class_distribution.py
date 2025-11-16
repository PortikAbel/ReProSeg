from pathlib import Path

import numpy as np
import torch

from data.dataloader import DataLoader
from utils.log import Log


def get_class_weights(dl: DataLoader, num_classes: int, cache_path: Path, log: Log) -> torch.Tensor:
    if cache_path.is_file():
        class_counts = np.load(cache_path)
        log.info(f"Loaded class counts from {cache_path}: {class_counts}")
    else:
        class_counts = count_class_distribution(dl, num_classes, cache_path)
        log.info(f"Calculated class counts: {class_counts}")
    class_weights = 1 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights


def count_class_distribution(dl: DataLoader, num_classes: int, save_path: Path) -> np.ndarray:
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dl:
        for c in range(num_classes):
            class_counts[c] += (label == c).sum()

    np.save(save_path, class_counts)

    return class_counts
