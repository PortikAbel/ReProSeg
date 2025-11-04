import numpy as np

from config import ReProSegConfig
from data.dataloader import DataLoader


def count_class_distribution(cfg: ReProSegConfig, save_path):
    dl = DataLoader("train", cfg)

    num_classes = len(dl.dataset.classes)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dl:
        for c in range(num_classes):
            class_counts[c] += (label == c).sum()

    np.save(save_path, class_counts)

    return class_counts
