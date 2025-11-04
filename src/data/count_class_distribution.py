import numpy as np
import torch

from config import ReProSegConfig
from data.dataloader import DataLoader
from utils.log import Log


def get_class_weights(self, cfg: ReProSegConfig, log: Log):
    class_counts_path = cfg.env.class_distribution_cache_path
    if class_counts_path.is_file():
        class_counts = np.load(class_counts_path)
        log.info(f"Loaded class counts from {class_counts_path}: {class_counts}")
    else:
        from data.count_class_distribution import count_class_distribution

        class_counts = count_class_distribution(cfg, class_counts_path)
        log.info(f"Calculated class counts: {class_counts}")
    class_weights = 1 / class_counts
    class_weights = torch.tensor(class_weights, device=self.device, dtype=torch.float32)
    return class_weights

def count_class_distribution(cfg: ReProSegConfig, save_path):
    dl = DataLoader("train", cfg)

    num_classes = len(dl.dataset.classes)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dl:
        for c in range(num_classes):
            class_counts[c] += (label == c).sum()

    np.save(save_path, class_counts)

    return class_counts
