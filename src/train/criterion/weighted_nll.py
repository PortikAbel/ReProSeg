from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import ReProSegConfig
from utils.log import Log


class WeightedNLLLoss(nn.Module):
    """
    NLLLoss with class weights inferred from dataset or .npy file.
    """

    def __init__(self, cfg: ReProSegConfig, log: Log, ignore_index=0, reduction="mean"):
        super().__init__()
        self.device = cfg.env.device
        self.class_weights = self._get_class_weights(cfg, log)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(
            weight=self.class_weights, ignore_index=self.ignore_index, reduction=self.reduction
        ).to(self.device)

    def _get_class_weights(self, cfg: ReProSegConfig, log: Log):
        class_counts_path = Path(__file__).parent.parent.parent / "data" / "class_counts.npy"
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

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = nn.functional.log_softmax(input, dim=1)
        return self.nll_loss(input, target)
