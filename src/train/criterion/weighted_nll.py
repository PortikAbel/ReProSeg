import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with optional class weights inferred from the dataset.
    """

    def __init__(
        self, device: torch.device, class_weights: torch.Tensor | None = None, ignore_index=0, reduction="mean"
    ):
        super().__init__()
        self.device = device
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=self.ignore_index, reduction=self.reduction
        ).to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(input, target)
