import torch
import torch.nn as nn


class WeightedNLLLoss(nn.Module):
    """
    NLLLoss with class weights inferred from dataset or .npy file.
    """

    def __init__(self, class_weights: torch.Tensor, ignore_index=0, reduction="mean"):
        super().__init__()
        self.device = class_weights.device
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(
            weight=self.class_weights, ignore_index=self.ignore_index, reduction=self.reduction
        ).to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = nn.functional.log_softmax(input, dim=1)
        return self.nll_loss(input, target)
