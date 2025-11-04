import torch.nn as nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = nn.functional.softmax(inputs, dim=1)
        inputs = inputs.permute(0, 2, 3, 1)  # Change to (batch_size, height, width, num_classes)
        if targets.dim() == inputs.dim() - 1:
            targets = nn.functional.one_hot(targets, num_classes=inputs.shape[-1])
        dims = (0, 1, 2)  # Sum over batch, height, and width
        dice = (
            (2.0 * (inputs * targets).sum(dim=dims) + self.smooth)
            / (inputs.sum(dim=dims) + targets.sum(dim=dims) + self.smooth)
        ).mean()
        return 1 - dice
