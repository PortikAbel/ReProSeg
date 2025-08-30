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
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice
