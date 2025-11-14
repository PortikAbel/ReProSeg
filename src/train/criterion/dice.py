import torch.nn as nn
from torch import Tensor, arange


class DiceLoss(nn.Module):
    def __init__(self, class_weights: Tensor, ignore_index=0, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.mask = arange(class_weights.size(0)) != ignore_index
        self.class_weights = class_weights[self.mask]

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = nn.functional.softmax(inputs, dim=1)
        inputs = inputs.permute(0, 2, 3, 1)  # Change to (batch_size, height, width, num_classes)
        if targets.dim() == inputs.dim() - 1:
            targets = nn.functional.one_hot(targets, num_classes=inputs.shape[-1])

        dims = (0, 1, 2)  # Sum over batch, height, and width
        dice_by_class = (2.0 * (inputs * targets).sum(dim=dims) + self.smooth) / (
            inputs.sum(dim=dims) + targets.sum(dim=dims) + self.smooth
        )
        dice_by_class = dice_by_class[self.mask]
        dice_by_class = dice_by_class * self.class_weights
        return 1 - dice_by_class.mean()
