import torch.nn as nn
from torch import Tensor, arange


class DiceLoss(nn.Module):
    def __init__(self, class_weights: Tensor, ignore_index=0, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.mask = arange(class_weights.size(0)) != ignore_index
        masked_weights = class_weights[self.mask]
        self.class_weights = masked_weights / masked_weights.sum()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = inputs.permute(0, 2, 3, 1)  # Change to (batch_size, height, width, num_classes)
        if targets.dim() == inputs.dim() - 1:
            targets = nn.functional.one_hot(targets, num_classes=inputs.shape[-1])

        dims = (0, 1, 2)  # Sum over batch, height, and width
        dice_by_class = (2.0 * (inputs * targets).sum(dim=dims) + self.smooth) / (
            inputs.sum(dim=dims) + targets.sum(dim=dims) + self.smooth
        )
        dice_by_class = dice_by_class[self.mask]
        weighted_dice_sum = (dice_by_class * self.class_weights).sum()
        return 1 - weighted_dice_sum
