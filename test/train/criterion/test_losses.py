"""Tests for classification criteria receiving raw class scores."""

import pytest
import torch
import torch.nn.functional as F

from train.criterion.dice import DiceLoss
from train.criterion.weighted_nll import WeightedCrossEntropyLoss


@pytest.mark.parametrize("class_weights", [None, torch.tensor([1.0, 2.0, 3.0])])
def test_cross_entropy_accepts_raw_scores(class_weights):
    scores = torch.tensor([[[[2.0]], [[0.0]], [[-1.0]]]])
    targets = torch.tensor([[[1]]])
    criterion = WeightedCrossEntropyLoss(torch.device("cpu"), class_weights=class_weights, ignore_index=0)

    actual = criterion(scores, targets)
    expected = F.cross_entropy(scores, targets, weight=class_weights, ignore_index=0)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("class_weights", [torch.ones(3), torch.tensor([1.0, 2.0, 3.0])])
def test_dice_applies_softmax_to_raw_scores(class_weights):
    scores = torch.tensor([[[[2.0]], [[0.0]], [[-1.0]]]])
    targets = torch.tensor([[[1]]])
    criterion = DiceLoss(class_weights, ignore_index=0)

    actual = criterion(scores, targets)

    probabilities = scores.softmax(dim=1).permute(0, 2, 3, 1)
    one_hot_targets = F.one_hot(targets, num_classes=scores.shape[1])
    dims = (0, 1, 2)
    dice_by_class = (2 * (probabilities * one_hot_targets).sum(dim=dims) + criterion.smooth) / (
        probabilities.sum(dim=dims) + one_hot_targets.sum(dim=dims) + criterion.smooth
    )
    expected = 1 - (dice_by_class[criterion.mask] * criterion.class_weights).sum()

    assert torch.allclose(actual, expected)
