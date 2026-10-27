from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.model import NonNegConv1x1
from train.test_step import eval


def test_eval_returns_sparsity_and_abstained_without_logging_scalars(mock_run_context):
    classifier = NonNegConv1x1(4, 2, bias=False)
    with torch.no_grad():
        classifier.weight.fill_(-1)
        classifier.weight[0, 0] = classifier.MIN_CLASSIFICATION_WEIGHT
        classifier.weight[:, 1] = classifier.MIN_CLASSIFICATION_WEIGHT + 0.0001

    class EvalNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Module()
            self.layers.classification_layer = classifier

        def forward(self, xs, inference=False):
            return xs, xs, xs

    cfg = SimpleNamespace(
        env=SimpleNamespace(device="cpu"),
        data=SimpleNamespace(require_num_classes=lambda: 2),
    )
    xs = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]]], [[[0.0, 0.0]], [[1.0, 1.0]]]])
    ys = torch.zeros((2, 1, 1, 2), dtype=torch.long)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=1)
    info = eval(cfg, EvalNet(), loader, epoch=7)

    assert info.num_non_zero_concepts == 2
    assert info.abstained == pytest.approx(0.25)
    assert info.sparsity == classifier.sparsity
    mock_run_context.tb_scalar.assert_not_called()
