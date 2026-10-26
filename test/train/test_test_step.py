from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, TensorDataset

from model.model import NonNegConv1x1
from train.test_step import eval
from utils.run_context import RunContext


def test_eval_logs_sparsity_and_abstained(tmp_path, monkeypatch):
    import utils.run_context as run_context

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
    run = RunContext(tmp_path)
    monkeypatch.setattr(run_context, "_run_context", run)
    try:
        info = eval(cfg, EvalNet(), loader, epoch=7)
    finally:
        run.close()

    assert info.num_non_zero_concepts == 2
    assert info.abstained == pytest.approx(0.25)
    events = EventAccumulator(str(run.tensorboard_dir)).Reload()
    for tag, expected in [
        ("sparsity/eval-epochs", classifier.sparsity),
        ("abstained/eval-epochs", info.abstained),
    ]:
        (scalar,) = events.Scalars(tag)
        assert scalar.step == 7
        assert scalar.value == pytest.approx(expected)
