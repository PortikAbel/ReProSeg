"""Regression tests for backbone BatchNorm statistics across training phases."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

import model.model as model_module
from config.schema.model import LossWeights
from model.model import ReProSeg, TrainPhase
from train.optim.manager import OptimizerSchedulerManager
from train.train_step import train


class _Backbone(nn.Sequential):
    def forward(self, xs):
        return {"out": super().forward(xs)}


def test_backbone_batchnorm_statistics_follow_training_phase(monkeypatch):
    backbone = _Backbone(nn.Sequential(nn.BatchNorm2d(1)))
    aspp_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))])
    monkeypatch.setitem(
        model_module.base_architecture_to_features,
        "fake",
        lambda **_kwargs: (backbone, aspp_convs),
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            backbone_network="fake",
            disable_pretrained=True,
            backbone_checkpoint=None,
            checkpoint=None,
            bias=False,
            loss_weights=LossWeights(),
        ),
        data=SimpleNamespace(require_num_classes=lambda: 2),
        env=SimpleNamespace(device=torch.device("cpu")),
    )
    net = ReProSeg(cfg)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    manager = MagicMock(spec=OptimizerSchedulerManager)
    manager.reset_gradients.side_effect = optimizer.zero_grad
    manager.step.side_effect = lambda _phase: optimizer.step()
    xs = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    ys = torch.ones(2, 1, 4, 4, dtype=torch.long)
    loader = [(xs, ys), (xs + 2, ys)]
    backbone_bn = backbone[0][0]
    head_bn = aspp_convs[0][1]

    for epoch, phase in enumerate((TrainPhase.FINETUNE, TrainPhase.FINETUNE, TrainPhase.FULL_TRAINING), start=1):
        if phase is TrainPhase.FINETUNE:
            net.finetune()
        else:
            net.full_train()
        net.eval()
        before = {name: buffer.clone() for name, buffer in backbone_bn.named_buffers()}
        head_batches_before = head_bn.num_batches_tracked.clone()

        train(cfg, net, loader, manager, nn.CrossEntropyLoss(), epoch)

        frozen = phase is TrainPhase.FINETUNE
        assert net.training
        assert backbone.training
        assert backbone_bn.training is not frozen
        assert all(param.requires_grad is not frozen for param in backbone.parameters())
        assert head_bn.training
        assert head_bn.num_batches_tracked == head_batches_before + len(loader)
        for name, buffer in backbone_bn.named_buffers():
            assert torch.equal(buffer, before[name]) is frozen, name
