"""Tests for shared ASPP convolution weights."""

from types import SimpleNamespace

import torch
import torch.nn as nn

import model.model as model_module


def _create_layers(monkeypatch):
    aspp_convs = nn.ModuleList(
        [
            nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
            )
            for _ in range(3)
        ]
    )

    monkeypatch.setitem(
        model_module.base_architecture_to_features,
        "fake",
        lambda **_kwargs: (nn.Identity(), aspp_convs),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            backbone_network="fake",
            disable_pretrained=True,
            backbone_checkpoint=None,
            bias=False,
        ),
        data=SimpleNamespace(require_num_classes=lambda: 2),
    )

    return model_module.ReProSegLayers(config)


def test_aspp_convolutions_share_parameter(monkeypatch):
    layers = _create_layers(monkeypatch)

    assert all(conv[0].weight is layers.shared_weights for conv in layers.aspp_convs)


def test_shared_parameter_accumulates_aspp_gradients(monkeypatch):
    layers = _create_layers(monkeypatch)
    inputs = torch.ones(1, 1, 2, 2)
    outputs = torch.cat([conv[0](inputs) for conv in layers.aspp_convs], dim=0)
    outputs.sum().backward()

    assert torch.allclose(layers.shared_weights.grad, torch.full_like(layers.shared_weights, 12.0))


def test_param_groups_include_each_aspp_parameter_once(monkeypatch):
    layers = _create_layers(monkeypatch)
    net = model_module.ReProSeg.__new__(model_module.ReProSeg)
    nn.Module.__init__(net)
    net.layers = layers
    net._cfg = SimpleNamespace(model=SimpleNamespace(bias=False))

    net._init_param_groups()

    classifier_params = net.param_groups["classifier_head"]
    aspp_params = list(layers.aspp_convs.parameters())

    assert all(any(parameter is candidate for candidate in classifier_params) for parameter in aspp_params)
    assert sum(parameter is layers.shared_weights for parameter in classifier_params) == 1
