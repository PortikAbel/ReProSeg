from pathlib import Path
import importlib

import torch
import torch.nn as nn

deeplab_module = importlib.import_module("model.segmentation_features.deeplab_v3_features")


class _FakeASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=1, bias=False), nn.BatchNorm2d(4)),
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(4)),
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(4)),
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(4)),
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(4, 4, kernel_size=1, bias=False), nn.BatchNorm2d(4)),
            ]
        )


class _FakeDeepLab(nn.Module):
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False), nn.BatchNorm2d(4))
        self.classifier = nn.ModuleList(
            [
                _FakeASPP(),
                nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Conv2d(4, num_classes, kernel_size=1),
            ]
        )


def _filled_like(tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
    if tensor.is_floating_point() or tensor.is_complex():
        return torch.full_like(tensor, fill_value)
    return torch.zeros_like(tensor)


def test_deeplab_v3_features_loads_external_checkpoint(monkeypatch):
    num_classes = 19
    checkpoint_model = _FakeDeepLab(num_classes=num_classes)
    checkpoint_state_dict = {
        f"model.{key}": _filled_like(value, float(index + 1))
        for index, (key, value) in enumerate(checkpoint_model.state_dict().items())
    }

    created_models: list[_FakeDeepLab] = []
    constructor_kwargs: dict[str, object] = {}

    def fake_deeplabv3_resnet50(**kwargs):
        constructor_kwargs.update(kwargs)
        model = _FakeDeepLab(num_classes=kwargs.get("num_classes", num_classes))
        created_models.append(model)
        return model

    monkeypatch.setattr(deeplab_module, "deeplabv3_resnet50", fake_deeplabv3_resnet50)
    monkeypatch.setattr(deeplab_module.torch, "load", lambda *args, **kwargs: {"state_dict": checkpoint_state_dict})

    backbone, aspp_convs = deeplab_module.deeplab_v3_features(
        pretrained=False,
        checkpoint_path=Path("/tmp/deeplab.ckpt"),
    )

    assert constructor_kwargs["weights"] is None
    assert constructor_kwargs["weights_backbone"] is None
    assert constructor_kwargs["num_classes"] == num_classes

    loaded_model = created_models[0]
    assert backbone is loaded_model.backbone
    expected_aspp_convs = loaded_model.classifier[0].convs[1:-1]
    assert len(aspp_convs) == len(expected_aspp_convs)
    assert all(returned is expected for returned, expected in zip(aspp_convs, expected_aspp_convs))
    assert torch.allclose(backbone[0].weight, checkpoint_state_dict["model.backbone.0.weight"])
    assert torch.allclose(aspp_convs[0][0].weight, checkpoint_state_dict["model.classifier.0.convs.1.0.weight"])
    assert torch.allclose(aspp_convs[1][0].weight, checkpoint_state_dict["model.classifier.0.convs.2.0.weight"])
    assert torch.allclose(aspp_convs[2][0].weight, checkpoint_state_dict["model.classifier.0.convs.3.0.weight"])