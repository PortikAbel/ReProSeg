"""Checkpoint loading and compatibility for supported segmentation models."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import torch
from torch import Tensor

from config.schema.model import ModelConfig
from model.model import ReProSeg
from model.proto_segmentation import PPNet

SupportedModel: TypeAlias = PPNet | ReProSeg


def _load_checkpoint(checkpoint_path: Path) -> SupportedModel | Mapping:
    """Load a checkpoint file: a serialized PPNet/ReProSeg object, or a state-dict mapping."""

    _register_legacy_checkpoint_modules()
    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def _build_supported_model(
    checkpoint: SupportedModel | Mapping, checkpoint_path: Path, model_cfg: ModelConfig | None = None
) -> SupportedModel:
    """Construct a PPNet or ReProSeg from an already-loaded training checkpoint.

    ``model_cfg`` supplies the architecture (backbone, etc.) when reconstructing a ReProSeg
    from a state dict; omit it to fall back to the default architecture.
    """

    if isinstance(checkpoint, (PPNet, ReProSeg)):
        return checkpoint

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected checkpoint {checkpoint_path} to contain a PPNet, ReProSeg, or state-dict mapping; "
            f"found {type(checkpoint).__name__}."
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a valid model_state_dict mapping.")

    classifier_weight = state_dict.get("layers.classification_layer.weight")
    if not isinstance(classifier_weight, Tensor) or classifier_weight.ndim != 4:
        raise TypeError(
            f"Checkpoint {checkpoint_path} is neither a serialized PPNet nor a recognizable ReProSeg checkpoint."
        )

    from config import BaseScenarioConfig

    config = BaseScenarioConfig(model=model_cfg) if model_cfg is not None else BaseScenarioConfig()
    config.env.device = torch.device("cpu")
    config.data.num_classes = int(classifier_weight.shape[0])
    config.model.checkpoint = None
    config.model.disable_pretrained = True
    config.model.bias = "layers.classification_layer.bias" in state_dict

    model = ReProSeg(cfg=config)
    model.load_state_dict(state_dict, strict=True)
    return model


def _register_legacy_checkpoint_modules() -> None:
    """Keep serialized ProtoSeg models loadable after dependency moves.

    Full-model ProtoSeg checkpoints contain pickle references to the original
    model module, DeepLab package, and segmentation utility module. Registering aliases
    before ``torch.load`` lets those trusted checkpoints resolve the same
    classes at their new canonical locations.
    """

    legacy_root = sys.modules.setdefault("proto_segmentation", ModuleType("proto_segmentation"))
    legacy_root.__path__ = []

    model_module = sys.modules.setdefault(
        "proto_segmentation.model", importlib.import_module("model.proto_segmentation")
    )
    legacy_root.__dict__["model"] = model_module

    current_package = "model.segmentation_features.deeplab_pytorch"
    legacy_package = "proto_segmentation.deeplab_pytorch"
    module_suffixes = (
        "",
        ".libs",
        ".libs.models",
        ".libs.models.deeplabv1",
        ".libs.models.deeplabv2",
        ".libs.models.deeplabv3",
        ".libs.models.deeplabv3plus",
        ".libs.models.msc",
        ".libs.models.resnet",
    )

    for suffix in module_suffixes:
        alias = legacy_package + suffix
        module = sys.modules.setdefault(alias, importlib.import_module(current_package + suffix))
        parent_name, _, child_name = alias.rpartition(".")
        setattr(sys.modules[parent_name], child_name, module)

    legacy_segmentation = sys.modules.setdefault(
        "proto_segmentation.segmentation", ModuleType("proto_segmentation.segmentation")
    )
    legacy_segmentation.__path__ = []
    legacy_root.__dict__["segmentation"] = legacy_segmentation
    legacy_utils = sys.modules.setdefault(
        "proto_segmentation.segmentation.utils", importlib.import_module("model.utils")
    )
    legacy_segmentation.__dict__["utils"] = legacy_utils
