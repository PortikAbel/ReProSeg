from pathlib import Path

import torch
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


def _load_deeplab_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint at {checkpoint_path} to deserialize to a dict, got {type(checkpoint)!r}")

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint at {checkpoint_path} to contain a state dict, got {type(state_dict)!r}")

    return state_dict


def _extract_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("model.") for key in state_dict):
        model_state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")
        }
    else:
        model_state_dict = state_dict

    if not model_state_dict:
        raise KeyError("Checkpoint does not contain any DeepLab weights.")

    return model_state_dict


def _infer_num_classes(model_state_dict: dict[str, torch.Tensor]) -> int | None:
    classifier_weight = model_state_dict.get("classifier.4.weight")
    if classifier_weight is None:
        return None
    return int(classifier_weight.shape[0])


def deeplab_v3_features(pretrained, checkpoint_path=None, **kwargs):
    weights = None
    if checkpoint_path is None and pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT

    kwargs.setdefault("weights_backbone", None)

    model_state_dict = None
    if checkpoint_path is not None:
        model_state_dict = _extract_model_state_dict(_load_deeplab_checkpoint_state_dict(checkpoint_path))
        inferred_num_classes = _infer_num_classes(model_state_dict)
        if inferred_num_classes is not None:
            kwargs["num_classes"] = inferred_num_classes

    original_model = deeplabv3_resnet50(weights=weights, **kwargs)

    if model_state_dict is not None:
        original_model.load_state_dict(model_state_dict, strict=True)

    # Extract the backbone and ASPP module
    backbone = original_model.backbone
    aspp_convs = original_model.classifier[0].convs[1:-1]  # ASPP module

    # Delete the original model to free up memory
    del original_model

    return backbone, aspp_convs
