from collections.abc import Callable
from typing import Any

from model.segmentation_features.deeplab_v2_features import deeplabv2_resnet101_features
from model.segmentation_features.deeplab_v3_features import deeplab_v3_features

base_architecture_to_features: dict[str, Callable[..., Any]] = {
    "deeplabv2_resnet101": deeplabv2_resnet101_features,
    "deeplab_v3": deeplab_v3_features,
}
