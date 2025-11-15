from model.segmentation_features.deeplab_v3_features import deeplab_v3_features
from proto_segmentation.deeplab_v2_features import deeplabv2_resnet101_features

base_architecture_to_features = {
    "deeplabv2_resnet101": deeplabv2_resnet101_features,
    "deeplab_v3": deeplab_v3_features,
}

# TODO set these values
base_architecture_to_layer_groups = {
    "deeplab_v3": {
        "to_train": [""],
        "to_freeze": [""],
        "backbone": ["backbone"],
    },
}
