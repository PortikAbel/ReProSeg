from model.segmentation_features.deeplab_v3_features import deeplab_v3_features

base_architecture_to_features = {
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
