from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


def deeplab_v3_features(pretrained, **kwargs):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    original_model = deeplabv3_resnet50(weights=weights, **kwargs)

    # Extract the backbone and ASPP module
    backbone = original_model.backbone
    aspp_convs = original_model.classifier[0].convs[1:-1]  # ASPP module

    # Delete the original model to free up memory
    del original_model

    return backbone, aspp_convs
