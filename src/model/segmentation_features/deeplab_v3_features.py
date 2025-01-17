from torchvision.models.segmentation import deeplabv3_resnet50

def deeplab_v3_features(pretrained):
    weights = "DEFAUT" if pretrained else None
    original_model = deeplabv3_resnet50(weights=weights)
        
    # Extract the backbone and ASPP module
    backbone = original_model.backbone
    aspp = original_model.classifier[0]  # ASPP module
    
    # Delete the original model to free up memory
    del original_model

    return backbone, aspp