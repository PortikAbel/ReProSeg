import torch.nn as nn


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for name, module in model.named_modules():
            if "layer" in name and isinstance(module, nn.Conv2d):
                yield from module.parameters()

    # For Conv weight in the ASPP module
    if key == "10x":
        for name, module in model.named_modules():
            if "aspp" in name and isinstance(module, nn.Conv2d):
                yield module.weight

    # For Conv bias in the ASPP module
    if key == "20x":
        for name, module in model.named_modules():
            if "aspp" in name and isinstance(module, nn.Conv2d) and module.bias is not None:
                yield module.bias
