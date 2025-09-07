from argparse import Namespace
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.config import get_dataset_config
from utils.log import Log
from utils.func import init_weights_xavier
from model.segmentation_features import (
    base_architecture_to_features,
    base_architecture_to_layer_groups,
)


class TrainPhase(Enum):
    """
    Enum to define the training phase.
    """

    PRETRAIN = 0
    """
    Pretraining phase.
    """

    FINETUNE = 1
    """
    Finetuning phase.
    """

    FREEZE_FIRST_LAYERS = 2
    """
    Freeze first layers of the backbone.
    """

    FULL_TRAINING = 3
    """
    Full training phase.
    """


class ReProSegLayers(nn.Module):
    def __init__(self, args: Namespace, log: Log):
        super().__init__()

        features, aspp_convs = base_architecture_to_features[args.net](
            pretrained=not args.disable_pretrained,
            in_channels=get_dataset_config(args.dataset)["color_channels"],
        )
        self.feature_net = features
        self.aspp_convs = aspp_convs

        self.shared_weights = self.aspp_convs[0][0].weight
        # set shared weights to all aspp convolutions
        for conv in self.aspp_convs:
            conv[0].weight.data = self.shared_weights

        first_add_on_layer_in_channels = [m for m in aspp_convs.modules() if isinstance(m, nn.Conv2d)][-1].out_channels

        # the sum of prototype activations should be 1 for each patch in each scale
        self.add_on_layers: nn.Module = nn.Softmax(dim=1)

        if args.num_features == 0:
            self.num_prototypes = first_add_on_layer_in_channels
            log.info(f"Number of prototypes: {self.num_prototypes}")
        else:
            self.num_prototypes = args.num_features
            log.info(
                f"Number of prototypes set from {first_add_on_layer_in_channels} to {self.num_prototypes}. "
                "Extra 1x1 conv layer added. Not recommended."
            )
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=first_add_on_layer_in_channels,
                    out_channels=self.num_prototypes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                self.add_on_layers,
            )

        self.max_pool = nn.AdaptiveMaxPool3d((1, None, None))
        self.classification_layer = NonNegConv1x1(self.num_prototypes, args.num_classes, bias=args.bias)


class ReProSeg(nn.Module):
    def __init__(
        self,
        args: Namespace,
        log: Log,
    ):
        super().__init__()
        assert args.num_classes > 0
        self._args = args
        self._log = log

        self.layers = ReProSegLayers(args, log)
        self.num_prototypes = self.layers.num_prototypes

        self._init_param_groups()

    def forward(self, xs, inference=False):
        backbone_features = self.layers.feature_net(xs)["out"]
        # (b x num_prototypes x num_scales x h x w)
        aspp_features = torch.cat([conv(backbone_features) for conv in self.layers.aspp_convs], dim=0)
        aspp_features = self.layers.add_on_layers(aspp_features)
        aspp_features = torch.stack(torch.chunk(aspp_features, len(self.layers.aspp_convs), dim=0), dim=2)
        pooled = torch.squeeze(self.layers.max_pool(aspp_features), dim=2)

        if inference:
            # ignore all prototypes that have 0.1 similarity or lower
            pooled = torch.where(pooled < 0.1, 0.0, pooled)

        out = self.layers.classification_layer(pooled)

        if inference:
            out = F.interpolate(out, size=xs.shape[2:], mode="bilinear")

        return aspp_features, pooled, out

    def interpolate_prototype_activations(self, xs: torch.Tensor) -> torch.Tensor:
        original_shape = xs.shape[2:]

        activations = self.forward(xs, inference=True)[0]  # (batch x num_prototypes x scales x h x w)
        activations = activations.permute(2, 0, 1, 3, 4)  # (scales x batch x num_prototypes x h x w)
        max_scale = torch.argmax(activations, dim=(0))
        scales = activations.shape[0]

        interpolated_activations = torch.zeros(original_shape, device=activations.device)
        upscale = transforms.Resize(size=original_shape, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        for scale in range(scales):
            activations[scale, max_scale != scale] = 0
            padding = scale + 1
            max_pool = torch.nn.MaxPool2d(kernel_size=2 * padding + 1, padding=padding, stride=1)
            scaled_activations = upscale(max_pool(activations[scale]))
            interpolated_activations = torch.maximum(interpolated_activations, scaled_activations)
        return interpolated_activations

    def init_add_on_weights(self):
        self.layers.add_on_layers.apply(init_weights_xavier)

    def init_classifier_weights(self):
        torch.nn.init.normal_(self.layers.classification_layer.weight, mean=1.0, std=0.1)
        self._log.info(
            f"Classification layer initialized with mean {torch.mean(self.layers.classification_layer.weight).item()}"
        )
        if self._args.bias:
            torch.nn.init.constant_(self.layers.classification_layer.bias, val=0.0)

    def _init_param_groups(self):
        self.param_groups = dict()
        self.param_groups["backbone"] = []
        self.param_groups["to_train"] = []
        self.param_groups["to_freeze"] = []

        if self._args.net in base_architecture_to_layer_groups.keys():
            layer_groups = base_architecture_to_layer_groups[self._args.net]
            for name, param in self.layers.feature_net.named_parameters():
                if any(layer in name for layer in layer_groups["to_train"]):
                    self.param_groups["to_train"].append(param)
                elif any(layer in name for layer in layer_groups["to_freeze"]):
                    self.param_groups["to_freeze"].append(param)
                elif any(layer in name for layer in layer_groups["backbone"]):
                    self.param_groups["backbone"].append(param)
                else:
                    param.requires_grad = False
        else:
            self._log.warning("Layer groups not implemented for selected backbone architecture.")

        self.param_groups["to_train"].append(self.layers.shared_weights)
        for name, param in self.layers.aspp_convs.named_parameters():
            if "weight" in name:
                self.param_groups["to_train"].append(param)
            elif "bias" in name and self._args.bias:
                self.param_groups["to_train"].append(param)
        self.param_groups["classification_weight"] = []
        self.param_groups["classification_bias"] = []
        for name, param in self.layers.classification_layer.named_parameters():
            if "weight" in name:
                self.param_groups["classification_weight"].append(param)
            elif self._args.bias:
                self.param_groups["classification_bias"].append(param)

    def get_optimizers(self):
        paramlist_net = [
            {
                "params": self.param_groups["backbone"],
                "lr": self._args.lr_net,
                "weight_decay_rate": self._args.weight_decay,
            },
            {
                "params": self.param_groups["to_freeze"],
                "lr": self._args.lr_block,
                "weight_decay_rate": self._args.weight_decay,
            },
            {
                "params": self.param_groups["to_train"],
                "lr": self._args.lr_block,
                "weight_decay_rate": self._args.weight_decay,
            },
            {
                "params": self.layers.add_on_layers.parameters(),
                "lr": self._args.lr_block * 10.0,
                "weight_decay_rate": self._args.weight_decay,
            },
        ]

        paramlist_classifier = [
            {
                "params": self.param_groups["classification_weight"],
                "lr": self._args.lr,
                "weight_decay_rate": self._args.weight_decay,
            },
            {
                "params": self.param_groups["classification_bias"],
                "lr": self._args.lr,
                "weight_decay_rate": 0,
            },
        ]

        if self._args.optimizer == "Adam":
            optimizer_net = torch.optim.AdamW(paramlist_net, lr=self._args.lr, weight_decay=self._args.weight_decay)
            optimizer_classifier = torch.optim.AdamW(
                paramlist_classifier,
                lr=self._args.lr,
                weight_decay=self._args.weight_decay,
            )
            return (optimizer_net, optimizer_classifier)
        else:
            raise ValueError("this optimizer type is not implemented")

    def pretrain(self):
        self.train_phase = TrainPhase.PRETRAIN

        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self.layers.add_on_layers.parameters():
            param.requires_grad = True
        for param in self.layers.classification_layer.parameters():
            param.requires_grad = False
        for param in self.param_groups["to_freeze"]:
            param.requires_grad = True  # can be set to False when you want to freeze more layers
        for param in self.param_groups["backbone"]:
            # can be set to True when you want to train whole backbone
            # (e.g. if dataset is very different from ImageNet)
            param.requires_grad = self._args.train_backbone_during_pretrain

    def finetune(self):
        self.train_phase = TrainPhase.FINETUNE

        for param in self.parameters():
            param.requires_grad = False
        for param in self.layers.classification_layer.parameters():
            param.requires_grad = True

    def freeze(self):
        self.train_phase = TrainPhase.FREEZE_FIRST_LAYERS

        for param in self.param_groups["to_freeze"]:
            # Can be set to False if you want
            # to train fewer layers of backbone
            param.requires_grad = True
        for param in self.layers.add_on_layers.parameters():
            param.requires_grad = True
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self.param_groups["backbone"]:
            param.requires_grad = False

    def unfreeze(self):
        self.train_phase = TrainPhase.FULL_TRAINING

        for param in self.layers.add_on_layers.parameters():
            param.requires_grad = True
        for param in self.param_groups["to_freeze"]:
            param.requires_grad = True
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self.param_groups["backbone"]:
            param.requires_grad = True


class NonNegConv1x1(nn.Module):
    """Applies a 1x1 convolution to the incoming data with non-negative weights"""

    MIN_CLASSIFICATION_WEIGHT = 10

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NonNegConv1x1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, 1, 1), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = torch.where(self.weight < self.MIN_CLASSIFICATION_WEIGHT, 0.0, self.weight)
        return F.conv2d(input_, weight, self.bias, stride=1, padding=0)

    @property
    def used_prototypes(self) -> torch.Tensor:
        return (self.weight >= self.MIN_CLASSIFICATION_WEIGHT).any(dim=0).squeeze().nonzero().squeeze()
