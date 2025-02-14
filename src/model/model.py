from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.config import DATASETS
from utils.log import Log
from model.segmentation_features import (
    base_architecture_to_features,
    base_architecture_to_layer_groups,
)


class ReProSeg(nn.Module):
    def __init__(
        self,
        args: Namespace,
        log: Log,
        num_classes: int,
        num_prototypes: int,
        feature_net: nn.Module,
        add_on_layers: nn.Module,
        aspp_convs: nn.Module,
        max_pool: nn.Module,
        classification_layer: nn.Module,
    ):
        super().__init__()
        assert num_classes > 0
        self._args = args
        self._log = log
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        
        self._net = feature_net
        self._add_on = add_on_layers
        self._aspp_convs = aspp_convs
        self._shared_weights = self._aspp_convs[0][0].weight
        for conv in self._aspp_convs:
            del conv[0].weight
        self._max_pool = max_pool
        
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier
        
        self._init_param_groups()

    def forward(self, xs, inference=False):
        # set shared weights to all aspp convolutions
        for conv in self._aspp_convs:
            conv[0].weight = self._shared_weights.clone()

        backbone_features = self._net(xs)["out"]
        aspp_features = torch.cat([conv(backbone_features) for conv in self._aspp_convs], dim=0)
        aspp_features = self._add_on(aspp_features)
        aspp_features = torch.stack(torch.chunk(aspp_features, len(self._aspp_convs), dim=0), dim=2)
        pooled = torch.squeeze(self._max_pool(aspp_features), dim=2)

        if inference:
            # ignore all prototypes that have 0.1 similarity or lower
            pooled = torch.where(pooled < 0.1, 0.0, pooled)
        
        out = self._classification(pooled)
        return aspp_features, pooled, out

    def _init_param_groups(self):
        self.param_groups = dict()
        self.param_groups["backbone"] = []
        self.param_groups["to_train"] = []
        self.param_groups["to_freeze"] = []

        if self._args.net in base_architecture_to_layer_groups.keys():
            layer_groups = base_architecture_to_layer_groups[self._args.net]
            for name, param in self._net.named_parameters():
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
        self.param_groups["classification_weight"] = []
        self.param_groups["classification_bias"] = []
        for name, param in self._classification.named_parameters():
            if "weight" in name:
                self.param_groups["classification_weight"].append(param)
            elif "multiplier" in name:
                param.requires_grad = False
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
                "params": self._add_on.parameters(),
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
            optimizer_net = torch.optim.AdamW(
                paramlist_net, lr=self._args.lr, weight_decay=self._args.weight_decay
            )
            optimizer_classifier = torch.optim.AdamW(
                paramlist_classifier,
                lr=self._args.lr,
                weight_decay=self._args.weight_decay,
            )
            return (optimizer_net, optimizer_classifier)
        else:
            raise ValueError("this optimizer type is not implemented")

    def pretrain(self):
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self._add_on.parameters():
            param.requires_grad = True
        for param in self._classification.parameters():
            param.requires_grad = False
        for param in self.param_groups["to_freeze"]:
            param.requires_grad = (
                True  # can be set to False when you want to freeze more layers
            )
        for param in self.param_groups["backbone"]:
            # can be set to True when you want to train whole backbone
            # (e.g. if dataset is very different from ImageNet)
            param.requires_grad = self._args.train_backbone_during_pretrain

    def finetune(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self._classification.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.param_groups["to_freeze"]:
            # Can be set to False if you want
            # to train fewer layers of backbone
            param.requires_grad = True
        for param in self._add_on.parameters():
            param.requires_grad = True
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self.param_groups["backbone"]:
            param.requires_grad = False

    def unfreeze(self):
        for param in self._add_on.parameters():
            param.requires_grad = True
        for param in self.param_groups["to_freeze"]:
            param.requires_grad = True
        for param in self.param_groups["to_train"]:
            param.requires_grad = True
        for param in self.param_groups["backbone"]:
            param.requires_grad = True


class NonNegConv1x1(nn.Module):
    """Applies a 1x1 convolution to the incoming data with non-negative weights"""

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
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels, 1, 1), **factory_kwargs)
        )
        #TODO check this if it is necessary
        self.normalization_multiplier = nn.Parameter(
            torch.ones((1,), requires_grad=True)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = torch.relu(self.weight)
        # TODO shouldn't be the bias also non-negative?
        return F.conv2d(input_, weight, self.bias, stride=1, padding=0)
    

def get_network(args: Namespace, log: Log, num_classes: int):
    feature_kwargs = {
        "in_channels": DATASETS[args.dataset]["color_channels"],
    }

    features, aspp_convs = base_architecture_to_features[args.net](
        pretrained=not args.disable_pretrained, **feature_kwargs,
    )
    first_add_on_layer_in_channels = [
        m for m in aspp_convs.modules() if isinstance(m, nn.Conv2d)
    ][-1].out_channels
    
    # the sum of prototype activations should be 1 for each patch in each scale
    add_on_layers = nn.Softmax(dim=1)

    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        log.info(f"Number of prototypes: {num_prototypes}")
    else:
        num_prototypes = args.num_features
        log.info(f"Number of prototypes set from {first_add_on_layer_in_channels} to {num_prototypes}. Extra 1x1 conv layer added. Not recommended.")
        add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=first_add_on_layer_in_channels,
                out_channels=num_prototypes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            add_on_layers
        )

    max_pool = nn.AdaptiveMaxPool3d((1, None, None))

    classification_layer = NonNegConv1x1(num_prototypes, num_classes, bias=args.bias)

    return (
        features,
        add_on_layers,
        aspp_convs,
        max_pool,
        classification_layer,
        num_prototypes,
    )