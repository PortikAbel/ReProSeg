"""Model configuration schemas."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

from config.schema.base import BaseConfig


class BackboneArchitecture(str, Enum):
    """Supported network backbones."""

    DEEPLAB_V3 = "deeplab_v3"
    RESNET = "resnet"
    VGG = "vgg"
    CONVNEXT = "convnext"


class LossCriterion(str, Enum):
    """Supported loss criteria."""

    DICE = "dice"
    WEIGHTED_DICE = "weighted_dice"
    WEIGHTED_NLL = "weighted_nll"


class LossWeights(BaseConfig):
    """Weights for different loss components."""

    alignment: float = Field(default=1.0, description="Ensures prototypes of similar images are aligned")
    jsd: float = Field(default=0.0, description="Jensen-Shannon Divergence: enforces prototype diversity")
    tanh: float = Field(default=0.0, description="Ensures each prototype is activated at least once per batch")
    uniformity: float = Field(default=0.0, description="Optional uniformity loss")
    variance: float = Field(default=0.0, description="Prototype feature variance regularizer")
    classification: float = Field(default=1.0, description="Standard classification loss weight")


class ModelConfig(BaseConfig):
    """Model architecture and parameters configuration."""

    # Network settings
    backbone_network: BackboneArchitecture = Field(
        default=BackboneArchitecture.DEEPLAB_V3, description="Backbone network"
    )
    checkpoint: Optional[Path] = Field(default=None, description="Path to ReProSeg checkpoint to resume from")

    # Architecture parameters
    num_prototypes: int = Field(default=0, ge=0, description="Number of prototypes. 0 = use backbone output channels")
    bias: bool = Field(default=False, description="Include a bias in classification layer if true")
    disable_pretrained: bool = Field(
        default=False, description="Initialize backbone randomly instead of using pretrained weights"
    )
    train_backbone_during_pretrain: bool = Field(
        default=False, description="Whether to train the full backbone during pretraining"
    )
    criterion: LossCriterion = Field(default=LossCriterion.DICE, description="Loss function type")
    loss_weights: LossWeights = Field(default_factory=LossWeights, description="Weights for different loss components")
