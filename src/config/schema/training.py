"""Training configuration schemas."""

from enum import Enum

from pydantic import Field, ValidationInfo, field_validator

from config.schema.base import BaseConfig


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "Adam"
    SGD = "SGD"
    ADAMW = "AdamW"


class EpochConfig(BaseConfig):
    """Epoch configuration for different training phases."""

    pretrain: int = Field(default=0, ge=0, description="Epochs for prototype pretraining (stage 1)")
    total: int = Field(default=0, gt=0, description="Total training epochs (stage 2)")
    finetune: int = Field(default=0, ge=0, description="Finetuning epochs with frozen backbone")
    freeze: int = Field(
        default=0, ge=0, description="Epochs where last layers of backbone are trained with the classifier."
    )
    start: int = Field(default=1, gt=0, description="Starting epoch (for resuming)")

    @field_validator("finetune")
    def validate_finetune(cls, v, values: ValidationInfo):
        if "total" in values.data and v > values.data["total"]:
            raise ValueError("finetune epochs cannot be greater than total epochs")
        return v

    @field_validator("freeze")
    def validate_freeze(cls, v, values: ValidationInfo):
        if "total" in values.data and v > values.data["total"]:
            raise ValueError("freeze epochs cannot be greater than total epochs")
        return v


class LearningRateConfig(BaseConfig):
    """Learning rate configuration for different network components."""

    classifier: float = Field(default=1.0, gt=0.0, description="Learning rate for prototype → class weights")
    backbone_end: float = Field(default=1.0, gt=0.0, description="Learning rate for final backbone layers")
    backbone_full: float = Field(default=1.0, gt=0.0, description="Learning rate for rest of backbone")


class TrainingConfig(BaseConfig):
    """Training parameters and optimization configuration."""

    skip_training: bool = Field(
        default=False, description="Skips training and only visualizes prototypes and predictions"
    )
    epochs: EpochConfig = Field(default_factory=lambda: EpochConfig(), description="Epoch configuration")
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW, description="Optimizer type")
    learning_rates: LearningRateConfig = Field(
        default_factory=lambda: LearningRateConfig(), description="Learning rate configuration"
    )
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Weight decay (L2 regularization) factor")
