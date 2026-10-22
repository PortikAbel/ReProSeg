"""Training configuration schemas."""

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from config.schema.base import BaseConfig


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "Adam"
    SGD = "SGD"
    ADAMW = "AdamW"


class EpochConfig(BaseConfig):
    """Epoch configuration for different training phases."""

    total: int = Field(default=0, gt=0, description="Total training epochs (stage 2)")
    finetune: int = Field(default=0, ge=0, description="Finetuning epochs with frozen backbone")
    start: int = Field(default=1, gt=0, description="Starting epoch (for resuming)")

    @field_validator("finetune")
    def validate_finetune(cls, v, values: ValidationInfo):
        if "total" in values.data and v > values.data["total"]:
            raise ValueError("finetune epochs cannot be greater than total epochs")
        return v


class LearningRateConfig(BaseConfig):
    """Learning rate configuration for different network components."""

    backbone: float = Field(default=1.0, gt=0.0, description="Learning rate for backbone network")
    classifier: float = Field(default=1.0, gt=0.0, description="Learning rate for concept → class weights")


class _SchedulerBase(BaseConfig):
    """Base for scheduler configs. Durations are expressed in epochs."""

    model_config = ConfigDict(extra="forbid")


class CosineWarmRestartsSchedulerConfig(_SchedulerBase):
    type: Literal["cosine_warm_restarts"] = "cosine_warm_restarts"
    T_0: int = Field(default=5, gt=0, description="Epochs until the first restart")
    T_mult: int = Field(default=1, ge=1, description="Restart period multiplier")
    eta_min: float = Field(default=0.0, ge=0.0, description="Minimum learning rate")


class CosineAnnealingSchedulerConfig(_SchedulerBase):
    type: Literal["cosine_annealing"] = "cosine_annealing"
    T_max: Optional[int] = Field(default=None, gt=0, description="Epochs of the cosine half-period (None: whole phase)")
    eta_min: float = Field(default=0.0, ge=0.0, description="Minimum learning rate")


class StepSchedulerConfig(_SchedulerBase):
    type: Literal["step"] = "step"
    step_size: int = Field(default=10, gt=0, description="Epochs between decays")
    gamma: float = Field(default=0.1, gt=0.0, description="Decay factor applied every `step_size` epochs")


class ExponentialSchedulerConfig(_SchedulerBase):
    type: Literal["exponential"] = "exponential"
    gamma: float = Field(default=0.95, gt=0.0, description="Decay factor applied every epoch")


class ConstantSchedulerConfig(_SchedulerBase):
    type: Literal["constant"] = "constant"


SchedulerConfig = Annotated[
    Union[
        CosineWarmRestartsSchedulerConfig,
        CosineAnnealingSchedulerConfig,
        StepSchedulerConfig,
        ExponentialSchedulerConfig,
        ConstantSchedulerConfig,
    ],
    Field(discriminator="type"),
]


class SchedulersConfig(BaseConfig):
    """Schedulers for the backbone network and the classifier head."""

    net: SchedulerConfig = Field(
        default_factory=lambda: CosineAnnealingSchedulerConfig(), description="Backbone scheduler"
    )
    classifier: SchedulerConfig = Field(
        # warm restarts let the classifier re-activate zeroed-out concepts
        default_factory=lambda: CosineWarmRestartsSchedulerConfig(),
        description="Classifier scheduler",
    )


class TrainingConfig(BaseConfig):
    """Training parameters and optimization configuration."""

    resume: bool = Field(
        default=False,
        description="Resume an in-progress run: also restore optimizer/scheduler state from model.checkpoint",
    )
    epochs: EpochConfig = Field(default_factory=lambda: EpochConfig(), description="Epoch configuration")
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW, description="Optimizer type")
    learning_rates: LearningRateConfig = Field(
        default_factory=lambda: LearningRateConfig(), description="Learning rate configuration"
    )
    schedulers: SchedulersConfig = Field(
        default_factory=lambda: SchedulersConfig(), description="Learning rate scheduler configuration"
    )
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Weight decay (L2 regularization) factor")
