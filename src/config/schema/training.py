"""Training configuration schemas."""

from enum import Enum
from typing import Optional

from pydantic import Field, ValidationInfo, field_validator, model_validator

from config.schema.base import BaseConfig


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "Adam"
    SGD = "SGD"
    ADAMW = "AdamW"


class EpochConfig(BaseConfig):
    """Epoch configuration for different training phases."""

    pretrain: int = Field(default=0, ge=0, description="Epochs for concept pretraining (stage 1)")
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


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""

    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    COSINE_ANNEALING = "cosine_annealing"
    STEP = "step"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"


# Parameters each scheduler type accepts; unset ones fall back to defaults at build time.
SCHEDULER_PARAMS: dict[SchedulerType, frozenset[str]] = {
    SchedulerType.COSINE_WARM_RESTARTS: frozenset({"T_0", "T_mult", "eta_min"}),
    SchedulerType.COSINE_ANNEALING: frozenset({"T_max", "eta_min"}),
    SchedulerType.STEP: frozenset({"step_size", "gamma"}),
    SchedulerType.EXPONENTIAL: frozenset({"gamma"}),
    SchedulerType.CONSTANT: frozenset(),
}


class SchedulerConfig(BaseConfig):
    """Learning rate scheduler configuration. Durations are expressed in epochs."""

    type: SchedulerType = Field(default=SchedulerType.COSINE_ANNEALING, description="Scheduler type")
    T_0: Optional[int] = Field(default=None, gt=0, description="Epochs until the first restart (default 5)")
    T_mult: Optional[int] = Field(default=None, ge=1, description="Restart period multiplier (default 1)")
    T_max: Optional[int] = Field(
        default=None, gt=0, description="Epochs of one cosine half-period (default: whole phase)"
    )
    eta_min: Optional[float] = Field(default=None, ge=0.0, description="Minimum learning rate (default 0.0)")
    step_size: Optional[int] = Field(default=None, gt=0, description="Epochs between decays (default 10)")
    gamma: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Decay factor per `step_size` epochs (step, default 0.1) or per epoch (exponential, default 0.95)",
    )

    @model_validator(mode="after")
    def validate_params_for_type(self):
        allowed = SCHEDULER_PARAMS[SchedulerType(self.type)]
        unused = sorted(
            name
            for name in ("T_0", "T_mult", "T_max", "eta_min", "step_size", "gamma")
            if getattr(self, name) is not None and name not in allowed
        )
        if unused:
            raise ValueError(f"parameters {unused} are not used by scheduler type '{self.type}'")
        return self


class SchedulersConfig(BaseConfig):
    """Schedulers for the backbone network and the classifier head."""

    net: SchedulerConfig = Field(default_factory=lambda: SchedulerConfig(), description="Backbone scheduler")
    classifier: SchedulerConfig = Field(
        # warm restarts let the classifier re-activate zeroed-out concepts
        default_factory=lambda: SchedulerConfig(type=SchedulerType.COSINE_WARM_RESTARTS, T_0=5),
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
