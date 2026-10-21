"""Top-level configuration schemas, one per entry-point scenario."""

from pydantic import Field

from .base import BaseConfig
from .data import DataConfig
from .environment import EnvironmentConfig
from .evaluate import EvaluateMetricsConfig
from .logging import LoggingConfig
from .model import ModelConfig
from .training import TrainingConfig
from .visualization import VisualizationConfig


class BaseScenarioConfig(BaseConfig):
    """Config shared by every scenario: enough to build the dataset and the net."""

    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class TrainConfig(BaseScenarioConfig):
    """Configuration for the training entry point."""

    training: TrainingConfig = Field(default_factory=TrainingConfig)


class VisualizeConfig(BaseScenarioConfig):
    """Configuration for the prototype-visualization entry point."""

    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)


class EvaluateConfig(BaseScenarioConfig):
    """Configuration for the interpretability-metric evaluation entry point."""

    evaluate: EvaluateMetricsConfig = Field(default_factory=EvaluateMetricsConfig)
