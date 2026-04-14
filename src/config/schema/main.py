"""Main configuration schema that combines all sub-configs."""

from pydantic import Field

from .base import BaseConfig
from .data import DataConfig
from .environment import EnvironmentConfig
from .evaluation import EvaluationConfig
from .logging import LoggingConfig
from .model import ModelConfig
from .training import TrainingConfig
from .visualization import VisualizationConfig


class ReProSegConfig(BaseConfig):
    """Complete ReProSeg configuration combining all sub-configurations."""

    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
