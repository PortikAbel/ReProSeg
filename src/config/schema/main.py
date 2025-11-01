"""Main configuration schema that combines all sub-configs."""

from pydantic import Field

from .base import BaseConfig
from .data import DataConfig
from .environment import EnvironmentConfig
from .logging import LoggingConfig
from .model import ModelConfig
from .training import TrainingConfig
from .visualization import VisualizationConfig
from .evaluation import EvaluationConfig


class ReProSegConfig(BaseConfig):
    """Complete ReProSeg configuration combining all sub-configurations."""

    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def _post_init_validation(self):
        # Validate that at least one loss is specified
        loss_weights = [
            self.model.loss_weights.jsd,
            self.model.loss_weights.tanh,
            self.model.loss_weights.uniformity,
            self.model.loss_weights.variance
        ]

        if all(weight == 0.0 for weight in loss_weights):
            # Set default JSD loss if no loss specified
            self.model.loss_weights.jsd = 5.0

        if self.model.checkpoint:
            self.logging.path = self.model.checkpoint.parent.parent

    class Config:
        extra = "allow"
        use_enum_values = True
