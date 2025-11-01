"""Evaluation metrics configuration schema."""

from pydantic import BaseModel, Field

class ConsistencyScoreConfig(BaseModel):
    """Consistency score evaluation configuration."""

    # Interpretability
    calculate: bool = Field(default=False, description="Whether to compute consistency score for interpretability")
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Threshold above which a prototype is considered consistent"
    )


class EvaluationConfig(BaseModel):
    """Evaluation metrics configuration."""

    consistency_score: ConsistencyScoreConfig = Field(
        default_factory=ConsistencyScoreConfig,
        description="Configuration for consistency score evaluation",
    )