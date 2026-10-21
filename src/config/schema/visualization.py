"""Visualization configuration schema."""

from pydantic import BaseModel, Field


class VisualizationConfig(BaseModel):
    """Visualization configuration."""

    # Concept visualization
    top_k: int = Field(default=10, gt=0, description="Number `k` for top-k prototypes per concept")
