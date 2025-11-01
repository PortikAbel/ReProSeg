"""Visualization configuration schema."""

from pydantic import BaseModel, Field

class VisualizationConfig(BaseModel):
    """Visualization configuration."""

    # Prototype visualization
    generate_explanations: bool = Field(default=False, description="Whether to generate model explanations")
    top_k: int = Field(default=10, gt=0, description="Number `k` for top-k activations to visualize per prototype")