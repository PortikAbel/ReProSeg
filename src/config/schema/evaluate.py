"""Evaluation metric configuration schemas."""

from pathlib import Path
from typing import Optional

from pydantic import Field

from .base import BaseConfig


class ConsistencyEvalConfig(BaseConfig):
    """Part-consistency score evaluation configuration."""

    quantile: float = Field(default=0.8, ge=0.0, le=1.0, description="Activation quantile used to binarize a map")
    threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Fraction of images a part must be hit in to count as consistent"
    )
    used_prototypes_only: bool = Field(
        default=False,
        description="For PPNet, restrict the denominator to prototypes with a positive final-layer connection",
    )
    parts_path: Optional[Path] = Field(
        default=None, description="Pascal validation TIFF directory; defaults to VOC2012/labels/val"
    )
    parts_spec: Optional[Path] = Field(
        default=None, description="Pascal PPP v2 specification; defaults to VOC2012/parts.yaml"
    )
    official_parts_only: bool = Field(
        default=False, description="Keep only documented dataset semantic/part pairs"
    )
    batch_size: int = Field(default=1, ge=1, description="Evaluation batch size")
    num_workers: int = Field(default=0, ge=0, description="Number of dataloader workers")
    image_shape: Optional[tuple[int, int]] = Field(
        default=None, description="Center crop (height, width); defaults to native resolution"
    )


class EvaluateMetricsConfig(BaseConfig):
    """Selects and configures which interpretability metric to compute."""

    metric_name: str = Field(default="consistency", description="Name of the registered metric to run")
    consistency: ConsistencyEvalConfig = Field(default_factory=ConsistencyEvalConfig)
