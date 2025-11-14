from pathlib import Path
from typing import Optional

import torch
from pydantic import Field, field_validator

from .base import BaseConfig


class EnvironmentConfig(BaseConfig):
    """Environment and hardware configuration."""

    gpu_id: Optional[int] = Field(
        default=None,
        description="ID of GPU to use",
        ge=0,
    )
    device: torch.device = Field(default=torch.device("cpu"), description="Computed device string (set at runtime)")

    pretrained_backbones_dir: Path = Field(
        default=Path("pretrained"), description="Directory to store pretrained backbone checkpoints"
    )
    class_distribution_cache_path: Path = Field(
        default=Path("data/class_counts.npy"),
        description="Path to cache class distribution counts",
    )

    seed: int = Field(default=1, description="Random seed. Note: nondeterminism may still occur")

    @field_validator("gpu_id")
    def validate_gpu_id(cls, v):
        if not torch.cuda.is_available():
            raise ValueError("GPU ID was specified but CUDA is not available.")
        if v >= torch.cuda.device_count():
            raise ValueError(f"GPU ID {v} is not available. Only {torch.cuda.device_count()} GPUs found.")
        return v

    def _post_init_setup(self):
        if self.gpu_id is not None:
            self.device = torch.device(f"cuda:{self.gpu_id}")

        import random

        import numpy as np

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
