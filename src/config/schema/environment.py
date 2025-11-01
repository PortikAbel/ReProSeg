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

    # Random seed
    seed: int = Field(default=1, description="Random seed. Note: nondeterminism may still occur")

    class Config:
        arbitrary_types_allowed = True

    @field_validator("gpu_id")
    def validate_gpu_id(cls, v):
        if not torch.cuda.is_available():
            raise ValueError("GPU ID was specified but CUDA is not available.")
        if v >= torch.cuda.device_count():
            raise ValueError(f"GPU ID {v} is not available. Only {torch.cuda.device_count()} GPUs found.")
        return v

    @field_validator("seed")
    def validate_seed(cls, v):
        import random

        import numpy as np

        random.seed(v)
        np.random.seed(v)
        torch.manual_seed(v)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(v)
        return v

    def _post_init_validation(self):
        if self.gpu_id is not None:
            self.device = torch.device(f"cuda:{self.gpu_id}")
