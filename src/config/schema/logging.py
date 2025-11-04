"""Logging and output configuration schemas."""

from pathlib import Path

from pydantic import Field

from .base import BaseConfig


class LoggingConfig(BaseConfig):
    """Logging and output configuration."""

    path: Path = Field(default=Path("logs"), description="Directory to save logs and outputs")
    save_all_models: bool = Field(default=False, description="Save the model at every epoch (default: only best model)")

    def _post_init_setup(self):
        try:
            self.path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create log directory {self.path}: {e}") from e
