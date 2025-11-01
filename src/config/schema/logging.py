"""Logging and output configuration schemas."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    """Logging and output configuration."""

    path: Path = Field(default=Path("logs"), description="Directory to save logs and outputs")
    save_all_models: bool = Field(default=False, description="Save the model at every epoch (default: only best model)")

    @field_validator("path")
    def validate_log_dir(cls, v):
        try:
            v.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create log directory {v}: {e}") from e
        return v
