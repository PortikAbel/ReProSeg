"""Data and dataset configuration schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    """Supported dataset types."""

    CITYSCAPES = "CityScapes"
    PASCAL_PARTS = "PascalParts"


class DataConfig(BaseModel):
    """Data and dataset configuration."""

    # Dataset settings
    dataset: DatasetType = Field(default=DatasetType.CITYSCAPES, description="Dataset to train ReProSeg on")
    validation_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Fraction of training data to use as validation"
    )
    disable_normalize: bool = Field(default=False, description="Disable normalization of images if set")

    # Dataloader settings
    num_workers: int = Field(default=8, ge=0, description="Number of workers in dataloaders")
    batch_size: int = Field(default=2, ge=2, description="Minibatch size (will be multiplied by the number of GPUs)")

    # Computed field based on dataset
    num_classes: int = Field(default=0, description="Number of classes (computed from dataset)")

    class Config:
        use_enum_values = True
