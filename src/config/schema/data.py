"""Data and dataset configuration schemas."""

from enum import Enum
from pathlib import Path

from pydantic import Field

from config.schema.base import BaseConfig


class DatasetType(str, Enum):
    """Supported dataset types."""

    CITYSCAPES = "CityScapes"
    PASCAL_PARTS = "PascalParts"


class DataConfig(BaseConfig):
    """Data and dataset configuration."""

    # Dataset settings
    dataset: DatasetType = Field(default=DatasetType.CITYSCAPES, description="Dataset to train ReProSeg on")
    path: Path = Field(default=Path("./data"), description="Root path to dataset")
    validation_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Fraction of training data to use as validation"
    )
    disable_normalize: bool = Field(default=False, description="Disable normalization of images if set")
    color_channels: int = Field(default=3, description="Number of color channels in the images")
    filter_classes: bool = Field(
        default=False, description="Whether to filter classes not used in training (if applicable)"
    )
    img_shape: tuple[int, int] = Field(default=(256, 256), description="Image shape (height, width) after resizing")
    mean: tuple[float, float, float] = Field(
        default=(0.5, 0.5, 0.5), description="Mean for each color channel for normalization"
    )
    std: tuple[float, float, float] = Field(
        default=(0.25, 0.25, 0.25), description="Standard deviation for each color channel for normalization"
    )

    # Dataloader settings
    num_workers: int = Field(default=8, ge=0, description="Number of workers in dataloaders")
    batch_size: int = Field(default=2, ge=2, description="Minibatch size (will be multiplied by the number of GPUs)")

    # Computed field based on dataset
    num_classes: int = Field(default=0, description="Number of classes (computed from dataset)")
