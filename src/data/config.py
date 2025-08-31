from pathlib import Path
from typing import TypedDict

from utils.environment import get_env


class DatasetConfig(TypedDict):
    data_dir: Path
    color_channels: int
    img_shape: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _get_cityscapes_config() -> DatasetConfig:
    """Get CityScapes dataset configuration with lazy environment variable loading."""
    return {
        "data_dir": Path(get_env("DATA_ROOT"), "Cityscapes"),
        "color_channels": 3,
        "img_shape": (1024 // 4, 2048 // 4),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }


DATASETS: dict[str, DatasetConfig] = {}


# Lazy loading of dataset configurations
def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name, loading it lazily if not already loaded."""
    if dataset_name not in DATASETS:
        if dataset_name == "CityScapes":
            DATASETS[dataset_name] = _get_cityscapes_config()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    return DATASETS[dataset_name]
