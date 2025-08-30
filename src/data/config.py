from pathlib import Path
from typing import TypedDict

from utils.environment import get_env


class DatasetConfig(TypedDict):
    data_dir: Path
    color_channels: int
    img_shape: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

DATASETS: dict[str, DatasetConfig] = {}

DATASETS["CityScapes"] = {
    "data_dir": Path(get_env("DATA_ROOT"), "Cityscapes"),
    "color_channels": 3,
    "img_shape": (1024 // 4, 2048 // 4),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
