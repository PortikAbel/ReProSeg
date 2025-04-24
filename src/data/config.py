from pathlib import Path

from utils.environment import get_env

DATASETS = {}

data_dir = Path(get_env("DATA_ROOT"), "Cityscapes")
DATASETS["CityScapes"] = {
    "data_dir": data_dir,
    "color_channels": 3,
    "img_shape": (1024 // 4, 2048 // 8),
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
