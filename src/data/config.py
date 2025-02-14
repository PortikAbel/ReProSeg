from pathlib import Path

from utils.environment import get_env

DATASETS = {}

# TODO: do some classes need to be merged?
cityscapes_classes = [
    "unlabeled", "ego vehicle", "rectification border", "out of roi", "static",
    "dynamic", "ground", "road", "sidewalk", "parking", "rail track", "building",
    "wall", "fence", "guard rail", "bridge", "tunnel", "pole", "polegroup", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle",
]
data_dir = Path(get_env("DATA_ROOT"), "Cityscapes")
DATASETS["CityScapes"] = {
    "data_dir": data_dir,
    "color_channels": 3,
    "img_shape": (1024 // 8, 2048 // 8),
    "num_classes": len(cityscapes_classes),
    "class_names": cityscapes_classes,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
