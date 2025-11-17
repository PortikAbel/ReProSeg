from pathlib import Path
from typing import Optional, cast

from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import Subset
from torchvision.datasets import Cityscapes

from config.schema.data import DataConfig, DatasetType
from data.dataset.base import Dataset


class PanopticPartsDataset(Dataset):
    def __init__(self, cfg: DataConfig, dataset: Optional[TorchDataset] = None):
        if cfg.dataset != DatasetType.CITYSCAPES:
            raise ValueError("PanopticPartsDataset only supports CITYSCAPES dataset type.")
        super().__init__(cfg, dataset)

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        panoptic_mask = self._get_panoptic_mask(index)

        return (image, target, panoptic_mask)

    def _get_panoptic_mask(self, index: int) -> Path:
        image_path = self._get_image_path(index)
        path_parts = list(image_path.parts)
        path_parts[-4] = "gtFinePanopticParts"
        path_parts[-1] = path_parts[-1].replace("leftImg8bit.png", "gtFinePanopticParts.tif")
        panoptic_mask_path = Path(*path_parts)

        panoptic_mask = Image.open(panoptic_mask_path)
        panoptic_mask = self.transform_set.base_target(panoptic_mask)
        panoptic_mask[panoptic_mask < 100_000] = 0
        panoptic_mask = panoptic_mask // 100_000 * 100 + panoptic_mask % 100

        return panoptic_mask

    def _get_image_path(self, index: int) -> Path:
        dataset: Cityscapes
        if isinstance(self.dataset, Subset):
            dataset = cast(Cityscapes, self.dataset.dataset)
            index = self.dataset.indices[index]
        else:
            dataset = cast(Cityscapes, self.dataset)
        return Path(dataset.images[index])

    @property
    def classes(self):
        return [
            "torso",
            "head",
            "arm",
            "leg",
            "torso",
            "head",
            "arm",
            "leg",
            "window",
            "wheel",
            "light",
            "license plate",
            "chassis",
            "window",
            "wheel",
            "light",
            "license plate",
            "chassis",
            "window",
            "wheel",
            "light",
            "license plate",
            "chassis",
        ]
