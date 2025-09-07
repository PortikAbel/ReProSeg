from pathlib import Path

from PIL import Image

from .base import Dataset


class PanopticPartsDataset(Dataset):
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        panoptic_mask = self._get_panoptic_mask(index)

        return (image, target, panoptic_mask)

    def _get_panoptic_mask(self, index: int) -> Path:
        image_path = Path(self.dataset.images[index])
        path_parts = list(image_path.parts)
        path_parts[-4] = "gtFinePanopticParts"
        path_parts[-1] = path_parts[-1].replace("leftImg8bit.png", "gtFinePanopticParts.tif")
        panoptic_mask_path = Path(*path_parts)

        panoptic_mask = Image.open(panoptic_mask_path)
        panoptic_mask = self.transforms.base_target(panoptic_mask)
        panoptic_mask = panoptic_mask % 100
        
        return panoptic_mask
