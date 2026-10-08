from pathlib import Path
from typing import Optional, cast

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import Subset
from torchvision.datasets import Cityscapes

from config.schema.data import DataConfig, DatasetType
from data.dataset.base import Dataset

# Native Cityscapes semantic IDs; part IDs are one-based in the listed order.
# https://github.com/pmeletis/panoptic_parts/blob/master/panoptic_parts/specs/dataset_specs/cpp_datasetspec.yaml
CITYSCAPES_PARTS = {
    24: ("torso", "head", "arm", "leg"),  # person
    25: ("torso", "head", "arm", "leg"),  # rider
    26: ("window", "wheel", "light", "license plate", "chassis"),  # car
    27: ("window", "wheel", "light", "license plate", "chassis"),  # truck
    28: ("window", "wheel", "light", "license plate", "chassis"),  # bus
}
_VALID_PART_LABELS = tuple(
    semantic_id * 100 + part_id
    for semantic_id, parts in CITYSCAPES_PARTS.items()
    for part_id in range(1, len(parts) + 1)
)


class PanopticPartsDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        dataset: Optional[TorchDataset] = None,
        *,
        official_parts_only: bool = True,
    ):
        """Select official parts or all positive decoded part IDs.

        Existing dataset callers keep official filtering by default. The
        consistency CLI explicitly supplies its runtime choice.
        """
        if cfg.dataset != DatasetType.CITYSCAPES:
            raise ValueError("PanopticPartsDataset only supports CITYSCAPES dataset type.")
        self.official_parts_only = official_parts_only
        super().__init__(cfg, dataset)

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        panoptic_mask = self._get_panoptic_mask(index)

        return (image, target, panoptic_mask)

    def _get_panoptic_mask(self, index: int) -> Tensor:
        """Return selected semantic-part labels; pixels without parts are ignored (0)."""
        image_path = self._get_image_path(index)
        path_parts = list(image_path.parts)
        path_parts[-4] = "gtFinePanopticParts"
        path_parts[-1] = path_parts[-1].replace("leftImg8bit.png", "gtFinePanopticParts.tif")
        panoptic_mask_path = Path(*path_parts)

        panoptic_mask = Image.open(panoptic_mask_path)
        panoptic_mask = self.transform_set.base_target(panoptic_mask)
        panoptic_mask = self.transform_set.random_crop(panoptic_mask)

        # Full UIDs encode semantic_id * 100_000 + instance_id * 100 + part_id.
        # Drop instance IDs, retaining the existing semantic_id * 100 + part_id format.
        part_labels = panoptic_mask // 100_000 * 100 + panoptic_mask % 100
        valid = (panoptic_mask >= 100_000) & (panoptic_mask % 100 > 0)
        if self.official_parts_only:
            valid_labels = part_labels.new_tensor(_VALID_PART_LABELS)
            valid &= torch.isin(part_labels, valid_labels)
        return part_labels.masked_fill(~valid, 0)

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
        return [part for parts in CITYSCAPES_PARTS.values() for part in parts]
