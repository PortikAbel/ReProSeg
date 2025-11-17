from typing import Optional

from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import Compose, Transform

from config.schema.data import DataConfig

from .base import Dataset


class DoubleAugmentDataset(Dataset):
    """Returns two similar augmented version of the image along with the labels."""

    def __init__(self, cfg: DataConfig, dataset: Optional[TorchDataset] = None):
        super().__init__(cfg, dataset)

        self.transform1 = self.transform_set.geometry_augmentation
        self.transform2 = Compose([self.transform_set.color_augmentation, self.transform_set.image_normalization])
        self.transform_shrink_target = self.transform_set.shrink_target

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        image, target = self.transform1(image, target)

        return (
            self.transform2(image),
            self.transform2(image),
            self.transform_shrink_target(target),
        )

    @property
    def transform(self) -> Transform:
        return self.transform_set.base_image

    @property
    def target_transform(self) -> Transform:
        return Compose([self.transform_set.base_target, self.transform_set.filter_classes])
