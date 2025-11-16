from torchvision.transforms.v2 import Compose

from config.schema.data import DataConfig
from data import SupportedSplit

from .base import Dataset


class DoubleAugmentDataset(Dataset):
    """Returns two similar augmented version of the image along with the labels."""

    def __init__(self, base_dataset: Dataset):
        self.config = base_dataset.config
        self.split = base_dataset.split
        self.dataset = base_dataset.dataset
        self.transforms = base_dataset.transforms
        
        self.dataset.transforms = None

        self.transform_base_image = self.transforms.base_image
        self.transform1 = self.transforms.geometry_augmentation
        self.transform2 = Compose([self.transforms.color_augmentation, self.transforms.image_normalization])
        self.transform_base_target = Compose([self.transforms.base_target, self.transforms.filter_classes])
        self.transform_shrink_target = self.transforms.shrink_target

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        image = self.transform_base_image(image)
        target = self.transform_base_target(target)

        image, target = self.transform1(image, target)

        return (
            self.transform2(image),
            self.transform2(image),
            self.transform_shrink_target(target),
        )
