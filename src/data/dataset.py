import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Lambda, ToImage
from torchvision.datasets import Cityscapes

from data.transforms import Transforms


class TwoAugSupervisedDataset(Dataset):
    """Returns two augmentation and no labels."""

    def __init__(self, dataset: Dataset, transforms: Transforms):
        self.dataset = dataset
        self.classes = dataset.classes
        self.transform_base_image = transforms.base_image
        self.transform1 = transforms.geometry_augmentation
        self.transform2 = Compose([
            transforms.color_augmentation,
            transforms.image_normalization
        ])
        self.transform_base_target = transforms.base_target
        self.transform_shrink_target = transforms.shrink_target

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

    def __len__(self):
        return len(self.dataset)
