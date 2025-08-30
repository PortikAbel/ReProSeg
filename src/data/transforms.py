from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torchvision.datasets import Cityscapes

from data.config import DatasetConfig


class AugmentGeometry(transforms.Compose):
    def __init__(self, img_shape: Tuple[int, int] = (256, 256)):
        super().__init__(
            [
                transforms.Resize(size=(img_shape[0] + 8, img_shape[1] + 8)),
                transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), shear=(0.5, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    size=(img_shape[0], img_shape[1]),
                    scale=(0.95, 1.0),
                ),
            ]
        )


class AugmentColor(transforms.Compose):
    def __init__(self, img_shape: Tuple[int, int] = (256, 256)):
        super().__init__(
            [
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=0.5),
                transforms.RandomPosterize(bits=4),
                transforms.RandomSolarize(threshold=0.5),
                transforms.RandomAutocontrast(),
                transforms.RandomEqualize(),
                transforms.RandomCrop(size=img_shape),
            ]
        )


class Transforms:
    base_image: transforms.Compose
    """
    Base image transform:
        - Converts to PIL Image
        - Converts to float32
        - Resizes to the target size
    """
    geometry_augmentation: AugmentGeometry
    """
    First step of augmentation:
        - Resizes to a larger size
        - Applies random affine transformation
        - Applies random horizontal flip
        - Applies random resized crop to the target size
    """
    color_augmentation: AugmentColor
    """
    Second step of augmentation:
        - Applies color jitter
        - Applies random adjust sharpness
        - Applies random posterize
        - Applies random solarize
        - Applies random autocontrast
        - Applies random equalize
        - Applies random crop to the target size
    """
    image_normalization: transforms.Compose
    """
    Final transform:
        - Converts to PIL Image
        - Converts to the default image dtype (float32)
        - Normalizes the image with mean and std
    """
    base_target: transforms.Compose
    """
    Base target transform:
        - Converts to PIL Image
        - Converts to int64
        - Resizes to the target size with nearest neighbor interpolation
    """
    shrink_target: transforms.Resize
    """
    Shrink target transform:
        - Resize target to the size of model output
    """

    def __init__(self, dataset_cfg: DatasetConfig):
        img_shape = dataset_cfg["img_shape"]
        mean = dataset_cfg["mean"]
        std = dataset_cfg["std"]

        self.base_image = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize(size=img_shape),
            ]
        )

        self.geometry_augmentation = AugmentGeometry(img_shape=img_shape)

        self.color_augmentation = AugmentColor(img_shape=img_shape)
        # applied twice on the result of transform1(p) to obtain two similar imgs

        self.image_normalization = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ConvertImageDtype(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.base_target = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.int64),
                transforms.Resize(size=img_shape, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
            ]
        )

        self.shrink_target = transforms.Resize(
            size=(img_shape[0] // 8, img_shape[1] // 8),
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
        )

    def filter_cityscapes_classes(self, classes: list[Cityscapes.CityscapesClass]):
        filtered_classes = [classes[0]] + [c for c in classes if not c.ignore_in_eval]
        map_classes: torch.Tensor = torch.tensor(
            [0 if c.ignore_in_eval else filtered_classes.index(c) for c in classes], dtype=torch.int64
        )
        self.base_target = transforms.Compose(
            [
                self.base_target,
                transforms.Lambda(np.vectorize(lambda c: map_classes[c])),
                transforms.Lambda(lambda x: x.transpose(1, 2, 0)),
                transforms.ToImage(),
            ]
        )

        return filtered_classes
