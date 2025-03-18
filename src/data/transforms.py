import argparse
from typing import Tuple

import torch
import torchvision.transforms.v2 as transforms

from data.config import DATASETS


def get_transforms(args: argparse.Namespace):
    img_shape = DATASETS[args.dataset]["img_shape"]
    mean = DATASETS[args.dataset]["mean"]
    std = DATASETS[args.dataset]["std"]

    transform_base_image = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size=img_shape),
    ])
    transform_base_target = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.int64),
        transforms.Resize(size=img_shape, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
    ])

    # transform1: first step of augmentation
    transform1 = AugmentGeometry(img_shape=img_shape)

    # transform2: second step of augmentation
    # applied twice on the result of transform1(p) to obtain two similar imgs
    transform2 = AugmentColor(img_shape=img_shape)

    transform_final = transforms.Compose([
        transforms.ToImage(),
        transforms.ConvertImageDtype(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return (
        transform_base_image,
        transform_base_target,
        transform1,
        transform2,
        transform_final,
    )


class AugmentGeometry(transforms.Compose):
    def __init__(self, img_shape: Tuple[int, int] = (256, 256)):
        super().__init__([
            transforms.Resize(size=(img_shape[0] + 8, img_shape[1] + 8)),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=(0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                size=(img_shape[0], img_shape[1]),
                scale=(0.95, 1.0),
            ),
        ])


class AugmentColor(transforms.Compose):
    def __init__(self, img_shape: Tuple[int, int] = (256, 256)):
        super().__init__([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5),
            transforms.RandomPosterize(bits=4),
            transforms.RandomSolarize(threshold=0.5),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.RandomCrop(size=img_shape),
        ])