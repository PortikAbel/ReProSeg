import argparse
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as transforms
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.log import Log

# TODO: move this to a separate file?
# TODO: do some classes need to be merged?
cityscapes_classes = [
    "unlabeled", "ego vehicle", "rectification border", "out of roi", "static",
    "dynamic", "ground", "road", "sidewalk", "parking", "rail track", "building",
    "wall", "fence", "guard rail", "bridge", "tunnel", "pole", "polegroup", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle",
]

def get_dataloaders(log: Log, args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        train_set,
        test_set,
        classes,
        train_indices,
    ) = get_datasets(log, args)

    # Determine if GPU should be used
    cuda = not args.disable_gpu and torch.cuda.is_available()
    sampler = None
    to_shuffle_train_set = True

    if args.weighted_loss:
        if targets is None:
            raise ValueError(
                "Weighted loss not implemented for this dataset. "
                "Targets should be restructured"
            )
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907 # noqa
        class_sample_count = torch.tensor(
            [
                (targets[train_indices] == t).sum()
                for t in torch.unique(targets, sorted=True)
            ]
        )
        weight = 1.0 / class_sample_count.float()
        log.info(f"Weights for weighted sampler: {weight}")
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True
        )
        to_shuffle_train_set = False

    def create_dataloader(dataset, batch_size, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dataset,
            # batch size is np.uint16, so we need to convert it to int
            batch_size=int(batch_size),
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=cuda,
            num_workers=args.num_workers,
            worker_init_fn=np.random.seed(args.seed),
            drop_last=drop_last,
        )

    # TODO: add weighted random sampler
    train_loader = create_dataloader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=to_shuffle_train_set,
        drop_last=True,
    )
    test_loader = create_dataloader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    log.info(f"Num classes (k) = {len(classes)} {classes[:5],} etc.")

    return (
        train_loader,
        test_loader,
        classes,
    )


def get_datasets(log: Log, args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    """
    (
        transform_no_augment,
        transform1,
        transform2,
        target_transform,
    ) = get_transforms(args)

    if args.dataset == "CityScapes":
        log.info("Loading CityScapes dataset")
        train_set = torchvision.datasets.Cityscapes(
            root='/tankstorage/data/Cityscapes', # TODO: add path to Cityscapes dataset as parameter
            split="train",
            mode="fine",
            target_type="semantic",
            transform=None, # will add it later in TwoAugSupervisedDataset
            target_transform=None,  # will add it later in TwoAugSupervisedDataset
        )

        train_indices = list(range(len(train_set)))

        train_set = torch.utils.data.Subset(
        TwoAugSupervisedDataset(
            train_set, transform1, transform2
        ),
            indices=train_indices,
        )
        
        test_set = torchvision.datasets.Cityscapes(
            root='/tankstorage/data/Cityscapes', # TODO: add path to Cityscapes dataset as parameter or to .env
            split="test",
            mode="fine",
            target_type="semantic",
            transform=transform_no_augment,
            target_transform=target_transform,
        )

    return (
        train_set,
        test_set,
        train_indices,
        cityscapes_classes
    )


def get_transforms(args: argparse.Namespace):

    mean = args.mean
    std = args.std
    img_shape = tuple(args.image_shape)

    normalize = transforms.Normalize(mean=mean, std=std)

    transform_no_augment = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=img_shape),
            # transforms.ToImage(),
            # transforms.ConvertImageDtype(),
            normalize,  # TODO: don't normalize the target
        ]
    )

    # transform1: first step of augmentation
    match args.dataset:
        case "CityScapes" | "Pascal-VOC":
            transform1 = transforms.Compose(
                [
                    transforms.Resize(size=(img_shape[0] + 8, img_shape[1] + 8)),
                    # TrivialAugmentWideNoColor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        size=(img_shape[0] + 4, img_shape[1] + 4),
                        scale=(0.95, 1.0),
                    ),
                ]
            )

    # transform2: second step of augmentation
    # applied twice on the result of transform1(p) to obtain two similar imgs
    transform2 = transforms.Compose(
        [
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=img_shape),  # includes crop
            transforms.ToImage(),
            transforms.ConvertImageDtype(),
            normalize,
        ]
    )

    target_transform = transforms.ToTensor()

    return (
        transform_no_augment,
        transform1,
        transform2,
        target_transform,
    )


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    """Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        # if isinstance(dataset, torchvision.datasets.Cityscapes):
        #     #TODO: do we need this? self.imgs and self.targets?
        #     # Create a DataLoader to load all data in a single batch
        #     data_loader = DataLoader(
        #         dataset,
        #         batch_size=10,  # TODO: for testing purposes, only working with 10 images; should be: batch_size=len(dataset),
        #         shuffle=False
        #     )

        #     # Retrieve all images and targets in one go
        #     self.imgs, self.targets = next(iter(data_loader))  # Get a single batch from the DataLoader

        # else:
        #     self.targets = dataset._labels
        #     self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]

        # adding the label (segmenation mask) to the RGB input images as a fourth channel in order to apply the same augmentation to both
        image_array = np.array(image)  # Shape: (H, W, 3)
        target_array = np.array(target)  # Shape: (H, W)

        # Ensure both images have the same size
        assert image_array.shape[:2] == target_array.shape, "Image and segmentation mask must have the same dimensions"

        # Add the grayscale image (segmentation mask) as the fourth channel
        image_target_array = np.dstack((image_array, target_array))  # Shape: (H, W, 4)

        # Convert back to a PIL image
        image_target_img = Image.fromarray(image_target_array, mode="RGBA")

        image_target_img = self.transform1(image_target_img)

        #TODO: only temporary!! target is set to a single pixel value
        target_transform = transforms.ToTensor()
        target = target_transform(target)
        return self.transform2(image), self.transform2(image), target[:, 0, 0]

    def __len__(self):
        return len(self.dataset)


# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted  # noqa
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
