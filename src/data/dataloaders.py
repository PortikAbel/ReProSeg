from argparse import Namespace
from typing import cast

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms.v2 import Compose

from data.config import DATASETS
from data.dataset import TwoAugSupervisedDataset
from data.transforms import Transforms
from utils.log import Log


def get_dataloaders(log: Log, args: Namespace) -> tuple[DataLoader, ...]:
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        train_set,
        test_set,
        train_set_visualization,
    ) = get_datasets(log, args)

    # Determine if GPU should be used
    cuda = not args.disable_gpu and torch.cuda.is_available()
    sampler = None
    to_shuffle_train_set = True

    def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
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
    train_loader_visualization = create_dataloader(
        dataset=train_set_visualization,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return (
        train_loader,
        test_loader,
        train_loader_visualization,
    )


def get_datasets(log: Log, args: Namespace) -> tuple[TwoAugSupervisedDataset, Dataset, Dataset]:
    """
    Load the proper dataset based on the parsed arguments
    """
    dataset_config = DATASETS[args.dataset]
    transforms = Transforms(dataset_config)

    if args.dataset == "CityScapes":
        log.info("Loading CityScapes dataset")
        train_set = torchvision.datasets.Cityscapes(
            root=dataset_config["data_dir"],
            split="train",
            mode="fine",
            target_type="semantic",
        )

        filtered_classes = transforms.filter_cityscapes_classes(train_set.classes)
        train_set.classes = filtered_classes

        train_set_augment = TwoAugSupervisedDataset(train_set, transforms)

        test_set = torchvision.datasets.Cityscapes(
            root=dataset_config["data_dir"],
            split="test",
            mode="fine",
            target_type="semantic",
            transform=Compose([transforms.base_image, transforms.image_normalization]),
            target_transform=transforms.base_target,
        )

        train_visualization_set = torchvision.datasets.Cityscapes(
            root=dataset_config["data_dir"],
            split="train",
            mode="fine",
            target_type="semantic",
            transform=Compose([transforms.base_image, transforms.image_normalization]),
            target_transform=transforms.base_target,
        )

    return (
        train_set_augment,
        test_set,
        train_visualization_set
    )

