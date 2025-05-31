from argparse import Namespace

import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms.v2 import Compose, Lambda, ToImage

from data.config import DATASETS
from data.dataset import TwoAugSupervisedDataset
from data.transforms import Transforms
from utils.log import Log


def get_dataloaders(log: Log, args: Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        train_set,
        test_set,
        train_set_visualization,
        train_indices,
    ) = get_datasets(log, args)

    # Determine if GPU should be used
    cuda = not args.disable_gpu and torch.cuda.is_available()
    sampler = None
    to_shuffle_train_set = True

    def create_dataloader(dataset, batch_size, shuffle, drop_last) -> DataLoader:
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
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    args.num_classes = len(train_loader.dataset.dataset.classes)

    log.info(f"Num classes (k) = {args.num_classes} {[c.name for c in train_loader.dataset.dataset.classes[:5]],} etc.")

    return (
        train_loader,
        test_loader,
        train_loader_visualization,
    )


def get_datasets(log: Log, args: Namespace) -> tuple[TwoAugSupervisedDataset, Dataset, list[int]]:
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

        train_indices = list(range(len(train_set)))

        train_set = torch.utils.data.Subset(
            TwoAugSupervisedDataset(train_set, transforms),
            indices=train_indices,
        )

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
        train_set,
        test_set,
        train_visualization_set,
        train_indices
    )

