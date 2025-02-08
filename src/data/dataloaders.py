import argparse

import numpy as np
import torch
import torch.optim
import torchvision
import torchvision.transforms.v2 as transforms

from torch.utils.data import DataLoader, SubsetRandomSampler

from data.config import DATASETS
from data.dataset import TwoAugSupervisedDataset
from utils.log import Log


def get_dataloaders(log: Log, args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        train_set,
        test_set,
        train_indices,
    ) = get_datasets(log, args)

    classes = DATASETS[args.dataset]["class_names"]

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

    dataset_config = DATASETS[args.dataset]

    if args.dataset == "CityScapes":
        log.info("Loading CityScapes dataset")
        train_set = torchvision.datasets.Cityscapes(
            root=dataset_config["data_dir"],
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
            root=dataset_config["data_dir"],
            split="test",
            mode="fine",
            target_type="semantic",
            transform=transform_no_augment,
            target_transform=target_transform,
        )

    return (
        train_set,
        test_set,
        train_indices
    )


# TODO: define separate transforms for image+target (geometrical transforms) and image only (e.g. color transforms, normalization)
def get_transforms(args: argparse.Namespace):
    
    mean = DATASETS[args.dataset]["mean"]
    std = DATASETS[args.dataset]["std"]
    img_shape = DATASETS[args.dataset]["img_shape"]

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
            # TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=img_shape),  # includes crop
            transforms.ToImage(),
            transforms.ConvertImageDtype(),
            normalize, # TODO: don't normalize the target
        ]
    )

    target_transform = transforms.ToTensor()

    return (
        transform_no_augment,
        transform1,
        transform2,
        target_transform,
    )

