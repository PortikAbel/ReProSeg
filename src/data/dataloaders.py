from argparse import Namespace

import numpy as np
import torch
import torch.optim
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as transforms

from data.config import DATASETS
from data.dataset import TwoAugSupervisedDataset
from data.augment import get_transforms
from utils.log import Log


def get_dataloaders(log: Log, args: Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
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


def get_datasets(log: Log, args: Namespace) -> tuple[TwoAugSupervisedDataset, Dataset, list[int]]:
    """
    Load the proper dataset based on the parsed arguments
    """
    (
        transform_base_image,
        transform_base_target,
        normalize,
        transform1,
        transform2,
    ) = get_transforms(args)

    dataset_config = DATASETS[args.dataset]

    if args.dataset == "CityScapes":
        log.info("Loading CityScapes dataset")
        train_set = torchvision.datasets.Cityscapes(
            root=dataset_config["data_dir"],
            split="train",
            mode="fine",
            target_type="semantic",
        )

        train_indices = list(range(len(train_set)))

        train_set = torch.utils.data.Subset(
            TwoAugSupervisedDataset(
                train_set,
                transform_base_image,
                transform_base_target,
                transform1,
                transforms.Compose([transform2, normalize]),
            ),
            indices=train_indices,
        )

        test_set = torchvision.datasets.Cityscapes(
            root=dataset_config["data_dir"],
            split="test",
            mode="fine",
            target_type="semantic",
            transform=transforms.Compose([transform_base_image, normalize]),
        )

    return (
        train_set,
        test_set,
        train_indices
    )

