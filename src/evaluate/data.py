"""Shared validation data loading for interpretability metrics."""

import torch
from torch.utils.data import DataLoader, Subset

from config.schema.data import DataConfig, DatasetType
from data import DataSplit, PanopticPartsDataset
from data.dataset.factory import DatasetFactory

CITYSCAPES_NATIVE_IMAGE_SHAPE = (1024, 2048)


def create_cityscapes_parts_dataloader(
    data: DataConfig,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    official_parts_only: bool = False,
    image_shape: tuple[int, int] = CITYSCAPES_NATIVE_IMAGE_SHAPE,
    device: torch.device | str = "cpu",
    sort_images: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Build the official validation split with semantic and part annotations.

    Preserve the source dataset order by default for existing consistency CSV
    image indices. Stability requests sorting by path and a private loader RNG
    so its two part modes can use identical image order and noise assignment.
    """
    if data.dataset != DatasetType.CITYSCAPES:
        raise ValueError("This loader supports Cityscapes only; select data=cityscapes.")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Batch size must be positive and worker count nonnegative.")
    data_config = DataConfig(
        dataset=DatasetType.CITYSCAPES,
        path=data.path,
        # The training config requires >=2, but the loader accepts one image.
        batch_size=max(2, batch_size),
        num_workers=num_workers,
        img_shape=image_shape,
        filter_classes=True,
        mean=data.mean,
        std=data.std,
    )
    validation_data = DatasetFactory.create(data_config, split=DataSplit.VAL)
    if sort_images:
        indices = sorted(range(len(validation_data)), key=lambda i: str(validation_data.images[i]))
        validation_data = Subset(validation_data, indices)
    parts_data = PanopticPartsDataset(data_config, validation_data, official_parts_only=official_parts_only)
    return DataLoader(
        parts_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        generator=None if seed is None else torch.Generator().manual_seed(seed),
    )
