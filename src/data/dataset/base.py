from typing import Optional, cast

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import Subset
from torchvision.datasets import Cityscapes
from torchvision.datasets.vision import StandardTransform
from torchvision.transforms.v2 import Compose, Transform

from config.schema.data import DataConfig, DatasetType
from data.data_split import DataSplit
from data.dataset.factory import DatasetFactory
from data.dataset.transform_set import TransformSet
from utils.errors import DatasetNotImplementedError


class Dataset(TorchDataset):
    dataset_type: DatasetType
    transform_set: TransformSet
    transforms: StandardTransform
    dataset: TorchDataset

    def __init__(self, cfg: DataConfig, dataset: Optional[TorchDataset] = None):
        self.dataset_type = cfg.dataset
        self.transform_set = TransformSet(cfg)
        self.dataset = dataset if dataset is not None else DatasetFactory.create(cfg, split=DataSplit.TRAIN)
        self.transforms = Compose([
            StandardTransform(
                self.transform,
                self.target_transform,
            ),
            self.transform_set.random_crop,
        ])

    def __getitem__(self, index: int):
        image, target = self.dataset.__getitem__(index)
        return self.transforms(image, target)

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        dataset = self.dataset
        if isinstance(self.dataset, Subset):
            dataset = self.dataset.dataset
        match self.dataset_type:
            case DatasetType.CITYSCAPES:
                city_scapes_dataset = cast(Cityscapes, dataset)
                return city_scapes_dataset.classes
            case _:
                raise DatasetNotImplementedError(self.dataset_type)

    @property
    def transform(self) -> Transform:
        """Transform to be applied to the dataset images"""
        return Compose([self.transform_set.base_image, self.transform_set.image_normalization])

    @property
    def target_transform(self) -> Transform:
        """Transform to be applied to the dataset targets"""
        match self.dataset_type:
            case DatasetType.CITYSCAPES:
                return Compose(
                    [
                        self.transform_set.base_target,
                        self.transform_set.filter_classes,
                    ]
                )
            case _:
                return self.transform_set.base_target
