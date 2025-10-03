from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Transform, Compose

from data import SupportedDataset, SupportedSplit
from data.config import get_dataset_config, DatasetConfig
from data.transforms import Transforms
from utils.errors import DatasetNotImplementedError


class Dataset(TorchDataset):
    name: SupportedDataset
    config: DatasetConfig
    split: SupportedSplit
    dataset: Cityscapes
    transforms: Transforms

    def __init__(self, dataset_name: SupportedDataset, split: SupportedSplit):
        # Validate dataset name first
        self.config = get_dataset_config(dataset_name)
        self.name = dataset_name
        self.split = split
        # Initialize transforms first, before creating the dataset
        self.transforms = Transforms(self.config)
        self.dataset = self.__getdata__()

    def __getitem__(self, index: int):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def __getdata__(self) -> Cityscapes:
        match self.name:
            case "CityScapes":
                data = Cityscapes(
                    root=self.config["data_dir"],
                    split=self.split,
                    mode="fine",
                    target_type="semantic",
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
                data.classes = self.transforms.classes

                return data
            case _:
                raise DatasetNotImplementedError(self.name)

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def transform(self) -> Transform:
        """Transform to be applied to the dataset images"""
        return Compose([self.transforms.base_image, self.transforms.image_normalization])

    @property
    def target_transform(self) -> Transform:
        """Transform to be applied to the dataset targets"""
        match self.name:
            case "CityScapes":
                return Compose(
                    [
                        self.transforms.base_target,
                        self.transforms.filter_classes,
                    ]
                )
            case _:
                return self.transforms.base_target
