from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Transform

from config.schema.data import DataConfig, DatasetType
from data import SupportedSplit
from data.transforms import Transforms
from utils.errors import DatasetNotImplementedError


class Dataset(TorchDataset):
    config: DataConfig
    split: SupportedSplit
    dataset: Cityscapes
    transforms: Transforms

    def __init__(self, cfg: DataConfig, split: SupportedSplit):
        # Validate dataset name first
        self.config = cfg
        self.split = split
        # Initialize transforms first, before creating the dataset
        self.transforms = Transforms(self.config)
        self.dataset = self.__getdata__()

    def __getitem__(self, index: int):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def __getdata__(self) -> Cityscapes:
        match self.config.dataset:
            case DatasetType.CITYSCAPES:
                data = Cityscapes(
                    root=self.config.path,
                    split=self.split,
                    mode="fine",
                    target_type="semantic",
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
                data.classes = self.transforms.classes

                return data
            case _:
                raise DatasetNotImplementedError(self.config.dataset)

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
        match self.config.dataset:
            case DatasetType.CITYSCAPES:
                return Compose(
                    [
                        self.transforms.base_target,
                        self.transforms.filter_classes,
                    ]
                )
            case _:
                return self.transforms.base_target
