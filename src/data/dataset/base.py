from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Transform, Compose

from data import SupportedDataset, SupportedSplit
from data.config import DATASETS, DatasetConfig
from data.transforms import Transforms


class Dataset(TorchDataset):
    name: SupportedDataset
    config: DatasetConfig
    split: SupportedSplit
    dataset: Cityscapes
    transforms: Transforms

    def __init__(self, dataset_name: SupportedDataset, split: SupportedSplit):
        # Validate dataset name first
        if dataset_name not in DATASETS:
            raise NotImplementedError(
                f"Dataset '{dataset_name}' not implemented. Available datasets: {list(DATASETS.keys())}"
            )

        self.name = dataset_name
        self.config = DATASETS[dataset_name]
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
                # self.log.info("Loading CityScapes test dataset")
                classes = self.transforms.filter_cityscapes_classes()
                data = Cityscapes(
                    root=self.config["data_dir"],
                    split=self.split,
                    mode="fine",
                    target_type="semantic",
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
                data.classes = classes

                return data
            case _:
                raise NotImplementedError(f"Dataset {self.name} not implemented")

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
        return self.transforms.base_target
