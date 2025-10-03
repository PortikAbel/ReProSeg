from argparse import Namespace

from data import SupportedDataset
from data.dataloader.base import DataLoader
from data.dataset import Dataset, DoubleAugmentDataset


class DoubleAugmentDataLoader(DataLoader):
    def __init__(self, args: Namespace):
        super().__init__("train", args)

    def _create_dataset(self, dataset_name: SupportedDataset) -> Dataset:
        return DoubleAugmentDataset(dataset_name)
