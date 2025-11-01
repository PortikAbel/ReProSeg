from config import ReProSegConfig
from config.schema.data import DatasetType
from data.dataloader.base import DataLoader
from data.dataset import Dataset, DoubleAugmentDataset


class DoubleAugmentDataLoader(DataLoader):
    def __init__(self, cfg: ReProSegConfig):
        super().__init__("train", cfg)

    def _create_dataset(self, dataset_name: DatasetType) -> Dataset:
        return DoubleAugmentDataset(dataset_name)
