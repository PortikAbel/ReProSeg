from config import ReProSegConfig
from config.schema.data import DataConfig
from data.dataloader.base import DataLoader
from data.dataset import Dataset, DoubleAugmentDataset


class DoubleAugmentDataLoader(DataLoader):
    def __init__(self, cfg: ReProSegConfig):
        super().__init__("train", cfg)

    def _create_dataset(self, cfg: DataConfig) -> Dataset:
        return DoubleAugmentDataset(cfg)
