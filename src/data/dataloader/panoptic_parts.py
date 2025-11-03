from config.schema.data import DataConfig
from data.dataloader.base import DataLoader
from data.dataset import Dataset, PanopticPartsDataset


class PanopticPartsDataLoader(DataLoader):
    def _create_dataset(self, cfg: DataConfig) -> Dataset:
        return PanopticPartsDataset(cfg, self.split)
