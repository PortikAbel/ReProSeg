from config.schema.data import DatasetType
from data.dataloader.base import DataLoader
from data.dataset import Dataset, PanopticPartsDataset


class PanopticPartsDataLoader(DataLoader):
    def _create_dataset(self, dataset_name: DatasetType) -> Dataset:
        return PanopticPartsDataset(dataset_name, self.split)
