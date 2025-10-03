from data import SupportedDataset
from data.dataloader.base import DataLoader
from data.dataset import Dataset, PanopticPartsDataset


class PanopticPartsDataLoader(DataLoader):
    def _create_dataset(self, dataset_name: SupportedDataset) -> Dataset:
        return PanopticPartsDataset(dataset_name, self.split)
