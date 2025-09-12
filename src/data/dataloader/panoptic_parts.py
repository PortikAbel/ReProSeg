from data.dataloader.base import DataLoader
from data.dataset import PanopticPartsDataset, Dataset
from data import SupportedDataset


class PanopticPartsDataLoader(DataLoader):
    def _create_dataset(self, dataset_name: SupportedDataset) -> Dataset:
        return PanopticPartsDataset(dataset_name, self.split)
