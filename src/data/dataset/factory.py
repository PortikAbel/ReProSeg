from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets import Cityscapes

from config.schema.data import DataConfig, DatasetType
from data.data_split import DataSplit
from data.dataset.class_filter import ClassFilter
from utils.errors import DatasetNotImplementedError


class DatasetFactory:
    @staticmethod
    def create(cfg: DataConfig, split: DataSplit) -> TorchDataset:
        match cfg.dataset:
            case DatasetType.CITYSCAPES:
                data = Cityscapes(
                    root=cfg.path,
                    split=split,
                    mode="fine",
                    target_type="semantic",
                )
                data.classes = ClassFilter.filter_cityscapes_classes(data.classes)

                return data
            case _:
                raise DatasetNotImplementedError(cfg.dataset)
