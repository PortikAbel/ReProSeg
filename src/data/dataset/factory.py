from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets import Cityscapes, VOCSegmentation

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
                cfg.set_num_classes(len(data.classes))

                return data
            case DatasetType.VOC_SEGMENTATION:
                return VOCSegmentation(
                    root=cfg.path,
                    year="2012",
                    image_set=split.value,
                    download=False,
                )
            case _:
                raise DatasetNotImplementedError(cfg.dataset)
