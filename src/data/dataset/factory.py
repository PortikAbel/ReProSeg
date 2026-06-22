from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets import Cityscapes, VOCSegmentation

from config.schema.data import DataConfig, DatasetType
from data.data_split import DataSplit
from data.dataset.label_mapping import LabelMapping
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
            case DatasetType.VOC_SEGMENTATION:
                data = VOCSegmentation(
                    root=cfg.path,
                    year="2012",
                    image_set=split.value,
                    download=False,
                )
            case _:
                raise DatasetNotImplementedError(cfg.dataset)

        data.classes = LabelMapping.get_classes(cfg.dataset)
        cfg.set_num_classes(len(data.classes))

        return data
