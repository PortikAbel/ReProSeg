from collections import namedtuple

import numpy as np
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Lambda, ToImage

from config.schema.data import DatasetType

CITYSCAPES_CLASSES = Cityscapes.classes

VOC_Class = namedtuple(
    "VOC_Class",
    ["name", "id", "color"],
)

VOC_CLASSES = [
    VOC_Class("background", 0, (0, 0, 0)),
    VOC_Class("aeroplane", 1, (0, 0, 0)),
    VOC_Class("bicycle", 2, (0, 0, 0)),
    VOC_Class("bird", 3, (0, 0, 0)),
    VOC_Class("boat", 4, (0, 0, 0)),
    VOC_Class("bottle", 5, (0, 0, 0)),
    VOC_Class("bus", 6, (0, 0, 0)),
    VOC_Class("car", 7, (0, 0, 0)),
    VOC_Class("cat", 8, (0, 0, 0)),
    VOC_Class("chair", 9, (0, 0, 0)),
    VOC_Class("cow", 10, (0, 0, 0)),
    VOC_Class("diningtable", 11, (0, 0, 0)),
    VOC_Class("dog", 12, (0, 0, 0)),
    VOC_Class("horse", 13, (0, 0, 0)),
    VOC_Class("motorbike", 14, (0, 0, 0)),
    VOC_Class("person", 15, (0, 0, 0)),
    VOC_Class("pottedplant", 16, (0, 0, 0)),
    VOC_Class("sheep", 17, (0, 0, 0)),
    VOC_Class("sofa", 18, (0, 0, 0)),
    VOC_Class("train", 19, (0, 0, 0)),
    VOC_Class("tvmonitor", 20, (0, 0, 0)),
]


class LabelMapping:
    """Handles label mapping logic independent of transforms."""

    @staticmethod
    def _get_cityscapes_classes() -> list[Cityscapes.CityscapesClass]:
        """Returns Cityscapes classes without ignored classes."""

        return [CITYSCAPES_CLASSES[0]] + [c for c in CITYSCAPES_CLASSES if not c.ignore_in_eval]

    @staticmethod
    def _get_pascal_voc_classes() -> list[VOC_Class]:
        """Returns the list of classes for Pascal VOC segmentation."""

        return VOC_CLASSES

    @staticmethod
    def get_classes(dataset_type: DatasetType) -> list[tuple]:
        """Returns the list of classes for the given dataset type."""
        match dataset_type:
            case DatasetType.CITYSCAPES:
                return LabelMapping._get_cityscapes_classes()
            case DatasetType.VOC_SEGMENTATION:
                return LabelMapping._get_pascal_voc_classes()

    @staticmethod
    def _get_cityscapes_transform() -> Compose:
        filtered_classes = LabelMapping._get_cityscapes_classes()
        map_labels = np.array(
            [0 if c.ignore_in_eval else filtered_classes.index(c) for c in CITYSCAPES_CLASSES], dtype=np.int64
        )

        filter_transform = Compose(
            [
                Lambda(lambda x: map_labels[x]),
                Lambda(lambda x: x.transpose(1, 2, 0)),
                ToImage(),
            ]
        )

        return filter_transform

    @staticmethod
    def _get_voc_transform() -> Compose:
        """Returns a transform that maps the VOC void label (255) to 0 (background)."""
        map_labels = np.arange(256, dtype=np.int64)
        map_labels[255] = 0

        return Compose(
            [
                Lambda(lambda x: map_labels[np.array(x, dtype=np.int64)]),
                Lambda(lambda x: x.transpose(1, 2, 0)),
                ToImage(),
            ]
        )

    @staticmethod
    def get_mapping(dataset_type: DatasetType) -> Compose:
        """Returns the label mapping transform for the given dataset type."""
        match dataset_type:
            case DatasetType.CITYSCAPES:
                return LabelMapping._get_cityscapes_transform()
            case DatasetType.VOC_SEGMENTATION:
                return LabelMapping._get_voc_transform()
