import numpy as np
import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Lambda, ToImage


class ClassFilter:
    """Handles class filtering logic independent of transforms."""

    @staticmethod
    def filter_cityscapes_classes(classes: list[Cityscapes.CityscapesClass]) -> list[Cityscapes.CityscapesClass]:
        """Returns Cityscapes classes without ignored classes."""

        return [classes[0]] + [c for c in classes if not c.ignore_in_eval]

    @staticmethod
    def get_cityscapes_transform(classes: list[Cityscapes.CityscapesClass]) -> Compose:
        filtered_classes = ClassFilter.filter_cityscapes_classes(classes)
        map_classes = torch.tensor(
            [0 if c.ignore_in_eval else filtered_classes.index(c) for c in classes], dtype=torch.int64
        )

        filter_transform = Compose(
            [
                Lambda(np.vectorize(lambda c: map_classes[c])),
                Lambda(lambda x: x.transpose(1, 2, 0)),
                ToImage(),
            ]
        )

        return filter_transform
