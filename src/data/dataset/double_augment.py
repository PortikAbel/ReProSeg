from torchvision.transforms.v2 import Compose

from .base import Dataset, SupportedDataset


class DoubleAugmentDataset(Dataset):
    """Returns two similar augmented version of the image along with the labels."""

    def __init__(self, dataset_name: SupportedDataset):
        super().__init__(dataset_name, split="train")
        self.transform_base_image = self.transforms.base_image
        self.transform1 = self.transforms.geometry_augmentation
        self.transform2 = Compose([self.transforms.color_augmentation, self.transforms.image_normalization])
        self.transform_base_target = self.transforms.base_target
        self.transform_shrink_target = self.transforms.shrink_target

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        image = self.transform_base_image(image)
        target = self.transform_base_target(target)

        image, target = self.transform1(image, target)

        return (
            self.transform2(image),
            self.transform2(image),
            self.transform_shrink_target(target),
        )

    @property
    def transform(self):
        return None

    @property
    def target_transform(self):
        return None
