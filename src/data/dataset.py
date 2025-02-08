import numpy as np
import torch
from PIL import Image
import torchvision.transforms.v2 as transforms


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    """Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        # if isinstance(dataset, torchvision.datasets.Cityscapes):
        #     #TODO: do we need this? self.imgs and self.targets?
        #     # Create a DataLoader to load all data in a single batch
        #     data_loader = DataLoader(
        #         dataset,
        #         batch_size=10,  # TODO: for testing purposes, only working with 10 images; should be: batch_size=len(dataset),
        #         shuffle=False
        #     )

        #     # Retrieve all images and targets in one go
        #     self.imgs, self.targets = next(iter(data_loader))  # Get a single batch from the DataLoader

        # else:
        #     self.targets = dataset._labels
        #     self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]

        # adding the label (segmenation mask) to the RGB input images as a fourth channel in order to apply the same augmentation to both
        image_array = np.array(image)  # Shape: (H, W, 3)
        target_array = np.array(target)  # Shape: (H, W)

        # Ensure both images have the same size
        assert image_array.shape[:2] == target_array.shape, "Image and segmentation mask must have the same dimensions"

        # Add the grayscale image (segmentation mask) as the fourth channel
        image_target_array = np.dstack((image_array, target_array))  # Shape: (H, W, 4)

        # Convert back to a PIL image
        image_target_img = Image.fromarray(image_target_array, mode="RGBA")

        # apply augmentation
        image_target_img = self.transform1(image_target_img)


        image_target_array = np.array(image_target_img)  # Shape: (H, W, 4)

        # Split the array
        image_array = image_target_array[:, :, :3]  # Shape: (H, W, 3) - RGB channels
        target_array = image_target_array[:, :, 3]  # Shape: (H, W) - Alpha channel (grayscale)

        # Convert back to PIL images
        image = Image.fromarray(image_array, mode="RGB")
        target = Image.fromarray(target_array, mode="L")

        #TODO: only temporary!! target is set to a single pixel value
        target_transform = transforms.ToTensor()
        target = target_transform(target)
        return self.transform2(image), self.transform2(image), target[:, 0, 0]  # return two augmented images and the target

    def __len__(self):
        return len(self.dataset)

