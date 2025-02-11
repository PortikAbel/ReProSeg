from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

class TwoAugSupervisedDataset(Dataset):
    """Returns two augmentation and no labels."""

    def __init__(self,
            dataset: Dataset,
            transform1: Transform,
            transform2: Transform):

        self.dataset = dataset
        self.classes = dataset.classes
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        image, target = self.transform1(image, target)
        return self.transform2(image),self.transform2(image), target

    def __len__(self):
        return len(self.dataset)
