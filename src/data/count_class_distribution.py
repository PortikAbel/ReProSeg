import numpy as np
import PIL

from data.dataloaders import get_dataloaders


def count_class_distribution(args, log):
    (
        train_loader,
        _test_loader,
        _train_loader_visualization,
    ) = get_dataloaders(log, args)

    num_classes = len(train_loader.dataset.dataset.classes)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for labels in train_loader.dataset.dataset.dataset.targets:
        labels = np.array(PIL.Image.open(labels[0]))
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()

    np.save("class_counts.npy", class_counts)

    return class_counts