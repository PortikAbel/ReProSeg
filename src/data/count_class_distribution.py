import numpy as np

from data.dataloaders import get_dataloaders


def count_class_distribution(args, log, save_path):
    (
        train_loader,
        _test_loader,
        train_loader_visualization,
    ) = get_dataloaders(log, args)

    num_classes = len(train_loader.dataset.dataset.classes)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in train_loader_visualization:
        for c in range(num_classes):
            class_counts[c] += (label == c).sum()

    np.save(save_path, class_counts)

    return class_counts