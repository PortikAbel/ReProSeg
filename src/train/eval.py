import torch


def compute_absained(out: torch.Tensor, ys: torch.Tensor) -> float:
    """
    Compute the number of pixels that were abstained from a decision
    :param out: model output
    :param ys: ground truth labels
    :param abstained: current number of abstained pixels
    :return: updated number of abstained pixels
    """
    max_out_score = torch.max(out, dim=1).values

    mask_labeled = ys.squeeze(1) == 0

    pixel_count = torch.prod(torch.tensor(max_out_score.shape))
    abstained_pixels = pixel_count - torch.count_nonzero(max_out_score[mask_labeled])

    return abstained_pixels / pixel_count


def compute_cm(out: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """
    Compute the confusion matrix
    :param out: model output
    :param ys: ground truth labels
    :return: confusion matrix
    """
    assert len(out.shape) == 4 and len(ys.shape) == 4
    assert out.shape[0] == ys.shape[0] and out.shape[2:] == ys.shape[2:]

    max_out_score, ys_pred = torch.max(out, dim=1)

    ys_true = ys.squeeze(1)

    mask_labeled = ys_true != 0
    
    y_true_flat = ys_true[mask_labeled]
    y_pred_flat = ys_pred[mask_labeled]

    n_classes = out.shape[1]

    cm = torch.bincount(
        y_true_flat * n_classes + y_pred_flat,
        minlength=n_classes**2
    ).view(n_classes, n_classes).to(ys.device)

    return cm[1:, 1:]  # Exclude the background class (0)


def acc_from_cm(cm: torch.Tensor) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = cm.diagonal().sum()
    total = cm.sum()

    return 1 if total == 0 else correct / total

def intersection_and_union_from_cm(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the intersection and union from the confusion matrix
    :param cm: confusion matrix
    :return: intersection and union tensors
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    intersection = cm.diagonal()
    union = (cm + cm.T).sum(dim=0) - intersection

    return intersection, union


def miou_from_cm(cm: torch.Tensor) -> float:
    """
    Compute the mean Intersection over Union (IoU) from the confusion matrix
    :param cm: confusion matrix
    :return: mean IoU score
    """
    intersection, union = intersection_and_union_from_cm(cm)
    iou = intersection / union

    return iou.mean().item() if union.sum() > 0 else 0.0