import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from model.model import ReProSeg
from utils.log import Log


@torch.no_grad()
def eval(
    args: argparse.Namespace,
    log: Log,
    net: nn.DataParallel[ReProSeg],
    test_loader: DataLoader,
    epoch,
    progress_prefix: str = "Eval Epoch",
) -> dict:
    net = net.to(args.device)
    net.eval()
    eval_info = dict()

    n_classes = net.module._num_classes
    cm = torch.zeros((n_classes, n_classes), dtype=int).to(args.device)
    abstained = 0.0
    
    test_iter = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=progress_prefix + " %s" % epoch,
        mininterval=5.0,
        ncols=0,
        file=log.tqdm_file,
    )
    (xs, ys) = next(iter(test_loader))
    
    for _, (xs, ys) in test_iter:
        xs, ys = xs.to(args.device), ys.to(args.device)

        with torch.no_grad():
            _, pooled, out = net(xs, inference=True)
            max_out_score, ys_pred = torch.max(out, dim=1)

            pixel_count = torch.prod(torch.tensor(max_out_score.shape))
            abstained_pixels = pixel_count - torch.count_nonzero(max_out_score)
            abstained += abstained_pixels / pixel_count
            
            y_true_flat = ys.squeeze(1).view(-1)
            y_pred_flat = ys_pred.view(-1)
            cm_batch = torch.bincount(y_true_flat * n_classes + y_pred_flat, minlength=n_classes**2).view(n_classes, n_classes).to(args.device)
            cm += cm_batch
            
            acc = acc_from_cm(cm_batch)
            test_iter.set_postfix_str(f"Acc: {acc:.3f}", refresh=False)

        del out
        del pooled
        del ys_pred

    abstained /= len(test_iter)
    log.info(f"model abstained from a decision for {abstained.item()*100}% of images")

    num_nonzero_prototypes = torch.count_nonzero(F.relu(net.module._classification.weight - 1e-3)).item()
    num_prototypes = torch.numel(net.module._classification.weight)
    log.info(f"sparsity ratio: {(num_prototypes - num_nonzero_prototypes) / num_prototypes}")

    eval_info["abstained"] = abstained.item()
    eval_info["num non-zero prototypes"] = num_nonzero_prototypes
    eval_info["confusion_matrix"] = cm
    eval_info["test_accuracy"] = acc_from_cm(cm)

    return eval_info


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
