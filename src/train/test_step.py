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
    cm = torch.zeros((net.module._num_classes, net.module._num_classes), dtype=int).to(args.device)
    abstained = 0
    
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
            abstained += np.prod(max_out_score.shape) - torch.count_nonzero(max_out_score)
            
            cm_batch = torch.zeros((net.module._num_classes, net.module._num_classes), dtype=int).to(args.device)
            for y_pred, y_true in zip(ys_pred, ys):
                cm[y_true][y_pred] += 1
                cm_batch[y_true][y_pred] += 1
            acc = acc_from_cm(cm_batch)

            test_iter.set_postfix_str(f"Acc: {acc:.3f}", refresh=False)

        del out
        del pooled
        del ys_pred

    abstained /= np.prod(xs.shape[-2:])
    log.info(f"model abstained from a decision for {abstained.item()} images")

    num_nonzero_prototypes = torch.count_nonzero(F.relu(net.module._classification.weight - 1e-3)).item()
    num_prototypes = torch.numel(net.module._classification.weight)
    log.info(f"sparsity ratio: {(num_prototypes - num_nonzero_prototypes) / num_prototypes}")

    eval_info["abstained"] = abstained.item()
    eval_info["num non-zero prototypes"] = num_nonzero_prototypes
    eval_info["confusion_matrix"] = cm.item()
    eval_info["test_accuracy"] = acc_from_cm(cm)

    return eval_info


def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = cm.diagonal().sum()
    total = np.sum(cm)

    return 1 if total == 0 else correct / total
