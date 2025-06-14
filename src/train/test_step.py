import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from model.model import ReProSeg
from utils.log import Log
from .eval import compute_absained, compute_cm, acc_from_cm, iou_from_cm


@torch.no_grad()
def eval(
    args: argparse.Namespace,
    log: Log,
    net: ReProSeg,
    test_loader: DataLoader,
    epoch,
    progress_prefix: str = "Eval Epoch",
) -> dict:
    net = net.to(args.device)
    net.eval()
    eval_info = dict()

    n_classes = args.num_classes
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

            abstained += compute_absained(out, ys)
            cm_batch = compute_cm(out, ys)
            cm_batch = cm_batch[1:, 1:]  # ignore unlabeled class
            cm += cm_batch

            acc = acc_from_cm(cm_batch)
            iou = iou_from_cm(cm_batch)

            test_iter.set_postfix_str(f"Acc: {acc:.3f}, mIoU: {iou:.3f}", refresh=False)

        del out
        del pooled

    abstained /= len(test_iter)
    log.info(f"model abstained from a decision for {abstained.item()*100}% of images")

    num_nonzero_prototypes = torch.count_nonzero(F.relu(net.layers.classification_layer.weight - 1e-3)).item()
    num_prototypes = torch.numel(net.layers.classification_layer.weight)
    log.info(f"sparsity ratio: {(num_prototypes - num_nonzero_prototypes) / num_prototypes}")

    eval_info["abstained"] = abstained.item()
    eval_info["num non-zero prototypes"] = num_nonzero_prototypes
    eval_info["confusion_matrix"] = cm
    eval_info["test_accuracy"] = acc_from_cm(cm)
    eval_info["test_miou"] = iou_from_cm(cm)

    return eval_info


