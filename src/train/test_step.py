from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ReProSegConfig
from model.model import ReProSeg
from utils.log import Log

from .eval import acc_from_cm, compute_absained, compute_cm, miou_from_cm


@dataclass
class EvalInfo:
    abstained: float
    num_non_zero_concepts: int
    confusion_matrix: np.ndarray
    accuracy: float
    miou: float


@torch.no_grad()
def eval(
    cfg: ReProSegConfig,
    log: Log,
    net: ReProSeg,
    valid_loader: DataLoader,
    epoch,
    progress_prefix: str = "Eval Epoch",
) -> EvalInfo:
    net = net.to(cfg.env.device)
    net.eval()

    n_classes: int = cfg.data.num_classes - 1
    cm = torch.zeros((n_classes, n_classes), dtype=torch.int32).to(cfg.env.device)
    abstained = 0.0

    test_iter = tqdm(
        enumerate(valid_loader),
        total=len(valid_loader),
        desc=progress_prefix + " %s" % epoch,
        mininterval=5.0,
        ncols=0,
        file=log.tqdm_file,
    )
    (xs, ys) = next(iter(valid_loader))

    for _, (xs, ys) in test_iter:
        xs, ys = xs.to(cfg.env.device), ys.to(cfg.env.device)

        with torch.no_grad():
            _, pooled, out = net(xs, inference=True)

            abstained += compute_absained(out, ys)
            cm_batch = compute_cm(out, ys)
            cm += cm_batch

            acc = acc_from_cm(cm_batch)
            miou = miou_from_cm(cm_batch)

            test_iter.set_postfix_str(f"Acc: {acc:.3f}, mIoU: {miou:.3f}", refresh=False)

        del out
        del pooled

    abstained /= len(test_iter)
    log.info(f"model abstained from a decision for {abstained * 100}% of images")

    num_nonzero_concepts = int(torch.count_nonzero(F.relu(net.layers.classification_layer.weight - 1e-3)).item())
    num_concepts = torch.numel(net.layers.classification_layer.weight)
    log.info(f"sparsity ratio: {(num_concepts - num_nonzero_concepts) / num_concepts}")

    return EvalInfo(
        abstained=abstained,
        num_non_zero_concepts=num_nonzero_concepts,
        confusion_matrix=cm.detach().cpu().numpy(),
        accuracy=acc_from_cm(cm),
        miou=miou_from_cm(cm),
    )
