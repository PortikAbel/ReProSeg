import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import TrainConfig
from model.model import ReProSeg, TrainPhase
from train.eval import acc_from_cm, compute_cm, intersection_and_union_from_cm
from train.loss import Loss, calculate_loss
from train.optim.manager import OptimizerSchedulerManager
from utils.run_context import get_run_context

logger = logging.getLogger(__name__)

CONFUSION_MATRIX_COMPUTE_INTERVAL = 10


@dataclass
class TrainInfo:
    loss: Loss
    accuracy: float
    miou: float
    iou_by_class: np.ndarray


def train(
    cfg: TrainConfig,
    net: ReProSeg,
    train_loader,
    optimizer_scheduler_manager: OptimizerSchedulerManager,
    criterion: nn.Module,
    epoch,
    progress_prefix: str = "Train Epoch",
) -> TrainInfo:
    # Make sure the model is in train mode
    net.train()
    if net.train_phase is TrainPhase.FINETUNE:
        net.freeze_backbone_batchnorm()

    loss_epoch: Loss = Loss.on_device(cfg.env.device)
    total_acc = 0.0
    n_classes: int = cfg.data.require_num_classes() - 1
    total_intersections_by_class = torch.zeros(n_classes).to(cfg.env.device)
    total_unions_by_class = torch.zeros(n_classes).to(cfg.env.device)

    iters = len(train_loader)
    # Show progress on progress bar.
    train_iter = tqdm(
        enumerate(train_loader),
        total=iters,
        desc=progress_prefix + "%s" % epoch,
        mininterval=2.0,
        ncols=0,
        file=get_run_context().tqdm_file,
    )

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    trainable_tensors = sum(1 for p in net.parameters() if p.requires_grad)
    logger.debug(f"Trainable parameters: {trainable_params:,} ({trainable_tensors} tensors)")

    logger.debug(f"Training phase: {net.train_phase.name}")

    accumulated_out = []
    accumulated_ys = []

    # Iterate through the data set to update leaves, concepts and network
    for i, (xs, ys) in train_iter:
        xs = xs.to(cfg.env.device, non_blocking=True)
        ys = ys.to(cfg.env.device, non_blocking=True)

        # Reset the gradients
        optimizer_scheduler_manager.reset_gradients()

        # Perform a forward pass through the network
        _aspp_features, pooled, out = net(xs)

        loss = calculate_loss(
            pooled,
            out,
            ys,
            cfg.model.loss_weights,
            net.train_phase,
            criterion,
        )

        train_iter.set_postfix_str(
            (
                f"LJ:{loss.jsd:.3f}, "
                + f"LT:{loss.tanh:.3f}, "
                + f"LC:{loss.classification:.3f}, "
                + f"L:{loss.total.item():.3f}"
            ),
            refresh=False,
        )

        # Compute the gradient
        loss.total.backward()

        optimizer_scheduler_manager.step(net.train_phase)

        with torch.no_grad():
            loss_epoch.total += loss.total
            loss_epoch.jsd += loss.jsd
            loss_epoch.tanh += loss.tanh
            loss_epoch.classification += loss.classification

            accumulated_out.append(out.detach())
            accumulated_ys.append(ys.detach())

            if (i + 1) % CONFUSION_MATRIX_COMPUTE_INTERVAL == 0 or (i + 1) == iters:
                batched_out = torch.cat(accumulated_out, dim=0)
                batched_ys = torch.cat(accumulated_ys, dim=0)

                cm = compute_cm(batched_out, batched_ys)
                total_acc += acc_from_cm(cm)
                intersections, unions = intersection_and_union_from_cm(cm)
                total_intersections_by_class += intersections
                total_unions_by_class += unions

                accumulated_out.clear()
                accumulated_ys.clear()

    # Average the losses over the epoch
    loss_epoch.total /= float(iters)
    loss_epoch.jsd /= float(iters)
    loss_epoch.tanh /= float(iters)
    loss_epoch.classification /= float(iters)

    return TrainInfo(
        loss=loss_epoch,
        accuracy=total_acc / np.ceil(iters / CONFUSION_MATRIX_COMPUTE_INTERVAL),
        miou=(total_intersections_by_class / total_unions_by_class).mean().item(),
        iou_by_class=(total_intersections_by_class / total_unions_by_class).detach().cpu().numpy(),
    )
