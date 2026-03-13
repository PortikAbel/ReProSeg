from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import ReProSegConfig
from model.model import ReProSeg, TrainPhase
from model.optimizers import OptimizerSchedulerManager
from train.eval import acc_from_cm, compute_cm, intersection_and_union_from_cm
from train.loss import Loss, calculate_loss
from utils.log import Log

CONFUSION_MATRIX_COMPUTE_INTERVAL = 10


@dataclass
class PrototypeStats:
    mean_activation: float
    active_ratio_1e3: float
    active_ratio_1e2: float
    dead_count: int
    total_prototypes: int


@dataclass
class TrainInfo:
    loss: Loss
    accuracy: float
    miou: float
    iou_by_class: np.ndarray
    prototype_stats: PrototypeStats | None = None


def compute_prototype_stats(pooled: torch.Tensor) -> PrototypeStats:
    """Compute statistics about prototype activation for monitoring."""
    with torch.no_grad():
        # pooled shape: (N, P, H, W)
        flat = pooled.flatten(2)  # (N, P, HW)
        proto_mean = flat.mean(dim=(0, 2))  # mean per prototype across batch and spatial

        # Active ratio at different thresholds
        per_pixel = pooled.permute(0, 2, 3, 1).reshape(-1, pooled.shape[1])  # (N*H*W, P)
        active_1e3 = (per_pixel > 1e-3).float().mean().item()
        active_1e2 = (per_pixel > 1e-2).float().mean().item()

        # Dead prototypes: mean activation below threshold
        dead_threshold = 1e-4
        dead_count = (proto_mean < dead_threshold).sum().item()

        return PrototypeStats(
            mean_activation=proto_mean.mean().item(),
            active_ratio_1e3=active_1e3,
            active_ratio_1e2=active_1e2,
            dead_count=dead_count,
            total_prototypes=pooled.shape[1],
        )


def train(
    cfg: ReProSegConfig,
    log: Log,
    net: ReProSeg,
    train_loader,
    optimizer_scheduler_manager: OptimizerSchedulerManager,
    criterion: nn.Module,
    epoch,
    progress_prefix: str = "Train Epoch",
) -> TrainInfo:
    # Make sure the model is in train mode
    net.train()

    if net.train_phase == TrainPhase.PRETRAIN:
        progress_prefix = "Pretrain Epoch"

    loss_epoch: Loss = Loss.on_device(cfg.env.device)
    total_acc = 0.0
    total_intersections_by_class = torch.zeros(cfg.data.num_classes - 1).to(cfg.env.device)
    total_unions_by_class = torch.zeros(cfg.data.num_classes - 1).to(cfg.env.device)

    # Prototype activation monitoring
    accumulated_pooled = []

    iters = len(train_loader)
    # Show progress on progress bar.
    train_iter = tqdm(
        enumerate(train_loader),
        total=iters,
        desc=progress_prefix + "%s" % epoch,
        mininterval=2.0,
        ncols=0,
        file=log.tqdm_file,
    )

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    trainable_tensors = sum(1 for p in net.parameters() if p.requires_grad)
    log.debug(f"Trainable parameters: {trainable_params:,} ({trainable_tensors} tensors)")

    log.debug(f"Training phase: {net.train_phase.name}")

    accumulated_out = []
    accumulated_ys = []

    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:
        xs1 = xs1.to(cfg.env.device, non_blocking=True)
        xs2 = xs2.to(cfg.env.device, non_blocking=True)
        ys = ys.to(cfg.env.device, non_blocking=True)

        # Reset the gradients
        optimizer_scheduler_manager.reset_gradients()

        # Perform a forward pass through the network
        aspp_features, pooled, out = net(torch.cat([xs1, xs2]))
        ys = torch.cat([ys, ys])

        loss = calculate_loss(
            aspp_features,
            pooled,
            out,
            ys,
            cfg.model.loss_weights,
            net.train_phase,
            criterion,
        )

        train_iter.set_postfix_str(
            (
                f"LA:{loss.alignment:.2f}, "
                + f"LJ:{loss.jsd:.3f}, "
                + f"LT:{loss.tanh:.3f}, "
                + f"LC:{loss.classification:.3f}, "
                + f"L:{loss.total.item():.3f}"
            ),
            refresh=False,
        )

        # Compute the gradient
        loss.total.backward()

        optimizer_scheduler_manager.step(net.train_phase, epoch - 1 + (i / iters))

        with torch.no_grad():
            loss_epoch.total += loss.total
            loss_epoch.alignment += loss.alignment
            loss_epoch.jsd += loss.jsd
            loss_epoch.tanh += loss.tanh
            loss_epoch.classification += loss.classification

            # Collect pooled activations for monitoring (sample periodically to avoid memory issues)
            if i % 10 == 0:
                accumulated_pooled.append(pooled.detach().cpu())

            if net.train_phase is not TrainPhase.PRETRAIN:
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
    loss_epoch.alignment /= float(iters)
    loss_epoch.jsd /= float(iters)
    loss_epoch.tanh /= float(iters)
    loss_epoch.classification /= float(iters)

    # Compute prototype statistics from accumulated pooled activations
    proto_stats = None
    if accumulated_pooled:
        pooled_sample = torch.cat(accumulated_pooled[:10], dim=0).to(cfg.env.device)  # Use first 10 batches
        proto_stats = compute_prototype_stats(pooled_sample)
        log.debug(
            f"Prototype stats: mean_act={proto_stats.mean_activation:.4f}, "
            f"active@1e-3={proto_stats.active_ratio_1e3:.2%}, "
            f"active@1e-2={proto_stats.active_ratio_1e2:.2%}, "
            f"dead={proto_stats.dead_count}/{proto_stats.total_prototypes}"
        )

    return TrainInfo(
        loss=loss_epoch,
        accuracy=total_acc / np.ceil(iters / CONFUSION_MATRIX_COMPUTE_INTERVAL),
        miou=(total_intersections_by_class / total_unions_by_class).mean().item(),
        iou_by_class=(total_intersections_by_class / total_unions_by_class).detach().cpu().numpy(),
        prototype_stats=proto_stats,
    )
