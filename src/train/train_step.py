import torch
from tqdm import tqdm

import argparse
from model.model import ReProSeg, TrainPhase
from model.optimizers import OptimizerSchedulerManager
from train.eval import compute_cm, acc_from_cm, intersection_and_union_from_cm
from train.loss import LossWeights, calculate_loss
from utils.log import Log

def train(
    args: argparse.Namespace,
    log: Log,
    net: ReProSeg,
    train_loader,
    optimizer_scheduler_manager: OptimizerSchedulerManager,
    criterion,
    epoch,
    progress_prefix: str = "Train Epoch",
):
    # Make sure the model is in train mode
    net.train()

    if net.train_phase == TrainPhase.PRETRAIN:
        progress_prefix = "Pretrain Epoch"

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0
    total_intersections_by_class = torch.zeros(args.num_classes - 1).to(args.device)
    total_unions_by_class = torch.zeros(args.num_classes - 1).to(args.device)

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

    count_param = 0
    for _, param in net.named_parameters():
        if param.requires_grad:
            count_param += 1
    log.debug(f"Number of parameters that require gradient: {count_param}")

    loss_weights = LossWeights(args)

    log.debug(f"Training phase: {net.train_phase.name}")

    prototype_activations = torch.empty(
        (iters, 2 * train_loader.batch_size, net.layers.num_prototypes, *args.wshape)
    )
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:
        xs1, xs2, ys = xs1.to(args.device), xs2.to(args.device), ys.to(args.device)
        
        # Reset the gradients
        optimizer_scheduler_manager.reset_gradients()
        
        # Perform a forward pass through the network
        aspp_features, pooled, out = net(torch.cat([xs1, xs2]))
        ys = torch.cat([ys, ys])
        prototype_activations[i, :, :, :, :] = pooled

        loss = calculate_loss(
            log,
            aspp_features,
            pooled,
            out,
            ys,
            loss_weights,
            net.train_phase,
            criterion,
            train_iter,
            len(train_iter) * (epoch - 1) + i,
            print=True,
        )

        # Compute the gradient
        loss.backward()

        optimizer_scheduler_manager.step(net.train_phase, epoch - 1 + (i / iters))

        with torch.no_grad():
            total_loss += loss.item()

            if net.train_phase is not TrainPhase.PRETRAIN:
                cm = compute_cm(out, ys)
                total_acc += acc_from_cm(cm)
                intersections, unions = intersection_and_union_from_cm(cm)
                total_intersections_by_class += intersections
                total_unions_by_class += unions

        if net.train_phase is not TrainPhase.PRETRAIN:
            with torch.no_grad():
                net.layers.classification_layer.weight.copy_(
                    torch.where(
                        net.layers.classification_layer.weight < 1e-3,
                        0.0,
                        net.layers.classification_layer.weight,
                    )
                )  # set weights in classification layer < 1e-3 to zero
                if net.layers.classification_layer.bias is not None:
                    net.layers.classification_layer.bias.copy_(
                        torch.clamp(net.layers.classification_layer.bias.data, min=0.0)
                    )

    train_info["loss"] = total_loss / float(i + 1)
    train_info["train_accuracy"] = total_acc / float(i + 1)
    train_info["train_miou"] = (total_intersections_by_class / total_unions_by_class).mean().item()
    train_info["train_iou_by_class"] = (total_intersections_by_class / total_unions_by_class).detach().cpu().numpy()
    train_info["prototype_activations"] = (
        prototype_activations.view((-1, net.layers.num_prototypes))
        .detach()
        .cpu()
        .numpy()
    )

    return train_info

