import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from tqdm import tqdm

import argparse
from model.model import ReProSeg
from train.loss import calculate_loss
from utils.log import Log

def train(
    args: argparse.Namespace,
    log: Log,
    net: nn.DataParallel[ReProSeg],
    train_loader,
    optimizer_net,
    optimizer_classifier,
    scheduler_net,
    scheduler_classifier,
    criterion,
    epoch,
    pretrain=False,
    finetune=False,
    progress_prefix: str = "Train Epoch",
):
    # Make sure the model is in train mode
    net.train()

    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = "Pretrain Epoch"
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0

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

    align_pf_weight = (epoch / args.epochs_pretrain) * 1.0
    t_weight = args.tanh_loss
    unif_weight = args.unif_loss
    var_weigth = args.variance_loss
    cl_weight = 0.0
    if not pretrain:
        align_pf_weight = 5.0
        t_weight = 2.0
        unif_weight = 2.0
        var_weigth = 2.0
        cl_weight = 2.0

    log.log_values(
        "log_loss_weights",
        epoch,
        align_pf_weight,
        t_weight,
        unif_weight,
        var_weigth,
        cl_weight,
    )
    log.debug(f"Pretrain - {'ON' if pretrain else 'OFF'}")
    log.debug(f"Finetune - {'ON' if finetune else 'OFF'}")

    lrs_net = []
    lrs_class = []
    prototype_activations = torch.empty(
        (iters, 2 * train_loader.batch_size, net.module._num_prototypes, *args.wshape)
    )
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:
        xs1, xs2, ys = xs1.to(args.device), xs2.to(args.device), ys.to(args.device)

        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Perform a forward pass through the network
        aspp_features, pooled, out = net(torch.cat([xs1, xs2]))
        prototype_activations[i, :, :, :, :] = pooled

        loss, acc = calculate_loss(
            log,
            aspp_features,
            pooled,
            out,
            ys,
            align_pf_weight,
            t_weight,
            unif_weight,
            var_weigth,
            cl_weight,
            pretrain,
            finetune,
            criterion,
            train_iter,
            len(train_iter) * (epoch - 1) + i,
            print=True,
        )

        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i / iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])

        if not finetune:
            optimizer_net.step()
            scheduler_net.step()
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.0)

        with torch.no_grad():
            total_acc += acc
            total_loss += loss.item()

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(
                    torch.where(
                        net.module._classification.weight < 1e-3,
                        0.0,
                        net.module._classification.weight,
                    )
                )  # set weights in classification layer < 1e-3 to zero
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(
                        torch.clamp(net.module._classification.bias.data, min=0.0)
                    )
    train_info["train_accuracy"] = total_acc / float(i + 1)
    train_info["loss"] = total_loss / float(i + 1)
    train_info["lrs_net"] = lrs_net
    train_info["lrs_class"] = lrs_class
    train_info["prototype_activations"] = (
        prototype_activations.view((-1, net.module._num_prototypes))
        .detach()
        .cpu()
        .numpy()
    )

    return train_info

