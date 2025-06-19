import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import argparse

from model.model import ReProSeg, TrainPhase
from model.optimizers import OptimizerSchedulerManager
from utils.log import Log
from train.criterion.dice import DiceLoss
from train.criterion.weighted_nll import WeightedNLLLoss
from train.train_step import train
from train.test_step import eval

def train_model(net:ReProSeg, train_loader: DataLoader, test_loader: DataLoader, log: Log, args: argparse.Namespace):
    optimizer_scheduler_manager = OptimizerSchedulerManager(
        net,
        len(train_loader) * args.epochs_pretrain,
        args.lr_block
    )

    # Initialize or load model
    with torch.no_grad():
        if args.model_checkpoint is not None:
            checkpoint = torch.load(args.model_checkpoint, map_location=args.device, weights_only=False)
            net.load_state_dict(checkpoint["model_state_dict"], strict=True)
            log.info("Pretrained network loaded")
            optimizer_scheduler_manager.load_state_dict(checkpoint)

            if (
                torch.mean(net.layers.classification_layer.weight).item() > 1.0
                and torch.mean(net.layers.classification_layer.weight).item() < 3.0
                and torch.count_nonzero(torch.relu(net.layers.classification_layer.weight - 1e-5)).float().item()
                    > 0.8 * (args.num_prototypes * args.num_classes)
            ):  # assume that the linear classification layer is not yet trained 
                # (e.g. when loading a pretrained backbone only)
                log.warning("We assume that the classification layer is not yet trained. We re-initialize it...")
                net.init_classifier_weights()

        else:
            net.init_add_on_weights()
            net.init_classifier_weights()

    if args.criterion == "dice":
        criterion = DiceLoss(args, log)
    else:
        criterion = WeightedNLLLoss(args, log)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(train_loader))
        xs1 = xs1.to(args.device)
        _aspp_features, pooled, _out = net(xs1)
        args.wshape = np.array(pooled.shape)[-2:]
        log.debug(f"ASPP features output shape: {_aspp_features.shape}")        
        log.debug(f"pooled ASPP output shape: {pooled.shape}")


    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain + 1):
        log.info(f"Pretrain Epoch {epoch} with batch size {train_loader.batch_size}")

        # Pretrain prototypes
        net.pretrain()
        train_info = train(
            args,
            log,
            net,
            train_loader,
            optimizer_scheduler_manager,
            criterion,
            epoch,
        )

    def get_checkpoint():
        checkpoint = optimizer_scheduler_manager.get_checkpoint()
        checkpoint["model_state_dict"] = net.state_dict()
        return checkpoint

    if args.model_checkpoint is None:
        net.eval()
        log.model_checkpoint(get_checkpoint(), "net_pretrained")
        net.train()

    # SECOND TRAINING PHASE re-initialize optimizers and schedulers
    # for second training phase
    optimizer_scheduler_manager = OptimizerSchedulerManager(
        net,
        len(train_loader) * args.epochs,
        args.lr_net,
    )

    best_acc = 0.0
    best_miou = 0.0

    for epoch in range(args.epoch_start, args.epochs + 1):
        if epoch <= args.epochs_finetune and (
            args.epochs_pretrain > 0 or args.model_checkpoint is not None
        ):
            # during fine-tuning, only train classification layer and freeze rest.
            # usually done for a few epochs (at least 1, more depends on size of dataset)
            net.finetune()
        elif epoch <= args.epochs_freeze:
            # freeze first layers of backbone, train rest
            net.freeze()
        else:
            # unfreeze backbone
            net.unfreeze()

        log.info(
            f"Epoch {epoch} first layers of backbone frozen: "
            f"{net.train_phase in [TrainPhase.FINETUNE, TrainPhase.FREEZE_FIRST_LAYERS]}"
        )
        if (epoch == args.epochs or epoch % 30 == 0) and args.epochs > 1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.layers.classification_layer.weight.copy_(
                    torch.clamp(net.layers.classification_layer.weight.data - 0.001, min=0.0)
                )
                cls_w = net.layers.classification_layer.weight[
                    net.layers.classification_layer.weight.nonzero(as_tuple=True)
                ]
                log.debug(f"Classifier weights:\n{cls_w}\n{cls_w.shape}")
                if args.bias:
                    cls_b = net.layers.classification_layer.bias
                    log.debug(f"Classifier bias: {cls_b}", flush=True)
                torch.set_printoptions(profile="default")

        train_info = train(
            args,
            log,
            net,
            train_loader,
            optimizer_scheduler_manager,
            criterion,
            epoch,
        )
        # Evaluate model
        eval_info = eval(args, log, net, test_loader, epoch)
        log.tb_scalar("Acc/eval-epochs", eval_info["test_accuracy"], epoch)
        log.tb_scalar("Acc/train-epochs", train_info["train_accuracy"], epoch)
        log.tb_scalar("mIoU/train-epochs", train_info["train_miou"], epoch)
        log.tb_scalar("mIoU/eval-epochs", eval_info["test_miou"], epoch)
        log.tb_scalar("Loss/train-epochs", train_info["loss"], epoch)
        log.tb_scalar("Abstained", eval_info["abstained"], epoch)

        with torch.no_grad():
            net.eval()
            log.model_checkpoint(get_checkpoint(), "net_trained_last")

            if train_info["train_accuracy"] > best_acc:
                best_acc = train_info["train_accuracy"]
                log.info(f"Best accuracy so far: {best_acc}")
                log.model_checkpoint(get_checkpoint(), "net_trained_best_acc")

            if train_info["train_miou"] > best_miou:
                best_miou = train_info["train_miou"]
                log.info(f"Best mIoU so far: {best_miou}")
                log.model_checkpoint(get_checkpoint(), "net_trained_best_miou")

    nonzero_weights = net.layers.classification_layer.weight[
        net.layers.classification_layer.weight.nonzero(as_tuple=True)
    ]
    log.debug(f"Classifier weights:\n{net.layers.classification_layer.weight}")
    log.debug(f"Classifier weights nonzero:\n{nonzero_weights}\n{nonzero_weights.shape}")
    log.debug(f"Classifier bias:\n{net.layers.classification_layer.bias}")
    # Print weights and relevant prototypes per class
    for c in range(net.layers.classification_layer.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.layers.classification_layer.weight[c, :]
        for p in range(net.layers.classification_layer.weight.shape[1]):
            if proto_weights[p] > 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))

    log.info("Done!")
