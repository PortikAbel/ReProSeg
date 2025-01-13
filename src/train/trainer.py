import numpy as np
import torch
import torch.nn as nn

import argparse

from data.dataloaders import get_dataloaders
# from model.model import ReProSeg, get_network  # TODO: add imports
from model.util.func import init_weights_xavier
from model.util.log import Log
# from train.train_step import train
# from train.test_step import eval


def train_model(log: Log, args: argparse.Namespace):

    # Log which device was actually used
    log.info(
        f"Device used: {args.device} "
        f"{f'with id {args.device_ids}' if len(args.device_ids) > 0 else ''}",
    )

    # Obtain the dataloaders
    (
        train_loader,
        test_loader,
        classes,
    ) = get_dataloaders(log, args)

    # Create a convolutional network based on arguments and add 1x1 conv layer
    (
        feature_net,
        add_on_layers,
        pool_layer,
        classification_layer,
        num_prototypes,
    ) = get_network(args, log, len(classes))

    # Create a ReProSeg model
    net = ReProSeg(
        args=args,
        log=log,
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )

    net = net.to(device=args.device)
    net = nn.DataParallel(net, device_ids=args.device_ids)

    optimizer_net, optimizer_classifier = net.module.get_optimizers()

    # Initialize or load model
    with torch.no_grad():
        if args.pretrained_net_state_dict_dir is not None:
            epoch = 0
            checkpoint = torch.load(args.pretrained_net_state_dict_dir, map_location=args.device)
            net.load_state_dict(checkpoint["model_state_dict"], strict=True)
            log.info("Pretrained network loaded")
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint["optimizer_net_state_dict"])
            except Exception:
                pass
            if (
                torch.mean(net.module._classification.weight).item() > 1.0
                and torch.mean(net.module._classification.weight).item() < 3.0
                and torch.count_nonzero(
                    torch.relu(net.module._classification.weight - 1e-5)
                )
                .float()
                .item()
                > 0.8 * (num_prototypes * len(classes))
            ):  # assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                log.warning("We assume that the classification layer is not yet trained. We re-initialize it...")
                torch.nn.init.normal_(
                    net.module._classification.weight, mean=1.0, std=0.1
                )
                torch.nn.init.constant_(net.module._multiplier, val=2.0)
                log.info(f"Classification layer initialized with mean {torch.mean(net.module._classification.weight).item()}")
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            else:
                if "optimizer_classifier_state_dict" in checkpoint.keys():
                    optimizer_classifier.load_state_dict(
                        checkpoint["optimizer_classifier_state_dict"]
                    )

        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            torch.nn.init.constant_(net.module._multiplier, val=2.0)
            net.module._multiplier.requires_grad = False

            log.info(f"Classification layer initialized with mean {torch.mean(net.module._classification.weight).item()}")

    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction="mean").to(args.device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(train_loader) * args.epochs_pretrain,
        eta_min=args.lr_block / 100.0,
        last_epoch=-1,
    )

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(train_loader))
        xs1 = xs1.to(args.device)
        proto_features, _, _ = net(xs1)
        wshape = np.array(proto_features.shape)[-2:]
        args.wshape = wshape  # needed for calculating image patch size
        log.debug("Output shape: {proto_features.shape}")

    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy,
        # F1-score, mean train accuracy and mean loss for each epoch
        log.create_log(
            "log_epoch_overview",
            "epoch",
            "test_top1_acc",
            "test_f1",
            "almost_sim_nonzeros",
            "local_size_all_classes",
            "almost_nonzeros_pooled",
            "num_nonzero_prototypes",
            "mean_train_acc",
            "mean_train_loss_during_epoch",
        )
        log.warning(
            "Your dataset only has two classes. "
            "Is the number of samples per class similar? "
            "If the data is imbalanced, we recommend to use "
            "the --weighted_loss flag to account for the imbalance."
        )
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5),
        # mean train accuracy and mean loss for each epoch
        log.create_log(
            "log_epoch_overview",
            "epoch",
            "test_top1_acc",
            "test_top5_acc",
            "almost_sim_nonzeros",
            "local_size_all_classes",
            "almost_nonzeros_pooled",
            "num_nonzero_prototypes",
            "mean_train_acc",
            "mean_train_loss_during_epoch",
        )
    log.create_log(
        "log_loss_weights"
        "Align",
        "Tanh",
        "Uniformity",
        "Variance",
        "Classification",
    )

    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain + 1):
        log.info(f"Pretrain Epoch {epoch} with batch size {train_loader.batch_size}")

        # Pretrain prototypes
        net.module.pretrain()
        train_info = train(
            args,
            log,
            net,
            train_loader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            None,
            criterion,
            epoch,
            pretrain=True,
            finetune=False,
        )
        log.log_values(
            "log_epoch_overview",
            epoch,
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            train_info["loss"],
        )

    def get_checkpoint(with_optimizer_classifier_state_dict: bool = True):
        if not with_optimizer_classifier_state_dict:
            return {
                "model_state_dict": net.state_dict(),
                "optimizer_net_state_dict": optimizer_net.state_dict(),
            }
        return {
            "model_state_dict": net.state_dict(),
            "optimizer_net_state_dict": optimizer_net.state_dict(),
            "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
        }

    if args.pretrained_net_state_dict_dir is None:
        net.eval()
        log.model_checkpoint(
            get_checkpoint(with_optimizer_classifier_state_dict=False),
            "net_pretrained",
        )
        net.train()

    # SECOND TRAINING PHASE re-initialize optimizers and schedulers
    # for second training phase
    optimizer_net, optimizer_classifier = net.module.get_optimizers()
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(train_loader) * args.epochs,
        eta_min=args.lr_net / 100.0,
    )
    # scheduler for the classification layer is with restarts,
    # such that the model can re-activated zeroed-out prototypes.
    # Hence, an intuitive choice.
    if args.epochs <= 30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier,
            T_0=5,
            eta_min=0.001,
            T_mult=1,
            verbose=False,
        )
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier,
            T_0=10,
            eta_min=0.001,
            T_mult=1,
            verbose=False,
        )

    frozen = True
    lrs_net = []
    lrs_classifier = []

    for epoch in range(1, args.epochs + 1):
        # during fine-tuning, only train classification layer and freeze rest.
        # usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= args.epochs_finetune and (
            args.epochs_pretrain > 0 or args.pretrained_net_state_dict_dir is not None
        ):
            net.module.finetune()
            finetune = True

        # freeze first layers of backbone, train rest
        elif epoch <= args.freeze_epochs:
            finetune = False
            net.module.freeze()
            frozen = True

        # unfreeze backbone
        else:
            net.module.unfreeze()
            frozen = False

        log.info(f"Epoch {epoch} frozen: {frozen}")
        if (epoch == args.epochs or epoch % 30 == 0) and args.epochs > 1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(
                    torch.clamp(net.module._classification.weight.data - 0.001, min=0.0)
                )
                cls_w = net.module._classification.weight[
                    net.module._classification.weight.nonzero(as_tuple=True)
                ]
                log.debug(f"Classifier weights:\n{cls_w}\n{cls_w.shape}")
                if args.bias:
                    cls_b = net.module._classification.bias
                    log.debug(f"Classifier bias: {cls_b}", flush=True)
                torch.set_printoptions(profile="default")

        train_info = train(
            args,
            log,
            net,
            train_loader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            scheduler_classifier,
            criterion,
            epoch,
            pretrain=False,
            finetune=finetune,
        )
        lrs_net += train_info["lrs_net"]
        lrs_classifier += train_info["lrs_class"]
        # Evaluate model
        eval_info = eval(args, log, net, test_loader, epoch)
        log.log_values(
            "log_epoch_overview",
            epoch,
            eval_info["top1_accuracy"],
            eval_info["top5_accuracy"],
            eval_info["almost_sim_nonzeros"],
            eval_info["local_size_all_classes"],
            eval_info["almost_nonzeros"],
            eval_info["num non-zero prototypes"],
            train_info["train_accuracy"],
            train_info["loss"],
        )
        log.tb_scalar("Acc/eval-epochs", eval_info["top1_accuracy"], epoch)
        log.tb_scalar("Acc/train-epochs", train_info["train_accuracy"], epoch)
        log.tb_scalar("Loss/train-epochs", train_info["loss"], epoch)
        log.tb_scalar("Num non-zero prototypes", eval_info["almost_nonzeros"], epoch)

        with torch.no_grad():
            net.eval()
            log.model_checkpoint(get_checkpoint(), "net_trained")

            if epoch % 30 == 0:
                net.eval()
                log.model_checkpoint(get_checkpoint(), f"net_trained_{epoch}")

    net.eval()
    log.model_checkpoint(get_checkpoint(), "net_trained_last")

    nonzero_weights = net.module._classification.weight[
        net.module._classification.weight.nonzero(as_tuple=True)
    ]
    log.debug(f"Classifier weights:\n{net.module._classification.weight}")
    log.debug(f"Classifier weights nonzero:\n{nonzero_weights}\n{nonzero_weights.shape}")
    log.debug(f"Classifier bias:\n{net.module._classification.bias}")
    # Print weights and relevant prototypes per class
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c, :]
        for p in range(net.module._classification.weight.shape[1]):
            if proto_weights[p] > 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))

    log.info("Done!")
