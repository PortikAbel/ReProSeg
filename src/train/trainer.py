import logging

import nni  # type: ignore[import-untyped]
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

from config import TrainConfig
from config.schema.model import LossCriterion
from data import AugmentedDataset, DataLoader, Dataset
from data.count_class_distribution import get_class_weights
from model.model import ReProSeg
from train.criterion.dice import DiceLoss
from train.criterion.weighted_nll import WeightedCrossEntropyLoss
from train.optim.manager import OptimizerSchedulerManager
from train.test_step import eval
from train.train_step import train
from utils.run_context import get_run_context
from utils.version import check_checkpoint_version, compat_version

logger = logging.getLogger(__name__)


def train_model(net: ReProSeg, train_data: TorchDataset, valid_data: TorchDataset, cfg: TrainConfig):
    run = get_run_context()
    train_set = AugmentedDataset(cfg.data, train_data)
    valid_set = Dataset(cfg.data, valid_data)
    train_loader = DataLoader(train_set, cfg.data)
    valid_loader = DataLoader(valid_set, cfg.data)

    last_epoch = 0
    best_acc = 0.0
    best_miou = 0.0

    optimizer_scheduler_manager = OptimizerSchedulerManager(
        net, cfg.training.schedulers, len(train_loader), cfg.training.epochs
    )
    if cfg.training.resume:
        checkpoint_path = cfg.model.checkpoint
        if checkpoint_path is None:
            raise ValueError("model.checkpoint must be set to resume training.")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.env.device, weights_only=False)
        check_checkpoint_version(checkpoint, checkpoint_path)
        optimizer_scheduler_manager.load_state_dict(checkpoint)
        last_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        best_miou = checkpoint["best_miou"]
        if checkpoint["total_epochs"] != cfg.training.epochs.total:
            logger.warning(
                f"epochs.total ({cfg.training.epochs.total}) differs from the checkpoint's "
                f"({checkpoint['total_epochs']}); the scheduler state is restored as saved."
            )
        if checkpoint.get("finetune_epochs", cfg.training.epochs.finetune) != cfg.training.epochs.finetune:
            logger.warning(
                f"epochs.finetune ({cfg.training.epochs.finetune}) differs from the checkpoint's "
                f"({checkpoint['finetune_epochs']}); the backbone scheduler state is restored as saved."
            )
        logger.info(f"Resuming training after epoch {last_epoch}")

    class_weights = get_class_weights(train_data, cfg.data).to(cfg.env.device)
    criterion: nn.Module
    match cfg.model.criterion:
        case LossCriterion.NLL:
            criterion = WeightedCrossEntropyLoss(device=cfg.env.device)
        case LossCriterion.WEIGHTED_NLL:
            criterion = WeightedCrossEntropyLoss(device=cfg.env.device, class_weights=class_weights)
        case LossCriterion.DICE:
            criterion = DiceLoss(torch.ones(cfg.data.require_num_classes(), device=cfg.env.device))
        case LossCriterion.WEIGHTED_DICE:
            criterion = DiceLoss(class_weights)
        case _:
            raise NotImplementedError(f"criterion {cfg.model.criterion} not implemented")

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs, _ = next(iter(train_loader))
        xs = xs.to(cfg.env.device)
        _aspp_features, pooled, _out = net(xs)
        logger.debug(f"ASPP features output shape: {_aspp_features.shape}")
        logger.debug(f"pooled ASPP output shape: {pooled.shape}")

    def get_checkpoint(epoch: int):
        checkpoint = optimizer_scheduler_manager.get_checkpoint()
        checkpoint["model_state_dict"] = net.state_dict()
        checkpoint["version"] = compat_version()
        checkpoint["epoch"] = epoch
        checkpoint["total_epochs"] = cfg.training.epochs.total
        checkpoint["finetune_epochs"] = cfg.training.epochs.finetune
        checkpoint["best_acc"] = best_acc
        checkpoint["best_miou"] = best_miou
        return checkpoint

    for epoch in range(last_epoch + 1, cfg.training.epochs.total + 1):
        if epoch <= cfg.training.epochs.finetune:
            net.finetune()
        else:
            # unfreeze backbone
            net.full_train()

        train_info = train(
            cfg,
            net,
            train_loader,
            optimizer_scheduler_manager,
            criterion,
            epoch,
        )

        run.tb_scalar("Acc/train-epochs", train_info.accuracy, epoch)
        run.tb_scalar("mIoU/train-epochs", train_info.miou, epoch)
        run.tb_scalar("loss-train/L", train_info.loss.total.item(), epoch)
        run.tb_scalar("loss-train/L_JSD", train_info.loss.jsd.item(), epoch)
        run.tb_scalar("loss-train/LT", train_info.loss.tanh.item(), epoch)
        run.tb_scalar("loss-train/LC", train_info.loss.classification.item(), epoch)

        eval_info = eval(cfg, net, valid_loader, epoch)

        run.tb_scalar("Acc/eval-epochs", eval_info.accuracy, epoch)
        run.tb_scalar("mIoU/eval-epochs", eval_info.miou, epoch)

        nni.report_intermediate_result(eval_info.miou)

        improved_acc = eval_info.accuracy > best_acc
        improved_miou = eval_info.miou > best_miou
        if improved_acc:
            best_acc = eval_info.accuracy
            logger.info(f"Best accuracy so far: {best_acc}")
        if improved_miou:
            best_miou = eval_info.miou
            logger.info(f"Best mIoU so far: {best_miou}")

        checkpoint = get_checkpoint(epoch)
        run.model_checkpoint(checkpoint, "net_trained_last")
        if improved_acc:
            run.model_checkpoint(checkpoint, "net_trained_best_acc")
        if improved_miou:
            run.model_checkpoint(checkpoint, "net_trained_best_miou")
            # Logged on every improvement so the HParams tab reflects progress while training is still running.
            run.log_hparams(cfg.model_dump(exclude={"logging": {"path"}}), {"best_miou": best_miou})

    nni.report_final_result(best_miou)
    logger.info("Done!")
