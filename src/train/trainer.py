import nni  # type: ignore[import-untyped]
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ReProSegConfig
from config.schema.model import LossCriterion
from data.count_class_distribution import get_class_weights
from model.model import ReProSeg, TrainPhase
from model.optimizers import OptimizerSchedulerManager
from train.criterion.dice import DiceLoss
from train.criterion.weighted_nll import WeightedNLLLoss
from train.test_step import eval
from train.train_step import train
from utils.log import Log


def train_model(net: ReProSeg, train_loader: DataLoader, test_loader: DataLoader, log: Log, cfg: ReProSegConfig):
    optimizer_scheduler_manager = OptimizerSchedulerManager(
        net, len(train_loader) * cfg.training.epochs.pretrain, cfg.training.learning_rates.backbone_end
    )
    if cfg.model.checkpoint is not None:
        optimizer_scheduler_manager.load_state_dict(cfg.model.checkpoint)

    class_weights = get_class_weights(cfg, log)
    criterion: nn.Module
    match cfg.model.criterion:
        case LossCriterion.DICE:
            criterion = DiceLoss()
        case LossCriterion.WEIGHTED_DICE:
            criterion = DiceLoss(class_weights)
        case LossCriterion.WEIGHTED_NLL:
            criterion = WeightedNLLLoss(class_weights)
        case _:
            raise NotImplementedError(f"criterion {cfg.model.criterion} not implemented")

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _ = next(iter(train_loader))
        xs1 = xs1.to(cfg.env.device)
        _aspp_features, pooled, _out = net(xs1)
        log.debug(f"ASPP features output shape: {_aspp_features.shape}")
        log.debug(f"pooled ASPP output shape: {pooled.shape}")

    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, cfg.training.epochs.pretrain + 1):
        log.info(f"Pretrain Epoch {epoch} with batch size {train_loader.batch_size}")

        # Pretrain prototypes
        net.pretrain()
        train_info = train(
            cfg,
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

    if cfg.model.checkpoint is None:
        net.eval()
        log.model_checkpoint(get_checkpoint(), "net_pretrained")
        net.train()

    # SECOND TRAINING PHASE re-initialize optimizers and schedulers
    # for second training phase
    if cfg.training.epochs.start <= 1:
        optimizer_scheduler_manager = OptimizerSchedulerManager(
            net,
            len(train_loader) * cfg.training.epochs.total,
            cfg.training.learning_rates.backbone_full,
        )

    best_acc = 0.0
    best_miou = 0.0

    for epoch in range(cfg.training.epochs.start, cfg.training.epochs.total + 1):
        if epoch <= cfg.training.epochs.finetune and (
            cfg.training.epochs.pretrain > 0 or cfg.model.checkpoint is not None
        ):
            # during fine-tuning, only train classification layer and freeze rest.
            # usually done for a few epochs (at least 1, more depends on size of dataset)
            net.finetune()
        elif epoch <= cfg.training.epochs.freeze:
            # freeze first layers of backbone, train rest
            net.freeze()
        else:
            # unfreeze backbone
            net.unfreeze()

        log.info(
            f"Epoch {epoch} first layers of backbone frozen: "
            f"{net.train_phase in [TrainPhase.FINETUNE, TrainPhase.FREEZE_FIRST_LAYERS]}"
        )
        if (epoch == cfg.training.epochs.total or epoch % 30 == 0) and cfg.training.epochs.total > 1:
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
                if cfg.model.bias:
                    cls_b = net.layers.classification_layer.bias
                    log.debug(f"Classifier bias: {cls_b}")
                torch.set_printoptions(profile="default")

        train_info = train(
            cfg,
            log,
            net,
            train_loader,
            optimizer_scheduler_manager,
            criterion,
            epoch,
        )
        # Evaluate model
        eval_info = eval(cfg, log, net, test_loader, epoch)
        log.tb_scalar("Acc/eval-epochs", eval_info["test_accuracy"], epoch)
        log.tb_scalar("Acc/train-epochs", train_info["train_accuracy"], epoch)
        log.tb_scalar("mIoU/train-epochs", train_info["train_miou"], epoch)
        log.tb_scalar("mIoU/eval-epochs", eval_info["test_miou"], epoch)
        log.tb_scalar("Loss/train-epochs", train_info["loss"], epoch)

        nni.report_final_result(train_info["train_accuracy"])

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
