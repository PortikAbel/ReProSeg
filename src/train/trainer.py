import nni  # type: ignore[import-untyped]
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

from config import ReProSegConfig
from config.schema.model import LossCriterion
from data import DataLoader, Dataset, DoubleAugmentDataset
from data.count_class_distribution import get_class_weights
from model.model import ReProSeg, TrainPhase
from model.optimizers import OptimizerSchedulerManager
from train.criterion.dice import DiceLoss
from train.criterion.weighted_nll import WeightedNLLLoss
from train.test_step import eval
from train.train_step import train
from utils.log import Log


def train_model(net: ReProSeg, train_data: TorchDataset, valid_data: TorchDataset, log: Log, cfg: ReProSegConfig):
    double_augment_set = DoubleAugmentDataset(cfg.data, train_data)
    valid_set = Dataset(cfg.data, valid_data)
    train_loader = DataLoader(double_augment_set, cfg)
    valid_loader = DataLoader(valid_set, cfg)

    optimizer_scheduler_manager = OptimizerSchedulerManager(
        net, len(train_loader) * cfg.training.epochs.pretrain, cfg.training.learning_rates.backbone_end
    )
    if cfg.model.checkpoint is not None:
        checkpoint = torch.load(cfg.model.checkpoint, map_location=cfg.env.device, weights_only=False)
        optimizer_scheduler_manager.load_state_dict(checkpoint)

    class_weights = get_class_weights(
        train_data, cfg.data.num_classes, cfg.env.class_distribution_cache_path, cfg, log
    ).to(cfg.env.device)
    criterion: nn.Module
    match cfg.model.criterion:
        case LossCriterion.NLL:
            criterion = WeightedNLLLoss()
        case LossCriterion.WEIGHTED_NLL:
            criterion = WeightedNLLLoss(class_weights)
        case LossCriterion.DICE:
            criterion = DiceLoss(torch.ones(cfg.data.num_classes, device=cfg.env.device))
        case LossCriterion.WEIGHTED_DICE:
            criterion = DiceLoss(class_weights)
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
        eval_info = eval(cfg, log, net, valid_loader, epoch)

        # Log to TensorBoard
        log.tb_scalar("Acc/train-epochs", train_info.accuracy, epoch)
        log.tb_scalar("mIoU/train-epochs", train_info.miou, epoch)
        log.tb_scalar("loss-train/L", train_info.loss.total.item(), epoch)
        log.tb_scalar("loss-train/LA", train_info.loss.alignment.item(), epoch)
        log.tb_scalar("loss-train/L_JSD", train_info.loss.jsd.item(), epoch)
        log.tb_scalar("loss-train/LT", train_info.loss.tanh.item(), epoch)
        log.tb_scalar("loss-train/LC", train_info.loss.classification.item(), epoch)
        
        # Log prototype activation statistics
        if train_info.prototype_stats is not None:
            stats = train_info.prototype_stats
            log.tb_scalar("prototype/mean_activation", stats.mean_activation, epoch)
            log.tb_scalar("prototype/active_ratio_1e-3", stats.active_ratio_1e3, epoch)
            log.tb_scalar("prototype/active_ratio_1e-2", stats.active_ratio_1e2, epoch)
            log.tb_scalar("prototype/dead_count", stats.dead_count, epoch)
            log.tb_scalar("prototype/alive_ratio", 1.0 - (stats.dead_count / stats.total_prototypes), epoch)

        log.tb_scalar("Acc/eval-epochs", eval_info.accuracy, epoch)
        log.tb_scalar("mIoU/eval-epochs", eval_info.miou, epoch)

        nni.report_intermediate_result(eval_info.miou)

        with torch.no_grad():
            net.eval()
            log.model_checkpoint(get_checkpoint(), "net_trained_last")

            if eval_info.accuracy > best_acc:
                best_acc = eval_info.accuracy
                log.info(f"Best accuracy so far: {best_acc}")
                log.model_checkpoint(get_checkpoint(), "net_trained_best_acc")

            if eval_info.miou > best_miou:
                best_miou = eval_info.miou
                log.info(f"Best mIoU so far: {best_miou}")
                log.model_checkpoint(get_checkpoint(), "net_trained_best_miou")

    nni.report_final_result(best_miou)
    log.info("Done!")
