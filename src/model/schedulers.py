from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LRScheduler,
    StepLR,
)

from config.schema.training import (
    ConstantSchedulerConfig,
    CosineAnnealingSchedulerConfig,
    CosineWarmRestartsSchedulerConfig,
    ExponentialSchedulerConfig,
    SchedulerConfig,
    StepSchedulerConfig,
)


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    iters_per_epoch: int,
    max_iterations: int,
) -> LRScheduler:
    """Build a scheduler that is stepped once per iteration.

    Config durations are in epochs and get converted to iterations here.
    """
    match cfg:
        case CosineWarmRestartsSchedulerConfig():
            return CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0 * iters_per_epoch, T_mult=cfg.T_mult, eta_min=cfg.eta_min
            )
        case CosineAnnealingSchedulerConfig():
            t_max = cfg.T_max * iters_per_epoch if cfg.T_max else max_iterations
            return CosineAnnealingLR(optimizer, T_max=max(t_max, 1), eta_min=cfg.eta_min)
        case StepSchedulerConfig():
            return StepLR(optimizer, step_size=cfg.step_size * iters_per_epoch, gamma=cfg.gamma)
        case ExponentialSchedulerConfig():
            return ExponentialLR(optimizer, gamma=cfg.gamma ** (1 / iters_per_epoch))
        case ConstantSchedulerConfig():
            return ConstantLR(optimizer, factor=1.0)
