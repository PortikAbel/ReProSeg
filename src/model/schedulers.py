from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LRScheduler,
    StepLR,
)

from config.schema.training import SchedulerConfig, SchedulerType


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    iters_per_epoch: int,
    max_iterations: int,
) -> LRScheduler:
    """Build a scheduler that is stepped once per iteration.

    Config durations are in epochs and get converted to iterations here.
    """
    eta_min = 0.0 if cfg.eta_min is None else cfg.eta_min

    match SchedulerType(cfg.type):
        case SchedulerType.COSINE_WARM_RESTARTS:
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=(cfg.T_0 or 5) * iters_per_epoch,
                T_mult=cfg.T_mult or 1,
                eta_min=eta_min,
            )
        case SchedulerType.COSINE_ANNEALING:
            t_max = cfg.T_max * iters_per_epoch if cfg.T_max else max_iterations
            return CosineAnnealingLR(optimizer, T_max=max(t_max, 1), eta_min=eta_min)
        case SchedulerType.STEP:
            return StepLR(
                optimizer,
                step_size=(cfg.step_size or 10) * iters_per_epoch,
                gamma=0.1 if cfg.gamma is None else cfg.gamma,
            )
        case SchedulerType.EXPONENTIAL:
            gamma = 0.95 if cfg.gamma is None else cfg.gamma
            return ExponentialLR(optimizer, gamma=gamma ** (1 / iters_per_epoch))
        case SchedulerType.CONSTANT:
            return ConstantLR(optimizer, factor=1.0)
