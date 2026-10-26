from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
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

WARMUP_START_FACTOR = 0.01


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    iters_per_epoch: int,
    max_iterations: int,
) -> LRScheduler:
    """Build a scheduler that is stepped once per iteration.

    Config durations are in epochs and get converted to iterations here.
    ``max_iterations`` is the number of steps the scheduler will be stepped in total, warmup included.
    """
    warmup_iterations = cfg.warmup_epochs * iters_per_epoch
    if warmup_iterations == 0:
        return _build_main_scheduler(optimizer, cfg, iters_per_epoch, max_iterations)
    main = _build_main_scheduler(optimizer, cfg, iters_per_epoch, max(max_iterations - warmup_iterations, 1))
    warmup = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, total_iters=warmup_iterations)
    return SequentialLR(optimizer, [warmup, main], milestones=[warmup_iterations])


def _build_main_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    iters_per_epoch: int,
    max_iterations: int,
) -> LRScheduler:
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
