import pytest
import torch
from pydantic import ValidationError

from config.schema.training import (
    ConstantSchedulerConfig,
    CosineAnnealingSchedulerConfig,
    CosineWarmRestartsSchedulerConfig,
    ExponentialSchedulerConfig,
    SchedulerConfig,
    SchedulersConfig,
    StepSchedulerConfig,
)
from train.optim.schedulers import build_scheduler


def _lrs(cfg: SchedulerConfig, iters_per_epoch: int, epochs: int, steps: int) -> list[float]:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = build_scheduler(optimizer, cfg, iters_per_epoch, iters_per_epoch * epochs)
    lrs = []
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_defaults_preserve_previous_behaviour():
    cfg = SchedulersConfig()
    assert cfg.net == CosineAnnealingSchedulerConfig()
    assert cfg.classifier == CosineWarmRestartsSchedulerConfig(T_0=5)


def test_scheduler_selected_by_type_from_dict():
    cfg = SchedulersConfig(net={"type": "step", "step_size": 3}, classifier={"type": "constant"})
    assert cfg.net == StepSchedulerConfig(step_size=3, gamma=0.1)
    assert cfg.classifier == ConstantSchedulerConfig()


def test_irrelevant_param_rejected():
    with pytest.raises(ValidationError):
        SchedulersConfig(classifier={"type": "constant", "gamma": 0.5})


def test_warm_restarts_period_is_in_epochs():
    lrs = _lrs(CosineWarmRestartsSchedulerConfig(T_0=2), iters_per_epoch=4, epochs=10, steps=8)
    assert lrs[-1] == pytest.approx(1.0)  # restarted after 2 epochs * 4 iterations


def test_step_decays_per_epoch():
    lrs = _lrs(StepSchedulerConfig(step_size=2, gamma=0.1), iters_per_epoch=3, epochs=10, steps=6)
    assert lrs[4] == pytest.approx(1.0)
    assert lrs[5] == pytest.approx(0.1)


def test_exponential_decays_gamma_per_epoch():
    lrs = _lrs(ExponentialSchedulerConfig(gamma=0.5), iters_per_epoch=4, epochs=10, steps=8)
    assert lrs[-1] == pytest.approx(0.25)


def test_constant_and_cosine_annealing():
    assert _lrs(ConstantSchedulerConfig(), 4, 2, 8) == [1.0] * 8
    assert _lrs(CosineAnnealingSchedulerConfig(), 4, 2, 8)[-1] == pytest.approx(0.0, abs=1e-9)


def test_warmup_ramps_up_then_follows_schedule():
    lrs = _lrs(CosineAnnealingSchedulerConfig(warmup_epochs=1), iters_per_epoch=4, epochs=3, steps=8)
    assert lrs[0] == pytest.approx(0.01 + (1 - 0.01) / 4)
    assert lrs[:3] == sorted(lrs[:3])
    assert lrs[3] == pytest.approx(1.0)
    assert lrs[-1] < lrs[3]
