"""Registry of interpretability metrics runnable via the evaluate entry point.

Adding a new metric: register a run function under a name, add a matching
config group yaml under config/hydra/evaluate/, and set evaluate.metric_name
to select it.
"""

from typing import Callable

from config import EvaluateConfig

_METRICS: dict[str, Callable[[EvaluateConfig], object]] = {}


def register(name: str) -> Callable[[Callable[[EvaluateConfig], object]], Callable[[EvaluateConfig], object]]:
    """Register a metric's run function under `name` for evaluate.metric_name dispatch."""

    def decorator(fn: Callable[[EvaluateConfig], object]) -> Callable[[EvaluateConfig], object]:
        _METRICS[name] = fn
        return fn

    return decorator


def run(cfg: EvaluateConfig) -> object:
    """Dispatch to the metric registered under cfg.evaluate.metric_name."""

    metric_name = cfg.evaluate.metric_name
    if metric_name not in _METRICS:
        raise ValueError(f"Unknown evaluation metric {metric_name!r}; registered metrics: {sorted(_METRICS)}")
    return _METRICS[metric_name](cfg)
