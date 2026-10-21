import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from config.schema.logging import LoggingConfig

logger = logging.getLogger(__name__)


def _flatten_hparams(value: Any, parent_key: str = "") -> Dict[str, Any]:
    """Flatten a nested config dict into dot-notated keys, coercing values to HParams-compatible types."""
    flat: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, sub_value in value.items():
            flat.update(_flatten_hparams(sub_value, f"{parent_key}.{key}" if parent_key else str(key)))
    elif value is None or isinstance(value, (bool, int, float, str)):
        flat[parent_key] = value
    else:
        flat[parent_key] = str(value)
    return flat


# Metric names ever logged via log_hparams, across every scenario. TensorBoard's hparams plugin
# does not merge the metric schema across sessions - it arbitrarily picks one session's Experiment
# proto (see tensorboard.plugins.hparams.backend_context._find_experiment_tag) - so every call must
# declare this same full set (missing ones filled with NaN, rendered blank) or some metrics will
# silently never appear as a HParams column, even though the scalar itself is still logged fine.
KNOWN_HPARAM_METRICS = ("best_miou", "consistency_score")


class RunContext:
    """Owns the artifacts of a run: checkpoints, TensorBoard events, tqdm output and prototypes."""

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.hparams_dir.mkdir(parents=True, exist_ok=True)

        self._tqdm_file = (self._log_dir / "tqdm.log").open(mode="a")
        self._tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self._hparams_writer = SummaryWriter(log_dir=str(self.hparams_dir))

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @property
    def tqdm_file(self):
        return self._tqdm_file

    @property
    def checkpoint_dir(self) -> Path:
        return self._log_dir / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self._log_dir / "tensorboard"

    @property
    def hparams_dir(self) -> Path:
        return self.tensorboard_dir / "hparams"

    @property
    def prototypes_dir(self) -> Path:
        return self._log_dir / "prototypes"

    def consistency_dir(self, official_parts_only: bool = False) -> Path:
        return self._log_dir / f"consistency{'_official_parts' if official_parts_only else ''}"

    def tb_scalar(self, tag, value, step):
        self._tensorboard_writer.add_scalar(tag, value, step)

    def log_hparams(self, config: Dict[str, Any], metrics: Dict[str, float], run_name: str = ".") -> None:
        """Write flattened config + metrics so TensorBoard's HParams tab can compare runs.

        `run_name="."` (the default) writes into this run's own row, safe to call repeatedly
        (e.g. once per improvement) to keep metrics up to date. Pass a distinct `run_name` to add
        a separate row nested under this same run (e.g. one per evaluation sweep parameter).

        Declares the full KNOWN_HPARAM_METRICS schema every time (missing ones as NaN) - see its
        comment for why a partial metric_dict would make that metric disappear as a column.
        Callers are responsible for passing the real value of any metric they can source
        authoritatively (e.g. reading best_miou back from a checkpoint) rather than relying on
        this call to remember a value logged by an earlier process.
        """
        full_metrics = {key: float("nan") for key in KNOWN_HPARAM_METRICS}
        full_metrics.update(metrics)
        self._hparams_writer.add_hparams(_flatten_hparams(config), full_metrics, run_name=run_name)

    def model_checkpoint(self, state_dict, name):
        torch.save(state_dict, self.checkpoint_dir / name)

    def close(self):
        self._tqdm_file.close()
        self._tensorboard_writer.close()
        self._hparams_writer.close()


_run_context: Optional[RunContext] = None


def init_run_context(cfg: LoggingConfig) -> RunContext:
    """Set up process wide logging tweaks and create the singleton run context."""
    global _run_context
    if _run_context is not None:
        raise RuntimeError("Run context has already been initialized.")

    logging.captureWarnings(True)
    if cfg.disable_console:
        root = logging.getLogger()
        for handler in list(root.handlers):
            # FileHandler is a StreamHandler subclass, so only bare stream handlers are dropped
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                root.removeHandler(handler)

    _run_context = RunContext(cfg.path)
    logger.info(f"Log dir: {_run_context.log_dir}")
    return _run_context


def get_run_context() -> RunContext:
    if _run_context is None:
        raise RuntimeError("Run context is not initialized. Call init_run_context() first.")
    return _run_context
