import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from config.schema.logging import LoggingConfig

logger = logging.getLogger(__name__)


class RunContext:
    """Owns the artifacts of a run: checkpoints, TensorBoard events, tqdm output and prototypes."""

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self._tqdm_file = (self._log_dir / "tqdm.log").open(mode="w")
        self._tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)

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
    def prototypes_dir(self) -> Path:
        return self._log_dir / "prototypes"

    def tb_scalar(self, tag, value, step):
        self._tensorboard_writer.add_scalar(tag, value, step)

    def model_checkpoint(self, state_dict, name):
        torch.save(state_dict, self.checkpoint_dir / name)

    def close(self):
        self._tqdm_file.close()
        self._tensorboard_writer.close()


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
