"""Hydra entry point for computing an interpretability metric on a trained checkpoint.

Works for any SupportedModel (ReProSeg, PPNet, ...); select the metric via
`evaluate=<name>` (see config/hydra/evaluate/).
"""

import logging
from typing import Any, Dict

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from config import EvaluateConfig
from evaluate.registry import run as run_metric
from utils.run_context import get_run_context, init_run_context

load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="evaluate")
def main(cfg_dict: DictConfig) -> None:
    cfg_object: Dict[str, Any] = OmegaConf.to_container(cfg_dict, resolve=True)  # type: ignore[assignment]
    cfg = EvaluateConfig(**cfg_object)

    init_run_context(cfg.logging)

    if cfg.model.checkpoint is None:
        raise ValueError("model.checkpoint must be set to the checkpoint to evaluate.")

    logger.info(f"Running evaluation metric: {cfg.evaluate.metric_name}")
    run_metric(cfg)

    get_run_context().close()


if __name__ == "__main__":
    main()
