"""Hydra entry point for rendering ReProSeg concept-prototype visualizations.

Set model.checkpoint to a trained checkpoint. To match the exact architecture
it was trained with, reuse its training run's resolved config, e.g.:
`uv run python -m visualize --config-path=<run_dir>/.hydra --config-name=config`
"""

import logging
from typing import Any, Dict

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from config import VisualizeConfig
from data import DataLoader, Dataset, get_train_val_split
from model.model import ReProSeg
from utils.run_context import get_run_context, init_run_context
from visualize.visualizer import ModelVisualizer

load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="visualize")
def main(cfg_dict: DictConfig) -> None:
    cfg_object: Dict[str, Any] = OmegaConf.to_container(cfg_dict, resolve=True)  # type: ignore[assignment]
    cfg = VisualizeConfig(**cfg_object)

    init_run_context(cfg.logging)

    if cfg.model.checkpoint is None:
        raise ValueError("model.checkpoint must be set to the checkpoint to visualize.")

    train_subset, _valid_subset = get_train_val_split(cfg)

    net = ReProSeg(cfg=cfg).to(device=cfg.env.device)

    visualize_set = Dataset(cfg.data, train_subset)
    visualize_loader = DataLoader(visualize_set, cfg.data)

    visualizer = ModelVisualizer(net, cfg)
    visualizer.visualize_prototypes(visualize_loader)

    get_run_context().close()


if __name__ == "__main__":
    main()
