"""Hydra entry point for training ReProSeg.

Produces checkpoints under the run's log directory. Visualize or evaluate a
trained checkpoint separately via `python -m visualize` / `python -m evaluate`.
"""

import logging
import os
import socket
from typing import Any, Dict

import hydra
import nni  # type: ignore[import-untyped]
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from config import TrainConfig
from data import get_train_val_split
from model.model import ReProSeg
from train.trainer import train_model
from utils.run_context import get_run_context, init_run_context

load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="train")
def main(cfg_dict: DictConfig) -> None:
    nni_trial_id = os.environ.get("NNI_TRIAL_JOB_ID")
    if nni_trial_id:
        if nni_params := nni.get_next_parameter():
            OmegaConf.set_struct(cfg_dict, False)
            nni_overrides = OmegaConf.from_dotlist([f"{k}={v}" for k, v in nni_params.items()])
            cfg_dict = OmegaConf.merge(cfg_dict, nni_overrides)  # type: ignore[assignment]
            OmegaConf.set_struct(cfg_dict, True)
    cfg_object: Dict[str, Any] = OmegaConf.to_container(cfg_dict, resolve=True)  # type: ignore[assignment]
    cfg = TrainConfig(**cfg_object)

    # Setup run artifacts (checkpoints, tensorboard, tqdm output)
    init_run_context(cfg.logging)

    logger.debug(f"Config: {OmegaConf.to_yaml(cfg_dict)}")
    logger.debug(f"Device used: {cfg.env.device}")
    if str.lower(cfg.env.device.type) != "cpu":
        logger.debug(f"Device name: {torch.cuda.get_device_name(cfg.env.device)}")
    logger.debug(f"Pytorch version: {torch.__version__}")
    logger.debug(f"Hostname: {socket.gethostname()}")
    if nni_trial_id:
        logger.info(f"NNI trial ID: {nni_trial_id}")

    train_subset, valid_subset = get_train_val_split(cfg)

    net = ReProSeg(cfg=cfg).to(device=cfg.env.device)

    train_model(net, train_subset, valid_subset, cfg)

    get_run_context().close()


if __name__ == "__main__":
    main()
