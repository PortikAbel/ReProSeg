import os
from typing import Any, Dict

import hydra
import nni  # type: ignore[import-untyped]
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch import Generator
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

from config import ReProSegConfig
from data import SupportedSplit
from data.dataloader import DataLoader
from data.dataset.base import Dataset
from data.dataset.double_augment import DoubleAugmentDataset
from data.dataset.panoptic_parts import PanopticPartsDataset
from model.model import ReProSeg
from utils.log import Log

load_dotenv()


@hydra.main(version_base=None, config_path="../config/yaml", config_name="config")
def main(cfg_dict: DictConfig):
    nni_trial_id = os.environ.get("NNI_TRIAL_JOB_ID")
    if nni_trial_id:
        if nni_params := nni.get_next_parameter():
            cfg_dict = OmegaConf.merge(cfg_dict, nni_params)  # type: ignore[assignment]
    cfg_object: Dict[str, Any] = OmegaConf.to_container(cfg_dict, resolve=True)  # type: ignore[assignment]
    cfg = ReProSegConfig(**cfg_object)

    # Setup logger
    log = Log(cfg.logging.path, __name__)

    log.info(f"Config: {cfg}")
    log.info(f"Device used: {cfg.env.device}")
    if nni_trial_id:
        log.info(f"NNI trial ID: {nni_trial_id}")

    # Create the dataloaders
    train_set = Dataset(cfg.data, SupportedSplit.TRAIN)
    train_subset: Subset[Dataset]
    valid_subset: Subset[Dataset]
    train_subset, valid_subset = random_split(
        train_set,
        [1 - cfg.data.validation_size, cfg.data.validation_size],
        generator=Generator().manual_seed(cfg.env.seed),
    )
    double_augment_subset: Subset[DoubleAugmentDataset] = Subset(
        DoubleAugmentDataset(train_set), train_subset.indices
    )
    train_loader = DataLoader(double_augment_subset, cfg)
    valid_loader = DataLoader(valid_subset, cfg)
    train_loader_visualization = DataLoader(train_subset, cfg)
    panoptic_parts_subset: Subset[PanopticPartsDataset] = Subset(
        PanopticPartsDataset(train_set), train_subset.indices
    )
    panoptic_parts_loader = DataLoader(panoptic_parts_subset, cfg)

    cfg.data.num_classes = len(train_set.classes)

    # Model
    net = ReProSeg(cfg=cfg, log=log).to(device=cfg.env.device)

    if not cfg.training.skip_training:
        from train.trainer import train_model

        try:
            train_model(net, train_loader, valid_loader, log, cfg)
        except Exception as e:
            log.exception(e)

    if cfg.visualization.generate_explanations:
        from visualize.visualizer import ModelVisualizer

        visualizer = ModelVisualizer(net, cfg, log)
        visualizer.visualize_prototypes(train_loader_visualization)

    if cfg.evaluation.consistency_score.calculate:
        from visualize.interpretability import ModelInterpretability

        interpretability = ModelInterpretability(net, cfg, log)
        interpretability.compute_prototype_consistency_score(panoptic_parts_loader)

    log.close()


if __name__ == "__main__":
    main()
