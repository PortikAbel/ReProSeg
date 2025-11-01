import os
from typing import Any, Dict

import hydra
import nni  # type: ignore[import-untyped]
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from config import ReProSegConfig
from data.dataloader import DataLoader, DoubleAugmentDataLoader, PanopticPartsDataLoader
from model.model import ReProSeg
from utils.log import Log

load_dotenv()  # loads .env into os.environ


@hydra.main(version_base=None, config_path="../config/yaml", config_name="config")
def main(cfg_dict: DictConfig):
    nni_trial_id = os.environ.get("NNI_TRIAL_JOB_ID")
    if nni_trial_id:
        if nni_params := nni.get_next_parameter():
            cfg_dict = OmegaConf.merge(cfg_dict, nni_params)  # type: ignore[assignment]
    cfg_object: Dict[str, Any] = OmegaConf.to_object(cfg_dict)  # type: ignore[assignment]
    cfg = ReProSegConfig(**cfg_object)

    # Setup logger
    log = Log(cfg.logging.path, __name__)

    log.info(f"Config: {cfg}")
    log.info(f"Device used: {cfg.env.device}")
    if nni_trial_id:
        log.info(f"NNI trial ID: {nni_trial_id}")

    # Create the dataloaders
    train_loader = DoubleAugmentDataLoader(cfg)
    test_loader = DataLoader("test", cfg)
    train_loader_visualization = DataLoader("train", cfg)
    panoptic_parts_loader = PanopticPartsDataLoader("train", cfg)

    cfg.data.num_classes = len(train_loader.dataset.classes)

    # Model
    net = ReProSeg(cfg=cfg, log=log).to(device=cfg.env.device)

    if not cfg.training.skip_training:
        from train.trainer import train_model

        try:
            train_model(net, train_loader, test_loader, log, cfg)
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
