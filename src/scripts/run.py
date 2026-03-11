import os
from typing import Any, Dict

import hydra
import nni  # type: ignore[import-untyped]
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from config import ReProSegConfig
from data import DataLoader, Dataset, PanopticPartsDataset, get_train_val_split
from model.model import ReProSeg
from utils.log import Log

load_dotenv()


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="config")
def main(cfg_dict: DictConfig):
    nni_trial_id = os.environ.get("NNI_TRIAL_JOB_ID")
    if nni_trial_id:
        if nni_params := nni.get_next_parameter():
            OmegaConf.set_struct(cfg_dict, False)
            nni_overrides = OmegaConf.from_dotlist([f"{k}={v}" for k, v in nni_params.items()])
            cfg_dict = OmegaConf.merge(cfg_dict, nni_overrides)  # type: ignore[assignment]
            OmegaConf.set_struct(cfg_dict, True)
    cfg_object: Dict[str, Any] = OmegaConf.to_container(cfg_dict, resolve=True)  # type: ignore[assignment]
    cfg = ReProSegConfig(**cfg_object)

    # Setup logger
    log = Log(cfg.logging.path, __name__)

    log.info(f"Config: {cfg}")
    log.info(f"Device used: {cfg.env.device}")
    if nni_trial_id:
        log.info(f"NNI trial ID: {nni_trial_id}")

    # Create the dataloaders
    train_subset, valid_subset = get_train_val_split(cfg)

    cfg.data.num_classes = len(train_subset.dataset.classes)  # type: ignore[attr-defined]

    # Model
    net = ReProSeg(cfg=cfg, log=log).to(device=cfg.env.device)

    if not cfg.training.skip_training:
        from train.trainer import train_model

        try:
            train_model(net, train_subset, valid_subset, log, cfg)
        except Exception as e:
            log.exception(e)

    if cfg.visualization.generate_explanations:
        from visualize.visualizer import ModelVisualizer

        visualize_set = Dataset(cfg.data, train_subset)
        visualize_loader = DataLoader(visualize_set, cfg)

        visualizer = ModelVisualizer(net, cfg, log)
        visualizer.visualize_prototypes(visualize_loader)

    if cfg.evaluation.consistency_score.calculate:
        from visualize.interpretability import ModelInterpretability

        panoptic_parts_subset = PanopticPartsDataset(cfg.data, train_subset)
        panoptic_parts_loader = DataLoader(panoptic_parts_subset, cfg)

        interpretability = ModelInterpretability(net, cfg, log)
        interpretability.compute_prototype_consistency_score(panoptic_parts_loader)

    log.close()


if __name__ == "__main__":
    main()
