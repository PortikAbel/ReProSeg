import os
from typing import Any, Dict

import hydra
import nni  # type: ignore[import-untyped]
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from config import ReProSegConfig
from data import DataLoader, PanopticPartsDataset, get_train_val_split
from proto_segmentation.model import PPNet
from utils.log import Log
from visualize.interpretability_protopnet import ModelInterpretability

load_dotenv()


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="config")
def main(cfg_dict: DictConfig):
    nni_trial_id = os.environ.get("NNI_TRIAL_JOB_ID")
    if nni_trial_id:
        if nni_params := nni.get_next_parameter():
            OmegaConf.set_struct(cfg_dict, False)
            nni_overrides = OmegaConf.from_dotlist([f"{key}={value}" for key, value in nni_params.items()])
            cfg_dict = OmegaConf.merge(cfg_dict, nni_overrides)  # type: ignore[assignment]
            OmegaConf.set_struct(cfg_dict, True)
    cfg_object: Dict[str, Any] = OmegaConf.to_container(cfg_dict, resolve=True)  # type: ignore[assignment]
    cfg = ReProSegConfig(**cfg_object)

    log = Log(cfg.logging.path, __name__)

    log.debug(f"Config: {OmegaConf.to_yaml(cfg_dict)}")
    log.debug(f"Device used: {cfg.env.device}")
    if nni_trial_id:
        log.info(f"NNI trial ID: {nni_trial_id}")

    train_subset, _valid_subset = get_train_val_split(cfg)
    cfg.data.num_classes = len(train_subset.dataset.classes)  # type: ignore[attr-defined]
    panoptic_parts_subset = PanopticPartsDataset(cfg.data, train_subset)
    panoptic_parts_loader = DataLoader(panoptic_parts_subset, cfg)

    if cfg.model.checkpoint is None:
        raise ValueError("A serialized PPNet checkpoint is required. Set model.checkpoint=/path/to/push_best.pth.")
    net = torch.load(
        cfg.model.checkpoint,
        map_location=cfg.env.device,
        weights_only=False,
    )
    if not isinstance(net, PPNet):
        raise TypeError(
            f"Expected checkpoint {cfg.model.checkpoint} to contain a PPNet, but found {type(net).__name__}."
        )
    net = net.to(device=cfg.env.device)

    interpretability = ModelInterpretability(net, cfg, log)
    interpretability.compute_prototype_consistency_score(panoptic_parts_loader)

    log.close()


if __name__ == "__main__":
    main()
