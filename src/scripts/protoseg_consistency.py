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
import torch
import proto_segmentation.model
from proto_segmentation.model import construct_PPNet


torch.serialization.safe_globals([proto_segmentation.model.PPNet])


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
    net = construct_PPNet().to(device=cfg.env.device)

    net = torch.load(
    '/home/annamari/ProtoSeg-checkpoints/cityscapes_no_kld_imnet_4_16/checkpoints/push_best.pth',
    map_location=cfg.env.device,
    weights_only=False,)

    
    from visualize.interpretability import ModelInterpretability

    interpretability = ModelInterpretability(net, cfg, log)
    interpretability.compute_prototype_consistency_score(panoptic_parts_loader)

    log.close()


if __name__ == "__main__":
    main()

