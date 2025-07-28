from data.dataloader import DataLoader, DoubleAugmentDataLoader, PanopticPartsDataLoader
from model.model import ReProSeg
from utils.args import ModelTrainerArgumentParser
from utils.log import Log

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import torch
import warnings


def set_rand_state(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_device(gpu_ids: str, disable_gpu: bool = False) -> tuple[str, list]:
    """
    Set the device to use for training.

    :param gpu_ids: GPU ids separated with comma
    :param disable_gpu: Whether to disable GPU. Defaults to ``False``.
    :return: The device to use for training
    """

    device_ids = [int(gpu_id) for gpu_id in (gpu_ids.split(",") if gpu_ids else [])]

    if disable_gpu or not torch.cuda.is_available():
        return "cpu", []
    if len(device_ids) == 1:
        return f"cuda:{gpu_ids}", device_ids
    if len(device_ids) == 0:
        device = "cuda"
        print("CUDA device set without id specification", flush=True)
        device_ids.append(torch.cuda.current_device())
        return device, device_ids
    print(
        "This code should work with multiple GPUs "
        "but we didn't test that, so we recommend to use only 1 GPU.",
        flush=True,
    )
    return "cuda:" + str(device_ids[0]), device_ids

@hydra.main(version_base=None, config_path="../utils", config_name="config.yaml")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)  # disable strict structure

    # Setup logger
    log = Log(cfg.log_dir, __name__)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    set_rand_state(cfg.seed)

    # Set device
    cfg.device, cfg.device_ids = set_device(cfg.gpu_ids, cfg.disable_gpu)
    log.info(f"Device used: {type(cfg.device)} {f'with id {cfg.device_ids}' if len(cfg.device_ids) > 0 else ''}")

    # Load checkpoint-specific logging directory
    if cfg.model_checkpoint is not None:
        cfg.log_dir = str(Path(cfg.model_checkpoint).parent.parent)
    
    # Handle default jsd loss fallback
    if (
        cfg.jsd_loss == 0.0
        and cfg.tanh_loss == 0.0
        and cfg.unif_loss == 0.0
        and cfg.variance_loss == 0.0
    ):
        log.info("No loss function specified. Using JSD loss by default")
        cfg.jsd_loss = 5.0

    # Create the dataloaders
    train_loader = DoubleAugmentDataLoader(cfg)
    test_loader = DataLoader("test", cfg)
    train_loader_visualization = DataLoader("train", cfg)
    panoptic_parts_loader = PanopticPartsDataLoader("train", cfg)

    cfg.num_classes = len(train_loader.dataset.classes)
    # Model
    net = ReProSeg(args=cfg, log=log).to(device=torch.device(cfg.device))

    if not cfg.skip_training:
        from train.trainer import train_model

        try:
            train_model(net, train_loader, test_loader, log, cfg)
        except Exception as e:
            log.exception(e)

    if cfg.visualize_prototypes:
        from visualize.visualizer import ModelVisualizer

        visualizer = ModelVisualizer(net, cfg, log, k=cfg.visualize_top_k)
        visualizer.visualize_prototypes(train_loader_visualization)

    if cfg.consistency_score:
        from visualize.interpretability import ModelInterpretability
        interpretability = ModelInterpretability(
            net, cfg, log, consistency_threshold=cfg.consistency_threshold)
        interpretability.compute_prototype_consistency_score(panoptic_parts_loader)

    log.close()


if __name__ == "__main__":
    main()