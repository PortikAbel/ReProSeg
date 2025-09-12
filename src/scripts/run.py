from data.dataloader import DataLoader, DoubleAugmentDataLoader, PanopticPartsDataLoader
from model.model import ReProSeg
from utils.log import Log
from utils.args import ConfigWrapper

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import torch
import nni


def set_rand_state(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_device(log: Log, gpu_ids: str, disable_gpu: bool = False) -> tuple[torch.device, list]:
    """
    Set the device to use for training.

    :param gpu_ids: GPU ids separated with comma
    :param disable_gpu: Whether to disable GPU. Defaults to ``False``.
    :return: The device to use for training
    """

    device_ids = [int(gpu_id) for gpu_id in (gpu_ids.split(",") if gpu_ids else [])]

    if disable_gpu or not torch.cuda.is_available():
        return torch.device("cpu"), []
    if len(device_ids) == 1:
        return torch.device(f"cuda:{gpu_ids}"), device_ids
    if len(device_ids) == 0:
        device = torch.device("cuda")
        log.info("CUDA device set without id specification")
        device_ids.append(torch.cuda.current_device())
        return device, device_ids
    log.info(
        "This code should work with multiple GPUs but we didn't test that, so we recommend to use only 1 GPU.",
        flush=True,
    )
    return torch.device("cuda:" + str(device_ids[0])), device_ids


def set_default_jsd_loss(args: ConfigWrapper, log: Log) -> ConfigWrapper:
    if args.jsd_loss == 0.0 and args.tanh_loss == 0.0 and args.unif_loss == 0.0 and args.variance_loss == 0.0:
        log.info("No loss function specified. Using JSD loss by default")
        args.jsd_loss = 5.0
    return args


# updates hydra omega conf params based on the nni parameters:
def update_hydra_config(cfg: DictConfig, nni_params: dict) -> DictConfig:
    for key, value in nni_params.items():
        if key in cfg:
            cfg[key] = value
    return cfg


@hydra.main(version_base=None, config_path="../utils", config_name="config.yaml")
def main(cfg: DictConfig):
    # Setup logger
    log = Log(cfg.log_dir, __name__)

    nni_params = nni.get_next_parameter()
    if nni_params:
        cfg = update_hydra_config(cfg, nni_params)

    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # wrap omegaconf object so custom type objects can be added later
    args = ConfigWrapper(cfg)

    # Set random seed
    set_rand_state(args.seed)

    # Set device
    args.device, args.device_ids = set_device(log, args.gpu_ids, args.disable_gpu)
    log.info(f"Device used: {args.device} {f'with id {args.device_ids}' if len(args.device_ids) > 0 else ''}")

    # Load checkpoint-specific logging directory
    if args.model_checkpoint is not None:
        args.log_dir = str(Path(args.model_checkpoint).parent.parent)

    # Handle default jsd loss fallback
    args = set_default_jsd_loss(args, log)

    # Create the dataloaders
    train_loader = DoubleAugmentDataLoader(args)
    test_loader = DataLoader("test", args)
    train_loader_visualization = DataLoader("train", args)
    panoptic_parts_loader = PanopticPartsDataLoader("train", args)

    args.num_classes = len(train_loader.dataset.classes)
    # Model
    net = ReProSeg(args=args, log=log).to(device=args.device)

    if not args.skip_training:
        from train.trainer import train_model

        try:
            train_model(net, train_loader, test_loader, log, args)
        except Exception as e:
            log.exception(e)

    if args.visualize_prototypes:
        from visualize.visualizer import ModelVisualizer

        visualizer = ModelVisualizer(net, args, log, k=args.visualize_top_k)
        visualizer.visualize_prototypes(train_loader_visualization)

    if args.consistency_score:
        from visualize.interpretability import ModelInterpretability

        interpretability = ModelInterpretability(net, args, log, consistency_threshold=args.consistency_threshold)
        interpretability.compute_prototype_consistency_score(panoptic_parts_loader)

    log.close()


if __name__ == "__main__":
    main()
