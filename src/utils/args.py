import argparse
from datetime import datetime
import warnings
from pathlib import Path
import pickle
from omegaconf import DictConfig

import random
import torch
import numpy as np

from utils.environment import get_env


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Train ReProSeg",
        description="Necessary parameters to train a ReProSeg",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed. Note that there will still be differences "
        "between runs due to nondeterminism. "
        "See https://pytorch.org/docs/stable/notes/randomness.html",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers in dataloaders.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Flag that skips training and only visualizes the prototypes and predictions.",
    )

    gpu_group = parser.add_argument_group("GPU", "Specifies the GPU settings")
    gpu_group.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="ID of gpu. Can be separated with comma",
    )
    gpu_group.add_argument(
        "--disable_gpu",
        action="store_true",
        help="Flag that disables GPU usage if set",
    )

    log_group = parser.add_argument_group(
        "Logging",
        "Specifies the directory where the log files and other outputs should be saved",
    )
    log_group.add_argument(
        "--log_dir",
        type=Path,
        default=(Path(get_env("LOG_ROOT")) / "reproseg" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")).resolve(),
        help="The directory in which train progress should be logged",
    )
    log_group.add_argument(
        "--save_all_models",
        action="store_true",
        help="Flag to save the model in each epoch",
    )

    dataset_group = parser.add_argument_group("Dataset", "Specifies the dataset to use and its hyperparameters")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="CityScapes",
        help="Data set on ReProSeg should be trained",
    )
    dataset_group.add_argument(
        "--validation_size",
        type=float,
        default=0.0,
        help="Split between training and validation set. Can be zero when "
        "there is a separate test or validation directory. "
        "Should be between 0 and 1. Used for partimagenet (e.g. 0.2)",
    )
    dataset_group.add_argument(
        "--disable_normalize",
        action="store_true",
        help="Flag that disables normalization of the images",
    )

    net_group = parser.add_mutually_exclusive_group()
    net_group.add_argument(
        "--net",
        type=str,
        default="deeplab_v3",
        help="Base network used as backbone of ReProSeg. Default is deeplab_v3",
    )
    net_group.add_argument("--model_checkpoint", type=Path, help="The state dict of (pre)trained ReProSeg.")

    net_parameter_group = parser.add_argument_group(
        "Network parameters", "Specifies the used network's hyperparameters"
    )
    net_parameter_group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size when training the model using minibatch gradient descent. "
        "Batch size is multiplied with number of available GPUs",
    )
    net_parameter_group.add_argument(
        "--train_backbone_during_pretrain",
        action="store_true",
        help="To train the whole backbone during pretrain (e.g. if dataset is very different from ImageNet)",
    )
    net_parameter_group.add_argument(
        "--epochs_pretrain",
        type=np.uint16,
        default=10,
        help="Number of epochs to pre-train the prototypes (first training stage). "
        "Recommended to train at least until the align loss < 1",
    )
    net_parameter_group.add_argument(
        "--epochs",
        type=np.uint16,
        default=60,
        help="The number of epochs ReProSeg should be trained (second training stage)",
    )
    net_parameter_group.add_argument(
        "--epochs_freeze",
        type=np.uint16,
        default=10,
        help="Number of epochs where pretrained features_net will be frozen while "
        "training classification layer (and last layer(s) of backbone)",
    )
    net_parameter_group.add_argument(
        "--epochs_finetune",
        type=np.uint16,
        default=3,
        help="During fine-tuning, only train classification layer and freeze rest. "
        "Usually done for a few epochs (at least 1, "
        "more depends on size of dataset)",
    )
    net_parameter_group.add_argument(
        "--epoch_start",
        type=np.uint16,
        default=1,
        help="The epoch to start training from. Useful when resuming training from a checkpoint.",
    )
    net_parameter_group.add_argument(
        "--num_features",
        type=int,
        default=0,
        help="Number of prototypes. When zero (default) the number of prototypes "
        "is the number of output channels of backbone. If this value is set, "
        "then a 1x1 conv layer will be added. Recommended to keep 0, but can "
        "be increased when number of classes > num output channels in backbone.",
    )
    net_parameter_group.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="When set, the backbone network is initialized with random weights "
        "instead of being pretrained on another dataset).",
    )
    net_parameter_group.add_argument(
        "--bias",
        action="store_true",
        help="Flag that indicates whether to include a trainable bias in the linear classification layer.",
    )

    optimizer_group = parser.add_argument_group("Optimizer", "Specifies the optimizer to use and its hyperparameters")
    optimizer_group.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer that should be used when training ReProSeg",
    )
    optimizer_group.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="The optimizer learning rate for training the weights from prototypes to classes",
    )
    optimizer_group.add_argument(
        "--lr_block",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for training the last convolutional layers of the backbone",
    )
    optimizer_group.add_argument(
        "--lr_net",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for the backbone. Usually similar as lr_block.",
    )
    optimizer_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used in the optimizer",
    )

    loss_group = parser.add_argument_group("Loss", "Specifies the loss function to use and its hyperparameters")
    loss_group.add_argument(
        "--align_loss",
        type=float,
        default=1.0,
        help="Align loss regulates that the prototypes of two similar images are aligned.",
    )
    loss_group.add_argument(
        "--jsd_loss",
        type=float,
        default=0.0,
        help="Jensen-Shannon divergence loss regulates that the distribution of the prototypes are pairwise disjoint.",
    )
    loss_group.add_argument(
        "--tanh_loss",
        type=float,
        default=0.0,
        help="tanh loss regulates that every prototype should be at least once present in a mini-batch.",
    )
    loss_group.add_argument(
        "--unif_loss",
        type=float,
        default=0.0,
        help="Our tanh-loss optimizes for uniformity and was sufficient for our "
        "experiments. However, if pretraining of the prototypes is not working "
        "well for your dataset, you may try to add another uniformity loss "
        "from https://www.tongzhouwang.info/hypersphere/",
    )
    loss_group.add_argument(
        "--variance_loss",
        type=float,
        default=0.0,
        help="Regularizer term that enforces variance of features from https://arxiv.org/abs/2105.04906",
    )
    loss_group.add_argument(
        "--classification_loss",
        type=float,
        default=1.0,
        help="Classification loss regulates that the classification layer predicts the labels correctly.",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        choices=["weighted_nll", "dice"],
        default="weighted_nll",
        help="Criterion to use for training.",
    )

    visualization_group = parser.add_argument_group(
        "Visualization", "Specifies which visualizations should be generated"
    )
    visualization_group.add_argument(
        "--visualize_prototypes",
        action="store_true",
        help="""Flag that indicates whether to visualize the top k
            activations of each prototype from test set.""",
    )
    visualization_group.add_argument(
        "--visualize_top_k",
        type=int,
        default=10,
        help="""Number of top activations of each prototype to visualize.
            Defaults to 10.""",
    )
    visualization_group.add_argument(
        "--visualize_predictions",
        action="store_true",
        help="""Flag that indicates whether to visualize the predictions
            on test data and the learned prototypes.""",
    )

    interpretability_group = parser.add_argument_group(
        "Interpretability", "Specifies which interpretability metrics should be generated"
    )
    interpretability_group.add_argument(
        "--consistency_score",
        action="store_true",
        help="""Flag that indicates whether to compute the consistency score""",
    )

    interpretability_group.add_argument(
        "--consistency_threshold",
        type=float,
        default=0.7,
        help="""Prototypes with at least one average per object part activation above 
        this threshold will be considered consistent.""",
    )

    return parser


def set_rand_state(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    :param seed: The random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_device(gpu_ids: str, disable_gpu: bool = False) -> tuple[torch.device, list]:
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
        print("CUDA device set without id specification", flush=True)
        device_ids.append(torch.cuda.current_device())
        return device, device_ids
    print(
        "This code should work with multiple GPUs but we didn't test that, so we recommend to use only 1 GPU.",
        flush=True,
    )
    return torch.device("cuda:" + str(device_ids[0])), device_ids


class ModelTrainerArgumentParser:
    def __init__(self):
        self._parser = define_parser()
        self._args = self._parser.parse_args()

    def get_args(self) -> argparse.Namespace:
        """
        Parse the arguments for the model.

        :param known_args_only: If ``True``, only known arguments are parsed.
            Defaults to ``True``.
        :return: specified arguments in the command line
        """

        set_rand_state(self._args.seed)
        self._args.device, self._args.device_ids = set_device(self._args.gpu_ids, self._args.disable_gpu)

        if (
            not self._args.jsd_loss
            and not self._args.tanh_loss
            and not self._args.unif_loss
            and not self._args.variance_loss
        ):
            warnings.warn("No loss function specified. Using JSD loss by default", stacklevel=2)
            self._args.jsd_loss = 5.0

        if self._args.model_checkpoint:
            warnings.warn("Logging directory is set to the parent of the model checkpoint directory", stacklevel=2)
            self._args.log_dir = self._args.model_checkpoint.parent.parent

        return self._args

    def save_args(self, directory_path: Path) -> None:
        """
        Save the arguments in the specified directory as
            - a text file called 'args.txt'
            - a pickle file called 'args.pickle'
        :param directory_path: The path to the directory where the
            arguments should be saved
        """
        # If the specified directory does not exist, create it
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
        file_name = "args"
        if (directory_path / f"{file_name}.txt").exists():
            idx = 1
            while (directory_path / f"{file_name}_{idx}.txt").exists():
                idx += 1
            file_name = f"{file_name}_{idx}"
        # Save the args in a text file
        with (directory_path / f"{file_name}.txt").open(mode="w") as f:
            for arg in vars(self._args):
                val = getattr(self._args, arg)
                if isinstance(val, str):
                    # Add quotation marks to indicate that
                    # the argument is of string type
                    val = f'"{val}"'
                f.write(f"{arg}: {val}\n")
        # Pickle the args for possible reuse
        with (directory_path / f"{file_name}.pickle").open(mode="wb") as f:
            pickle.dump(self._args, f)


class ConfigWrapper:
    def __init__(self, cfg: DictConfig):
        # Copy all config entries as attributes
        for key, value in cfg.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"