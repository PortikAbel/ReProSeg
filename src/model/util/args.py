import argparse
from datetime import datetime
import warnings
from pathlib import Path
import pickle

import random
import torch
import numpy as np


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
        "Specifies the directory where the log files "
        "and other outputs should be saved",
    )
    log_group.add_argument(
        "--log_dir",
        type=Path,
        default=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        help="The directory in which train progress should be logged",
    )
    log_group.add_argument(
        "--save_all_models",
        action="store_true",
        help="Flag to save the model in each epoch",
    )

    dataset_group = parser.add_argument_group(
        "Dataset", "Specifies the dataset to use and its hyperparameters"
    )
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

    image_size_group = dataset_group.add_argument_group(
        "Image size",
        "Specifies the size of the images. At least one of them is required",
    )
    image_size_group.add_argument(
        "--image_width",
        type=np.uint16,
        help="The width of the images in the dataset",
    )
    image_size_group.add_argument(
        "--image_height",
        type=np.uint16,
        help="The height of the images in the dataset",
    )

    net_group = parser.add_mutually_exclusive_group()
    net_group.add_argument(
        "--net",
        type=str,
        default="convnext_tiny_26",
        help="Base network used as backbone of ReProSeg. "
        "Default is convnext_tiny_26 with adapted strides "
        "to output 26x26 latent representations. "
        "Other option is convnext_tiny_13 that outputs 13x13 " 
        "(smaller and faster to train, less fine-grained). "
        "Pretrained network on iNaturalist is only available for resnet50_inat. "
        "Options are: "
        "resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, "
        "convnext_tiny_13 and convnext_tiny_26.",
    )
    net_group.add_argument(
        "--pretrained_net_state_dict_dir",
        type=Path,
        help="The directory containing a state dict with a pretrained ReProSeg."
    )

    net_parameter_group = parser.add_argument_group(
        "Network parameters", "Specifies the used network's hyperparameters"
    )
    net_parameter_group.add_argument(
        "--batch_size",
        type=np.uint16,
        default=64,
        help="Batch size when training the model using minibatch gradient descent. "
        "Batch size is multiplied with number of available GPUs",
    )
    net_parameter_group.add_argument(
        "--train_backbone_during_pretrain",
        action="store_true",
        help="To train the whole backbone during pretrain "
        "(e.g. if dataset is very different from ImageNet)",
    )
    net_parameter_group.add_argument(
        "--epochs",
        type=np.uint16,
        default=60,
        help="The number of epochs ReProSeg should be trained (second training stage)",
    )
    net_parameter_group.add_argument(
        "--epochs_pretrain",
        type=np.uint16,
        default=10,
        help="Number of epochs to pre-train the prototypes (first training stage). "
        "Recommended to train at least until the align loss < 1",
    )
    net_parameter_group.add_argument(
        "--freeze_epochs",
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
        help="Flag that indicates whether to include a trainable bias in the "
        "linear classification layer.",
    )

    optimizer_group = parser.add_argument_group(
        "Optimizer", "Specifies the optimizer to use and its hyperparameters"
    )
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
        help="The optimizer learning rate for training the weights "
        "from prototypes to classes",
    )
    optimizer_group.add_argument(
        "--lr_block",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for training the last convolutional "
        "layers of the backbone",
    )
    optimizer_group.add_argument(
        "--lr_net",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for the backbone. "
        "Usually similar as lr_block.",
    )
    optimizer_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used in the optimizer",
    )

    loss_group = parser.add_argument_group(
        "Loss", "Specifies the loss function to use and its hyperparameters"
    )
    loss_group.add_argument(
        "--weighted_loss",
        action="store_true",
        help="Flag that weights the loss based on the class balance of the dataset. "
        "Recommended to use when data is imbalanced.",
    )
    loss_group.add_argument(
        "--tanh_loss",
        type=float,
        default=0.0,
        help="tanh loss regulates that every prototype should be at "
        "least once present in a mini-batch.",
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
        help="Regularizer term that enforces variance of features from "
        "https://arxiv.org/abs/2105.04906",
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
        "This code should work with multiple GPUs "
        "but we didn't test that, so we recommend to use only 1 GPU.",
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

        if self._args.image_height is None and self._args.image_width is None:
            self._parser.error("Both image_height and image_width cannot be None")

        self._args.image_height = self._args.image_height or self._args.image_width
        self._args.image_width = self._args.image_width or self._args.image_height

        self._args.image_shape = np.array(
            (self._args.image_height, self._args.image_width)
        )

        if (
            not self._args.tanh_loss
            and not self._args.unif_loss
            and not self._args.variance_loss
        ):
            warnings.warn(f"No loss function specified. Using tanh loss by default")
            self._args.tanh_loss = 5.0

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
        # Save the args in a text file
        with (directory_path / "args.txt").open(mode="w") as f:
            for arg in vars(self._args):
                val = getattr(self._args, arg)
                if isinstance(val, str):
                    # Add quotation marks to indicate that
                    # the argument is of string type
                    val = f'"{val}"'
                f.write(f"{arg}: {val}\n")
        # Pickle the args for possible reuse
        with (directory_path / "args.pickle").open(mode="wb") as f:
            pickle.dump(self._args, f)
