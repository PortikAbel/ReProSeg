import argparse
from collections import defaultdict
import heapq
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


from model.model import ReProSeg
from utils.log import Log
from data.config import DATASETS
from .utils import activations_to_alpha, prototype_text, draw_activation_minmax_text_on_image


class ModelInterpretability:
    topks: dict[int, list]
    i_to_p: dict
    tensors_per_prototype: dict[int, list]

    MIN_CLASSIFICATION_WEIGHT = 10
    MIN_ACTIVATION_SCORE = 0.1

    def __init__(self, net: ReProSeg, args: argparse.Namespace, log: Log, k: int = 10):
        self.net = net
        self.args = args
        self.log = log
        self.image_shape = DATASETS[args.dataset]["img_shape"]
        self.k = k

    @torch.no_grad()
    def get_panoptic_parts_ids(self, train_loader_visualization):
        self.log.info("Collecting the panoptic object part ids...")
        self.net.eval()
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Collecting panoptic object part ids",
            ncols=0,
            file=self.log.tqdm_file,
        )

        object_part_ids = set()
        for i, (xs, ys) in img_iter:
            img_to_open = train_loader_visualization.dataset.images[i]
            # self.log.info(f"Processing image {i} ({img_to_open})")
            # Replace 'leftImg8bit' in the directory path to find panoptic labels
            parts = list(Path(img_to_open).parts)
            parts[parts.index("leftImg8bit")] = "gtFinePanopticParts"

            # Create the new path object with updated directory
            path_to_panoptic_labels = Path(*parts)

            # Replace 'leftImg8bit' in the filename and change extension to .tif
            new_filename = path_to_panoptic_labels.name.replace("leftImg8bit", "gtFinePanopticParts").replace(".png", ".tif")

            # Combine updated directory with new filename
            panoptic_labels = path_to_panoptic_labels.with_name(new_filename)
            if panoptic_labels.exists():
                # self.log.info(f"panoptic labels found: {panoptic_labels}")

                # panoptic labels format:
                # 10^5 * semantic_id + 10^2 * instance_id + part_id
                panoptic_labels_image = Image.open(panoptic_labels)
                panoptic_labels_array = np.array(panoptic_labels_image)
                panoptic_labels_array_filtered = ((panoptic_labels_array//100000)*100000 + 
                                                  panoptic_labels_array%100).astype(np.int32)
                
                self.log.info(f"Processing panoptic labels {panoptic_labels} with shape {panoptic_labels_array_filtered.shape}")

                # Flatten and get unique values with their counts
                unique_values, counts = np.unique(panoptic_labels_array_filtered, return_counts=True)
                # Filter out values that belong to semantic class void (i.e., those less than or equal to 100000)
                object_part_ids.update({x for x in unique_values if x > 100000})

        self.panoptic_parts_ids = sorted(object_part_ids)     

    def calculate_consistency_score(self, train_loader_visualization):
        self.get_panoptic_parts_ids(train_loader_visualization)
        # self.log.info(self.panoptic_parts_ids)

        resize_image = transforms.Resize(size=tuple(self.image_shape))
        pil_to_tensor = transforms.ToTensor()

        self.log.info("Computing consistency scores of prototypes...")
        self.net.eval()
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Collecting panoptic object part ids",
            ncols=0,
            file=self.log.tqdm_file,
        )

        for i, (xs, ys) in img_iter:
            img_to_open = train_loader_visualization.dataset.images[i]
            self.log.info(f"Processing image {i} ({img_to_open}) with all the prototypes")
            image = pil_to_tensor(Image.open(img_to_open).convert("RGB"))
            image = resize_image(image)
            xs, ys = xs.to(self.args.device), ys.to(self.args.device)
            prototype_activations = self.net.interpolate_prototype_activations(xs).to(image.device)
            for p in range(self.net.num_prototypes):
                # getting activation of prototype p for image i
                alpha = activations_to_alpha(prototype_activations[p])
                self.log.info(f"Prototype {p} activation shape: {alpha.shape}")

