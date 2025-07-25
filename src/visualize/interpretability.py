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
from torchvision.transforms import InterpolationMode
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

    def __init__(self, net: ReProSeg, args: argparse.Namespace, log: Log, consistency_threshold: float = 0.7):
        self.net = net
        self.args = args
        self.log = log
        self.image_shape = DATASETS[args.dataset]["img_shape"]
        self.consisistency_threshold = consistency_threshold


    # TODO: move this to dataloader?
    # returns the panoptic labels for a given image
    def get_panoptic_mask_for_img(self, img_to_open):

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
            return panoptic_labels_array_filtered
        else:
            return None

    def get_number_of_active_prototypes(self):
        self.number_of_active_prototypes = 0
        for p in range(self.net.num_prototypes):
            # skip prototype if it has low classification weight
            if torch.max(self.net.layers.classification_layer.weight[:, p]) < self.MIN_CLASSIFICATION_WEIGHT:
                # self.log.info(f"Prototype {p} skipped due to low classification weight")
                continue
            self.number_of_active_prototypes += 1


    def compute_average_activation_per_prototype(self):
        for d in self.prototype_object_part_activations:
            for label, scores in d.items():
                if len(scores) > 0:
                    d[label] = np.mean(scores)
                else:
                    d[label] = 0.0

        self.log.info(f"Average object part activation scores per prototypes computed")

    @torch.no_grad()
    def get_per_image_object_part_prototype_activations(self, train_loader_visualization):
        # self.get_panoptic_parts_ids(train_loader_visualization)
        # for each prototype, we will maintain a dictionary of average activation scores
        #   for each panoptic part id (dict. key: part id, value: list of per image average activation scores)
        self.prototype_object_part_activations = [defaultdict(list) for _ in range(self.net.num_prototypes)] 

        resize_image = transforms.Resize(size=tuple(self.image_shape))
        resize_panoptic_labels = transforms.Resize(size=tuple(self.image_shape), interpolation=InterpolationMode.NEAREST_EXACT)
        pil_to_tensor = transforms.ToTensor()

        self.log.info("Computing per image average object part activations of prototypes...")
        self.net.eval()
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Computing per image average object part activations of prototypes",
            ncols=0,
            file=self.log.tqdm_file,
        )

        for i, (xs, ys) in img_iter:
            img_to_open = train_loader_visualization.dataset.images[i]
            panoptic_labels = self.get_panoptic_mask_for_img(img_to_open)
            panoptic_labels = torch.tensor(panoptic_labels, dtype=torch.int32).unsqueeze(0) if panoptic_labels is not None else None
            panoptic_labels = resize_panoptic_labels(panoptic_labels) if panoptic_labels is not None else None
        
            image = pil_to_tensor(Image.open(img_to_open).convert("RGB"))
            image = resize_image(image)
            xs, ys = xs.to(self.args.device), ys.to(self.args.device)
            prototype_activations = self.net.interpolate_prototype_activations(xs).to(image.device)
            for p in range(self.net.num_prototypes):
                # skip prototype if it has low classification weight
                # if torch.max(self.net.layers.classification_layer.weight[:, p]) < self.MIN_CLASSIFICATION_WEIGHT:
                #     continue

                # getting activation of prototype p for image i
                alpha = activations_to_alpha(prototype_activations[p])

                alpha_flat = alpha.view(-1)
                labels_flat = panoptic_labels.view(-1)

                # Create a mask for labels > 100000 (values less than 100000 are belong to semantic class void)
                mask = labels_flat > 100000

                # Apply the mask
                filtered_alpha = alpha_flat[mask]
                filtered_labels = labels_flat[mask]

                # Get unique labels and inverse indices
                unique_labels, inverse_indices = torch.unique(filtered_labels, return_inverse=True)

                # Sum alpha values per object part label
                sum_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
                sum_alpha = sum_alpha.index_add(0, inverse_indices, filtered_alpha)

                # Count occurrences per object part label
                count_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
                count_alpha = count_alpha.index_add(0, inverse_indices, torch.ones_like(filtered_alpha))

                # Compute average
                average_alpha = sum_alpha / count_alpha

                # label → average dict
                label_to_avg = dict(zip(unique_labels.tolist(), average_alpha.tolist()))

                for label, avg_value in label_to_avg.items():
                    self.prototype_object_part_activations[p][label].append(avg_value)
        
        self.log.info(f"Per image prototype object part activations updated")


    def compute_prototype_consistency_score(self, train_loader_visualization):
        self.log.info("Computing prototype consistency score...")
        self.get_per_image_object_part_prototype_activations(train_loader_visualization)
        self.compute_average_activation_per_prototype()

        count = sum(
            any(value > self.consisistency_threshold for value in d.values())
            for d in self.prototype_object_part_activations
        )

        self.get_number_of_active_prototypes()
        self.log.info(f"Number of active prototypes: {self.number_of_active_prototypes}")
        normalized_consistency_score = count / self.number_of_active_prototypes
        self.log.info(f"Prototype consistency score: {count} prototypes with per object part activation > {self.consisistency_threshold}")


