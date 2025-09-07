import argparse

import numpy as np
import torch
from tqdm import tqdm


from data.dataloader import PanopticPartsDataLoader
from model.model import ReProSeg
from utils.log import Log
from .utils import activations_to_alpha


class ModelInterpretability:
    target_classes_with_panoptic_labels = torch.tensor([11, 12, 13, 14, 15])
    """
    check if these semantic labels are present (we only have object part notations for these)
    11/24: person, 12/25: rider, 13/26: car, 14/27: truck, 15/28: bus
    """
    prototype_object_part_activations: torch.Tensor
    """
    Average prototype activations per object part label.
    """

    def __init__(self, net: ReProSeg, args: argparse.Namespace, log: Log, consistency_threshold: float = 0.7):
        self.net = net
        self.args = args
        self.log = log
        self.consistency_threshold = consistency_threshold

    def compute_average_activation_per_prototype(self):
        for d in self.prototype_object_part_activations:
            for label, scores in d.items():
                if len(scores) > 0:
                    d[label] = np.mean(scores)
                else:
                    d[label] = 0.0

        self.log.info("Average object part activation scores per prototypes computed")

    @torch.no_grad()
    def _compute_prototype_activations_by_object_parts(self, panoptic_parts_loader: PanopticPartsDataLoader):
        # for each prototype, we will maintain a dictionary of average activation scores
        #   for each panoptic part id (dict. key: part id, value: list of per image average activation scores)

        self.log.info("Computing per image average object part activations of prototypes...")
        self.net.eval()
        img_iter = tqdm(
            enumerate(panoptic_parts_loader),
            total=len(panoptic_parts_loader),
            mininterval=100.0,
            desc="Computing per image average object part activations of prototypes",
            ncols=0,
            file=self.log.tqdm_file,
        )

        for _, (xs, ys, pps) in img_iter:
            if not torch.isin(ys, self.target_classes_with_panoptic_labels).any():
                print("Image skipped because none of the semantic classes with object part labels available found.")
                continue

            xs, ys, pps = xs.to(self.args.device), ys.to(self.args.device), pps.to(self.args.device)
            prototype_activations = self.net.interpolate_prototype_activations(xs)
            for p in self.net.layers.classification_layer.used_prototypes:
                alpha = activations_to_alpha(prototype_activations[p])
                self.prototype_object_part_activations[p] += self._compute_prototype_activations_for_image_parts(alpha, pps)
        self.log.info("Per image prototype object part activations updated")

    def _compute_prototype_activations_for_image_parts(self, alpha: torch.Tensor, pps: torch.Tensor) -> torch.Tensor:
        """
        Compute average activation scores for a single prototype across different object parts in an image.
        
        Args:
            prototype_idx: Index of the prototype
            alpha: Activation values tensor
            pps: Panoptic parts tensor
            
        Returns:
            Dictionary mapping object part labels to their average activation scores
        """
        alpha_flat = alpha.view(-1)
        part_labels_flat = pps.view(-1)

        mask = part_labels_flat != 0  # ignore unlabeled parts

        filtered_alpha = alpha_flat[mask]
        filtered_part_labels = part_labels_flat[mask]

        unique_labels, inverse_indices = torch.unique(filtered_part_labels, return_inverse=True)

        sum_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
        sum_alpha = sum_alpha.index_add(0, inverse_indices, filtered_alpha)

        count_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
        count_alpha = count_alpha.index_add(0, inverse_indices, torch.ones_like(filtered_alpha))

        average_alpha = sum_alpha / count_alpha

        result = torch.zeros(self.prototype_object_part_activations.shape[1])
        result[unique_labels] = average_alpha
        return result

    def compute_prototype_consistency_score(self, panoptic_parts_loader: PanopticPartsDataLoader):
        self.log.info("Computing prototype consistency score...")
        self.prototype_object_part_activations = torch.zeros((
            self.net.layers.classification_layer.num_prototypes,
            len(panoptic_parts_loader.dataset.classes)
        ))
        self._compute_prototype_activations_by_object_parts(panoptic_parts_loader)
        self.compute_average_activation_per_prototype()

        count = sum(
            any(value > self.consistency_threshold for value in d.values())
            for d in self.prototype_object_part_activations
        )

        self.get_number_of_active_prototypes()
        self.log.info(f"Number of active prototypes: {self.number_of_active_prototypes}")
        normalized_consistency_score = count / self.number_of_active_prototypes
        self.log.info(
            f"Prototype consistency score: {count} prototypes "
            f"with per object part activation > {self.consistency_threshold}"
        )

        return normalized_consistency_score
