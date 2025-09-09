import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


from data.dataloader import PanopticPartsDataLoader
from model.model import ReProSeg
from utils.log import Log
from .utils import activations_to_alpha


class ModelInterpretability:
    _target_classes_with_panoptic_labels = torch.tensor([11, 12, 13, 14, 15])
    """
    check if these semantic labels are present (we only have object part notations for these)
    11/24: person, 12/25: rider, 13/26: car, 14/27: truck, 15/28: bus
    """
    _part_activations: list[dict[int, list[float]]]
    """
    For each prototype,
    a dictionary mapping panoptic part labels
    to lists of average activation scores in relevant images.
    """

    def __init__(self, net: ReProSeg, args: argparse.Namespace, log: Log, consistency_threshold: float = 0.7):
        self.net = net
        self.args = args
        self.log = log
        self.consistency_threshold = consistency_threshold
        self._part_activations = [defaultdict(list) for _ in range(self.net.num_prototypes)]

    @torch.no_grad()
    def compute_prototype_consistency_score(self, panoptic_parts_loader: PanopticPartsDataLoader):
        self.log.info("Computing prototype consistency score...")
        self._collect_prototype_activations_by_object_parts(panoptic_parts_loader)
        is_consistent = self._compute_if_prototype_consistent()

        num_consistent_prototypes = sum(is_consistent)

        self.log.info(
            f"Found {num_consistent_prototypes} consistent prototypes "
            f"with per object part activation > {self.consistency_threshold} "
            f"out of {len(is_consistent)}."
        )
        return num_consistent_prototypes / len(is_consistent)

    def _collect_prototype_activations_by_object_parts(self, panoptic_parts_loader: PanopticPartsDataLoader):
        self.log.info("Collecting average object part activations of prototypes from images...")
        self.net.eval()
        img_iter = tqdm(
            enumerate(panoptic_parts_loader),
            total=len(panoptic_parts_loader),
            mininterval=100.0,
            desc="Collecting average object part activations of prototypes from images",
            ncols=0,
            file=self.log.tqdm_file,
        )

        for _, (xs, ys, pps) in img_iter:
            if not torch.isin(ys, self._target_classes_with_panoptic_labels).any():
                print("Image skipped because none of the semantic classes with object part labels available found.")
                continue

            xs, ys, pps = xs.to(self.args.device), ys.to(self.args.device), pps.to(self.args.device)
            prototype_activations = self.net.interpolate_prototype_activations(xs)
            for p in self.net.layers.classification_layer.used_prototypes:
                alpha = activations_to_alpha(prototype_activations[:, p])
                for label, avg_value in self._compute_part_activation_averages(alpha, pps):
                    self._part_activations[p][label].append(avg_value)
        self.log.info("Collected average object part activations of prototypes from images.")

    def _compute_part_activation_averages(self, alpha: torch.Tensor, pps: torch.Tensor) -> zip[tuple[int, float]]:
        """
        Compute average activation scores for a single prototype across different object parts in an image.
        
        Args:
            alpha: Activation values tensor of shape (H, W) containing prototype activations
            pps: Panoptic parts tensor of shape (H, W) containing part labels for each pixel
            
        Returns:
            zip: Iterator of tuples (part_label, average_activation) where:
                - part_label (int): Unique panoptic part label 
                - average_activation (float): Mean activation score for that part
        """
        alpha_flat = alpha.view(-1)
        part_labels_flat = pps.view(-1)

        mask = part_labels_flat != 0  # ignore unlabeled parts

        filtered_alpha = alpha_flat[mask]
        filtered_part_labels = part_labels_flat[mask]

        unique_labels, inverse_indices, count = torch.unique(filtered_part_labels, return_inverse=True, return_counts=True)

        sum_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
        sum_alpha = sum_alpha.index_add(0, inverse_indices, filtered_alpha)

        average_alpha = sum_alpha / count

        return zip(unique_labels.tolist(), average_alpha.tolist())
    
    def _compute_if_prototype_consistent(self) -> list[bool]:
        return list([
            any(
                np.mean(avgs) > self.consistency_threshold
                for avgs in avg_part_activations.values()
            ) for avg_part_activations in self._part_activations
        ])
