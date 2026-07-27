from collections import defaultdict
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ReProSegConfig
from data import DataLoader
from proto_segmentation.model import PPNet
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

    def __init__(self, net: PPNet, cfg: ReProSegConfig, log: Log):
        self.net = net
        self.device = cfg.env.device
        self.consistency_score = cfg.evaluation.consistency_score.threshold
        self.log = log
        self._part_activations = [defaultdict(list) for _ in range(self.net.num_prototypes)]

    @torch.no_grad()
    def compute_prototype_consistency_score(self, panoptic_parts_loader: DataLoader):
        self.log.info("Computing prototype consistency score...")
        self._part_activations = [defaultdict(list) for _ in range(self.net.num_prototypes)]
        self._collect_prototype_activations_by_object_parts(panoptic_parts_loader)
        is_consistent = self._compute_if_prototype_consistent()

        used_prototypes = self._get_used_prototypes()
        num_used_prototypes = len(used_prototypes)
        num_consistent_prototypes = sum(is_consistent[prototype] for prototype in used_prototypes)

        self.log.info(
            f"Found {num_consistent_prototypes} consistent prototypes "
            f"with per object part activation > {self.consistency_score} "
            f"out of {num_used_prototypes} used prototypes."
        )
        if num_used_prototypes == 0:
            self.log.warning("No used prototypes found; returning a consistency score of 0.")
            return 0.0
        return num_consistent_prototypes / num_used_prototypes

    def _collect_prototype_activations_by_object_parts(self, panoptic_parts_loader: DataLoader):
        self.log.info("Collecting average object part activations of prototypes from images...")
        self.net.eval()
        used_prototypes = self._get_used_prototypes()
        img_iter = tqdm(
            enumerate(panoptic_parts_loader),
            total=len(panoptic_parts_loader),
            mininterval=100.0,
            desc="Collecting average object part activations of prototypes from images",
            ncols=0,
            file=self.log.tqdm_file,
        )

        for _, (xs, ys, pps) in img_iter:
            target_classes = self._target_classes_with_panoptic_labels.to(ys.device)
            if not torch.isin(ys, target_classes).any():
                print("Image skipped because none of the semantic classes with object part labels available found.")
                continue

            xs, ys, pps = xs.to(self.device), ys.to(self.device), pps.to(self.device)
            if pps.dim() == 4 and pps.shape[1] == 1:
                pps = pps.squeeze(1)

            prototype_activations = self._interpolate_prototype_activations(
                xs,
                output_size=tuple(pps.shape[-2:]),
            )
            prototype_alphas = activations_to_alpha(prototype_activations)

            for p in used_prototypes:
                alpha = prototype_alphas[:, p]
                for label, avg_value in self._compute_part_activation_averages(alpha, pps):
                    self._part_activations[p][label].append(avg_value)
        self.log.info("Collected average object part activations of prototypes from images.")

    def _get_used_prototypes(self) -> list[int]:
        """
        Return prototypes that provide positive evidence for at least one class.

        PPNet uses its signed final-layer weights directly. A zero connection is
        therefore unused, while a negative connection represents inhibitory
        rather than positive explanatory evidence.
        """
        weights = self.net.last_layer.weight.detach()
        return (weights > 0).any(dim=0).nonzero(as_tuple=True)[0].cpu().tolist()

    def _interpolate_prototype_activations(
        self,
        xs: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Compute full spatial prototype similarities and resize them to the mask.

        Normal PPNet forward propagation globally pools prototype distances when
        patch classification is disabled. ``push_forward`` is used here because
        it retains the spatial distance map for every prototype.
        """
        push_result = self.net.push_forward(xs)
        if isinstance(push_result, list):
            raise RuntimeError(
                "Expected one PPNet distance tensor in evaluation mode, but received multi-scale outputs."
            )

        _conv_features, prototype_distances = push_result
        prototype_activations = self.net.distance_2_similarity(prototype_distances)
        if prototype_activations.dim() != 4:
            raise ValueError(
                "Expected prototype activations with shape (B, P, H, W), "
                f"but received {tuple(prototype_activations.shape)}."
            )
        if torch.any(prototype_activations < 0):
            raise ValueError(
                "Consistency scoring requires non-negative prototype activations; "
                "use PPNet's log activation instead of linear negative distance."
            )

        return F.interpolate(
            prototype_activations,
            size=output_size,
            mode="nearest-exact",
        )

    def _compute_part_activation_averages(self, alpha: torch.Tensor, pps: torch.Tensor) -> Iterator[tuple[int, float]]:
        """
        Compute average activation scores for a single prototype across different object parts in an image.

        Args:
            alpha: Activation values tensor with shape (B, H, W)
            pps: Panoptic parts tensor with shape (B, H, W)

        Returns:
            Iterator[tuple[int, float]]: Iterator of tuples (part_label, average_activation) where:
                - part_label (int): Unique panoptic part label
                - average_activation (float): Mean activation score for that part
        """
        if alpha.shape != pps.shape:
            raise ValueError(
                "Prototype activations and panoptic-part masks must have matching shapes, "
                f"but received alpha={tuple(alpha.shape)} and pps={tuple(pps.shape)}."
            )

        alpha_flat = alpha.reshape(-1)
        part_labels_flat = pps.reshape(-1)

        mask = part_labels_flat != 0  # ignore unlabeled parts

        filtered_alpha = alpha_flat[mask]
        filtered_part_labels = part_labels_flat[mask]

        unique_labels, inverse_indices, count = torch.unique(
            filtered_part_labels, return_inverse=True, return_counts=True
        )

        sum_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
        sum_alpha = sum_alpha.index_add(0, inverse_indices, filtered_alpha)

        average_alpha = sum_alpha / count

        return zip(unique_labels.tolist(), average_alpha.tolist(), strict=False)

    def _compute_if_prototype_consistent(self) -> list[bool]:
        return list(
            [
                any(
                    (np.mean(avgs) if len(avgs) > 0 else 0) > self.consistency_score
                    for avgs in avg_part_activations.values()
                )
                for avg_part_activations in self._part_activations
            ]
        )
