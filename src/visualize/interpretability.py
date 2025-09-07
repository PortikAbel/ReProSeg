import argparse

import numpy as np
import torch
from tqdm import tqdm


from data.dataloader import PanopticPartsDataLoader
from model.model import ReProSeg
from utils.log import Log
from .utils import activations_to_alpha


class ModelInterpretability:
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
    def get_per_image_object_part_prototype_activations(self, panoptic_parts_loader: PanopticPartsDataLoader):
        # self.get_panoptic_parts_ids(train_loader_visualization)
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

        # check if these semantic labels are present (we only have object part notations for these)
        # 11/24: person, 12/25: rider, 13/26: car, 14/27: truck, 15/28: bus
        target_classes_with_panoptic_labels = torch.tensor([11, 12, 13, 14, 15])

        for _, (xs, ys, pps) in img_iter:
            # Check if labels contain any of the target classes with panoptic labels
            found = torch.isin(ys, target_classes_with_panoptic_labels).any()

            if not found:
                print("Image skipped because none of the semantic classes with object part labels available found.")
                continue

            xs, ys, pps = xs.to(self.args.device), ys.to(self.args.device), pps.to(self.args.device)
            prototype_activations = self.net.interpolate_prototype_activations(xs)
            for p in self.net.layers.classification_layer.used_prototypes:
                # TODO: do we need this?
                # skip prototype if it has low classification weight
                # if torch.max(self.net.layers.classification_layer.weight[:, p]) < self.MIN_CLASSIFICATION_WEIGHT:
                #     continue

                alpha = activations_to_alpha(prototype_activations[p])

                alpha_flat = alpha.view(-1)
                part_labels_flat = pps.view(-1)

                mask = part_labels_flat != 0  # ignore unlabeled parts

                filtered_alpha = alpha_flat[mask]
                filtered_part_labels = part_labels_flat[mask]

                # Get unique labels and inverse indices
                unique_labels, inverse_indices = torch.unique(filtered_part_labels, return_inverse=True)

                # Sum alpha values per object part label
                sum_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
                sum_alpha = sum_alpha.index_add(0, inverse_indices, filtered_alpha)

                # Count occurrences per object part label
                count_alpha = torch.zeros_like(unique_labels, dtype=torch.float)
                count_alpha = count_alpha.index_add(0, inverse_indices, torch.ones_like(filtered_alpha))

                # Compute average
                average_alpha = sum_alpha / count_alpha

                self.prototype_object_part_activations[p] = {
                    label: avg_value
                    for label, avg_value in zip(unique_labels.tolist(), average_alpha.tolist(), strict=False)
                }
        self.log.info("Per image prototype object part activations updated")

    def compute_prototype_consistency_score(self, panoptic_parts_loader: PanopticPartsDataLoader):
        self.log.info("Computing prototype consistency score...")
        self.get_per_image_object_part_prototype_activations(panoptic_parts_loader)
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
