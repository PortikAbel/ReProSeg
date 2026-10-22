"""Shared part observations for interpretability metrics, without scoring.

Keep class assignment, spatial activations, and annotation geometry identical
between consistency and stability. Thresholding and centroids are implemented
by evaluate.utils; checkpoint compatibility remains in evaluate.checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from evaluate.checkpoint import SupportedModel
from evaluate.utils import _part_centroids, quantile_activation_mask
from model.model import ReProSeg
from model.proto_segmentation import PPNet


@dataclass(frozen=True)
class PartPresence:
    """One prototype/part observation in one image."""

    image_index: int
    class_id: int
    prototype_id: int
    part_id: int
    present: bool


class PartObservationExtractor:
    """Extract prototype/concept part presence independently of metric aggregation."""

    def __init__(
        self,
        model: SupportedModel,
        *,
        activation_quantile: float = 0.8,
        device: torch.device | str | None = None,
        used_prototypes_only: bool = False,
        semantic_class_offset: int | None = None,
    ):
        if not isinstance(model, (PPNet, ReProSeg)):
            raise TypeError(f"Expected a PPNet or ReProSeg model, received {type(model).__name__}.")
        if not 0.0 <= activation_quantile <= 1.0:
            raise ValueError(f"Activation quantile must be in [0, 1], received {activation_quantile}.")

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.activation_quantile = activation_quantile
        self.used_prototypes_only = used_prototypes_only

        if isinstance(self.model, PPNet):
            self.num_components = self.model.num_prototypes
            self.semantic_class_offset = 1
            identity = self.model.prototype_class_identity.detach().to(device="cpu", dtype=torch.bool)
            if identity.ndim != 2 or identity.shape[0] != self.num_components:
                raise ValueError(
                    "PPNet.prototype_class_identity must have shape "
                    f"({self.num_components}, C), received {tuple(identity.shape)}."
                )

            if used_prototypes_only:
                weights = self.model.last_layer.weight.detach().to(device="cpu")
                used_component_mask = (weights > 0).any(dim=0)
            else:
                used_component_mask = torch.ones(self.num_components, dtype=torch.bool)
        else:
            self.num_components = self.model.num_concepts
            self.semantic_class_offset = 0
            classifier_weights = (
                self.model.layers.classification_layer.weight.detach().to(device="cpu").squeeze(-1).squeeze(-1)
            )
            if classifier_weights.ndim != 2 or classifier_weights.shape[1] != self.num_components:
                raise ValueError(
                    "ReProSeg classification weights must have shape "
                    f"(C, {self.num_components}, 1, 1), received "
                    f"{tuple(self.model.layers.classification_layer.weight.shape)}."
                )

            identity = classifier_weights.T >= self.model.layers.classification_layer.MIN_CLASSIFICATION_WEIGHT
            used_concepts = self.model.layers.classification_layer.used_concepts.detach().to(device="cpu")
            used_component_mask = torch.zeros(self.num_components, dtype=torch.bool)
            used_component_mask[used_concepts] = True

        self.component_class_identity = identity
        self.used_component_mask = used_component_mask
        if semantic_class_offset is not None:
            if semantic_class_offset not in (0, 1):
                raise ValueError("Semantic class offset must be 0 (with background) or 1 (foreground only).")
            self.semantic_class_offset = semantic_class_offset

    def _prototype_activations(self, images: Tensor) -> Tensor:
        if isinstance(self.model, PPNet):
            push_result = self.model.push_forward(images)
            if isinstance(push_result, list):
                raise RuntimeError(
                    "Expected one PPNet distance tensor in evaluation mode, but received multi-scale outputs."
                )

            _features, distances = push_result
            activations = self.model.distance_2_similarity(distances)
        else:
            activations = self._reproseg_concept_activations(images)

        if activations.ndim != 4:
            raise ValueError(
                "Expected spatial prototype/concept activations with shape "
                f"(B, K, H, W), received {tuple(activations.shape)}."
            )
        if activations.shape[1] != self.num_components:
            raise ValueError(f"Expected {self.num_components} activation maps, received {activations.shape[1]}.")
        if not torch.isfinite(activations).all():
            raise ValueError("Prototype/concept activations contain non-finite values.")
        if torch.any(activations < 0):
            raise ValueError("Part evaluation requires non-negative prototype/concept activations.")
        return activations

    def _reproseg_concept_activations(self, images: Tensor) -> Tensor:
        """Return ReProSeg's multi-scale concept maps before image-size interpolation.

        This is equivalent to ``ReProSeg.interpolate_concept_activations`` up
        to its final nearest-neighbor resize.  Keeping the maps at feature-map
        resolution lets the evaluator resize only concepts assigned to a
        part-annotated class, instead of materializing all concepts at
        1024x2048.
        """

        aspp_features, _pooled, _out = self.model(images, inference=False)
        if aspp_features.ndim != 5:
            raise ValueError(
                "Expected ReProSeg ASPP concept activations with shape "
                f"(B, K, S, H, W), received {tuple(aspp_features.shape)}."
            )
        if aspp_features.shape[1] != self.num_components:
            raise ValueError(f"Expected {self.num_components} ReProSeg concepts, received {aspp_features.shape[1]}.")

        scale_activations = aspp_features.permute(2, 0, 1, 3, 4)
        max_scale = torch.argmax(scale_activations, dim=0)
        concept_activations = torch.zeros_like(scale_activations[0])

        for scale in range(scale_activations.shape[0]):
            selected_scale = scale_activations[scale].masked_fill(max_scale != scale, 0)
            padding = scale + 1
            selected_scale = F.max_pool2d(
                selected_scale,
                kernel_size=2 * padding + 1,
                padding=padding,
                stride=1,
            )
            concept_activations = torch.maximum(concept_activations, selected_scale)

        return concept_activations

    def _image_part_presence(
        self,
        *,
        activation_maps: Sequence[Tensor],
        semantic_mask: Tensor,
        part_mask: Tensor,
        image_index: int,
    ) -> tuple[list[PartPresence], ...]:
        """Observe one or more versions of an image using the same part centroids.

        Consistency supplies one activation tensor; stability supplies clean
        and noisy tensors. Each version is thresholded independently, while
        class assignments and annotation geometry are computed only once.
        """

        observations: tuple[list[PartPresence], ...] = tuple([] for _ in activation_maps)
        semantic_mask = semantic_mask.detach().to(device="cpu", dtype=torch.long)
        part_mask = part_mask.detach().to(device="cpu", dtype=torch.long)
        output_size = tuple(part_mask.shape)

        for class_value in torch.unique(semantic_mask):
            class_id = int(class_value.item())
            if class_id <= 0:
                continue

            prototype_class_index = class_id - self.semantic_class_offset
            if prototype_class_index < 0 or prototype_class_index >= self.component_class_identity.shape[1]:
                continue

            class_mask = semantic_mask == class_id
            part_centroids = _part_centroids(part_mask, class_mask)
            if not part_centroids:
                continue

            prototype_ids = self.component_class_identity[:, prototype_class_index].nonzero(as_tuple=True)[0].tolist()
            prototype_ids = [prototype_id for prototype_id in prototype_ids if self.used_component_mask[prototype_id]]
            if not prototype_ids:
                continue

            for activations, version_observations in zip(activation_maps, observations, strict=True):
                resized_activations = F.interpolate(
                    activations[prototype_ids].unsqueeze(0),
                    size=output_size,
                    # Match cv2.INTER_NEAREST used by ScaleProtoSeg's metrics.
                    mode="nearest",
                ).squeeze(0)
                active_regions = quantile_activation_mask(
                    resized_activations,
                    class_mask.to(self.device),
                    self.activation_quantile,
                ).to(device="cpu")

                for part_id, centroids in part_centroids.items():
                    rows = torch.tensor([centroid[0] for centroid in centroids], dtype=torch.long)
                    columns = torch.tensor([centroid[1] for centroid in centroids], dtype=torch.long)
                    part_present = active_regions[:, rows, columns].any(dim=1)

                    for local_index, prototype_id in enumerate(prototype_ids):
                        version_observations.append(
                            PartPresence(
                                image_index=image_index,
                                class_id=class_id,
                                prototype_id=prototype_id,
                                part_id=part_id,
                                present=bool(part_present[local_index].item()),
                            )
                        )

        return observations
