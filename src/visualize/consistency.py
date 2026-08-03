"""Part-consistency evaluation for ProtoSeg and ReProSeg models.

This module adapts the consistency metric used by ScaleProtoSeg to the
``proto_segmentation.model.PPNet`` and ``model.model.ReProSeg``
implementations in this repository.  ReProSeg concepts are associated with
classes through active concept-to-class classifier connections.  For every
class-specific prototype or class-concept assignment, the evaluator:

1. obtains the prototype/concept spatial activation map;
2. calculates a per-image quantile over pixels belonging to its semantic
   class;
3. binarizes the activation and masks it to that semantic class;
4. checks whether the centroids of annotated object-part components fall in
   the active region; and
5. calls the prototype or concept assignment consistent when the same part is
   hit in more than the configured fraction of images in which that part is
   annotated.

The score is the fraction of evaluated prototypes or class-concept
assignments that are consistent.  Only components associated with semantic
classes having part annotations enter the denominator.  For output-schema
compatibility, ReProSeg concept indices are stored in ``prototype_id`` fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.schema.data import DataConfig
from data import DataSplit, PanopticPartsDataset
from data.dataset.factory import DatasetFactory
from model.model import ReProSeg
from proto_segmentation.model import PPNet
from utils.log import Log

Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]
AccumulatorKey: TypeAlias = tuple[int, int, int]
ComponentClassKey: TypeAlias = tuple[int, int]
SupportedModel: TypeAlias = PPNet | ReProSeg
CITYSCAPES_NATIVE_IMAGE_SHAPE = (1024, 2048)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PartPresence:
    """One prototype/part observation in one image."""

    image_index: int
    class_id: int
    prototype_id: int
    part_id: int
    present: bool


@dataclass(frozen=True)
class PartConsistency:
    """Mean presence of one object part for one prototype."""

    class_id: int
    prototype_id: int
    part_id: int
    mean_presence: float
    num_images: int


@dataclass(frozen=True)
class PrototypeConsistency:
    """Consistency decision for a single evaluated prototype."""

    class_id: int
    prototype_id: int
    consistent: bool
    max_part_consistency: float


@dataclass(frozen=True)
class ConsistencyResult:
    """Structured result returned by :func:`run_consistency`."""

    score: float
    activation_quantile: float
    consistency_threshold: float
    num_consistent_prototypes: int
    num_evaluated_prototypes: int
    prototype_consistency: tuple[PrototypeConsistency, ...]
    part_consistency: tuple[PartConsistency, ...]
    observations: tuple[PartPresence, ...]

    def save(self, output_dir: Path) -> None:
        """Persist source-compatible score and CSV artifacts."""

        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"th_{self.consistency_threshold:g}_qt_{self.activation_quantile:g}"

        (output_dir / f"consistency_score_{suffix}.txt").write_text(f"{self.score}\n")

        with (output_dir / f"part_presence_{suffix}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["image_index", "class_id", "prototype_id", "part_id", "present"],
            )
            writer.writeheader()
            writer.writerows(asdict(observation) for observation in self.observations)

        prototype_by_key = {
            (prototype.class_id, prototype.prototype_id): prototype for prototype in self.prototype_consistency
        }
        with (output_dir / f"part_presence_mean_{suffix}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "class_id",
                    "prototype_id",
                    "part_id",
                    "mean_presence",
                    "num_images",
                    "is_consistent",
                ],
            )
            writer.writeheader()
            for part in self.part_consistency:
                row = asdict(part)
                row["is_consistent"] = prototype_by_key[(part.class_id, part.prototype_id)].consistent
                writer.writerow(row)

        summary = {
            "score": self.score,
            "activation_quantile": self.activation_quantile,
            "consistency_threshold": self.consistency_threshold,
            "num_consistent_prototypes": self.num_consistent_prototypes,
            "num_evaluated_prototypes": self.num_evaluated_prototypes,
            "prototype_consistency": [asdict(prototype) for prototype in self.prototype_consistency],
        }
        (output_dir / f"consistency_summary_{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")


def connected_component_centroids(mask: Tensor) -> list[tuple[int, int]]:
    """Return rounded ``(y, x)`` centroids of 8-connected foreground regions.

    A small NumPy flood fill avoids adding OpenCV solely for this metric.  In
    contrast to ``cv2.connectedComponentsWithStats``, this helper never returns
    a centroid for the background component.
    """

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D component mask, received shape {tuple(mask.shape)}.")

    foreground = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=np.bool_)
    centroids: list[tuple[int, int]] = []

    for start_y, start_x in np.argwhere(foreground & ~visited):
        start_y = int(start_y)
        start_x = int(start_x)
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        sum_y = 0
        sum_x = 0
        count = 0

        while queue:
            y, x = queue.popleft()
            sum_y += y
            sum_x += x
            count += 1

            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if not visited[neighbor_y, neighbor_x] and foreground[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))

        centroid_y = int(np.rint(sum_y / count))
        centroid_x = int(np.rint(sum_x / count))
        centroids.append((centroid_y, centroid_x))

    return centroids


def quantile_activation_mask(
    prototype_activations: Tensor,
    semantic_mask: Tensor,
    quantile: float,
) -> Tensor:
    """Quantile-threshold prototype activations within a semantic class.

    The quantile for each prototype is calculated only from pixels belonging
    to the ground-truth semantic class.  Excluding out-of-class pixels avoids
    a zero threshold when the class occupies less than ``1 - quantile`` of
    the image.
    """

    if prototype_activations.ndim != 3:
        raise ValueError(
            f"Expected prototype activations with shape (P, H, W), received {tuple(prototype_activations.shape)}."
        )
    if semantic_mask.ndim != 2:
        raise ValueError(f"Expected a semantic mask with shape (H, W), received {tuple(semantic_mask.shape)}.")
    if prototype_activations.shape[-2:] != semantic_mask.shape:
        raise ValueError(
            "Prototype activations and semantic mask must have matching spatial shapes, "
            f"received {tuple(prototype_activations.shape[-2:])} and {tuple(semantic_mask.shape)}."
        )
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Activation quantile must be in [0, 1], received {quantile}.")

    class_mask = semantic_mask.to(
        device=prototype_activations.device,
        dtype=torch.bool,
    )
    if not class_mask.any():
        raise ValueError("Semantic mask must contain at least one class pixel.")

    class_activations = prototype_activations[:, class_mask]
    thresholds = torch.quantile(
        class_activations,
        quantile,
        dim=1,
        keepdim=True,
    ).reshape(-1, 1, 1)

    return (prototype_activations > thresholds) & class_mask.unsqueeze(0)


class ConsistencyEvaluator:
    """Compute ScaleProtoSeg-style consistency for PPNet prototypes or ReProSeg concepts."""

    def __init__(
        self,
        model: SupportedModel,
        *,
        activation_quantile: float = 0.8,
        consistency_threshold: float = 0.8,
        device: torch.device | str | None = None,
        used_prototypes_only: bool = False,
    ):
        if not isinstance(model, (PPNet, ReProSeg)):
            raise TypeError(f"Expected a PPNet or ReProSeg model, received {type(model).__name__}.")
        if not 0.0 <= activation_quantile <= 1.0:
            raise ValueError(f"Activation quantile must be in [0, 1], received {activation_quantile}.")
        if not 0.0 <= consistency_threshold <= 1.0:
            raise ValueError(f"Consistency threshold must be in [0, 1], received {consistency_threshold}.")

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.activation_quantile = activation_quantile
        self.consistency_threshold = consistency_threshold
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

            # ReProSeg's NonNegConv1x1 uses the sign of the raw parameter to
            # identify active concept-to-class connections.  A concept may be
            # connected to more than one class, so consistency is tracked per
            # class-concept pair.
            identity = classifier_weights.T >= self.model.layers.classification_layer.MIN_CLASSIFICATION_WEIGHT
            used_component_mask = identity.any(dim=1)

        self.component_class_identity = identity
        self.used_component_mask = used_component_mask

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: Iterable[Batch],
        *,
        show_progress: bool = True,
    ) -> ConsistencyResult:
        """Evaluate consistency over a loader of image/semantic/part batches."""

        observations: list[PartPresence] = []
        presence_sum: defaultdict[AccumulatorKey, int] = defaultdict(int)
        presence_count: defaultdict[AccumulatorKey, int] = defaultdict(int)
        evaluated_prototypes: set[ComponentClassKey] = set()
        image_index = 0

        total = len(data_loader) if isinstance(data_loader, Sized) else None
        batches = tqdm(
            data_loader,
            total=total,
            disable=not show_progress,
            desc="Computing prototype/concept consistency",
        )

        for images, semantic_masks, part_masks in batches:
            semantic_masks = _squeeze_mask_channel(semantic_masks, "semantic")
            part_masks = _squeeze_mask_channel(part_masks, "panoptic-part")
            if semantic_masks.shape != part_masks.shape:
                raise ValueError(
                    "Semantic and panoptic-part masks must have matching shapes, "
                    f"received {tuple(semantic_masks.shape)} and {tuple(part_masks.shape)}."
                )
            if images.shape[0] != semantic_masks.shape[0]:
                raise ValueError(
                    "Images and masks must have the same batch size, "
                    f"received {images.shape[0]} and {semantic_masks.shape[0]}."
                )

            images = images.to(self.device)
            activations = self._prototype_activations(images)

            for batch_index in range(images.shape[0]):
                self._collect_image_observations(
                    activations=activations[batch_index],
                    semantic_mask=semantic_masks[batch_index],
                    part_mask=part_masks[batch_index],
                    image_index=image_index,
                    observations=observations,
                    presence_sum=presence_sum,
                    presence_count=presence_count,
                    evaluated_prototypes=evaluated_prototypes,
                )
                image_index += 1

        return self._build_result(
            observations,
            presence_sum,
            presence_count,
            evaluated_prototypes,
        )

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
            raise ValueError("Consistency evaluation requires non-negative prototype/concept activations.")
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
            raise ValueError(
                f"Expected {self.num_components} ReProSeg concepts, received {aspp_features.shape[1]}."
            )

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

    def _collect_image_observations(
        self,
        *,
        activations: Tensor,
        semantic_mask: Tensor,
        part_mask: Tensor,
        image_index: int,
        observations: list[PartPresence],
        presence_sum: defaultdict[AccumulatorKey, int],
        presence_count: defaultdict[AccumulatorKey, int],
        evaluated_prototypes: set[ComponentClassKey],
    ) -> None:
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

            resized_activations = F.interpolate(
                activations[prototype_ids].unsqueeze(0),
                size=output_size,
                mode="nearest-exact",
            ).squeeze(0)
            active_regions = quantile_activation_mask(
                resized_activations,
                class_mask.to(self.device),
                self.activation_quantile,
            ).to(device="cpu")

            for prototype_id in prototype_ids:
                evaluated_prototypes.add((class_id, prototype_id))

            for part_id, centroids in part_centroids.items():
                rows = torch.tensor([centroid[0] for centroid in centroids], dtype=torch.long)
                columns = torch.tensor([centroid[1] for centroid in centroids], dtype=torch.long)
                part_present = active_regions[:, rows, columns].any(dim=1)

                for local_index, prototype_id in enumerate(prototype_ids):
                    present = bool(part_present[local_index].item())
                    key = (class_id, prototype_id, part_id)
                    presence_sum[key] += int(present)
                    presence_count[key] += 1
                    observations.append(
                        PartPresence(
                            image_index=image_index,
                            class_id=class_id,
                            prototype_id=prototype_id,
                            part_id=part_id,
                            present=present,
                        )
                    )

    def _build_result(
        self,
        observations: Sequence[PartPresence],
        presence_sum: dict[AccumulatorKey, int],
        presence_count: dict[AccumulatorKey, int],
        evaluated_prototypes: set[ComponentClassKey],
    ) -> ConsistencyResult:
        part_results: list[PartConsistency] = []
        part_scores_by_prototype: defaultdict[ComponentClassKey, list[float]] = defaultdict(list)

        for class_id, prototype_id, part_id in sorted(presence_count):
            key = (class_id, prototype_id, part_id)
            mean_presence = presence_sum[key] / presence_count[key]
            part_scores_by_prototype[(class_id, prototype_id)].append(mean_presence)
            part_results.append(
                PartConsistency(
                    class_id=class_id,
                    prototype_id=prototype_id,
                    part_id=part_id,
                    mean_presence=mean_presence,
                    num_images=presence_count[key],
                )
            )

        prototype_results: list[PrototypeConsistency] = []
        for class_id, prototype_id in sorted(evaluated_prototypes):
            part_scores = part_scores_by_prototype[(class_id, prototype_id)]
            max_part_consistency = max(part_scores, default=0.0)
            prototype_results.append(
                PrototypeConsistency(
                    class_id=class_id,
                    prototype_id=prototype_id,
                    # Keep the strict comparison used by the ScaleProtoSeg code.
                    consistent=max_part_consistency > self.consistency_threshold,
                    max_part_consistency=max_part_consistency,
                )
            )

        num_evaluated = len(prototype_results)
        num_consistent = sum(prototype.consistent for prototype in prototype_results)
        score = num_consistent / num_evaluated if num_evaluated else 0.0

        return ConsistencyResult(
            score=score,
            activation_quantile=self.activation_quantile,
            consistency_threshold=self.consistency_threshold,
            num_consistent_prototypes=num_consistent,
            num_evaluated_prototypes=num_evaluated,
            prototype_consistency=tuple(prototype_results),
            part_consistency=tuple(part_results),
            observations=tuple(observations),
        )


def _squeeze_mask_channel(mask: Tensor, name: str) -> Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.ndim != 3:
        raise ValueError(f"Expected {name} masks with shape (B, H, W) or (B, 1, H, W), received {tuple(mask.shape)}.")
    return mask


def _part_centroids(
    part_mask: Tensor,
    semantic_class_mask: Tensor,
) -> dict[int, list[tuple[int, int]]]:
    """Extract component centroids keyed by the decoded Cityscapes part ID."""

    centroids_by_part: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    encoded_parts = torch.unique(part_mask[semantic_class_mask])

    for encoded_part_value in encoded_parts:
        encoded_part = int(encoded_part_value.item())
        part_id = encoded_part % 100
        if encoded_part <= 0 or part_id <= 0:
            continue

        component_mask = (part_mask == encoded_part) & semantic_class_mask
        centroids_by_part[part_id].extend(connected_component_centroids(component_mask))

    return dict(centroids_by_part)


def run_consistency(
    model: SupportedModel,
    data_loader: Iterable[Batch],
    *,
    activation_quantile: float = 0.8,
    consistency_threshold: float = 0.8,
    device: torch.device | str | None = None,
    used_prototypes_only: bool = False,
    show_progress: bool = True,
) -> ConsistencyResult:
    """Convenience wrapper matching the original evaluator's entry point."""

    evaluator = ConsistencyEvaluator(
        model,
        activation_quantile=activation_quantile,
        consistency_threshold=consistency_threshold,
        device=device,
        used_prototypes_only=used_prototypes_only,
    )
    return evaluator.evaluate(data_loader, show_progress=show_progress)


def _load_supported_model(checkpoint_path: Path) -> SupportedModel:
    """Load a serialized PPNet or construct ReProSeg from its training checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(checkpoint, (PPNet, ReProSeg)):
        return checkpoint

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected checkpoint {checkpoint_path} to contain a PPNet, ReProSeg, or state-dict mapping; "
            f"found {type(checkpoint).__name__}."
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a valid model_state_dict mapping.")

    classifier_weight = state_dict.get("layers.classification_layer.weight")
    if not isinstance(classifier_weight, Tensor) or classifier_weight.ndim != 4:
        raise TypeError(
            f"Checkpoint {checkpoint_path} is neither a serialized PPNet nor a recognizable ReProSeg checkpoint."
        )

    from config import ReProSegConfig

    config = ReProSegConfig()
    config.env.device = torch.device("cpu")
    config.data.num_classes = int(classifier_weight.shape[0])
    config.model.checkpoint = None
    config.model.disable_pretrained = True
    config.model.bias = "layers.classification_layer.bias" in state_dict

    model = ReProSeg(
        cfg=config,
        log=cast(Log, logging.getLogger(f"{__name__}.checkpoint")),
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def _default_data_path() -> Path | None:
    data_root = os.environ.get("DATA_ROOT")
    return Path(data_root) / "Cityscapes" if data_root else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ScaleProtoSeg-style prototype/concept consistency on Cityscapes."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Trusted serialized PPNet or ReProSeg training checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=_default_data_path(),
        help="Cityscapes root containing leftImg8bit, gtFine, and gtFinePanopticParts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("consistency_results"),
        help="Directory for the score, per-image observations, and aggregate CSV.",
    )
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size. Native-resolution evaluation defaults to one image per batch.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=CITYSCAPES_NATIVE_IMAGE_SHAPE,
        help=(
            "Evaluation image shape. Defaults to Cityscapes' native 1024x2048 "
            "resolution; passing another shape applies a center crop."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu, cuda, or cuda:1.",
    )
    parser.add_argument(
        "--used-prototypes-only",
        action="store_true",
        help=(
            "For PPNet, restrict the denominator to prototypes with a positive final-layer connection. "
            "ReProSeg concepts are always selected through active concept-to-class connections."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point using ReProSeg's official Cityscapes validation loader."""

    load_dotenv()
    args = _parse_args()
    if args.data_path is None:
        raise ValueError("Set DATA_ROOT or pass --data-path with the Cityscapes dataset root.")
    if args.batch_size < 1:
        raise ValueError(f"Batch size must be positive, received {args.batch_size}.")
    if args.num_workers < 0:
        raise ValueError(f"Number of workers cannot be negative, received {args.num_workers}.")

    device = torch.device(args.device)
    model = _load_supported_model(args.checkpoint)

    data_config = DataConfig(
        path=args.data_path,
        batch_size=max(2, args.batch_size),
        num_workers=args.num_workers,
        img_shape=tuple(args.image_shape),
        filter_classes=True,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    validation_data = DatasetFactory.create(data_config, split=DataSplit.VAL)
    parts_data = PanopticPartsDataset(data_config, validation_data)
    data_loader = DataLoader(
        parts_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    result = run_consistency(
        model,
        data_loader,
        activation_quantile=args.quantile,
        consistency_threshold=args.threshold,
        device=device,
        used_prototypes_only=args.used_prototypes_only,
    )
    result.save(args.output_dir)

    component_label = "concept assignments" if isinstance(model, ReProSeg) else "prototypes"
    print(
        f"Consistency score: {result.score:.6f} "
        f"({result.num_consistent_prototypes}/{result.num_evaluated_prototypes} {component_label})"
    )
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
