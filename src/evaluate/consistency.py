"""Part-consistency evaluation for ProtoSeg and ReProSeg models.

This module adapts the consistency metric used by ScaleProtoSeg to the
``model.proto_segmentation.PPNet`` and ``model.model.ReProSeg``
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

For ReProSeg, the score is the number of distinct concepts consistent for at
least one evaluated class divided by the number of distinct concepts in the
evaluated assignments. Detailed results retain separate class-concept pairs.
PPNet uses the fraction of evaluated prototype/class assignments that are
consistent. Only assignments with part observations are evaluated. For output-schema
compatibility, ReProSeg concept indices are stored in ``prototype_id`` fields.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import EvaluateConfig
from config.schema.data import DataConfig, DatasetType
from data import DataSplit, PanopticPartsDataset
from data.dataset.factory import DatasetFactory
from data.dataset.pascal_parts import PascalPartsDataset
from evaluate.checkpoint import SupportedModel, _build_supported_model, _load_checkpoint
from evaluate.parts import PartObservationExtractor, PartPresence
from evaluate.registry import register
from evaluate.utils import _squeeze_mask_channel
from model.model import ReProSeg
from model.proto_segmentation import PPNet
from utils.run_context import get_run_context

logger = logging.getLogger(__name__)

Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]
AccumulatorKey: TypeAlias = tuple[int, int, int]
ComponentClassKey: TypeAlias = tuple[int, int]
CITYSCAPES_NATIVE_IMAGE_SHAPE = (1024, 2048)


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


class ConsistencyEvaluator(PartObservationExtractor):
    """Compute ScaleProtoSeg-style consistency for PPNet prototypes or ReProSeg concepts."""

    def __init__(
        self,
        model: SupportedModel,
        *,
        activation_quantile: float = 0.8,
        consistency_threshold: float = 0.8,
        device: torch.device | str | None = None,
        used_prototypes_only: bool = False,
        semantic_class_offset: int | None = None,
    ):
        if not 0.0 <= consistency_threshold <= 1.0:
            raise ValueError(f"Consistency threshold must be in [0, 1], received {consistency_threshold}.")
        super().__init__(
            model,
            activation_quantile=activation_quantile,
            device=device,
            used_prototypes_only=used_prototypes_only,
            semantic_class_offset=semantic_class_offset,
        )
        self.consistency_threshold = consistency_threshold

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
            ncols=0,
            file=get_run_context().tqdm_file,
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
        (image_observations,) = self._image_part_presence(
            activation_maps=(activations,),
            semantic_mask=semantic_mask,
            part_mask=part_mask,
            image_index=image_index,
        )
        for observation in image_observations:
            key = (observation.class_id, observation.prototype_id, observation.part_id)
            presence_sum[key] += int(observation.present)
            presence_count[key] += 1
            evaluated_prototypes.add((observation.class_id, observation.prototype_id))
        observations.extend(image_observations)

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

        if isinstance(self.model, ReProSeg):
            num_evaluated = len({prototype.prototype_id for prototype in prototype_results})
            num_consistent = len({prototype.prototype_id for prototype in prototype_results if prototype.consistent})
        else:
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


def run_consistency(
    model: SupportedModel,
    data_loader: Iterable[Batch],
    *,
    activation_quantile: float = 0.8,
    consistency_threshold: float = 0.8,
    device: torch.device | str | None = None,
    used_prototypes_only: bool = False,
    show_progress: bool = True,
    semantic_class_offset: int | None = None,
) -> ConsistencyResult:
    """Convenience wrapper matching the original evaluator's entry point."""

    evaluator = ConsistencyEvaluator(
        model,
        activation_quantile=activation_quantile,
        consistency_threshold=consistency_threshold,
        device=device,
        used_prototypes_only=used_prototypes_only,
        semantic_class_offset=semantic_class_offset,
    )
    return evaluator.evaluate(data_loader, show_progress=show_progress)


def _pascal_class_offset(model: SupportedModel) -> int:
    """VOC models may have 20 foreground classes or 21 including background."""
    num_classes = (
        model.prototype_class_identity.shape[1]
        if isinstance(model, PPNet)
        else model.layers.classification_layer.weight.shape[0]
    )
    if num_classes not in (20, 21):
        raise ValueError(f"Pascal VOC requires a 20- or 21-class checkpoint; received {num_classes} classes.")
    return 21 - num_classes


def _training_hparams(log_dir: Path) -> dict:
    """Best-effort read of the original training run's resolved config snapshot.

    Schema changes since training don't matter here: this is read as a plain dict for display,
    never validated against the current config schema.
    """
    snapshot = log_dir / ".hydra" / "config.yaml"
    if not snapshot.is_file():
        return {}
    try:
        return OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)  # type: ignore[return-value]
    except Exception:
        logger.warning(f"Could not read training config snapshot at {snapshot}; omitting it from hparams.")
        return {}


@register("consistency")
def run_consistency_evaluation(cfg: EvaluateConfig) -> ConsistencyResult:
    """Hydra-driven consistency evaluation: load a checkpoint and compute its part-consistency score."""

    consistency_cfg = cfg.evaluate.consistency
    if consistency_cfg.batch_size < 1:
        raise ValueError(f"Batch size must be positive, received {consistency_cfg.batch_size}.")
    if consistency_cfg.num_workers < 0:
        raise ValueError(f"Number of workers cannot be negative, received {consistency_cfg.num_workers}.")
    if (
        cfg.data.dataset == DatasetType.VOC_SEGMENTATION
        and consistency_cfg.image_shape is None
        and consistency_cfg.batch_size != 1
    ):
        raise ValueError(
            "Native Pascal images have variable sizes; use evaluate.consistency.batch_size=1 "
            "or set evaluate.consistency.image_shape."
        )

    device = torch.device(cfg.env.device)
    checkpoint_path = cfg.model.checkpoint
    if checkpoint_path is None:
        raise ValueError("model.checkpoint must be set to the checkpoint to evaluate.")
    checkpoint = _load_checkpoint(checkpoint_path)
    model = _build_supported_model(checkpoint, checkpoint_path, model_cfg=cfg.model)

    semantic_class_offset = None
    if cfg.data.dataset == DatasetType.VOC_SEGMENTATION:
        semantic_class_offset = _pascal_class_offset(model)
        parts_data = PascalPartsDataset(
            cfg.data.path,
            parts_path=consistency_cfg.parts_path,
            parts_spec=consistency_cfg.parts_spec,
            image_shape=consistency_cfg.image_shape,
            official_parts_only=consistency_cfg.official_parts_only,
            mean=cfg.data.mean,
            std=cfg.data.std,
        )
        logger.info(
            f"Pascal validation: {len(parts_data)} images with parts; {parts_data.num_missing_parts} without TIFFs."
        )
    else:
        image_shape = consistency_cfg.image_shape or CITYSCAPES_NATIVE_IMAGE_SHAPE
        data_config = DataConfig(
            path=cfg.data.path,
            batch_size=max(2, consistency_cfg.batch_size),
            num_workers=consistency_cfg.num_workers,
            img_shape=image_shape,
            filter_classes=True,
            mean=cfg.data.mean,
            std=cfg.data.std,
        )
        validation_data = DatasetFactory.create(data_config, split=DataSplit.VAL)
        parts_data = PanopticPartsDataset(
            data_config, validation_data, official_parts_only=consistency_cfg.official_parts_only
        )

    data_loader: DataLoader[Batch] = DataLoader(
        parts_data,
        batch_size=consistency_cfg.batch_size,
        shuffle=False,
        num_workers=consistency_cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    result = run_consistency(
        model,
        data_loader,
        activation_quantile=consistency_cfg.quantile,
        consistency_threshold=consistency_cfg.threshold,
        device=device,
        used_prototypes_only=consistency_cfg.used_prototypes_only,
        semantic_class_offset=semantic_class_offset,
    )

    output_dir = get_run_context().consistency_dir(official_parts_only=consistency_cfg.official_parts_only)
    result.save(output_dir)
    if cfg.data.dataset == DatasetType.VOC_SEGMENTATION:
        # CSV image_index refers to this order after intersecting the two datasets.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "image_ids.txt").write_text("\n".join(parts_data.image_ids) + "\n")
        (output_dir / "pascal_evaluation.json").write_text(
            json.dumps(
                {
                    "dataset": "pascal_voc",
                    "voc_root": str(parts_data.voc_root.resolve()),
                    "parts_path": str(parts_data.parts_path.resolve()),
                    "parts_spec": str(parts_data.parts_spec.resolve()),
                    "num_images": len(parts_data),
                    "num_missing_parts": parts_data.num_missing_parts,
                    "image_shape": consistency_cfg.image_shape,
                    "semantic_class_offset": semantic_class_offset,
                    "official_parts_only": consistency_cfg.official_parts_only,
                    "part_names": {f"{sid}:{pid}": name for (sid, pid), name in parts_data.part_names.items()},
                },
                indent=2,
            )
            + "\n"
        )

    component_label = "distinct concepts" if isinstance(model, ReProSeg) else "prototypes"
    logger.info(
        f"Consistency score: {result.score:.6f} "
        f"({result.num_consistent_prototypes}/{result.num_evaluated_prototypes} consistent {component_label})"
    )
    logger.info(f"Results written to {output_dir.resolve()}")

    run = get_run_context()
    hparams = {**_training_hparams(run.log_dir), **cfg.model_dump(exclude={"logging": {"path"}})}
    run_name = f"consistency_qt_{consistency_cfg.quantile:g}_th_{consistency_cfg.threshold:g}" + (
        "_official_parts" if consistency_cfg.official_parts_only else ""
    )
    metrics = {"consistency_score": result.score}
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("best_miou"), (int, float)):
        metrics["best_miou"] = checkpoint["best_miou"]
    run.log_hparams(hparams, metrics, run_name=run_name)

    return result
