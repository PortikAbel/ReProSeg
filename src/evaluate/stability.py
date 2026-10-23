"""Part stability for PPNet and ReProSeg on clean and noisy Cityscapes images.

ReProSeg uses consistency's active concept-to-class connections, multi-scale
concept maps, class-masked thresholds, and part centres. Each annotated
image/class/concept combination contributes one exact vector comparison.
A concept assigned to multiple classes contributes separately for each class;
scores are averaged over comparisons, not over distinct concepts or images.
Matching all-zero vectors count as stable. Classes without annotated parts,
background, and concepts without active class connections contribute no score.
"""

import csv
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import EvaluateConfig
from config.schema.evaluate import StabilityEvalConfig
from config.schema.model import ModelConfig
from evaluate.checkpoint import SupportedModel, _build_supported_model, _load_checkpoint
from evaluate.data import create_cityscapes_parts_dataloader
from evaluate.parts import PartObservationExtractor, PartPresence
from evaluate.registry import register
from evaluate.utils import _squeeze_mask_channel
from model.model import ReProSeg
from model.proto_segmentation import PPNet
from utils.run_context import get_run_context

logger = logging.getLogger(__name__)


def create_stability_dataloader(cfg: EvaluateConfig) -> DataLoader:
    """Load native-resolution Cityscapes validation images and part annotations.

    Uses data.path, data.mean/std, and evaluate.stability's part mode, batch
    size, worker count, and seed. Batches contain normalized RGB images
    (B, 3, 1024, 2048), semantic masks (B, 1, H, W), and encoded part masks
    (B, 1, H, W), with the same transforms as consistency evaluation.

    Images are ordered by path in both part modes, without shuffling or dropping
    the final batch. Missing annotations raise an error when their image is read.
    The returned dataset's _get_image_path(index) identifies each image for audits.
    """
    settings = cfg.evaluate.stability
    return create_cityscapes_parts_dataloader(
        cfg.data,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        official_parts_only=settings.official_parts_only,
        device=cfg.env.device,
        sort_images=True,
        seed=settings.seed,
    )


@dataclass(frozen=True)
class StabilityImagePair:
    """Normalized clean/noisy inputs to reuse for every activation threshold."""

    clean: Tensor
    noisy: Tensor


@dataclass(frozen=True)
class PartStability:
    """Exact comparison for one image/class/component.

    ``prototype_id`` denotes a PPNet prototype or a ReProSeg concept, retaining
    the same field name as consistency's observations.
    """

    image_index: int
    class_id: int
    prototype_id: int
    part_ids: tuple[int, ...]
    clean_presence: tuple[bool, ...]
    noisy_presence: tuple[bool, ...]
    score: int


@dataclass(frozen=True)
class StabilitySummary:
    """Mean agreement over image/class/component combinations at one quantile."""

    activation_quantile: float
    score: float
    num_evaluated_pairs: int
    num_stable_pairs: int
    num_both_empty_pairs: int


@dataclass(frozen=True)
class StabilityResult:
    """All comparisons and their summary for one dataset run."""

    summary: StabilitySummary
    comparisons: tuple[PartStability, ...]
    num_images: int
    std_dev: float
    seed: int

    def save(self, output_dir: Path) -> None:
        """Write the score, run summary, and each image/prototype comparison."""
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"qt_{self.summary.activation_quantile:g}_std_{self.std_dev:g}_seed_{self.seed}"
        (output_dir / f"stability_score_{suffix}.txt").write_text(f"{self.summary.score}\n")
        summary = {**asdict(self.summary), "num_images": self.num_images, "std_dev": self.std_dev, "seed": self.seed}
        (output_dir / f"stability_summary_{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (output_dir / f"part_presence_{suffix}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image_index", "class_id", "prototype_id", "part_ids",
                    "clean_presence", "noisy_presence", "score",
                ],
            )
            writer.writeheader()
            for comparison in self.comparisons:
                row = asdict(comparison)
                for field in ("part_ids", "clean_presence", "noisy_presence"):
                    row[field] = json.dumps(row[field])
                writer.writerow(row)


def _validate_stability_model(model: SupportedModel, *, cityscapes: bool = False) -> int:
    """Validate classifier layout and return the semantic-label offset.

    Masks reserve zero for background/ignored pixels. PPNet omits that output
    (label 1 maps to classifier row 0); ReProSeg includes it (label 1 maps to
    row 1). The CLI additionally requires the Cityscapes class count.
    """
    if isinstance(model, PPNet):
        identity = model.prototype_class_identity
        weights = model.last_layer.weight
        if identity.ndim != 2 or identity.shape[0] != model.num_prototypes:
            raise ValueError("PPNet prototype_class_identity must have shape (num_prototypes, num_classes).")
        num_classes = identity.shape[1]
        if weights.ndim != 2 or tuple(weights.shape) != (num_classes, model.num_prototypes):
            raise ValueError("PPNet classifier dimensions must match its prototype class assignments.")
        offset, expected_classes = 1, 19
    elif isinstance(model, ReProSeg):
        weights = model.layers.classification_layer.weight
        if weights.ndim != 4 or tuple(weights.shape[1:]) != (model.num_concepts, 1, 1):
            raise ValueError("ReProSeg classifier must have shape (num_classes, num_concepts, 1, 1).")
        num_classes = weights.shape[0]
        offset, expected_classes = 0, 20
    else:
        raise TypeError("Stability evaluation requires a PPNet or ReProSeg checkpoint.")
    if num_classes < (1 if offset else 2):
        raise ValueError("Stability requires at least one foreground class.")
    if cityscapes and num_classes != expected_classes:
        layout = (
            "PPNet with 19 foreground outputs"
            if offset else "ReProSeg with 20 outputs including background at index 0"
        )
        raise ValueError(f"Cityscapes stability requires {layout}; received {num_classes} outputs.")
    return offset


@torch.no_grad()
def run_stability(
    model: SupportedModel,
    data_loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    *,
    quantile: float = 0.8,
    std_dev: float = 0.2,
    seed: int = 0,
    max_images: int | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> StabilityResult:
    """Process every image and average over annotated image/class/components.

    Images must already be normalized. Noise is drawn per image in loader order
    with a new private stream per run, so changing the quantile alone preserves
    the perturbations. Images without part observations contribute no pairs.
    ReProSeg concepts assigned to multiple annotated classes contribute once
    for each class. All eligible comparisons have equal weight, including
    matching all-zero vectors; images may contribute different numbers of pairs.
    max_images optionally limits processing to the first N images in loader order.
    """
    semantic_class_offset = _validate_stability_model(model)
    settings = StabilityEvalConfig(quantile=quantile, std_dev=std_dev, seed=seed, max_images=max_images)
    observer = StabilityObserver(
        model, activation_quantile=settings.quantile, device=device, semantic_class_offset=semantic_class_offset
    )
    noise = StabilityNoise(std_dev=settings.std_dev, seed=settings.seed, device=observer.device)
    comparisons: list[PartStability] = []
    image_index = 0
    for images, semantics, parts in tqdm(data_loader, desc="Computing stability", disable=not show_progress):
        semantics = _squeeze_mask_channel(semantics, "semantic")
        parts = _squeeze_mask_channel(parts, "part")
        if semantics.shape != parts.shape:
            raise ValueError("Semantic and part masks must have matching shapes.")
        if images.ndim != 4 or images.shape[0] != semantics.shape[0]:
            raise ValueError("Images must have shape (B, C, H, W) with the same batch size as masks.")
        for index in range(images.shape[0]):
            pair = noise.prepare_pair(images[index])
            comparisons.extend(
                observer.compare_pair(
                    pair.clean,
                    pair.noisy,
                    semantic_mask=semantics[index],
                    part_mask=parts[index],
                    image_index=image_index,
                )
            )
            image_index += 1
            if settings.max_images is not None and image_index >= settings.max_images:
                break
        if settings.max_images is not None and image_index >= settings.max_images:
            break
    return StabilityResult(
        summary=summarize_stability(comparisons, activation_quantile=settings.quantile),
        comparisons=tuple(comparisons),
        num_images=image_index,
        std_dev=settings.std_dev,
        seed=settings.seed,
    )


@register("stability")
def run_stability_evaluation(cfg: EvaluateConfig) -> StabilityResult:
    """Load the configured PPNet/ReProSeg and Cityscapes data, then evaluate one quantile."""
    checkpoint_path = cfg.model.checkpoint
    if checkpoint_path is None:
        raise ValueError("model.checkpoint must be set to the checkpoint to evaluate.")
    model = _build_supported_model(_load_checkpoint(checkpoint_path), checkpoint_path, model_cfg=cfg.model)
    _validate_stability_model(model, cityscapes=True)
    data_loader = create_stability_dataloader(cfg)
    settings = cfg.evaluate.stability
    result = run_stability(
        model,
        data_loader,
        quantile=settings.quantile,
        std_dev=settings.std_dev,
        seed=settings.seed,
        device=cfg.env.device,
        max_images=settings.max_images,
    )
    output_dir = get_run_context().stability_dir(
        model_name="protoseg" if isinstance(model, PPNet) else "reproseg",
        dataset_name=cfg.data.dataset.value.lower(),
        official_parts_only=settings.official_parts_only,
    )
    if settings.max_images is not None:
        output_dir = output_dir / f"first_{settings.max_images}_images"
    result.save(output_dir)
    logger.info(
        "Stability score: %.6f (%d/%d pairs)",
        result.summary.score,
        result.summary.num_stable_pairs,
        result.summary.num_evaluated_pairs,
    )
    logger.info("Results written to %s", output_dir.resolve())
    return result


def summarize_stability(
    comparisons: Sequence[PartStability],
    *,
    activation_quantile: float,
) -> StabilitySummary:
    """Average exact matches over image/class/components, including empty matches.

    Keep classes separate when a ReProSeg concept has multiple assignments.
    Require at least one annotated comparison; missing annotations are excluded
    by the shared observation extractor rather than counted as empty matches.
    """
    if not 0.0 <= activation_quantile <= 1.0:
        raise ValueError("Activation quantile must be in [0, 1].")
    if not comparisons:
        raise ValueError("Stability is undefined because there are no evaluated image/prototype pairs.")

    stable_count = sum(comparison.score for comparison in comparisons)
    both_empty_count = sum(
        not any(comparison.clean_presence) and not any(comparison.noisy_presence) for comparison in comparisons
    )
    return StabilitySummary(
        activation_quantile=activation_quantile,
        score=stable_count / len(comparisons),
        num_evaluated_pairs=len(comparisons),
        num_stable_pairs=stable_count,
        num_both_empty_pairs=both_empty_count,
    )


def compare_part_presence(clean: Sequence[PartPresence], noisy: Sequence[PartPresence]) -> tuple[PartStability, ...]:
    """Score 1 only when every annotated part agrees; two all-zero vectors agree.

    Observations are keyed by image, class, prototype, and part ID. There must
    be a matching observation on both sides so missing data cannot look stable.
    An image/class without annotated parts has no observations and contributes
    no comparison.
    """
    indexed: list[dict[tuple[int, int, int], dict[int, bool]]] = []
    for observations in (clean, noisy):
        by_pair: dict[tuple[int, int, int], dict[int, bool]] = defaultdict(dict)
        for observation in observations:
            key = (observation.image_index, observation.class_id, observation.prototype_id)
            if observation.part_id in by_pair[key]:
                raise ValueError(f"Duplicate part observation for {key}, part {observation.part_id}.")
            by_pair[key][observation.part_id] = observation.present
        indexed.append(by_pair)

    clean_by_pair, noisy_by_pair = indexed
    if clean_by_pair.keys() != noisy_by_pair.keys():
        raise ValueError("Clean and noisy observations must have identical image/class/prototype pairs.")

    comparisons = []
    for key in sorted(clean_by_pair):
        clean_parts = clean_by_pair[key]
        noisy_parts = noisy_by_pair[key]
        if clean_parts.keys() != noisy_parts.keys():
            raise ValueError(f"Clean and noisy observations must have identical annotated parts for {key}.")
        part_ids = tuple(sorted(clean_parts))
        clean_vector = tuple(clean_parts[part_id] for part_id in part_ids)
        noisy_vector = tuple(noisy_parts[part_id] for part_id in part_ids)
        comparisons.append(PartStability(*key, part_ids, clean_vector, noisy_vector, int(clean_vector == noisy_vector)))
    return tuple(comparisons)


class StabilityNoise:
    """A private, reproducible Gaussian noise stream for one evaluation run.

    Create one instance per run and call prepare_pair once per image, in dataset
    order. Each evaluation run uses its configured activation quantile.
    A new instance with the same seed repeats the stream on the same device
    backend; CPU and CUDA streams are not guaranteed to match each other.
    """

    def __init__(
        self,
        *,
        std_dev: float = 0.2,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ):
        settings = StabilityEvalConfig(std_dev=std_dev, seed=seed)
        self.std_dev = settings.std_dev
        self.seed = settings.seed
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device).manual_seed(self.seed)

    @torch.no_grad()
    def prepare_pair(self, normalized_image: Tensor) -> StabilityImagePair:
        """Add N(0, std_dev**2) after normalization, without clipping.

        Input is one floating-point (C, H, W) image, already normalized by the
        dataset. It is copied to the configured device and left unchanged.
        Keeping this operation per image makes noise independent of batching.
        """
        if normalized_image.ndim != 3 or normalized_image.numel() == 0:
            raise ValueError("Expected a nonempty normalized image shaped (C, H, W).")
        if not normalized_image.is_floating_point():
            raise TypeError("Normalized images must contain floating-point values.")
        if not torch.isfinite(normalized_image).all():
            raise ValueError("Normalized images must contain finite values.")

        clean = normalized_image.detach().to(self.device).clone()
        noise = torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=self._generator,
        )
        return StabilityImagePair(clean=clean, noisy=clean + self.std_dev * noise)


class StabilityObserver(PartObservationExtractor):
    """Reuse consistency's activations, active class assignments, and geometry.

    For ReProSeg, the inherited extractor selects used concepts and their active
    classifier connections, and merges their multi-scale spatial activations.
    Stability only adds the second observation and exact vector comparison.
    """

    def compare_pair(
        self,
        clean_image: Tensor,
        perturbed_image: Tensor,
        *,
        semantic_mask: Tensor,
        part_mask: Tensor,
        image_index: int = 0,
    ) -> tuple[PartStability, ...]:
        """Compare the part vectors returned by the shared observation logic."""
        clean, perturbed = self.observe_pair(
            clean_image,
            perturbed_image,
            semantic_mask=semantic_mask,
            part_mask=part_mask,
            image_index=image_index,
        )
        return compare_part_presence(clean, perturbed)

    @torch.no_grad()
    def observe_pair(
        self,
        clean_image: Tensor,
        perturbed_image: Tensor,
        *,
        semantic_mask: Tensor,
        part_mask: Tensor,
        image_index: int = 0,
    ) -> tuple[list[PartPresence], list[PartPresence]]:
        """Return clean/perturbed part observations, without comparing or scoring.

        Images must already be normalized and shaped (C, H, W). Masks are
        matching (H, W) annotation arrays, shared by both versions. Activation
        maps are thresholded independently using the same quantile.
        """
        if clean_image.ndim != 3 or clean_image.shape != perturbed_image.shape:
            raise ValueError("Clean and perturbed images must have matching (C, H, W) shapes.")
        if semantic_mask.ndim != 2 or semantic_mask.shape != part_mask.shape:
            raise ValueError("Semantic and part masks must have matching (H, W) shapes.")
        clean_activations = self._prototype_activations(clean_image.unsqueeze(0).to(self.device))[0]
        perturbed_activations = self._prototype_activations(perturbed_image.unsqueeze(0).to(self.device))[0]
        clean, perturbed = self._image_part_presence(
            activation_maps=(clean_activations, perturbed_activations),
            semantic_mask=semantic_mask,
            part_mask=part_mask,
            image_index=image_index,
        )
        return clean, perturbed


def load_stability_observer(
    checkpoint_path: Path,
    *,
    activation_quantile: float = 0.8,
    device: torch.device | str | None = None,
    model_cfg: ModelConfig | None = None,
    semantic_class_offset: int | None = None,
) -> StabilityObserver:
    """Reuse checkpoint support, all PPNet prototypes, and active ReProSeg concepts."""
    checkpoint = _load_checkpoint(checkpoint_path)
    model = _build_supported_model(checkpoint, checkpoint_path, model_cfg=model_cfg)
    return StabilityObserver(
        model,
        activation_quantile=activation_quantile,
        device=device,
        used_prototypes_only=False,
        semantic_class_offset=semantic_class_offset,
    )
