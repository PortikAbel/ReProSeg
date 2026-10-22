"""Preparation and part observations for the future stability evaluator.

Noise is added to already normalized images. Prepare each clean/noisy pair once
and reuse it across activation thresholds. This module scores each observed
image/class/prototype pair; dataset aggregation and CLI registration come later.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from config import EvaluateConfig
from config.schema.evaluate import StabilityEvalConfig
from config.schema.model import ModelConfig
from evaluate.checkpoint import _build_supported_model, _load_checkpoint
from evaluate.data import create_cityscapes_parts_dataloader
from evaluate.parts import PartObservationExtractor, PartPresence


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
    """Exact part-presence comparison for one image, class, and prototype."""

    image_index: int
    class_id: int
    prototype_id: int
    part_ids: tuple[int, ...]
    clean_presence: tuple[bool, ...]
    noisy_presence: tuple[bool, ...]
    score: int


@dataclass(frozen=True)
class StabilitySummary:
    """Mean exact part-vector agreement for one activation quantile."""

    activation_quantile: float
    score: float
    num_evaluated_pairs: int
    num_stable_pairs: int
    num_both_empty_pairs: int


def summarize_stability(
    comparisons: Sequence[PartStability],
    *,
    activation_quantile: float,
) -> StabilitySummary:
    """Average pair scores; require at least one annotated pair."""
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
    order. Reuse the returned pair for all quantiles (0.7/0.8/0.9 by default).
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
    """Observe both versions using consistency's class assignments and geometry."""

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
    """Reuse consistency's checkpoint compatibility and all-prototype selection."""
    checkpoint = _load_checkpoint(checkpoint_path)
    model = _build_supported_model(checkpoint, checkpoint_path, model_cfg=model_cfg)
    return StabilityObserver(
        model,
        activation_quantile=activation_quantile,
        device=device,
        used_prototypes_only=False,
        semantic_class_offset=semantic_class_offset,
    )
