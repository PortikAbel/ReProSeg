"""Preparation and part observations for the future stability evaluator.

Noise is added to already normalized images. Prepare each clean/noisy pair once
and reuse it across activation thresholds. Stability scoring, dataset iteration,
and CLI registration belong to later steps.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from config.schema.evaluate import StabilityEvalConfig
from config.schema.model import ModelConfig
from evaluate.checkpoint import _build_supported_model, _load_checkpoint
from evaluate.parts import PartObservationExtractor, PartPresence


@dataclass(frozen=True)
class StabilityImagePair:
    """Normalized clean/noisy inputs to reuse for every activation threshold."""

    clean: Tensor
    noisy: Tensor


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
