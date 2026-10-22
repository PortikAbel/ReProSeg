"""Preparation and part observations for the future stability evaluator.

This module accepts an already perturbed image. Noise generation, stability
scoring, dataset iteration, and CLI registration belong to later steps.
"""

from pathlib import Path

import torch
from torch import Tensor

from config.schema.model import ModelConfig
from evaluate.checkpoint import _build_supported_model, _load_checkpoint
from evaluate.parts import PartObservationExtractor, PartPresence


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
