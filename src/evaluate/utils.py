"""Tensor and mask helpers for segmentation evaluation."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch
from torch import Tensor


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
    """Extract component centroids keyed by the decoded dataset part ID."""

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
