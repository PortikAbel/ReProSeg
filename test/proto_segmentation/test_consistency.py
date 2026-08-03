import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model.model import NonNegConv1x1, ReProSeg
from proto_segmentation.model import PPNet
from visualize.consistency import (
    CITYSCAPES_NATIVE_IMAGE_SHAPE,
    ConsistencyEvaluator,
    _parse_args,
    connected_component_centroids,
    quantile_activation_mask,
)


class DummyPPNet(PPNet):
    """Minimal PPNet whose first two image channels are activation maps."""

    def __init__(self):
        nn.Module.__init__(self)
        self.prototype_vectors = nn.Parameter(torch.zeros(2, 1, 1, 1))
        self.prototype_class_identity = torch.ones(2, 1)
        self.last_layer = nn.Linear(2, 1, bias=False)
        self.last_layer.weight.data.fill_(1)

    def push_forward(self, images):
        activations = images[:, :2]
        return torch.empty(0, device=images.device), activations

    def distance_2_similarity(self, distances):
        return distances


class DummyReProSeg(ReProSeg):
    """Minimal ReProSeg exposing input channels as one-scale concept maps."""

    def __init__(self, num_concepts=1, num_classes=3):
        nn.Module.__init__(self)
        self.num_concepts = num_concepts
        self.layers = nn.Module()
        self.layers.classification_layer = NonNegConv1x1(num_concepts, num_classes, bias=False)
        self.layers.classification_layer.weight.data.fill_(-1)

    def forward(self, images, inference=False):
        concept_activations = images[:, : self.num_concepts]
        aspp_features = concept_activations.unsqueeze(2)
        output = torch.zeros(
            images.shape[0],
            self.layers.classification_layer.out_channels,
            *images.shape[-2:],
            device=images.device,
        )
        return aspp_features, concept_activations, output


def test_connected_component_centroids_uses_eight_connectivity_without_background():
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[3, 3] = True

    assert connected_component_centroids(mask) == [(0, 0), (3, 3)]


def test_quantile_activation_mask_is_computed_per_prototype():
    activations = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    semantic_mask = torch.ones(2, 2, dtype=torch.bool)

    active = quantile_activation_mask(activations, semantic_mask, quantile=0.5)

    assert torch.equal(active[0], torch.tensor([[False, False], [True, True]]))
    assert not active[1].any()


def test_quantile_activation_mask_excludes_out_of_class_pixels_from_quantile():
    activations = torch.zeros(2, 5, 5)
    semantic_mask = torch.zeros(5, 5, dtype=torch.bool)
    semantic_mask[:2, :2] = True
    activations[0, :2, :2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    activations[1, :2, :2] = torch.tensor([[4.0, 3.0], [2.0, 1.0]])

    active = quantile_activation_mask(activations, semantic_mask, quantile=0.8)

    # The class occupies only 16% of the image. Including the 21 out-of-class
    # zeros would produce a zero threshold and activate all four class pixels.
    assert active[0].sum() == 1
    assert active[0, 1, 1]
    assert active[1].sum() == 1
    assert active[1, 0, 0]
    assert not active[:, ~semantic_mask].any()


def test_cli_defaults_to_native_cityscapes_resolution(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["visualize.consistency", "checkpoint.pth"])

    args = _parse_args()

    assert tuple(args.image_shape) == CITYSCAPES_NATIVE_IMAGE_SHAPE
    assert args.batch_size == 1


def test_reproseg_uses_active_class_concept_assignments():
    model = DummyReProSeg()
    model.layers.classification_layer.weight.data[1, 0] = 1
    model.layers.classification_layer.weight.data[2, 0] = 2
    images = torch.zeros(2, 3, 5, 5)
    images[:, 0, 2, 2] = 1
    semantic_masks = torch.empty(2, 1, 5, 5, dtype=torch.long)
    semantic_masks[0] = 1
    semantic_masks[1] = 2
    part_masks = torch.zeros(2, 1, 5, 5, dtype=torch.long)
    part_masks[0, 0, 2, 2] = 2401
    part_masks[1, 0, 2, 2] = 2501

    evaluator = ConsistencyEvaluator(
        model,
        activation_quantile=0.5,
        consistency_threshold=0.8,
        device="cpu",
    )
    result = evaluator.evaluate(
        [(images, semantic_masks, part_masks)],
        show_progress=False,
    )

    assert result.score == 1
    assert result.num_evaluated_prototypes == 2
    assert {
        (component.class_id, component.prototype_id) for component in result.prototype_consistency
    } == {(1, 0), (2, 0)}


def test_reproseg_concept_aggregation_matches_scale_aware_pooling():
    model = DummyReProSeg(num_concepts=2)
    model.layers.classification_layer.weight.data[1] = 1
    images = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    evaluator = ConsistencyEvaluator(model, device="cpu")

    activations = evaluator._prototype_activations(images)

    assert torch.equal(activations, F.max_pool2d(images, kernel_size=3, padding=1, stride=1))


def test_evaluator_computes_per_image_part_consistency(tmp_path: Path):
    model = DummyPPNet()
    images = torch.zeros(2, 3, 4, 4)
    images[:, 0, 0, 0] = 1
    images[0, 1, 0, 0] = 1
    images[1, 1, 3, 3] = 1

    semantic_masks = torch.ones(2, 1, 4, 4, dtype=torch.long)
    part_masks = torch.zeros(2, 1, 4, 4, dtype=torch.long)
    part_masks[:, 0, 0, 0] = 2401
    part_masks[:, 0, 3, 3] = 2402

    evaluator = ConsistencyEvaluator(
        model,
        activation_quantile=0.8,
        consistency_threshold=0.75,
        device="cpu",
    )
    result = evaluator.evaluate(
        [(images, semantic_masks, part_masks)],
        show_progress=False,
    )

    assert result.score == 0.5
    assert result.num_consistent_prototypes == 1
    assert result.num_evaluated_prototypes == 2
    assert [prototype.consistent for prototype in result.prototype_consistency] == [
        True,
        False,
    ]
    assert {(part.prototype_id, part.part_id): part.mean_presence for part in result.part_consistency} == {
        (0, 1): 1.0,
        (0, 2): 0.0,
        (1, 1): 0.5,
        (1, 2): 0.5,
    }
    assert len(result.observations) == 8

    result.save(tmp_path)
    suffix = "th_0.75_qt_0.8"
    assert (tmp_path / f"consistency_score_{suffix}.txt").is_file()
    assert (tmp_path / f"part_presence_{suffix}.csv").is_file()
    assert (tmp_path / f"part_presence_mean_{suffix}.csv").is_file()
    assert (tmp_path / f"consistency_summary_{suffix}.json").is_file()
