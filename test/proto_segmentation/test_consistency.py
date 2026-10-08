import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from config import EvaluateConfig
from config.schema.data import DatasetType
from evaluate.consistency import (
    CITYSCAPES_NATIVE_IMAGE_SHAPE,
    ConsistencyEvaluator,
    _pascal_class_offset,
    _register_legacy_checkpoint_modules,
    connected_component_centroids,
    quantile_activation_mask,
)
from model.model import NonNegConv1x1, ReProSeg
from model.proto_segmentation import PPNet


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


def test_pascal_batch_guard_rejects_non_unit_batch_without_image_shape():
    from evaluate import consistency

    cfg = EvaluateConfig()
    cfg.data.dataset = DatasetType.VOC_SEGMENTATION
    cfg.model.checkpoint = Path("checkpoint.pth")
    cfg.evaluate.consistency.batch_size = 2

    with pytest.raises(ValueError, match="variable sizes"):
        consistency.run_consistency_evaluation(cfg)


def test_cityscapes_defaults_to_native_image_shape_when_unset(monkeypatch, tmp_path):
    from evaluate import consistency

    cfg = EvaluateConfig()
    cfg.model.checkpoint = Path("checkpoint.pth")
    cfg.env.device = torch.device("cpu")

    monkeypatch.setattr(consistency, "_load_checkpoint", lambda *args, **kwargs: DummyPPNet())
    captured = {}

    def fake_create(data_config, split):
        captured["img_shape"] = data_config.img_shape
        return object()

    monkeypatch.setattr(consistency.DatasetFactory, "create", fake_create)
    monkeypatch.setattr(consistency, "PanopticPartsDataset", MagicMock())
    monkeypatch.setattr(consistency, "DataLoader", MagicMock())
    result = MagicMock()
    result.score = 0.5
    monkeypatch.setattr(consistency, "run_consistency", MagicMock(return_value=result))

    consistency.run_consistency_evaluation(cfg)

    assert captured["img_shape"] == CITYSCAPES_NATIVE_IMAGE_SHAPE


@pytest.mark.parametrize("num_classes", [20, 21])
@pytest.mark.parametrize("model_type", [DummyPPNet, DummyReProSeg])
def test_pascal_class_mapping_including_last_class(num_classes, model_type):
    if model_type is DummyPPNet:
        model = DummyPPNet()
        model.prototype_class_identity = torch.zeros(2, num_classes)
        model.prototype_class_identity[0, -1] = 1
    else:
        model = DummyReProSeg(num_classes=num_classes)
        model.layers.classification_layer.weight.data[-1, 0] = 1
    images = torch.zeros(1, 3, 5, 5)
    images[0, 0, 2, 2] = 1
    semantic = torch.full((1, 1, 5, 5), 20, dtype=torch.long)
    parts = torch.zeros_like(semantic)
    parts[0, 0, 2, 2] = 2001
    evaluator = ConsistencyEvaluator(
        model,
        device="cpu",
        activation_quantile=0.5,
        semantic_class_offset=_pascal_class_offset(model),
    )
    result = evaluator.evaluate([(images, semantic, parts)], show_progress=False)
    assert result.score == 1
    assert result.num_evaluated_prototypes == 1
    assert result.observations[0].class_id == 20


def test_pascal_rejects_checkpoint_for_another_dataset():
    with pytest.raises(ValueError, match="20- or 21-class checkpoint"):
        _pascal_class_offset(DummyPPNet())


def test_legacy_checkpoint_modules_resolve_to_new_locations(monkeypatch):
    aliases = {
        "proto_segmentation.model": "model.proto_segmentation",
        "proto_segmentation.segmentation.utils": "model.utils",
        "proto_segmentation.deeplab_pytorch.libs.models.deeplabv2": (
            "model.segmentation_features.deeplab_pytorch.libs.models.deeplabv2"
        ),
    }
    for legacy_module in (*aliases, "proto_segmentation", "proto_segmentation.segmentation"):
        monkeypatch.delitem(sys.modules, legacy_module, raising=False)

    _register_legacy_checkpoint_modules()

    for legacy_module, current_module in aliases.items():
        assert sys.modules[legacy_module] is sys.modules[current_module]
        assert importlib.import_module(legacy_module) is sys.modules[current_module]


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
    assert {(component.class_id, component.prototype_id) for component in result.prototype_consistency} == {
        (1, 0),
        (2, 0),
    }


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


@pytest.mark.parametrize("official_parts_only", [False, True])
def test_evaluation_passes_part_selection_to_loader_and_separates_results(
    monkeypatch, tmp_path, official_parts_only, mock_run_context
):
    from evaluate import consistency

    output = tmp_path / "results"
    mock_run_context.consistency_dir.side_effect = (
        lambda official_parts_only=False: output / "official_parts" if official_parts_only else output
    )

    cfg = EvaluateConfig()
    cfg.env.device = torch.device("cpu")
    cfg.model.checkpoint = Path("checkpoint.pth")
    cfg.evaluate.consistency.official_parts_only = official_parts_only
    cfg.evaluate.consistency.image_shape = CITYSCAPES_NATIVE_IMAGE_SHAPE

    monkeypatch.setattr(consistency, "_load_checkpoint", lambda *args, **kwargs: DummyPPNet())
    validation_data = object()
    monkeypatch.setattr(consistency.DatasetFactory, "create", lambda *args, **kwargs: validation_data)
    dataset = MagicMock()
    monkeypatch.setattr(consistency, "PanopticPartsDataset", dataset)
    loader = MagicMock()
    monkeypatch.setattr(consistency, "DataLoader", loader)
    evaluate_mock = MagicMock()
    evaluate_mock.return_value.score = 0.5
    monkeypatch.setattr(consistency, "run_consistency", evaluate_mock)

    consistency.run_consistency_evaluation(cfg)

    assert dataset.call_args.args[1] is validation_data
    assert dataset.call_args.kwargs["official_parts_only"] is official_parts_only
    assert loader.call_args.args[0] is dataset.return_value
    assert evaluate_mock.call_args.args[1] is loader.return_value
    expected_output = output / "official_parts" if official_parts_only else output
    evaluate_mock.return_value.save.assert_called_once_with(expected_output)
