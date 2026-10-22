from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from evaluate import parts, stability
from evaluate.consistency import ConsistencyEvaluator
from evaluate.parts import PartPresence
from evaluate.stability import StabilityObserver
from test.proto_segmentation.test_consistency import DummyPPNet, DummyReProSeg


def test_pair_reuses_centroids_but_thresholds_each_version_independently(monkeypatch):
    clean = torch.zeros(3, 5, 5)
    clean[0, 0, 0] = 1
    clean[1, 2, 2] = 1  # Background centroid must not count as a part.
    perturbed = clean.clone()
    perturbed[0, 0, 0] = 0
    perturbed[0, 4, 4] = 2
    semantic = torch.ones(5, 5, dtype=torch.long)
    annotations = torch.zeros_like(semantic)
    annotations[0, 0] = 2401
    annotations[4, 4] = 2402
    centroids = MagicMock(wraps=parts._part_centroids)
    monkeypatch.setattr(parts, "_part_centroids", centroids)
    observer = StabilityObserver(DummyPPNet(), device="cpu")

    before, after = observer.observe_pair(
        clean,
        perturbed,
        semantic_mask=semantic,
        part_mask=annotations,
        image_index=7,
    )

    assert centroids.call_count == 1
    assert before == [
        PartPresence(7, 1, 0, 1, True),
        PartPresence(7, 1, 1, 1, False),
        PartPresence(7, 1, 0, 2, False),
        PartPresence(7, 1, 1, 2, False),
    ]
    assert after == [
        PartPresence(7, 1, 0, 1, False),
        PartPresence(7, 1, 1, 1, False),
        PartPresence(7, 1, 0, 2, True),
        PartPresence(7, 1, 1, 2, False),
    ]
    assert not hasattr(observer, "evaluate")  # No consistency scoring inherited.


@pytest.mark.parametrize("model_type", [DummyPPNet, DummyReProSeg])
@pytest.mark.parametrize("quantile", [0.7, 0.8, 0.9])
def test_pair_observations_match_consistency_for_identical_images(model_type, quantile):
    model = model_type()
    if isinstance(model, DummyReProSeg):
        model.layers.classification_layer.weight.data[1] = 1
    image = torch.arange(75, dtype=torch.float32).reshape(3, 5, 5)
    semantic = torch.ones(5, 5, dtype=torch.long)
    annotations = torch.zeros_like(semantic)
    annotations[0, 0] = 2401
    annotations[4, 4] = 2402
    expected = ConsistencyEvaluator(model, activation_quantile=quantile, device="cpu").evaluate(
        [(image[None], semantic[None], annotations[None])],
        show_progress=False,
    )
    observer = StabilityObserver(model, activation_quantile=quantile, device="cpu")
    clean, perturbed = observer.observe_pair(
        image,
        image,
        semantic_mask=semantic,
        part_mask=annotations,
    )
    assert tuple(clean) == tuple(perturbed) == expected.observations


def test_stability_loader_uses_shared_checkpoint_support_and_keeps_all_prototypes(monkeypatch):
    model = DummyPPNet()
    model.last_layer.weight.data[0, 1] = -1
    checkpoint_path = Path("push_best.pth")
    load = MagicMock(return_value=model)
    build = MagicMock(return_value=model)
    monkeypatch.setattr(stability, "_load_checkpoint", load)
    monkeypatch.setattr(stability, "_build_supported_model", build)

    observer = stability.load_stability_observer(checkpoint_path, device="cpu")

    load.assert_called_once_with(checkpoint_path)
    build.assert_called_once_with(model, checkpoint_path, model_cfg=None)
    assert observer.model is model
    assert not model.training
    assert observer.used_component_mask.tolist() == [True, True]
    assert torch.equal(observer.component_class_identity, model.prototype_class_identity.bool())


def test_pair_requires_matching_image_and_annotation_shapes():
    observer = StabilityObserver(DummyPPNet(), device="cpu")
    image = torch.zeros(3, 5, 5)
    mask = torch.ones(5, 5, dtype=torch.long)
    with pytest.raises(ValueError, match="images must have matching"):
        observer.observe_pair(image, image[:, :4], semantic_mask=mask, part_mask=mask)
    with pytest.raises(ValueError, match="masks must have matching"):
        observer.observe_pair(image, image, semantic_mask=mask, part_mask=mask[:4])
