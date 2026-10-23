"""Exercise ReProSeg's shared observations and class-specific stability averaging."""

import pytest
import torch

from evaluate.consistency import ConsistencyEvaluator
from evaluate.stability import StabilityImagePair, StabilityNoise, StabilityObserver, run_stability
from test.proto_segmentation.test_consistency import DummyReProSeg


class MultiScaleReProSeg(DummyReProSeg):
    def forward(self, images, inference=False):
        features, pooled, output = super().forward(images, inference=inference)
        features = features.relu()  # Match the real model's nonnegative concept maps.
        # The right-hand class selects a different scale, exercising the shared
        # scale selection and spatial pooling rather than supplying final maps.
        second_scale = torch.zeros_like(features)
        second_scale[..., 20:] = 2 * features[..., 20:]
        return torch.cat((features, second_scale), dim=2), pooled, output


@pytest.fixture
def example():
    model = MultiScaleReProSeg(num_concepts=3, num_classes=20)
    weights = model.layers.classification_layer.weight.data
    weights[1, 0] = weights[2, 0] = 1  # Concept 0 belongs to two classes.
    weights[1, 1] = 1  # Connected concept with an all-zero activation map.
    # Concept 2 has no active connection and must contribute no comparisons.
    clean = torch.zeros(3, 20, 40)
    clean[0, 3, 3] = clean[0, 3, 33] = 1
    clean[2, 3, 3] = 1
    noisy = clean.clone()
    noisy[0, 3, 3] = 0
    noisy[0, 13, 3] = 1
    semantic = torch.ones(20, 40, dtype=torch.long)
    semantic[:, 20:] = 2
    parts = torch.zeros_like(semantic)
    parts[3, 3], parts[13, 3] = 2401, 2402
    parts[3, 33], parts[13, 33] = 2501, 2502
    return model, clean, noisy, semantic, parts


@pytest.mark.parametrize("quantile", [0.7, 0.8, 0.9])
def test_reproseg_clean_and_noisy_observations_reuse_consistency(example, quantile):
    model, clean, noisy, semantic, parts = example
    observer = StabilityObserver(model, activation_quantile=quantile)
    actual = observer.observe_pair(clean, noisy, semantic_mask=semantic, part_mask=parts)
    consistency = ConsistencyEvaluator(model, activation_quantile=quantile)
    for image, observations in zip((clean, noisy), actual, strict=True):
        expected = consistency.evaluate([(image[None], semantic[None], parts[None])], show_progress=False)
        assert tuple(observations) == expected.observations
    comparisons = observer.compare_pair(clean, noisy, semantic_mask=semantic, part_mask=parts)
    assert [(c.class_id, c.prototype_id, c.score) for c in comparisons] == [(1, 0, 0), (1, 1, 1), (2, 0, 1)]
    assert comparisons[0].clean_presence == (True, False)
    assert comparisons[0].noisy_presence == (False, True)
    assert comparisons[1].clean_presence == comparisons[1].noisy_presence == (False, False)
    assert comparisons[2].clean_presence == comparisons[2].noisy_presence == (True, False)


@pytest.mark.parametrize("quantile", [0.7, 0.8, 0.9])
def test_reproseg_runner_weights_each_image_class_concept_combination(example, monkeypatch, quantile):
    model, clean, noisy, semantic, parts = example
    # Image 0 has three comparisons; image 1 only two. A third image has no
    # parts and contributes no comparisons. Fixed perturbations isolate scoring.
    inputs = iter(
        (StabilityImagePair(clean, noisy), StabilityImagePair(clean, clean), StabilityImagePair(clean, clean))
    )
    monkeypatch.setattr(StabilityNoise, "prepare_pair", lambda self, image: next(inputs))
    class_one_parts = parts.clone()
    class_one_parts[:, 20:] = 0
    images = torch.stack((clean, clean, clean))
    semantics = torch.stack((semantic, semantic, semantic))
    annotations = torch.stack((parts, class_one_parts, torch.zeros_like(parts)))
    result = run_stability(model, [(images, semantics, annotations)], quantile=quantile, show_progress=False)
    assert result.num_images == 3
    assert result.summary.num_evaluated_pairs == 5
    assert result.summary.num_stable_pairs == 4
    assert result.summary.num_both_empty_pairs == 2
    assert result.summary.score == pytest.approx(4 / 5)
    assert [(c.image_index, c.class_id, c.prototype_id) for c in result.comparisons] == [
        (0, 1, 0), (0, 1, 1), (0, 2, 0), (1, 1, 0), (1, 1, 1),
    ]


@pytest.mark.parametrize("quantile", [0.7, 0.8, 0.9])
def test_zero_noise_reproseg_runner_is_perfectly_stable(example, quantile):
    model, clean, _, semantic, parts = example
    result = run_stability(
        model, [(clean[None], semantic[None], parts[None])],
        quantile=quantile, std_dev=0, show_progress=False,
    )
    assert result.summary.score == 1
    assert result.summary.num_evaluated_pairs == 3
    assert result.summary.num_both_empty_pairs == 1
    assert all(c.clean_presence == c.noisy_presence for c in result.comparisons)


def test_reproseg_real_noise_is_repeatable_and_independent_of_batching(example):
    model, clean, _, semantic, parts = example
    batch = tuple(torch.stack((tensor, tensor)) for tensor in (clean, semantic, parts))
    state = torch.random.get_rng_state()
    first = run_stability(model, [batch], seed=0, std_dev=0.2, show_progress=False)
    repeated = run_stability(model, [batch], seed=0, std_dev=0.2, show_progress=False)
    split = run_stability(
        model, [tuple(t[i:i + 1] for t in batch) for i in range(2)],
        seed=0, std_dev=0.2, show_progress=False,
    )
    assert first == repeated == split
    assert first.summary.num_evaluated_pairs == 6
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("limit", [1, 2])
def test_image_limit_stops_within_or_at_batch_boundary(example, limit):
    model, clean, _, semantic, parts = example
    batch = tuple(torch.stack((tensor, tensor)) for tensor in (clean, semantic, parts))

    def loader():
        yield batch
        pytest.fail("The runner read another batch after reaching max_images")

    result = run_stability(model, loader(), max_images=limit, std_dev=0, show_progress=False)
    assert result.num_images == limit
    assert result.summary.num_evaluated_pairs == 3 * limit
    assert {c.image_index for c in result.comparisons} == set(range(limit))
