from unittest.mock import MagicMock

import pytest
import torch

from config.schema.evaluate import StabilityEvalConfig
from evaluate.stability import StabilityNoise, StabilityObserver
from test.proto_segmentation.test_consistency import DummyPPNet


def test_noise_is_added_in_normalized_space_without_clipping_or_input_mutation():
    image = torch.linspace(-3, 3, 75, dtype=torch.float64).reshape(3, 5, 5)
    original = image.clone()
    pair = StabilityNoise().prepare_pair(image)
    expected = original + 0.2 * torch.randn(
        original.shape,
        dtype=original.dtype,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(pair.noisy, expected, rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    torch.testing.assert_close(pair.clean, original, rtol=0, atol=0)
    assert pair.noisy.min() < 0 and pair.noisy.max() > 1
    assert pair.noisy.dtype == image.dtype
    assert pair.noisy.device == pair.clean.device == torch.device("cpu")
    image.zero_()
    torch.testing.assert_close(pair.clean, original, rtol=0, atol=0)
    torch.testing.assert_close(pair.noisy, expected, rtol=0, atol=0)


def test_private_stream_repeats_across_runs_but_advances_between_images():
    image = torch.ones(3, 5, 5)
    global_state = torch.random.get_rng_state()
    first_run = StabilityNoise(seed=0)
    first = first_run.prepare_pair(image)
    second = first_run.prepare_pair(image)
    assert not torch.equal(first.noisy, second.noisy)
    assert torch.equal(global_state, torch.random.get_rng_state())

    repeated_run = StabilityNoise(seed=0)
    torch.testing.assert_close(first.noisy, repeated_run.prepare_pair(image).noisy, rtol=0, atol=0)
    torch.testing.assert_close(second.noisy, repeated_run.prepare_pair(image).noisy, rtol=0, atol=0)
    assert not torch.equal(first.noisy, StabilityNoise(seed=1).prepare_pair(image).noisy)


def test_zero_noise_returns_equal_inputs_without_gradients():
    image = torch.ones(3, 5, 5, requires_grad=True)
    pair = StabilityNoise(std_dev=0).prepare_pair(image)
    torch.testing.assert_close(pair.clean, pair.noisy, rtol=0, atol=0)
    assert not pair.clean.requires_grad and not pair.noisy.requires_grad


def test_one_prepared_pair_uses_the_configured_threshold(monkeypatch):
    settings = StabilityEvalConfig()
    model = DummyPPNet()
    forward = MagicMock(wraps=model.push_forward)
    monkeypatch.setattr(model, "push_forward", forward)
    noise = StabilityNoise(std_dev=settings.std_dev, seed=settings.seed)
    image = torch.full((3, 5, 5), 2.0)
    pair = noise.prepare_pair(image)
    semantic = torch.ones(5, 5, dtype=torch.long)
    parts = torch.zeros_like(semantic)
    parts[0, 0] = 2401

    observer = StabilityObserver(model, activation_quantile=settings.quantile, device="cpu")
    observer.observe_pair(pair.clean, pair.noisy, semantic_mask=semantic, part_mask=parts)

    assert forward.call_count == 2
    for clean_call, noisy_call in zip(forward.call_args_list[::2], forward.call_args_list[1::2], strict=True):
        torch.testing.assert_close(clean_call.args[0][0], pair.clean, rtol=0, atol=0)
        torch.testing.assert_close(noisy_call.args[0][0], pair.noisy, rtol=0, atol=0)
    # Observing the image pair consumes no extra random draws.
    reference = StabilityNoise()
    reference.prepare_pair(image)
    torch.testing.assert_close(noise.prepare_pair(image).noisy, reference.prepare_pair(image).noisy, rtol=0, atol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"std_dev": -0.1},
        {"std_dev": float("nan")},
        {"std_dev": float("inf")},
        {"seed": -1},
    ],
)
def test_invalid_noise_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        StabilityNoise(**settings)


@pytest.mark.parametrize(
    "image",
    [
        torch.ones(5, 5),
        torch.ones(1, 3, 5, 5),
        torch.empty(3, 0, 5),
        torch.full((3, 5, 5), float("nan")),
        torch.full((3, 5, 5), float("inf")),
    ],
)
def test_invalid_images_are_rejected_without_consuming_noise(image):
    noise = StabilityNoise()
    with pytest.raises(ValueError):
        noise.prepare_pair(image)
    valid = torch.ones(3, 5, 5)
    torch.testing.assert_close(
        noise.prepare_pair(valid).noisy, StabilityNoise().prepare_pair(valid).noisy, rtol=0, atol=0
    )


def test_integer_images_are_rejected():
    with pytest.raises(TypeError, match="floating-point"):
        StabilityNoise().prepare_pair(torch.ones(3, 5, 5, dtype=torch.uint8))
