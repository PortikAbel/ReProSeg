from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from config import EvaluateConfig
from config.schema.evaluate import StabilityEvalConfig
from evaluate import stability
from evaluate.registry import run as run_metric
from test.proto_segmentation.test_consistency import DummyPPNet, DummyReProSeg


def batch():
    images = torch.full((3, 3, 5, 5), 2.0)
    semantics = torch.ones(3, 1, 5, 5, dtype=torch.long)
    parts = torch.zeros_like(semantics)
    parts[:2, :, 0, 0] = 2401
    return images, semantics, parts


def test_all_images_processed_with_no_parts_excluded_from_denominator():
    model = DummyPPNet()
    model.push_forward = MagicMock(wraps=model.push_forward)
    result = stability.run_stability(model, [batch()], quantile=0.7, std_dev=0, show_progress=False)
    assert result.num_images == 3
    assert result.summary.activation_quantile == 0.7
    assert result.summary.num_evaluated_pairs == 4
    assert result.summary.score == 1
    assert {p.image_index for p in result.comparisons} == {0, 1}
    assert model.push_forward.call_count == 6


def test_changing_quantile_preserves_noise_and_results_match_selected_threshold():
    model = DummyPPNet()
    model.push_forward = MagicMock(wraps=model.push_forward)
    images, semantics, parts = batch()
    first = stability.run_stability(model, [(images, semantics, parts)], quantile=0.7, show_progress=False)
    second = stability.run_stability(model, [(images, semantics, parts)], quantile=0.9, show_progress=False)
    assert first.summary.activation_quantile == 0.7
    assert second.summary.activation_quantile == 0.9
    calls = model.push_forward.call_args_list
    for index in range(6):
        torch.testing.assert_close(calls[index].args[0], calls[index + 6].args[0], rtol=0, atol=0)
    expected = []
    observer = stability.StabilityObserver(model, activation_quantile=0.9)
    for index in range(3):
        expected.extend(
            observer.compare_pair(
                calls[2 * index].args[0][0],
                calls[2 * index + 1].args[0][0],
                semantic_mask=semantics[index, 0],
                part_mask=parts[index, 0],
                image_index=index,
            )
        )
    assert second.comparisons == tuple(expected)


def test_result_is_independent_of_batch_size():
    data = batch()
    combined = stability.run_stability(DummyPPNet(), [data], show_progress=False)
    individual = stability.run_stability(
        DummyPPNet(),
        [tuple(t[i : i + 1] for t in data) for i in range(3)],
        show_progress=False,
    )
    assert combined == individual


@pytest.mark.parametrize("official_parts_only", [False, True])
@pytest.mark.parametrize("model_name", ["protoseg", "reproseg"])
@pytest.mark.parametrize("dataset_as_string", [False, True])
def test_cli_registration_passes_quantile_and_saves_results(
    monkeypatch, temp_dir, official_parts_only, model_name, dataset_as_string
):
    cfg = EvaluateConfig()
    if dataset_as_string:
        cfg.data.dataset = "CityScapes"
    cfg.model.checkpoint = Path("push_best.pth")
    cfg.evaluate.stability.quantile = 0.9
    cfg.evaluate.metric_name = "stability"
    cfg.evaluate.stability.official_parts_only = official_parts_only
    if model_name == "protoseg":
        model = DummyPPNet()
        model.prototype_class_identity = torch.ones(2, 19)
        model.last_layer = torch.nn.Linear(2, 19, bias=False)
    else:
        model = DummyReProSeg(num_classes=20)
    loader = object()
    monkeypatch.setattr(stability, "create_stability_dataloader", lambda cfg: loader)
    monkeypatch.setattr(stability, "_load_checkpoint", lambda path: model)
    comparison = stability.PartStability(0, 1, 0, (2401,), (False,), (False,), 1)
    result = stability.StabilityResult(
        stability.StabilitySummary(0.9, 1.0, 1, 1, 1), (comparison,), 1, 0.2, 0
    )
    run = MagicMock(return_value=result)
    monkeypatch.setattr(stability, "run_stability", run)
    assert run_metric(cfg) == result
    assert run.call_args.args == (model, loader)
    assert run.call_args.kwargs["quantile"] == 0.9
    parts_mode = "official_parts" if official_parts_only else "all_parts"
    output_dir = temp_dir / f"stability_{model_name}_cityscapes_{parts_mode}"
    assert (output_dir / "stability_score_qt_0.9_std_0.2_seed_0.txt").read_text() == "1.0\n"
    assert '"num_both_empty_pairs": 1' in (output_dir / "stability_summary_qt_0.9_std_0.2_seed_0.json").read_text()
    assert ",[false],[false],1" in (output_dir / "part_presence_qt_0.9_std_0.2_seed_0.csv").read_text()


@pytest.mark.parametrize("quantile", [-0.1, 1.1, float("nan")])
def test_invalid_run_quantile_rejected(quantile):
    with pytest.raises(ValueError):
        StabilityEvalConfig(quantile=quantile)


def test_empty_run_has_no_defined_score():
    with pytest.raises(ValueError, match="no evaluated"):
        stability.run_stability(DummyPPNet(), [], show_progress=False)


@pytest.mark.parametrize("model_type", [DummyPPNet, DummyReProSeg])
def test_runner_maps_first_last_class_and_excludes_background(model_type):
    if model_type is DummyPPNet:
        model = DummyPPNet()
        model.prototype_class_identity = torch.zeros(2, 19)
        model.prototype_class_identity[0, 0] = 1
        model.prototype_class_identity[1, 18] = 1
        model.last_layer = torch.nn.Linear(2, 19, bias=False)
    else:
        model = DummyReProSeg(num_concepts=3, num_classes=20)
        model.layers.classification_layer.weight.data[1, 0] = 1
        model.layers.classification_layer.weight.data[19, 1] = 1
        model.layers.classification_layer.weight.data[0, 2] = 1
    images = torch.full((1, 3, 5, 5), 2.0)
    semantics = torch.zeros(1, 1, 5, 5, dtype=torch.long)
    semantics[:, :, 0, :] = 1
    semantics[:, :, -1, :] = 19
    parts = torch.full_like(semantics, 2401)
    result = stability.run_stability(model, [(images, semantics, parts)], std_dev=0, show_progress=False)
    assert {(c.class_id, c.prototype_id) for c in result.comparisons} == {(1, 0), (19, 1)}
    assert result.summary.num_evaluated_pairs == 2
    assert result.summary.score == 1


@pytest.mark.parametrize("model_type,num_classes", [(DummyPPNet, 20), (DummyReProSeg, 19), (DummyReProSeg, 21)])
def test_cli_rejects_incompatible_cityscapes_class_layout(monkeypatch, model_type, num_classes):
    cfg = EvaluateConfig()
    cfg.model.checkpoint = Path("checkpoint.pth")
    if model_type is DummyPPNet:
        model = DummyPPNet()
        model.prototype_class_identity = torch.ones(2, num_classes)
        model.last_layer = torch.nn.Linear(2, num_classes, bias=False)
    else:
        model = DummyReProSeg(num_classes=num_classes)
    monkeypatch.setattr(stability, "_load_checkpoint", lambda path: model)
    loader = MagicMock()
    monkeypatch.setattr(stability, "create_stability_dataloader", loader)
    with pytest.raises(ValueError, match="Cityscapes stability requires"):
        stability.run_stability_evaluation(cfg)
    loader.assert_not_called()


def test_runner_rejects_inconsistent_ppnet_classifier():
    model = DummyPPNet()
    model.last_layer = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="classifier dimensions"):
        stability.run_stability(model, [])


def test_runner_rejects_malformed_reproseg_classifier():
    model = DummyReProSeg()
    model.layers.classification_layer = torch.nn.Conv2d(1, 3, 3)
    with pytest.raises(ValueError, match="classifier must have shape"):
        stability.run_stability(model, [])


def test_runner_rejects_unsupported_model():
    with pytest.raises(TypeError, match="PPNet or ReProSeg"):
        stability.run_stability(torch.nn.Identity(), [])


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_image_limit_rejected(limit):
    with pytest.raises(ValueError):
        StabilityEvalConfig(max_images=limit)
