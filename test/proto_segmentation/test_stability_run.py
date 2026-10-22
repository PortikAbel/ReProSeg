from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from config import EvaluateConfig
from config.schema.evaluate import StabilityEvalConfig
from evaluate import stability
from evaluate.registry import run as run_metric
from test.proto_segmentation.test_consistency import DummyPPNet


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


def test_cli_registration_passes_quantile_and_saves_results(monkeypatch, temp_dir):
    cfg = EvaluateConfig()
    cfg.model.checkpoint = Path("push_best.pth")
    cfg.evaluate.stability.quantile = 0.9
    cfg.evaluate.metric_name = "stability"
    model = DummyPPNet()
    model.prototype_class_identity = torch.ones(2, 19)
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
    output_dir = temp_dir / "stability"
    assert (output_dir / "stability_score_qt_0.9_std_0.2_seed_0.txt").read_text() == "1.0\n"
    assert '"num_both_empty_pairs": 1' in (output_dir / "stability_summary_qt_0.9_std_0.2_seed_0.json").read_text()
    assert ",[false],[false],1" in (output_dir / "stability_pairs_qt_0.9_std_0.2_seed_0.csv").read_text()


@pytest.mark.parametrize("quantile", [-0.1, 1.1, float("nan")])
def test_invalid_run_quantile_rejected(quantile):
    with pytest.raises(ValueError):
        StabilityEvalConfig(quantile=quantile)


def test_empty_run_has_no_defined_score():
    with pytest.raises(ValueError, match="no evaluated"):
        stability.run_stability(DummyPPNet(), [], show_progress=False)
