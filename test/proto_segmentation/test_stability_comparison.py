import pytest
import torch

from evaluate.parts import PartPresence
from evaluate.stability import StabilityObserver, compare_part_presence, summarize_stability
from test.proto_segmentation.test_consistency import DummyPPNet


def test_observed_part_change_fails_and_two_empty_vectors_pass():
    clean = torch.zeros(3, 5, 5)
    clean[0, 0, 0] = clean[0, 4, 4] = 1
    noisy = clean.clone()
    noisy[0, 4, 4] = 0
    noisy[0, 2, 2] = 1  # Active on background, not the annotated part.
    semantic = torch.ones(5, 5, dtype=torch.long)
    parts = torch.zeros_like(semantic)
    parts[0, 0] = 2401
    parts[4, 4] = 2402

    results = StabilityObserver(DummyPPNet(), device="cpu").compare_pair(
        clean,
        noisy,
        semantic_mask=semantic,
        part_mask=parts,
        image_index=9,
    )

    assert len(results) == 2
    assert (results[0].image_index, results[0].class_id, results[0].prototype_id) == (9, 1, 0)
    assert results[0].part_ids == (1, 2)
    assert results[0].clean_presence == (True, True)
    assert results[0].noisy_presence == (True, False)
    assert results[0].score == 0
    assert results[1].clean_presence == results[1].noisy_presence == (False, False)
    assert results[1].score == 1


def test_observations_align_by_identity_independent_of_order():
    clean = [
        PartPresence(0, 1, 0, 2, False),
        PartPresence(0, 1, 0, 1, True),
        PartPresence(1, 2, 0, 3, False),
    ]
    noisy = [
        PartPresence(1, 2, 0, 3, False),
        PartPresence(0, 1, 0, 1, True),
        PartPresence(0, 1, 0, 2, False),
    ]
    results = compare_part_presence(clean, noisy)
    assert [(r.image_index, r.class_id, r.prototype_id, r.score) for r in results] == [
        (0, 1, 0, 1),
        (1, 2, 0, 1),
    ]
    assert results[0].part_ids == (1, 2)


def test_missing_data_cannot_be_counted_as_stable():
    clean = [PartPresence(0, 1, 0, 1, False), PartPresence(0, 1, 0, 2, False)]
    assert compare_part_presence([], []) == ()
    with pytest.raises(ValueError, match="identical image/class/prototype pairs"):
        compare_part_presence(clean, [])
    with pytest.raises(ValueError, match="identical annotated parts"):
        compare_part_presence(clean, clean[:1])
    with pytest.raises(ValueError, match="Duplicate part observation"):
        compare_part_presence(clean + clean[:1], clean)


def test_stability_summary_averages_all_image_prototype_pairs_including_empty_vectors():
    comparisons = compare_part_presence(
        [
            PartPresence(0, 1, 0, 1, True),
            PartPresence(0, 1, 1, 1, False),
            PartPresence(1, 1, 0, 1, False),
        ],
        [
            PartPresence(0, 1, 0, 1, True),  # Stable positive vector.
            PartPresence(0, 1, 1, 1, False),  # Stable all-zero vector.
            PartPresence(1, 1, 0, 1, True),  # Changed vector.
        ],
    )

    summary = summarize_stability(comparisons, activation_quantile=0.8)

    assert summary.activation_quantile == 0.8
    assert summary.num_evaluated_pairs == 3
    assert summary.num_stable_pairs == 2
    assert summary.num_both_empty_pairs == 1
    assert summary.score == pytest.approx(2 / 3)


def test_stability_summary_rejects_undefined_empty_score_and_invalid_quantile():
    with pytest.raises(ValueError, match="no evaluated image/prototype pairs"):
        summarize_stability((), activation_quantile=0.8)
    with pytest.raises(ValueError, match="quantile"):
        summarize_stability((), activation_quantile=1.1)
