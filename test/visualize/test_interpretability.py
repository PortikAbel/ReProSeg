"""Unit tests for the ModelInterpretability class methods."""

from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import torch

from visualize.interpretability import ModelInterpretability


class TestModelInterpretabilityMethods:
    """Test cases for ModelInterpretability private methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock objects
        self.mock_net = MagicMock()
        self.mock_net.num_prototypes = 5
        self.mock_log = MagicMock()

    def _create_interpretability_instance(self, consistency_threshold=0.7):
        """Helper to create ModelInterpretability instance with custom threshold."""
        from config import ReProSegConfig
        from config.schema.evaluation import ConsistencyScoreConfig, EvaluationConfig

        mock_config = ReProSegConfig(
            evaluation=EvaluationConfig(
                consistency_score=ConsistencyScoreConfig(calculate=True, threshold=consistency_threshold)
            )
        )

        return ModelInterpretability(net=self.mock_net, cfg=mock_config, log=self.mock_log)

    def sort_function(self, x):
        return x[0]

    @pytest.mark.parametrize(
        "alpha_values, part_labels, expected_results",
        [
            # Test case 1: Simple 2x2 tensors with two distinct parts
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1, 1], [2, 2]]),
                [
                    (1, 1.5),  # part 1: (1.0 + 2.0) / 2 = 1.5
                    (2, 3.5),  # part 2: (3.0 + 4.0) / 2 = 3.5
                ],
            ),
            # Test case 2: Mixed parts with zeros (unlabeled)
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[0, 1], [1, 2]]),
                [
                    (1, 2.5),  # part 1: (2.0 + 3.0) / 2 = 2.5,
                    (2, 4.0),  # part 2: 4.0 / 1 = 4.0
                ],
            ),
            # Test case 3: Single part
            (
                torch.tensor([[5.0, 10.0], [15.0, 20.0]]),
                torch.tensor([[3, 3], [3, 3]]),
                [(3, 12.5)],  # part 3: (5.0 + 10.0 + 15.0 + 20.0) / 4 = 12.5
            ),
            # Test case 4: All zeros (unlabeled) - should return empty
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[0, 0], [0, 0]]),
                [],  # No labeled parts
            ),
            # Test case 5: Larger tensor with multiple parts
            (
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
                torch.tensor([[1, 1, 2], [1, 2, 2], [3, 3, 3]]),
                [
                    (1, 7 / 3),  # part 1: (1+2+4)/3
                    (2, 14 / 3),  # part 2: (3+5+6)/3
                    (3, 24 / 3),  # part 3: (7+8+9)/3
                ],
            ),
            # Test case 6: Parts with single pixel
            (
                torch.tensor([[7.5, 2.1], [9.3, 4.8]]),
                torch.tensor([[1, 2], [3, 4]]),  # Each pixel is a different part
                [(1, 7.5), (2, 2.1), (3, 9.3), (4, 4.8)],
            ),
            # Test case 7: floating point precision
            (
                torch.tensor([[0.12345678, 0.87654321], [0.11111111, 0.99999999]]),
                torch.tensor([[1, 1], [1, 2]]),
                [(1, (0.12345678 + 0.87654321 + 0.11111111) / 3), (2, 0.99999999)],
            ),
        ],
    )
    def test_compute_part_activation_averages(self, alpha_values, part_labels, expected_results):
        """Test _compute_part_activation_averages with various input configurations."""
        interpretability = self._create_interpretability_instance()
        result = list(interpretability._compute_part_activation_averages(alpha_values, part_labels))

        # Sort both results by part label for consistent comparison
        result = sorted(result, key=self.sort_function)
        expected_results = sorted(expected_results, key=self.sort_function)

        assert len(result) == len(expected_results)

        for (actual_label, actual_avg), (expected_label, expected_avg) in zip(result, expected_results, strict=False):
            assert actual_label == expected_label
            assert torch.isclose(torch.tensor(actual_avg), torch.tensor(expected_avg), atol=1e-6)

    @pytest.mark.parametrize(
        "part_activations, consistency_threshold, expected",
        [
            # Test case 1: All prototypes consistent
            (
                [
                    {1: [0.8, 0.9, 0.75], 2: [0.6, 0.7, 0.65]},  # prototype 0: max avg = 0.8167 > 0.7
                    {3: [0.85, 0.8, 0.9], 4: [0.5, 0.6, 0.55]},  # prototype 1: max avg = 0.85 > 0.7
                    {1: [0.72, 0.78, 0.74], 5: [0.4, 0.3, 0.2]},  # prototype 2: max avg = 0.7467 > 0.7
                ],
                0.7,
                [True, True, True],
            ),
            # Test case 2: No prototypes consistent
            (
                [
                    {1: [0.5, 0.6, 0.4], 2: [0.3, 0.4, 0.2]},  # prototype 0: max avg = 0.5 < 0.7
                    {3: [0.6, 0.65, 0.55], 4: [0.1, 0.2, 0.15]},  # prototype 1: max avg = 0.6 < 0.7
                    {1: [0.45, 0.35, 0.25], 5: [0.0, 0.1, 0.05]},  # prototype 2: max avg = 0.35 < 0.7
                ],
                0.7,
                [False, False, False],
            ),
            # Test case 3: Mixed consistency
            (
                [
                    {1: [0.8, 0.9, 0.7]},  # prototype 0: avg = 0.8 > 0.7 -> True
                    {2: [0.5, 0.6, 0.4]},  # prototype 1: avg = 0.5 < 0.7 -> False
                    {3: [0.75, 0.72, 0.73]},  # prototype 2: avg = 0.7333 > 0.7 -> True
                    {4: [0.6, 0.65]},  # prototype 3: avg = 0.625 < 0.7 -> False
                ],
                0.7,
                [True, False, True, False],
            ),
            # Test case 4: Empty activations
            (
                [
                    {},  # prototype 0: no activations -> False
                    {1: []},  # prototype 1: empty list -> False (np.mean([]) = nan)
                    {2: [0.8, 0.9]},  # prototype 2: avg = 0.85 > 0.7 -> True
                ],
                0.7,
                [False, False, True],
            ),
            # Test case 5: Single value lists
            (
                [
                    {1: [0.8]},  # prototype 0: single value 0.8 > 0.7 -> True
                    {2: [0.6]},  # prototype 1: single value 0.6 < 0.7 -> False
                    {3: [0.75]},  # prototype 2: single value 0.75 > 0.7 -> True
                ],
                0.7,
                [True, False, True],
            ),
        ],
    )
    def test_compute_if_prototype_consistent(self, part_activations, consistency_threshold, expected):
        """Test _compute_if_prototype_consistent with various activation patterns."""
        # Set up the interpretability instance with the test data
        interpretability = self._create_interpretability_instance(consistency_threshold)
        interpretability._part_activations = [defaultdict(list) for _ in range(len(part_activations))]

        # Populate the part activations
        for i, prototype_activations in enumerate(part_activations):
            for part_label, activation_list in prototype_activations.items():
                interpretability._part_activations[i][part_label] = activation_list

        result = interpretability._compute_if_prototype_consistent()

        assert result == expected

    def test_compute_if_prototype_consistent_multiple_parts_per_prototype(self):
        """Test _compute_if_prototype_consistent where prototypes activate on multiple parts."""
        interpretability = self._create_interpretability_instance(0.7)
        # Prototype has multiple parts: one above threshold, one below
        interpretability._part_activations = [
            {
                1: [0.6, 0.65, 0.55],  # avg = 0.6 < 0.7
                2: [0.8, 0.85, 0.75],  # avg = 0.8 > 0.7 -> should make prototype consistent
                3: [0.4, 0.3, 0.5],  # avg = 0.4 < 0.7
            },
        ]

        result = interpretability._compute_if_prototype_consistent()

        # Should be True because part 2 has average > threshold
        assert result == [True]

    def test_compute_if_prototype_consistent_high_threshold(self):
        """Test _compute_if_prototype_consistent with a high consistency threshold."""
        interpretability = self._create_interpretability_instance(0.95)
        interpretability._part_activations = [
            {1: [0.9, 0.92, 0.88]},  # avg = 0.9 < 0.95 -> False
            {2: [0.96, 0.98, 0.94]},  # avg = 0.96 > 0.95 -> True
        ]

        result = interpretability._compute_if_prototype_consistent()

        assert result == [False, True]

    def test_compute_if_prototype_consistent_with_nan_values(self):
        """Test _compute_if_prototype_consistent handles NaN values gracefully."""
        interpretability = self._create_interpretability_instance(0.7)
        # This test covers the edge case where np.mean([]) returns nan
        interpretability._part_activations = [
            {1: []},  # Empty list -> np.mean([]) = nan
            {2: [0.8, 0.9]},  # Normal case
        ]

        result = interpretability._compute_if_prototype_consistent()

        # Empty list should result in False (nan is not > threshold)
        assert result == [False, True]

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
        ],
    )
    def test_compute_part_activation_averages_device_compatibility(self, device):
        """Test _compute_part_activation_averages works on different devices."""
        interpretability = self._create_interpretability_instance()
        alpha = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(device)
        pps = torch.tensor([[1, 1], [2, 2]]).to(device)

        result = list(interpretability._compute_part_activation_averages(alpha, pps))
        result = sorted(result, key=self.sort_function)

        expected = [(1, 1.5), (2, 3.5)]

        assert len(result) == len(expected)
        for (actual_label, actual_avg), (expected_label, expected_avg) in zip(result, expected, strict=False):
            assert actual_label == expected_label
            assert abs(actual_avg - expected_avg) < 1e-6

    def test_compute_part_activation_averages_large_tensor(self):
        """Test _compute_part_activation_averages with larger tensors."""
        interpretability = self._create_interpretability_instance()
        alpha = torch.rand(10, 10)

        # Create parts: 4 quadrants
        pps = torch.zeros(10, 10, dtype=torch.long)
        pps[:5, :5] = 1  # Top-left quadrant
        pps[:5, 5:] = 2  # Top-right quadrant
        pps[5:, :5] = 3  # Bottom-left quadrant
        pps[5:, 5:] = 4  # Bottom-right quadrant

        result = list(interpretability._compute_part_activation_averages(alpha, pps))
        result = sorted(result, key=self.sort_function)

        # Verify we got 4 parts
        assert len(result) == 4

        # Verify labels are correct
        labels = [r[0] for r in result]
        assert labels == [1, 2, 3, 4]

        # Manually verify one quadrant
        expected_avg_part1 = alpha[:5, :5].mean()
        actual_avg_part1 = result[0][1]  # Part 1 is first after sorting
        assert torch.isclose(torch.tensor(actual_avg_part1), expected_avg_part1, atol=1e-6)
