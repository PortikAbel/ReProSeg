from collections import defaultdict
from io import StringIO
from unittest.mock import MagicMock

import pytest
import torch

from config import ReProSegConfig
from config.schema.evaluation import ConsistencyScoreConfig, EvaluationConfig
from visualize.interpretability_protopnet import ModelInterpretability


class TestProtoPNetInterpretability:
    def setup_method(self):
        self.net = MagicMock()
        self.net.num_prototypes = 5
        self.net.last_layer.weight = torch.nn.Parameter(torch.ones(2, 5))
        self.log = MagicMock()
        self.log.tqdm_file = StringIO()
        cfg = ReProSegConfig(
            evaluation=EvaluationConfig(consistency_score=ConsistencyScoreConfig(calculate=True, threshold=0.7))
        )
        self.interpretability = ModelInterpretability(self.net, cfg, self.log)

    def test_get_used_prototypes_requires_positive_class_connection(self):
        self.net.last_layer.weight = torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, -1.0, 0.0],
                    [-0.5, 0.0, -2.0, 0.3],
                ]
            )
        )

        assert self.interpretability._get_used_prototypes() == [0, 3]

    def test_compute_score_uses_only_used_prototypes(self):
        weights = torch.full((2, 5), -1.0)
        weights[0, 1] = 1.0
        weights[1, 4] = 1.0
        self.net.last_layer.weight = torch.nn.Parameter(weights)
        self.interpretability._collect_prototype_activations_by_object_parts = MagicMock()
        self.interpretability._compute_if_prototype_consistent = MagicMock(
            return_value=[False, True, True, True, False]
        )

        result = self.interpretability.compute_prototype_consistency_score(MagicMock())

        assert result == 0.5

    def test_compute_score_without_used_prototypes_returns_zero(self):
        self.net.last_layer.weight = torch.nn.Parameter(torch.zeros(2, 5))
        self.interpretability._collect_prototype_activations_by_object_parts = MagicMock()
        self.interpretability._compute_if_prototype_consistent = MagicMock(return_value=[False] * 5)

        result = self.interpretability.compute_prototype_consistency_score(MagicMock())

        assert result == 0.0
        self.log.warning.assert_called_once()

    def test_interpolate_prototype_activations_uses_spatial_distances(self):
        distances = torch.rand(2, 5, 2, 3)
        activations = torch.rand(2, 5, 2, 3)
        self.net.push_forward.return_value = (torch.rand(2, 4, 2, 3), distances)
        self.net.distance_2_similarity.return_value = activations

        result = self.interpretability._interpolate_prototype_activations(
            torch.rand(2, 3, 8, 12),
            output_size=(8, 12),
        )

        assert result.shape == (2, 5, 8, 12)
        self.net.distance_2_similarity.assert_called_once_with(distances)

    def test_interpolate_prototype_activations_rejects_negative_similarity(self):
        distances = torch.rand(1, 5, 2, 2)
        self.net.push_forward.return_value = (torch.rand(1, 4, 2, 2), distances)
        self.net.distance_2_similarity.return_value = -distances

        with pytest.raises(ValueError, match="non-negative prototype activations"):
            self.interpretability._interpolate_prototype_activations(
                torch.rand(1, 3, 4, 4),
                output_size=(4, 4),
            )

    def test_collection_handles_mask_channel_and_only_used_prototypes(self):
        weights = torch.zeros(2, 5)
        weights[0, 0] = 1.0
        weights[1, 3] = 1.0
        self.net.last_layer.weight = torch.nn.Parameter(weights)
        self.interpretability._interpolate_prototype_activations = MagicMock(return_value=torch.ones(2, 5, 4, 4))
        xs = torch.rand(2, 3, 4, 4)
        ys = torch.full((2, 1, 4, 4), 11)
        pps = torch.ones(2, 1, 4, 4, dtype=torch.long)

        self.interpretability._collect_prototype_activations_by_object_parts([(xs, ys, pps)])

        assert self.interpretability._part_activations[0][1] == [1.0]
        assert self.interpretability._part_activations[3][1] == [1.0]
        assert self.interpretability._part_activations[1] == defaultdict(list)
        assert self.interpretability._part_activations[2] == defaultdict(list)
        assert self.interpretability._part_activations[4] == defaultdict(list)

    def test_part_activation_averages_reject_shape_mismatch(self):
        with pytest.raises(ValueError, match="must have matching shapes"):
            self.interpretability._compute_part_activation_averages(
                torch.ones(2, 4, 4),
                torch.ones(2, 1, 4, 4, dtype=torch.long),
            )
