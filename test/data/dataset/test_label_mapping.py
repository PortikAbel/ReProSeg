"""Unit tests for the LabelMapping class."""

import pytest
import torch
import torchvision

from config.schema.data import DatasetType
from data.dataset.label_mapping import LabelMapping


class TestLabelMapping:
    """Test cases for the LabelMapping class."""

    def test_get_cityscapes_classes(self):
        """Test filtering removes ignored classes but keeps unlabeled."""

        filtered = LabelMapping._get_cityscapes_classes()

        assert filtered[0].name == "unlabeled"
        assert all(not c.ignore_in_eval for c in filtered[1:])

    @pytest.mark.parametrize("dataset_type", list(DatasetType))
    def test_get_classes_returns_list(self, dataset_type):
        """Test get_classes returns a non-empty list for each dataset type."""

        classes = LabelMapping.get_classes(dataset_type)

        assert isinstance(classes, list)
        assert len(classes) > 0

    @pytest.mark.parametrize("dataset_type", list(DatasetType))
    def test_get_mapping_returns_compose(self, dataset_type):
        """Test that get_mapping returns a Compose transform for each dataset type."""

        transform = LabelMapping.get_mapping(dataset_type)

        assert transform is not None
        assert callable(transform)
        assert isinstance(transform, torchvision.transforms.v2.Compose)

    def test_get_mapping_maps_labels_correctly(self, label_mapping_case):
        """Test that the label mapping transform remaps labels correctly for each dataset."""

        transform, sample_input, expected = label_mapping_case

        result = transform(sample_input)

        assert isinstance(result, torch.Tensor)
        assert result.shape == expected.shape
        assert torch.equal(result, expected)
