"""Unit tests for the DoubleAugmentDataset class."""

import pytest
import torch

from data import DoubleAugmentDataset


class TestDoubleAugmentDataset:
    """Test cases for the DoubleAugmentDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_cityscapes_constructor, mock_transform_set_constructor):
        """Run before each test method to create a fresh dataset instance."""
        self.dataset = DoubleAugmentDataset(mock_config.data)

    def test_getitem_transforms_sequence(self, sample_image, sample_target):
        """Test __getitem__ method applies transforms in the correct sequence."""

        result = self.dataset[0]

        self.dataset.dataset.__getitem__.assert_called_once_with(0)
        assert len(result) == 3

        self.dataset.transform_set.base_image.assert_called_once_with(sample_image)
        self.dataset.transform_set.base_target.assert_called_once_with(sample_target)
        self.dataset.transform_set.filter_classes.assert_called_once()

        self.dataset.transform_set.geometry_augmentation.assert_called_once()

        self.dataset.transform_set.color_augmentation.assert_called()
        self.dataset.transform_set.image_normalization.assert_called()
        assert self.dataset.transform_set.color_augmentation.call_count == 2
        assert self.dataset.transform_set.image_normalization.call_count == 2

        # Verify that the target shrink transform was applied
        self.dataset.transform_set.shrink_target.assert_called_once()

        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)
        assert isinstance(result[2], torch.Tensor)
