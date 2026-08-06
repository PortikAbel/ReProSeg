"""
Integration tests for dataset classes.

These tests verify that the dataset classes work together correctly
and can handle real-world scenarios.
"""

import pytest

from config.schema.data import DatasetType
from data import Dataset, DoubleAugmentDataset


class TestDatasetIntegration:
    """Integration tests for dataset classes."""

    @pytest.fixture(
        params=[
            pytest.param(DatasetType.CITYSCAPES, id="cityscapes"),
            pytest.param(DatasetType.VOC_SEGMENTATION, id="pascal_voc"),
        ],
        autouse=True,
    )
    def setup(self, request, mock_config, mock_cityscapes_constructor, mock_voc_constructor):
        """Run before each test method to create a fresh dataset instance."""
        self.dataset_type = request.param
        mock_config.data.dataset = request.param
        self.base_dataset = Dataset(mock_config.data)
        self.double_augment_dataset = DoubleAugmentDataset(mock_config.data, self.base_dataset.dataset)

    def test_base_and_double_augment_same_config(self):
        """Test that base and double augment datasets use the same config."""

        assert self.base_dataset.dataset_type == self.double_augment_dataset.dataset_type
        assert self.base_dataset.dataset == self.double_augment_dataset.dataset

    def test_transforms_object_consistency(self):
        """Test that transforms objects are consistent between datasets."""

        # Both should have the same transforms object structure
        base_transforms = self.base_dataset.transform_set
        double_transforms = self.double_augment_dataset.transform_set

        assert type(base_transforms) is type(double_transforms)

        # The individual transforms should be the same type or equivalent
        assert type(base_transforms.base_image) is type(double_transforms.base_image)
        assert type(base_transforms.base_target) is type(double_transforms.base_target)
        assert type(base_transforms.geometry_augmentation) is type(double_transforms.geometry_augmentation)

    def test_dataset_compatibility(self, sample_image, sample_target):
        """Test that both dataset types can work with the same data."""

        # Mock both datasets to return the same sample data
        self.base_dataset.dataset.__getitem__.return_value = (sample_image, sample_target)
        self.double_augment_dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        # Both should be able to access the same index
        base_result = self.base_dataset[0]
        double_result = self.double_augment_dataset[0]

        # Base dataset returns 2 items, double augment returns 3
        assert len(base_result) == 2
        assert len(double_result) == 3

    def test_length_consistency(self):
        """Test that both dataset types report the same length."""

        # Mock both to return the same length
        self.base_dataset.dataset.__len__.return_value = 100
        self.double_augment_dataset.dataset.__len__.return_value = 100

        assert len(self.base_dataset) == len(self.double_augment_dataset)

    def test_classes_consistency(self):
        """Test that base and double augment datasets report the same classes."""

        assert self.base_dataset.classes == self.double_augment_dataset.classes
