"""
Integration tests for dataset classes.

These tests verify that the dataset classes work together correctly
and can handle real-world scenarios.
"""

import pytest

from config.schema.data import DatasetType
from data import AugmentedDataset, Dataset


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
        self.augmented_dataset = AugmentedDataset(mock_config.data, self.base_dataset.dataset)

    def test_base_and_augmented_same_config(self):
        """Test that base and augmented datasets use the same config."""

        assert self.base_dataset.dataset_type == self.augmented_dataset.dataset_type
        assert self.base_dataset.dataset == self.augmented_dataset.dataset

    def test_transforms_object_consistency(self):
        """Test that transforms objects are consistent between datasets."""

        # Both should have the same transforms object structure
        base_transforms = self.base_dataset.transform_set
        augmented_transforms = self.augmented_dataset.transform_set

        assert type(base_transforms) is type(augmented_transforms)

        # The individual transforms should be the same type or equivalent
        assert type(base_transforms.base_image) is type(augmented_transforms.base_image)
        assert type(base_transforms.base_target) is type(augmented_transforms.base_target)
        assert type(base_transforms.geometry_augmentation) is type(augmented_transforms.geometry_augmentation)

    def test_dataset_compatibility(self, sample_image, sample_target):
        """Test that both dataset types can work with the same data."""

        # Mock both datasets to return the same sample data
        self.base_dataset.dataset.__getitem__.return_value = (sample_image, sample_target)
        self.augmented_dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        # Both should be able to access the same index
        base_result = self.base_dataset[0]
        augmented_result = self.augmented_dataset[0]

        # Both datasets return an (image, target) pair
        assert len(base_result) == 2
        assert len(augmented_result) == 2

    def test_length_consistency(self):
        """Test that both dataset types report the same length."""

        # Mock both to return the same length
        self.base_dataset.dataset.__len__.return_value = 100
        self.augmented_dataset.dataset.__len__.return_value = 100

        assert len(self.base_dataset) == len(self.augmented_dataset)

    def test_classes_consistency(self):
        """Test that base and augmented datasets report the same classes."""

        assert self.base_dataset.classes == self.augmented_dataset.classes
