"""
Integration tests for dataset classes.

These tests verify that the dataset classes work together correctly
and can handle real-world scenarios.
"""

from data.dataset.base import Dataset
from data.dataset.double_augment import DoubleAugmentDataset


class TestDatasetIntegration:
    """Integration tests for dataset classes."""

    def test_base_and_double_augment_same_config(self, mock_config, mock_cityscapes_constructor):
        """Test that base and double augment datasets use the same config."""
        base_dataset = Dataset(mock_config.data, "train")
        double_augment_dataset = DoubleAugmentDataset(mock_config.data)

        assert base_dataset.config == double_augment_dataset.config
        assert base_dataset.config.dataset == double_augment_dataset.config.dataset

    def test_transforms_object_consistency(self, mock_config, mock_cityscapes_constructor):
        """Test that transforms objects are consistent between datasets."""
        base_dataset = Dataset(mock_config.data, "train")
        double_augment_dataset = DoubleAugmentDataset(mock_config.data)

        # Both should have the same transforms object structure
        base_transforms = base_dataset.transforms
        double_transforms = double_augment_dataset.transforms

        assert type(base_transforms) is type(double_transforms)

        # The individual transforms should be the same type or equivalent
        assert type(base_transforms.base_image) is type(double_transforms.base_image)
        assert type(base_transforms.base_target) is type(double_transforms.base_target)
        assert type(base_transforms.geometry_augmentation) is type(double_transforms.geometry_augmentation)

    def test_dataset_compatibility(self, mock_config, mock_cityscapes_constructor, sample_image, sample_target):
        """Test that both dataset types can work with the same data."""
        base_dataset = Dataset(mock_config.data, "train")
        double_augment_dataset = DoubleAugmentDataset(mock_config.data)

        # Mock both datasets to return the same sample data
        base_dataset.dataset.__getitem__.return_value = (sample_image, sample_target)
        double_augment_dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        # Both should be able to access the same index
        base_result = base_dataset[0]
        double_result = double_augment_dataset[0]

        # Base dataset returns 2 items, double augment returns 3
        assert len(base_result) == 2
        assert len(double_result) == 3

    def test_length_consistency(self, mock_config, mock_cityscapes_constructor):
        """Test that both dataset types report the same length."""
        base_dataset = Dataset(mock_config.data, "train")
        double_augment_dataset = DoubleAugmentDataset(mock_config.data)

        # Mock both to return the same length
        base_dataset.dataset.__len__.return_value = 100
        double_augment_dataset.dataset.__len__.return_value = 100

        assert len(base_dataset) == len(double_augment_dataset)

    def test_classes_consistency(self, mock_config, mock_cityscapes_constructor):
        """Test that both dataset types report the same classes."""
        base_dataset = Dataset(mock_config.data, "train")
        double_augment_dataset = DoubleAugmentDataset(mock_config.data)

        assert base_dataset.classes == double_augment_dataset.classes
