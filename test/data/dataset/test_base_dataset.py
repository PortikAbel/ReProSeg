"""Unit tests for the base Dataset class."""

from unittest.mock import patch

import pytest
import torch
from torchvision.transforms.v2 import Compose, Transform

from config.schema.data import DatasetType
from data import SupportedSplit
from data.dataset.base import Dataset


class TestBaseDataset:
    """Test cases for the base Dataset class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.dataset_name = DatasetType.CITYSCAPES
        self.split = SupportedSplit.TRAIN

    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_cityscapes_constructor):
        self.dataset = Dataset(mock_config.data, self.split)

    @staticmethod
    def create_mock_filter_classes_method(filtered_classes):
        """Create a mock filter classes method that sets both classes and filter_classes attributes."""

        def mock_filter_classes_method(self):
            self.classes = filtered_classes
            self.filter_classes = Transform()
            return filtered_classes

        return mock_filter_classes_method

    def test_init_valid_dataset(self):
        """Test Dataset initialization with valid parameters."""

        assert self.dataset.config.dataset == self.dataset_name
        assert self.dataset.split == self.split
        assert self.dataset.dataset is not None
        assert self.dataset.transforms is not None

    def test_supported_dataset_literal(self):
        """Test that the SupportedDataset literal type works correctly."""
        # This should work without any issues
        assert self.dataset.config.dataset == DatasetType.CITYSCAPES

    def test_getitem(self, sample_image, sample_target):
        """Test __getitem__ method."""

        result = self.dataset[0]

        self.dataset.dataset.__getitem__.assert_called_once_with(0)
        assert len(result) == 2
        assert result[0] == sample_image
        assert result[1] == sample_target

    def test_len(self):
        """Test __len__ method."""

        # Mock the underlying self.dataset's __len__
        self.dataset.dataset.__len__.return_value = 100

        assert len(self.dataset) == 100
        self.dataset.dataset.__len__.assert_called_once()

    def test_getdata_cityscapes(self, mock_cityscapes_constructor):
        """Test __getdata__ method for CityScapes self.dataset."""

        # Verify that Cityscapes was called once with correct basic parameters
        mock_cityscapes_constructor.assert_called_once()
        call_kwargs = mock_cityscapes_constructor.call_args.kwargs

        assert call_kwargs["split"] == self.split
        assert call_kwargs["mode"] == "fine"
        assert call_kwargs["target_type"] == "semantic"
        assert call_kwargs["transforms"] is not None

    def test_classes_property(self):
        """Test classes property."""

        mock_classes = ["class1", "class2", "class3"]
        self.dataset.dataset.classes = mock_classes

        assert self.dataset.classes == mock_classes

    def test_transform_property(self):
        """Test transform property."""

        transform = self.dataset.transform
        assert transform is not None
        assert callable(transform)  # Should be callable

    def test_target_transform_property(self):
        """Test target_transform property."""

        target_transform = self.dataset.target_transform
        assert target_transform is not None
        assert callable(target_transform)  # Should be callable

    def test_transforms_object_creation(self):
        """Test that Transforms object is created correctly."""

        transforms = self.dataset.transforms
        assert transforms is not None
        assert hasattr(transforms, "base_image")
        assert hasattr(transforms, "image_normalization")
        assert hasattr(transforms, "base_target")
        assert hasattr(transforms, "geometry_augmentation")
        assert hasattr(transforms, "color_augmentation")
        assert hasattr(transforms, "shrink_target")

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_multiple_splits(self, split, mock_config, mock_cityscapes_constructor):
        """Test Dataset initialization with different splits."""
        dataset = Dataset(mock_config.data, split)
        assert dataset.split == split

    def test_transform_integration(self, sample_image):
        """Test that transforms can be applied to sample data."""

        transformed_image = self.dataset.transform(sample_image)

        assert transformed_image is not None
        assert isinstance(transformed_image, torch.Tensor)

    def test_target_transform_integration(self, sample_target):
        """Test that target transforms can be applied to sample data."""

        transformed_target = self.dataset.target_transform(sample_target)

        assert transformed_target is not None
        assert isinstance(transformed_target, torch.Tensor)

    def test_filtered_classes_assigned_to_dataset(self, mock_config):
        """Test that filtered classes are properly assigned to the self.dataset."""
        # Mock the _filter_cityscapes_classes method to return specific classes
        mock_filtered_classes = ["filtered_class_1", "filtered_class_2"]

        mock_filter_method = self.create_mock_filter_classes_method(mock_filtered_classes)

        with patch("data.transforms.Transforms._filter_cityscapes_classes", mock_filter_method):
            dataset = Dataset(mock_config.data, self.split)

            # Verify that the filtered classes were assigned to the underlying dataset
            assert dataset.dataset.classes == mock_filtered_classes

    def test_class_filtering_with_ignores(self, mock_config, mock_cityscapes_classes, mock_cityscapes_dataset):
        """Test class filtering correctly handles ignore_in_eval flag."""
        # Set the classes on our existing mock dataset
        mock_cityscapes_dataset.classes = mock_cityscapes_classes

        non_ignored_classes = [c for c in mock_cityscapes_classes if not c.ignore_in_eval]
        expected_filtered_classes = [mock_cityscapes_classes[0]] + non_ignored_classes
        mock_filter_method = self.create_mock_filter_classes_method(expected_filtered_classes)

        with (
            patch("data.dataset.base.Cityscapes") as mock_cityscapes_constructor,
            patch("data.transforms.Transforms._filter_cityscapes_classes", mock_filter_method),
        ):
            mock_cityscapes_constructor.return_value = mock_cityscapes_dataset

            dataset = Dataset(mock_config.data, self.split)

            # Verify the filtered classes were assigned
            assert dataset.dataset.classes == expected_filtered_classes

    def test_target_transform_contains_class_filtering(self):
        """Test that class filtering updates the target transform."""

        # Verify that target_transform contains the filter_classes transform
        assert isinstance(self.dataset.target_transform, Compose)
        assert self.dataset.target_transform.transforms[0] == self.dataset.transforms.base_target
        assert self.dataset.target_transform.transforms[1] == self.dataset.transforms.filter_classes

    def test_class_filtering_preserves_other_dataset_properties(self, mock_config):
        """Test that class filtering doesn't interfere with other self.dataset properties."""

        # Verify that other properties are still accessible
        assert self.dataset.config == mock_config.data
        assert self.dataset.split == self.split
        assert self.dataset.transforms is not None

        # Verify that the dataset can still be indexed
        result = self.dataset[0]
        assert result is not None
        assert len(result) == 2
