"""Unit tests for the base Dataset class."""

from unittest.mock import patch

import pytest
import torch

from data.dataset.base import Dataset
from data.config import DATASETS


class TestBaseDataset:
    """Test cases for the base Dataset class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.dataset_name = "CityScapes"
        self.split = "train"

    def test_init_valid_dataset(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test Dataset initialization with valid parameters."""
        dataset = Dataset(self.dataset_name, self.split)

        assert dataset.name == self.dataset_name
        assert dataset.split == self.split
        assert dataset.config == DATASETS[self.dataset_name]
        assert dataset.dataset is not None
        assert dataset.transforms is not None

    def test_init_invalid_dataset(self, mock_env_data_root):
        """Test Dataset initialization with invalid dataset name."""
        with patch("torchvision.datasets.Cityscapes"):
            try:
                Dataset("InvalidDataset", self.split)  # type: ignore
                raise AssertionError("Should have raised NotImplementedError")
            except NotImplementedError as e:
                assert "'InvalidDataset'" in str(e)
                assert "not implemented" in str(e)

    def test_supported_dataset_literal(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that the SupportedDataset literal type works correctly."""
        # This should work without any issues
        dataset = Dataset("CityScapes", self.split)
        assert dataset.name == "CityScapes"

    def test_getitem(self, mock_env_data_root, mock_cityscapes_constructor, sample_image, sample_target):
        """Test __getitem__ method."""
        dataset = Dataset(self.dataset_name, self.split)

        result = dataset[0]

        dataset.dataset.__getitem__.assert_called_once_with(0)
        assert len(result) == 2
        assert result[0] == sample_image
        assert result[1] == sample_target

    def test_len(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test __len__ method."""
        dataset = Dataset(self.dataset_name, self.split)

        # Mock the underlying dataset's __len__
        dataset.dataset.__len__.return_value = 100

        assert len(dataset) == 100
        dataset.dataset.__len__.assert_called_once()

    def test_getdata_cityscapes(self, mock_env_data_root, mock_cityscapes_constructor, temp_data_dir):
        """Test __getdata__ method for CityScapes dataset."""
        _dataset = Dataset(self.dataset_name, self.split)

        # Verify that Cityscapes was called once with correct basic parameters
        mock_cityscapes_constructor.assert_called_once()
        call_kwargs = mock_cityscapes_constructor.call_args.kwargs

        assert call_kwargs["split"] == self.split
        assert call_kwargs["mode"] == "fine"
        assert call_kwargs["target_type"] == "semantic"
        assert call_kwargs["transform"] is not None
        assert call_kwargs["target_transform"] is not None

    def test_classes_property(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test classes property."""
        dataset = Dataset(self.dataset_name, self.split)

        mock_classes = ["class1", "class2", "class3"]
        dataset.dataset.classes = mock_classes

        assert dataset.classes == mock_classes

    def test_transform_property(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test transform property."""
        dataset = Dataset(self.dataset_name, self.split)

        transform = dataset.transform
        assert transform is not None
        assert callable(transform)  # Should be callable

    def test_target_transform_property(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test target_transform property."""
        dataset = Dataset(self.dataset_name, self.split)

        target_transform = dataset.target_transform
        assert target_transform is not None
        assert callable(target_transform)  # Should be callable

    def test_transforms_object_creation(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that Transforms object is created correctly."""
        dataset = Dataset(self.dataset_name, self.split)

        transforms = dataset.transforms
        assert transforms is not None
        assert hasattr(transforms, "base_image")
        assert hasattr(transforms, "image_normalization")
        assert hasattr(transforms, "base_target")
        assert hasattr(transforms, "geometry_augmentation")
        assert hasattr(transforms, "color_augmentation")
        assert hasattr(transforms, "shrink_target")

    def test_dataset_config_access(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that dataset configuration is accessible."""
        dataset = Dataset(self.dataset_name, self.split)

        config = dataset.config
        assert config is not None
        assert "data_dir" in config
        assert "color_channels" in config
        assert "img_shape" in config
        assert "mean" in config
        assert "std" in config

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_multiple_splits(self, split, mock_env_data_root, mock_cityscapes_constructor):
        """Test Dataset initialization with different splits."""
        dataset = Dataset(self.dataset_name, split)
        assert dataset.split == split

    def test_transform_integration(self, mock_env_data_root, mock_cityscapes_constructor, sample_image):
        """Test that transforms can be applied to sample data."""
        dataset = Dataset(self.dataset_name, self.split)

        transformed_image = dataset.transform(sample_image)

        assert transformed_image is not None
        assert isinstance(transformed_image, torch.Tensor)

    def test_target_transform_integration(self, mock_env_data_root, mock_cityscapes_constructor, sample_target):
        """Test that target transforms can be applied to sample data."""
        dataset = Dataset(self.dataset_name, self.split)

        transformed_target = dataset.target_transform(sample_target)

        assert transformed_target is not None
        assert isinstance(transformed_target, torch.Tensor)

    def test_class_filtering_called(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that filter_cityscapes_classes is called during dataset creation."""
        dataset = Dataset(self.dataset_name, self.split)

        # Verify that filter_cityscapes_classes was called
        with patch.object(dataset.transforms, "filter_cityscapes_classes", return_value=[]) as mock_filter:
            # Call __getdata__ explicitly to test the filtering
            dataset.__getdata__()
            mock_filter.assert_called_once()

    def test_filtered_classes_assigned_to_dataset(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that filtered classes are properly assigned to the dataset."""
        # Mock the filter_cityscapes_classes method to return specific classes
        mock_filtered_classes = ["filtered_class_1", "filtered_class_2"]

        with patch("data.transforms.Transforms.filter_cityscapes_classes", return_value=mock_filtered_classes):
            dataset = Dataset(self.dataset_name, self.split)

            # Verify that the filtered classes were assigned to the underlying dataset
            assert dataset.dataset.classes == mock_filtered_classes

    def test_class_filtering_with_ignores(self, mock_env_data_root, mock_cityscapes_classes, mock_cityscapes_dataset):
        """Test class filtering correctly handles ignore_in_eval flag."""
        # Set the classes on our existing mock dataset
        mock_cityscapes_dataset.classes = mock_cityscapes_classes

        with patch("data.dataset.base.Cityscapes") as mock_cityscapes_constructor:
            mock_cityscapes_constructor.return_value = mock_cityscapes_dataset

            # Mock the filter_cityscapes_classes to return expected filtered classes
            with patch("data.transforms.Transforms.filter_cityscapes_classes") as mock_filter:
                # From our mock_cityscapes_classes fixture:
                # Classes with ignore_in_eval=True: "unlabeled", "wall", "fence" (3 classes)
                # Classes with ignore_in_eval=False: "road", "sidewalk", "building", "vegetation", "sky" (5 classes)
                # Expected: first class + non-ignored = unlabeled + 5 non-ignored = 6 total
                non_ignored_classes = [c for c in mock_cityscapes_classes if not c.ignore_in_eval]
                expected_filtered_classes = [mock_cityscapes_classes[0]] + non_ignored_classes
                mock_filter.return_value = expected_filtered_classes

                dataset = Dataset(self.dataset_name, self.split)

                # Verify the filtered classes were assigned
                assert dataset.dataset.classes == expected_filtered_classes

    def test_class_filtering_integration_with_transforms(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that class filtering integrates properly with transforms."""
        dataset = Dataset(self.dataset_name, self.split)

        # Verify that transforms object has the filter method
        assert hasattr(dataset.transforms, "filter_cityscapes_classes")
        assert callable(dataset.transforms.filter_cityscapes_classes)

        # Verify that the method can be called without error
        filtered_classes = dataset.transforms.filter_cityscapes_classes()
        assert filtered_classes is not None

    def test_class_filtering_updates_base_target_transform(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that class filtering updates the base_target transform."""
        dataset = Dataset(self.dataset_name, self.split)

        # Store original transform
        original_transform = dataset.transforms.base_target

        # Call filter_cityscapes_classes
        dataset.transforms.filter_cityscapes_classes()

        # Verify that base_target transform has been updated (composition should be different)
        updated_transform = dataset.transforms.base_target
        assert updated_transform is not None
        # The transforms should be different objects after filtering
        assert id(original_transform) != id(updated_transform)

    def test_class_filtering_preserves_other_dataset_properties(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that class filtering doesn't interfere with other dataset properties."""
        dataset = Dataset(self.dataset_name, self.split)

        # Verify that other properties are still accessible
        assert dataset.name == self.dataset_name
        assert dataset.split == self.split
        assert dataset.config == DATASETS[self.dataset_name]
        assert dataset.transforms is not None

        # Verify that the dataset can still be indexed
        result = dataset[0]
        assert result is not None
        assert len(result) == 2
