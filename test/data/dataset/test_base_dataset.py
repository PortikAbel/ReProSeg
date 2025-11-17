"""Unit tests for the base Dataset class."""


import pytest
import torch
from torch.utils.data import Dataset as TorchDataset

from config.schema.data import DatasetType
from data import Dataset, DataSplit
from data.dataset.transform_set import TransformSet


class TestBaseDataset:
    """Test cases for the base Dataset class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.dataset_name = DatasetType.CITYSCAPES
        self.split = DataSplit.TRAIN

    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_cityscapes_constructor):
        self.dataset = Dataset(mock_config.data)

    def test_init_valid_dataset(self):
        """Test Dataset initialization with valid parameters."""

        assert self.dataset.dataset_type == self.dataset_name
        assert isinstance(self.dataset.dataset, TorchDataset)
        assert isinstance(self.dataset.transform_set, TransformSet)

    def test_supported_dataset_literal(self):
        """Test that the SupportedDataset literal type works correctly."""
        # This should work without any issues
        assert self.dataset.dataset_type == DatasetType.CITYSCAPES

    def test_getitem(self, sample_image, sample_target):
        """Test __getitem__ method."""

        result = self.dataset[0]

        self.dataset.dataset.__getitem__.assert_called_once_with(0)
        assert len(result) == 2
        # Verify that transforms were applied - results should be tensors
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

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

        transforms = self.dataset.transform_set
        assert transforms is not None
        assert hasattr(transforms, "base_image")
        assert hasattr(transforms, "image_normalization")
        assert hasattr(transforms, "base_target")
        assert hasattr(transforms, "geometry_augmentation")
        assert hasattr(transforms, "color_augmentation")
        assert hasattr(transforms, "shrink_target")

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
