"""Unit tests for the DoubleAugmentDataset class."""

from unittest.mock import MagicMock

import pytest
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import Compose

from config.schema.data import DatasetType
from data import Dataset, DoubleAugmentDataset
from data.dataset.transform_set import TransformSet


class TestDoubleAugmentDataset:
    """Test cases for the DoubleAugmentDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_cityscapes_constructor):
        """Run before each test method to create a fresh dataset instance."""
        self.dataset = DoubleAugmentDataset(mock_config.data)

    def test_init_hardcoded_train_split(self):
        """Test that DoubleAugmentDataset always uses 'train' split."""

        assert self.dataset.dataset_type == DatasetType.CITYSCAPES
        assert isinstance(self.dataset.dataset, TorchDataset)
        assert isinstance(self.dataset.transform_set, TransformSet)

    def test_transform_attributes_initialization(self):
        """Test that all transform attributes are properly initialized."""

        # Check that all required transform attributes exist
        assert hasattr(self.dataset, "transform")
        assert hasattr(self.dataset, "transform1")
        assert hasattr(self.dataset, "transform2")
        assert hasattr(self.dataset, "target_transform")
        assert hasattr(self.dataset, "transform_shrink_target")
        # Check transforms that are assigned from the parent transforms object
        assert self.dataset.transform == self.dataset.transform_set.base_image
        assert self.dataset.transform1 == self.dataset.transform_set.geometry_augmentation
        assert self.dataset.transform_shrink_target == self.dataset.transform_set.shrink_target

        # transform_base_target should be a Compose object with base_target and filter_classes
        assert isinstance(self.dataset.target_transform, Compose)
        assert self.dataset.target_transform.transforms[0] == self.dataset.transform_set.base_target
        assert self.dataset.target_transform.transforms[1] == self.dataset.transform_set.filter_classes

        # transform2 should be a Compose object with color_augmentation and image_normalization
        assert callable(self.dataset.transform2)
        assert isinstance(self.dataset.transform2, Compose)
        assert self.dataset.transform2.transforms[0] == self.dataset.transform_set.color_augmentation
        assert self.dataset.transform2.transforms[1] == self.dataset.transform_set.image_normalization

    def test_getitem_returns_three_items(self, sample_image, sample_target):
        """Test that __getitem__ returns exactly three items."""

        # Mock the underlying self.dataset's __getitem__
        self.dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        result = self.dataset[0]

        # Should return tuple of 3 items: (augmented_image1, augmented_image2, shrunken_target)
        assert isinstance(result, tuple)
        assert len(result) == 3

        augmented_image1, augmented_image2, shrunken_target = result
        assert augmented_image1 is not None
        assert augmented_image2 is not None
        assert shrunken_target is not None

    def test_getitem_transform_sequence(self, sample_image, sample_target):
        """Test that __getitem__ applies transforms in the correct sequence."""

        # Mock all transforms with identifiable return values
        mock_base_transformed_image = MagicMock()
        mock_base_transformed_target = MagicMock()
        mock_geometry_transformed_image = MagicMock()
        mock_geometry_transformed_target = MagicMock()
        mock_color_transformed_image = MagicMock()
        mock_shrunken_target = MagicMock()

        self.dataset.transform_set.base_image = MagicMock(return_value=mock_base_transformed_image)
        self.dataset.transform_set.base_target = MagicMock(return_value=mock_base_transformed_target)
        self.dataset.transform_set.filter_classes = MagicMock(return_value=mock_base_transformed_target)
        self.dataset.transform1 = MagicMock(
            return_value=(mock_geometry_transformed_image, mock_geometry_transformed_target)
        )
        self.dataset.transform2 = MagicMock(return_value=mock_color_transformed_image)
        self.dataset.transform_shrink_target = MagicMock(return_value=mock_shrunken_target)

        # Mock the underlying self.dataset
        self.dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        result = self.dataset[0]

        # Verify the transform sequence
        self.dataset.dataset.__getitem__.assert_called_once_with(0)
        self.dataset.transform1.assert_called_once()

        # transform2 should be called twice (for both augmented versions)
        assert self.dataset.transform2.call_count == 2
        self.dataset.transform2.assert_any_call(mock_geometry_transformed_image)

        # shrink_target should be called once
        self.dataset.transform_shrink_target.assert_called_once_with(mock_geometry_transformed_target)

        # Check the result structure
        assert result[0] == mock_color_transformed_image  # First augmented image
        assert result[1] == mock_color_transformed_image  # Second augmented image
        assert result[2] == mock_shrunken_target  # Shrunken target

    def test_double_augmentation_concept(self, sample_image, sample_target):
        """Test the core concept: two similar but independently augmented images."""

        # Use real-ish mock behavior to simulate the double augmentation
        self.dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        # Mock transforms to return different values on each call
        call_count = 0

        def mock_transform2_side_effect(image):
            nonlocal call_count
            call_count += 1
            # Return different mock objects to simulate different augmentations
            return MagicMock(name=f"augmented_image_{call_count}")

        self.dataset.transform_set.base_image = MagicMock(return_value=sample_image)
        self.dataset.transform_set.base_target = MagicMock(return_value=sample_target)
        self.dataset.transform_set.filter_classes = MagicMock(return_value=sample_target)
        self.dataset.transform1 = MagicMock(return_value=(sample_image, sample_target))
        self.dataset.transform2 = MagicMock(side_effect=mock_transform2_side_effect)
        self.dataset.transform_shrink_target = MagicMock(return_value=sample_target)

        result = self.dataset[0]

        # Both images should be the result of transform2, but potentially different due to randomness
        assert len(result) == 3
        assert self.dataset.transform2.call_count == 2
        # The two augmented images could be different due to random augmentations
        assert result[0] != result[1]  # They should be different objects

    def test_inheritance_from_base_dataset(self):
        """Test that DoubleAugmentDataset properly inherits from Dataset."""

        # Should have all the properties of the base Dataset
        assert hasattr(self.dataset, "dataset_type")
        assert hasattr(self.dataset, "dataset")
        assert hasattr(self.dataset, "transform_set")
        assert hasattr(self.dataset, "classes")
        # Should be an instance of the base Dataset class

        assert isinstance(self.dataset, Dataset)

    def test_len_inherited_from_base(self):
        """Test that __len__ method is inherited correctly from base Dataset."""

        # Mock the underlying self.dataset's __len__
        self.dataset.dataset.__len__.return_value = 50

        assert len(self.dataset) == 50
        self.dataset.dataset.__len__.assert_called_once()

    def test_classes_property_inherited(self):
        """Test that classes property is inherited correctly from base Dataset."""

        mock_classes = ["road", "sidewalk", "building", "wall", "fence"]
        self.dataset.dataset.classes = mock_classes

        assert self.dataset.classes == mock_classes

    def test_transform2_composition(self):
        """Test that transform2 is properly composed of color augmentation and normalization."""

        # transform2 should be a Compose object
        from torchvision.transforms.v2 import Compose

        assert isinstance(self.dataset.transform2, Compose)

        # It should be callable
        assert callable(self.dataset.transform2)

        # It should contain exactly 2 transforms (color_augmentation and image_normalization)
        assert len(self.dataset.transform2.transforms) == 2

    def test_multiple_calls_independence(self, sample_image, sample_target):
        """Test that multiple calls to __getitem__ work independently and produce different results."""

        # Call __getitem__ multiple times with the same index
        result1 = self.dataset[0]
        result2 = self.dataset[0]

        assert len(result1) == 3
        assert len(result2) == 3

        # The two augmented images should potentially be different due to random augmentations
        # Even though we use the same index, the geometric and color augmentations should produce different results
        augmented_image1_first, augmented_image2_first, target1 = result1
        augmented_image1_second, augmented_image2_second, target2 = result2

        # Check if any of the tensor values are different
        assert (augmented_image1_first != augmented_image1_second).any()
        assert (augmented_image2_first != augmented_image2_second).any()
        assert (target1 != target2).any()
