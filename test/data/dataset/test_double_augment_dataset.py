"""Unit tests for the DoubleAugmentDataset class."""

from unittest.mock import MagicMock

from torchvision.transforms.v2 import Compose

from data.dataset import DoubleAugmentDataset


class TestDoubleAugmentDataset:
    """Test cases for the DoubleAugmentDataset class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.dataset_name = "CityScapes"

    def test_init_hardcoded_train_split(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that DoubleAugmentDataset always uses 'train' split."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        assert dataset.name == self.dataset_name
        assert dataset.split == "train"  # Should always be "train"
        assert dataset.dataset is not None
        assert dataset.transforms is not None

    def test_transform_attributes_initialization(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that all transform attributes are properly initialized."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # Check that all required transform attributes exist
        assert hasattr(dataset, "transform_base_image")
        assert hasattr(dataset, "transform1")
        assert hasattr(dataset, "transform2")
        assert hasattr(dataset, "transform_base_target")
        assert hasattr(dataset, "transform_shrink_target")

        # Check transforms that are assigned from the parent transforms object
        assert dataset.transform_base_image == dataset.transforms.base_image
        assert dataset.transform1 == dataset.transforms.geometry_augmentation
        assert dataset.transform_base_target == dataset.transforms.base_target
        assert dataset.transform_shrink_target == dataset.transforms.shrink_target

        # transform2 should be a Compose object with color_augmentation and image_normalization
        assert callable(dataset.transform2)
        assert isinstance(dataset.transform2, Compose)
        assert dataset.transform2.transforms[0] == dataset.transforms.color_augmentation
        assert dataset.transform2.transforms[1] == dataset.transforms.image_normalization

    def test_getitem_returns_three_items(
        self, mock_env_data_root, mock_cityscapes_constructor, sample_image, sample_target
    ):
        """Test that __getitem__ returns exactly three items."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # Mock the underlying dataset's __getitem__
        dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        result = dataset[0]

        # Should return tuple of 3 items: (augmented_image1, augmented_image2, shrunken_target)
        assert isinstance(result, tuple)
        assert len(result) == 3

        augmented_image1, augmented_image2, shrunken_target = result
        assert augmented_image1 is not None
        assert augmented_image2 is not None
        assert shrunken_target is not None

    def test_getitem_transform_sequence(
        self, mock_env_data_root, mock_cityscapes_constructor, sample_image, sample_target
    ):
        """Test that __getitem__ applies transforms in the correct sequence."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # Mock all transforms with identifiable return values
        mock_base_transformed_image = MagicMock()
        mock_base_transformed_target = MagicMock()
        mock_geometry_transformed_image = MagicMock()
        mock_geometry_transformed_target = MagicMock()
        mock_color_transformed_image = MagicMock()
        mock_shrunken_target = MagicMock()

        dataset.transform_base_image = MagicMock(return_value=mock_base_transformed_image)
        dataset.transform_base_target = MagicMock(return_value=mock_base_transformed_target)
        dataset.transform1 = MagicMock(return_value=(mock_geometry_transformed_image, mock_geometry_transformed_target))
        dataset.transform2 = MagicMock(return_value=mock_color_transformed_image)
        dataset.transform_shrink_target = MagicMock(return_value=mock_shrunken_target)

        # Mock the underlying dataset
        dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        result = dataset[0]

        # Verify the transform sequence
        dataset.dataset.__getitem__.assert_called_once_with(0)
        dataset.transform_base_image.assert_called_once_with(sample_image)
        dataset.transform_base_target.assert_called_once_with(sample_target)
        dataset.transform1.assert_called_once_with(mock_base_transformed_image, mock_base_transformed_target)

        # transform2 should be called twice (for both augmented versions)
        assert dataset.transform2.call_count == 2
        dataset.transform2.assert_any_call(mock_geometry_transformed_image)

        # shrink_target should be called once
        dataset.transform_shrink_target.assert_called_once_with(mock_geometry_transformed_target)

        # Check the result structure
        assert result[0] == mock_color_transformed_image  # First augmented image
        assert result[1] == mock_color_transformed_image  # Second augmented image
        assert result[2] == mock_shrunken_target  # Shrunken target

    def test_double_augmentation_concept(
        self, mock_env_data_root, mock_cityscapes_constructor, sample_image, sample_target
    ):
        """Test the core concept: two similar but independently augmented images."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # Use real-ish mock behavior to simulate the double augmentation
        dataset.dataset.__getitem__.return_value = (sample_image, sample_target)

        # Mock transforms to return different values on each call
        call_count = 0

        def mock_transform2_side_effect(image):
            nonlocal call_count
            call_count += 1
            # Return different mock objects to simulate different augmentations
            return MagicMock(name=f"augmented_image_{call_count}")

        dataset.transform_base_image = MagicMock(return_value=sample_image)
        dataset.transform_base_target = MagicMock(return_value=sample_target)
        dataset.transform1 = MagicMock(return_value=(sample_image, sample_target))
        dataset.transform2 = MagicMock(side_effect=mock_transform2_side_effect)
        dataset.transform_shrink_target = MagicMock(return_value=sample_target)

        result = dataset[0]

        # Both images should be the result of transform2, but potentially different due to randomness
        assert len(result) == 3
        assert dataset.transform2.call_count == 2
        # The two augmented images could be different due to random augmentations
        assert result[0] != result[1]  # They should be different objects

    def test_transform_property_returns_none(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that transform property returns None (overrides parent)."""
        dataset = DoubleAugmentDataset(self.dataset_name)
        assert dataset.transform is None

    def test_target_transform_property_returns_none(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that target_transform property returns None (overrides parent)."""
        dataset = DoubleAugmentDataset(self.dataset_name)
        assert dataset.target_transform is None

    def test_inheritance_from_base_dataset(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that DoubleAugmentDataset properly inherits from Dataset."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # Should have all the properties of the base Dataset
        assert hasattr(dataset, "name")
        assert hasattr(dataset, "config")
        assert hasattr(dataset, "split")
        assert hasattr(dataset, "dataset")
        assert hasattr(dataset, "transforms")
        assert hasattr(dataset, "classes")

        # Should be an instance of the base Dataset class
        from data.dataset.base import Dataset

        assert isinstance(dataset, Dataset)

    def test_len_inherited_from_base(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that __len__ method is inherited correctly from base Dataset."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # Mock the underlying dataset's __len__
        dataset.dataset.__len__.return_value = 50

        assert len(dataset) == 50
        dataset.dataset.__len__.assert_called_once()

    def test_classes_property_inherited(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that classes property is inherited correctly from base Dataset."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        mock_classes = ["road", "sidewalk", "building", "wall", "fence"]
        dataset.dataset.classes = mock_classes

        assert dataset.classes == mock_classes

    def test_transform2_composition(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that transform2 is properly composed of color augmentation and normalization."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # transform2 should be a Compose object
        from torchvision.transforms.v2 import Compose

        assert isinstance(dataset.transform2, Compose)

        # It should be callable
        assert callable(dataset.transform2)

        # It should contain exactly 2 transforms (color_augmentation and image_normalization)
        assert len(dataset.transform2.transforms) == 2

    def test_multiple_calls_independence(
        self, mock_env_data_root, mock_cityscapes_constructor, sample_image, sample_target
    ):
        """Test that multiple calls to __getitem__ work independently and produce different results."""
        dataset = DoubleAugmentDataset(self.dataset_name)

        # Call __getitem__ multiple times with the same index
        result1 = dataset[0]
        result2 = dataset[0]

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

    def test_supported_dataset_literal_type(self, mock_env_data_root, mock_cityscapes_constructor):
        """Test that DoubleAugmentDataset works with SupportedDataset literal type."""
        # This should work without any type errors
        dataset = DoubleAugmentDataset("CityScapes")
        assert dataset.name == "CityScapes"
        assert dataset.split == "train"

        # Test that an invalid dataset name would raise an error
        try:
            DoubleAugmentDataset("InvalidDataset")  # type: ignore
            raise AssertionError("Should have raised NotImplementedError")
        except NotImplementedError as e:
            assert "'InvalidDataset'" in str(e)
