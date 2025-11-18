"""Unit tests for the PanopticPartsDataset class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from config.schema.data import DatasetType
from data import PanopticPartsDataset


class TestPanopticPartsDataset:
    """Test cases for the PanopticPartsDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config, mock_cityscapes_constructor, mock_transform_set_constructor):
        """Run before each test method to create a fresh dataset instance."""
        self.dataset = PanopticPartsDataset(mock_config.data)

    def test_init_requires_cityscapes(self, mock_config, mock_cityscapes_constructor, mock_transform_set_constructor):
        """Test that PanopticPartsDataset only accepts CITYSCAPES dataset type."""
        
        assert self.dataset.dataset_type == DatasetType.CITYSCAPES
        
        # Should raise error with other dataset types
        mock_config.data.dataset = DatasetType.PASCAL_PARTS
        with pytest.raises(ValueError, match="only supports CITYSCAPES"):
            PanopticPartsDataset(mock_config.data)

    def test_getitem_returns_three_elements(self, sample_image, sample_target):
        """Test __getitem__ method returns image, target, and panoptic mask."""
        
        # Mock the panoptic mask path and image
        mock_panoptic_mask = torch.randint(0, 20, (256, 512), dtype=torch.int64)
        
        with patch.object(self.dataset, '_get_panoptic_mask', return_value=mock_panoptic_mask):
            result = self.dataset[0]
            
            assert len(result) == 3
            assert isinstance(result[0], torch.Tensor)  # image
            assert isinstance(result[1], torch.Tensor)  # target
            assert isinstance(result[2], torch.Tensor)  # panoptic mask

    def test_get_image_path_from_regular_dataset(self, mock_config):
        """Test _get_image_path with regular dataset (not Subset)."""
        
        # Mock the dataset.images attribute
        mock_image_path = "/data/cityscapes/leftImg8bit/train/city/image_000000_leftImg8bit.png"
        self.dataset.dataset.images = [mock_image_path]
        
        result = self.dataset._get_image_path(0)
        
        assert isinstance(result, Path)
        assert str(result) == mock_image_path

    def test_get_image_path_from_subset(self, mock_config):
        """Test _get_image_path with Subset dataset."""
        
        from torch.utils.data.dataset import Subset
        
        # Create a mock subset
        mock_base_dataset = MagicMock()
        mock_base_dataset.images = [
            "/data/cityscapes/leftImg8bit/train/city/image_000000_leftImg8bit.png",
            "/data/cityscapes/leftImg8bit/train/city/image_000001_leftImg8bit.png",
        ]
        
        mock_subset = MagicMock(spec=Subset)
        mock_subset.dataset = mock_base_dataset
        mock_subset.indices = [1, 0]  # Reversed indices
        
        self.dataset.dataset = mock_subset
        
        result = self.dataset._get_image_path(0)
        
        # Should use the index from subset.indices[0] = 1
        assert isinstance(result, Path)
        assert "image_000001" in str(result)

    def test_get_panoptic_mask_path_transformation(self):
        """Test that panoptic mask path is correctly transformed from image path."""

        mock_image_path = Path("/data/cityscapes/leftImg8bit/train/city/image_000000_leftImg8bit.png")
        expected_panoptic_path = Path(
            "/data/cityscapes/gtFinePanopticParts/train/city/image_000000_gtFinePanopticParts.tif"
        )

        mock_panoptic_image = MagicMock()
        mock_panoptic_tensor = torch.tensor([[150_023, 100_045], [200_067, 175_099]], dtype=torch.int64)
        expected_result = torch.tensor([[123, 145], [267, 199]], dtype=torch.int64)

        with patch.object(self.dataset, "_get_image_path", return_value=mock_image_path):
            with patch("data.dataset.panoptic_parts.Image.open", return_value=mock_panoptic_image) as mock_open:
                self.dataset.transform_set.base_target.return_value = mock_panoptic_tensor.clone()

                result = self.dataset._get_panoptic_mask(0)

                # Verify the path transformation
                mock_open.assert_called_once()
                called_path = mock_open.call_args[0][0]
                assert called_path == expected_panoptic_path
                assert torch.equal(result, expected_result)

    def test_get_panoptic_mask_processing(self):
        """Test panoptic mask processing logic."""
        
        mock_image_path = Path("/data/cityscapes/leftImg8bit/train/city/image_000000_leftImg8bit.png")
        mock_panoptic_image = MagicMock()
        
        # Create tensor with values: some < 100_000 (should become 0), some >= 100_000
        # Formula: value // 100_000 * 100 + value % 100
        mock_panoptic_tensor = torch.tensor([
            [50_000, 150_023],  # 50k -> 0, 150023 -> 100 + 23 = 123
            [100_045, 200_067]  # 100045 -> 100 + 45 = 145, 200067 -> 200 + 67 = 267
        ], dtype=torch.int64)
        
        with patch.object(self.dataset, '_get_image_path', return_value=mock_image_path):
            with patch('data.dataset.panoptic_parts.Image.open', return_value=mock_panoptic_image):
                self.dataset.transform_set.base_target.return_value = mock_panoptic_tensor.clone()
                
                result = self.dataset._get_panoptic_mask(0)
                
                # Verify the processing
                assert result[0, 0] == 0  # 50_000 < 100_000 -> 0
                assert result[0, 1] == 123  # 150_023 -> 1 * 100 + 23
                assert result[1, 0] == 145  # 100_045 -> 1 * 100 + 45
                assert result[1, 1] == 267  # 200_067 -> 2 * 100 + 67

    def test_classes_property(self):
        """Test classes property returns expected part classes."""
        
        classes = self.dataset.classes
        
        assert isinstance(classes, list)
        assert len(classes) == 23
        # Check some expected classes
        assert "torso" in classes
        assert "head" in classes
        assert "wheel" in classes
        assert "window" in classes
        assert "chassis" in classes
