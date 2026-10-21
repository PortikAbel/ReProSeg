"""Unit tests for the PanopticPartsDataset class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from config.schema.data import DataConfig, DatasetType
from data import PanopticPartsDataset
from data.dataset.transform_set import TransformSet


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
        mock_config.data.dataset = DatasetType.VOC_SEGMENTATION
        with pytest.raises(ValueError, match="only supports CITYSCAPES"):
            PanopticPartsDataset(mock_config.data)

    def test_getitem_returns_three_elements(self, sample_image, sample_target):
        """Test __getitem__ method returns image, target, and panoptic mask."""

        # Mock the panoptic mask path and image
        mock_panoptic_mask = torch.randint(0, 20, (256, 512), dtype=torch.int64)

        with patch.object(self.dataset, "_get_panoptic_mask", return_value=mock_panoptic_mask):
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
        mock_panoptic_tensor = torch.tensor([[2_400_101, 2_500_204], [2_600_305, 2_800_403]], dtype=torch.int64)
        expected_result = torch.tensor([[2401, 2504], [2605, 2803]], dtype=torch.int64)

        with patch.object(self.dataset, "_get_image_path", return_value=mock_image_path):
            with patch("data.dataset.panoptic_parts.Image.open", return_value=mock_panoptic_image) as mock_open:
                self.dataset.transform_set.base_target.return_value = mock_panoptic_tensor.clone()
                self.dataset.transform_set.random_crop.side_effect = lambda mask: mask

                result = self.dataset._get_panoptic_mask(0)

                # Verify the path transformation
                mock_open.assert_called_once()
                called_path = mock_open.call_args[0][0]
                assert called_path == expected_panoptic_path
                assert torch.equal(result, expected_result)

    @pytest.mark.parametrize("official_parts_only", [False, True])
    def test_get_panoptic_mask_processing(self, official_parts_only):
        """Test panoptic mask processing logic."""

        mock_image_path = Path("/data/cityscapes/leftImg8bit/train/city/image_000000_leftImg8bit.png")
        mock_panoptic_image = MagicMock()

        self.dataset.official_parts_only = official_parts_only
        # Missing parts are always excluded; unsupported pairs depend on the option.
        mock_panoptic_tensor = torch.tensor(
            [
                [24, 24_001, 2_400_100, 2_400_105],
                [2_600_106, 3_100_101, 2_600_105, 2_400_104],
                [-1, 0, 2_500_199, 2_700_102],
            ],
            dtype=torch.int64,
        )

        with patch.object(self.dataset, "_get_image_path", return_value=mock_image_path):
            with patch("data.dataset.panoptic_parts.Image.open", return_value=mock_panoptic_image):
                self.dataset.transform_set.base_target.return_value = mock_panoptic_tensor.clone()
                self.dataset.transform_set.random_crop.side_effect = lambda mask: mask

                result = self.dataset._get_panoptic_mask(0)

                expected = (
                    [[0, 0, 0, 0], [0, 0, 2605, 2404], [0, 0, 0, 2702]]
                    if official_parts_only
                    else [[0, 0, 0, 2405], [2606, 3101, 2605, 2404], [0, 0, 2599, 2702]]
                )
                assert torch.equal(result, torch.tensor(expected, dtype=torch.int64))

    @pytest.mark.parametrize("official_parts_only", [False, True])
    def test_tiff_filters_all_semantic_part_pairs(self, tmp_path, official_parts_only):
        """Read real encoded TIFF pixels using the production transforms."""
        self.dataset.official_parts_only = official_parts_only
        # Cover every possible native semantic/part pair, including unsupported classes.
        semantic_ids = np.arange(34, dtype=np.int32)[:, None]
        part_ids = np.arange(100, dtype=np.int32)[None, :]
        uids = semantic_ids * 100_000 + 999 * 100 + part_ids
        mask_path = tmp_path / "gtFinePanopticParts/val/city/image_gtFinePanopticParts.tif"
        mask_path.parent.mkdir(parents=True)
        Image.fromarray(uids).save(mask_path)
        image_path = tmp_path / "leftImg8bit/val/city/image_leftImg8bit.png"
        self.dataset.transform_set = TransformSet(DataConfig(img_shape=(34, 100)))

        with patch.object(self.dataset, "_get_image_path", return_value=image_path):
            result = self.dataset._get_panoptic_mask(0)

        expected = torch.zeros((1, 34, 100), dtype=torch.int64)
        if official_parts_only:
            pairs = ((24, 4), (25, 4), (26, 5), (27, 5), (28, 5))
        else:
            pairs = ((sid, 99) for sid in range(1, 34))
        for sid, max_part in pairs:
            expected[0, sid, 1 : max_part + 1] = sid * 100 + torch.arange(1, max_part + 1)
        assert result.dtype == torch.int64
        assert torch.equal(result, expected)

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
