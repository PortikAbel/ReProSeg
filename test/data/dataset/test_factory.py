"""Unit tests for the DatasetFactory class."""

from unittest.mock import MagicMock, patch

import pytest

from config.schema.data import DataConfig, DatasetType
from data.data_split import DataSplit
from data.dataset.factory import DatasetFactory
from utils.errors import DatasetNotImplementedError

_DATASET_PARAMS = pytest.mark.parametrize(
    "dataset_type,patch_path,expected_kwargs_fn",
    [
        pytest.param(
            DatasetType.CITYSCAPES,
            "data.dataset.factory.Cityscapes",
            lambda path, split: {
                "root": path,
                "split": split,
                "mode": "fine",
                "target_type": "semantic",
            },
            id="cityscapes",
        ),
        pytest.param(
            DatasetType.VOC_SEGMENTATION,
            "data.dataset.factory.VOCSegmentation",
            lambda path, split: {
                "root": path,
                "year": "2012",
                "image_set": split.value,
                "download": False,
            },
            id="pascal_voc",
        ),
    ],
)


class TestDatasetFactory:
    """Test cases for the DatasetFactory class."""

    @_DATASET_PARAMS
    def test_create_dataset(
        self,
        mock_config,
        mock_cityscapes_dataset,
        mock_voc_dataset,
        dataset_type,
        patch_path,
        expected_kwargs_fn,
    ):
        """Test factory creates the correct dataset with correct parameters."""
        mock_config.data.dataset = dataset_type
        mock_dataset = mock_voc_dataset if dataset_type == DatasetType.VOC_SEGMENTATION else mock_cityscapes_dataset

        with patch(patch_path, return_value=mock_dataset) as mock_constructor:
            dataset = DatasetFactory.create(mock_config.data, split=DataSplit.TRAIN)

            mock_constructor.assert_called_once_with(**expected_kwargs_fn(mock_config.data.path, DataSplit.TRAIN))
            assert dataset == mock_dataset

    @pytest.mark.parametrize("dataset_type", list(DatasetType))
    def test_create_sets_dataset_classes(self, mock_config, dataset_type):
        """Test factory sets classes on the created dataset."""
        mock_classes = ["cls_a", "cls_b"]
        mock_config.data.dataset = dataset_type
        mock_dataset = MagicMock()

        with (
            patch("data.dataset.factory.Cityscapes", return_value=mock_dataset),
            patch("data.dataset.factory.VOCSegmentation", return_value=mock_dataset),
            patch("data.dataset.factory.LabelMapping.get_classes", return_value=mock_classes)
        ):
            dataset = DatasetFactory.create(mock_config.data, split=DataSplit.TRAIN)

            assert dataset.classes == mock_classes

    @_DATASET_PARAMS
    def test_create_with_validation_split(
        self,
        mock_config,
        mock_cityscapes_dataset,
        mock_voc_dataset,
        dataset_type,
        patch_path,
        expected_kwargs_fn,
    ):
        """Test factory works with validation split for each dataset type."""
        mock_config.data.dataset = dataset_type
        mock_dataset = mock_voc_dataset if dataset_type == DatasetType.VOC_SEGMENTATION else mock_cityscapes_dataset

        with patch(patch_path, return_value=mock_dataset) as mock_constructor:
            dataset = DatasetFactory.create(mock_config.data, split=DataSplit.VAL)

            mock_constructor.assert_called_once_with(**expected_kwargs_fn(mock_config.data.path, DataSplit.VAL))
            assert dataset is not None

    def test_create_raises_for_unsupported_dataset(self):
        """Test factory raises error for unsupported dataset types."""
        mock_data_config = MagicMock(spec=DataConfig)
        mock_data_config.dataset = "UNSUPPORTED_DATASET_TYPE"

        with pytest.raises(DatasetNotImplementedError):
            DatasetFactory.create(mock_data_config, split=DataSplit.TRAIN)
