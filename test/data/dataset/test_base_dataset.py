"""Unit tests for the base Dataset class."""

from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset as TorchDataset

from config.schema.data import DatasetType
from data import Dataset, DataSplit
from data.dataset.factory import DatasetFactory
from data.dataset.transform_set import TransformSet
from utils.errors import DatasetNotImplementedError


class TestBaseDataset:
    """Test cases for the base Dataset class."""

    @pytest.fixture(
        params=[
            pytest.param(DatasetType.CITYSCAPES, id="cityscapes"),
            pytest.param(DatasetType.VOC_SEGMENTATION, id="pascal_voc"),
        ],
        autouse=True,
    )
    def setup(self, request, mock_config, mock_cityscapes_constructor, mock_voc_constructor, mock_transform_set_constructor):
        self.dataset_type = request.param
        self.split = DataSplit.TRAIN
        mock_config.data.dataset = request.param
        self.dataset = Dataset(mock_config.data)

    def test_init_valid_dataset(self):
        """Test Dataset initializes all attributes."""

        assert self.dataset.dataset_type == self.dataset_type
        assert isinstance(self.dataset.dataset, TorchDataset)
        assert isinstance(self.dataset.transform_set, TransformSet)

    def test_getitem(self, sample_image, sample_target):
        """Test __getitem__ method returns tensors."""

        result = self.dataset[0]

        self.dataset.dataset.__getitem__.assert_called_once_with(0)
        assert len(result) == 2
        # Verify that transforms were applied (transform_set methods should be mocked in fixtures)
        self.dataset.transform_set.base_image.assert_called_once_with(sample_image)
        self.dataset.transform_set.image_normalization.assert_called_once()
        self.dataset.transform_set.base_target.assert_called_once_with(sample_target)

        if self.dataset_type == DatasetType.CITYSCAPES:
            self.dataset.transform_set.filter_classes.assert_called_once()

        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_len(self):
        """Test __len__ method."""

        # Mock the underlying self.dataset's __len__
        self.dataset.dataset.__len__.return_value = 100

        assert len(self.dataset) == 100
        self.dataset.dataset.__len__.assert_called_once()

    def test_dataset_factory_called_by_default(self, mock_config, mock_cityscapes_dataset):
        """Test that the factory creates the default dataset based on config."""
        with patch.object(DatasetFactory, "create", return_value=mock_cityscapes_dataset) as mock_create:
            Dataset(mock_config.data)
            mock_create.assert_called_once_with(mock_config.data, split=self.split)

    def test_classes_property_cityscapes(self):
        """Test classes property for Cityscapes dataset."""

        if self.dataset_type != DatasetType.CITYSCAPES:
            pytest.skip("Classes property only supported for Cityscapes dataset")

        mock_classes = ["class1", "class2", "class3"]
        self.dataset.dataset.classes = mock_classes

        assert self.dataset.classes == mock_classes

    def test_classes_property_raises_for_non_cityscapes(self):
        """Test classes property raises DatasetNotImplementedError for non-Cityscapes datasets."""

        if self.dataset_type == DatasetType.CITYSCAPES:
            pytest.skip("This test is for non-Cityscapes datasets only")

        with pytest.raises(DatasetNotImplementedError):
            _ = self.dataset.classes

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
