"""Unit tests for the DatasetFactory class."""

from unittest.mock import patch

import pytest

from config.schema.data import DatasetType
from data.data_split import DataSplit
from data.dataset.factory import DatasetFactory
from utils.errors import DatasetNotImplementedError


class TestDatasetFactory:
    """Test cases for the DatasetFactory class."""

    def test_create_cityscapes_dataset(self, mock_config, mock_cityscapes_dataset):
        """Test factory creates Cityscapes dataset with correct parameters."""
        
        with patch("data.dataset.factory.Cityscapes", return_value=mock_cityscapes_dataset) as mock_constructor:
            dataset = DatasetFactory.create(mock_config.data, split=DataSplit.TRAIN)
            
            mock_constructor.assert_called_once_with(
                root=mock_config.data.path,
                split=DataSplit.TRAIN,
                mode="fine",
                target_type="semantic",
            )
            assert dataset == mock_cityscapes_dataset

    def test_create_filters_cityscapes_classes(self, mock_config, mock_cityscapes_dataset, mock_cityscapes_classes):
        """Test factory filters Cityscapes classes."""
        
        with patch("data.dataset.factory.Cityscapes", return_value=mock_cityscapes_dataset):
            with patch("data.dataset.factory.ClassFilter.filter_cityscapes_classes") as mock_filter:
                filtered_classes = mock_cityscapes_classes[:6]  # Filtered classes
                mock_filter.return_value = filtered_classes
                
                dataset = DatasetFactory.create(mock_config.data, split=DataSplit.TRAIN)
                
                mock_filter.assert_called_once()
                assert dataset.classes == filtered_classes

    def test_create_with_validation_split(self, mock_config, mock_cityscapes_dataset):
        """Test factory works with validation split."""
        
        with patch("data.dataset.factory.Cityscapes", return_value=mock_cityscapes_dataset) as mock_constructor:
            dataset = DatasetFactory.create(mock_config.data, split=DataSplit.VAL)
            
            mock_constructor.assert_called_once_with(
                root=mock_config.data.path,
                split=DataSplit.VAL,
                mode="fine",
                target_type="semantic",
            )
            
            assert dataset is not None

    def test_create_raises_for_unsupported_dataset(self, mock_config):
        """Test factory raises error for unsupported dataset types."""
        
        # Set an unsupported dataset type
        mock_config.data.dataset = DatasetType.PASCAL_PARTS
        
        with pytest.raises(DatasetNotImplementedError):
            DatasetFactory.create(mock_config.data, split=DataSplit.TRAIN)
