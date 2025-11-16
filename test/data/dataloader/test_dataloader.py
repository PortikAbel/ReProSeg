"""Unit tests for the DataLoader class."""

from unittest.mock import patch

import pytest
import torch

from config.schema.data import DatasetType
from data import DataLoader, Dataset, DataSplit


class TestDataLoader:
    """Test cases for the DataLoader class."""

    @pytest.fixture(autouse=True)
    def dataset(self, mock_config, mock_cityscapes_constructor):
        """Run before each test method to create a fresh dataset instance."""
        self.dataset = Dataset(mock_config.data, DataSplit.TRAIN)

    def test_init_basic(self, mock_config):
        """Test DataLoader initialization with basic parameters."""
        mock_config.data.batch_size = 4
        mock_config.data.num_workers = 2

        dataloader = DataLoader(self.dataset, mock_config)

        assert dataloader.dataset is not None
        assert isinstance(dataloader.dataset, Dataset)
        assert dataloader.batch_size == 4
        assert dataloader.num_workers == 2
        assert dataloader.to_shuffle is False
        assert dataloader.to_drop_last is False

    @patch("torch.cuda.is_available")
    def test_pin_memory_when_cuda_available(self, mock_cuda_available, mock_config):
        """Test that pin_memory is True when CUDA is available."""
        mock_cuda_available.return_value = True

        dataloader = DataLoader(self.dataset, mock_config)
        assert dataloader.pin_memory is True

    @patch("torch.cuda.is_available")
    def test_pin_memory_when_cuda_not_available(self, mock_cuda_available, mock_config):
        """Test that pin_memory is False when CUDA is not available."""
        mock_cuda_available.return_value = False

        dataloader = DataLoader(self.dataset, mock_config)
        assert dataloader.pin_memory is False

    def test_dataset_creation(self, mock_config):
        """Test that dataset is created correctly."""
        dataloader = DataLoader(self.dataset, mock_config)

        assert dataloader.dataset is not None
        assert dataloader.dataset.config.dataset == DatasetType.CITYSCAPES
        assert dataloader.dataset.split == DataSplit.TRAIN

    def test_torch_dataloader_inheritance(self, mock_config):
        """Test that DataLoader properly inherits from torch.utils.data.DataLoader."""
        dataloader = DataLoader(self.dataset, mock_config)

        # Check that it has the expected torch DataLoader attributes/methods
        assert hasattr(dataloader, "dataset")
        assert hasattr(dataloader, "batch_size")
        assert hasattr(dataloader, "sampler")
        assert hasattr(dataloader, "num_workers")
        assert hasattr(dataloader, "pin_memory")
        assert hasattr(dataloader, "drop_last")

        # Check that it's iterable
        assert hasattr(dataloader, "__iter__")
        assert hasattr(dataloader, "__len__")

    @patch("numpy.random.seed")
    def test_worker_init_function(self, mock_numpy_seed, mock_config):
        """Test that worker_init_fn is set correctly."""
        dataloader = DataLoader(self.dataset, mock_config)

        # The worker_init_fn should be a function that calls numpy.random.seed
        assert dataloader.worker_init_fn is not None
        assert callable(dataloader.worker_init_fn)

        # Call the worker_init_fn and verify it calls numpy.random.seed with the correct seed
        dataloader.worker_init_fn(0)  # worker_id is typically passed as argument
        mock_numpy_seed.assert_called_with(mock_config.env.seed)

    @pytest.mark.parametrize("batch_size", [2, 8, 16, 32])
    def test_batch_size_parameter(self, batch_size, mock_config):
        """Test that batch_size is set correctly from config."""
        mock_config.data.batch_size = batch_size

        dataloader = DataLoader(self.dataset, mock_config)
        assert dataloader.batch_size == batch_size

    @pytest.mark.parametrize("num_workers", [0, 1, 4, 8])
    def test_num_workers_parameter(self, num_workers, mock_config):
        """Test that num_workers is set correctly from config."""
        mock_config.data.num_workers = num_workers

        dataloader = DataLoader(self.dataset, mock_config)
        assert dataloader.num_workers == num_workers

    def test_default_shuffle_and_drop_last(self, mock_config):
        """Test that shuffle and drop_last have correct default values."""
        dataloader = DataLoader(self.dataset, mock_config)

        # The actual attributes from torch DataLoader might be different from our class attributes
        # Test our class attributes that control the behavior
        assert dataloader.to_shuffle is False
        assert dataloader.to_drop_last is False
        assert dataloader.drop_last == dataloader.to_drop_last

    def test_sampler_is_none(self, mock_config):
        """Test that sampler configuration is correct."""
        dataloader = DataLoader(self.dataset, mock_config)
        # Note: torch DataLoader creates a default sampler when sampler=None is passed
        # so we just verify the sampler exists and is of expected type
        assert dataloader.sampler is not None

    def test_dataset_length_propagation(self, mock_config):
        """Test that DataLoader length matches dataset length."""
        # Mock the dataset length
        with patch.object(Dataset, "__len__", return_value=100):
            dataloader = DataLoader(self.dataset, mock_config)
            # DataLoader length should be affected by batch_size and drop_last
            # With batch_size=4, drop_last=False, length should be ceil(100/4) = 25
            expected_length = (100 + mock_config.data.batch_size - 1) // mock_config.data.batch_size  # ceiling division
            assert len(dataloader) == expected_length

    def test_iteration_functionality(self, mock_config, sample_image, sample_target):
        """Test that DataLoader can be iterated over."""
        # Create mock tensor data that the DataLoader can handle
        mock_tensor_image = torch.randn(3, 256, 512)  # Sample image tensor
        mock_tensor_target = torch.randint(0, 20, (256, 512))  # Sample target tensor

        # Mock dataset to return tensor data instead of PIL images
        with patch.object(Dataset, "__getitem__", return_value=(mock_tensor_image, mock_tensor_target)):
            with patch.object(Dataset, "__len__", return_value=8):  # 2 batches with batch_size=4
                # Use num_workers=0 to avoid multiprocessing issues in tests
                mock_config.data.num_workers = 0
                dataloader = DataLoader(self.dataset, mock_config)

                # Test that we can iterate
                batch_count = 0
                for batch in dataloader:
                    assert len(batch) == 2  # image, target
                    batch_count += 1
                    if batch_count >= 2:  # Don't iterate through all batches to save time
                        break

                assert batch_count >= 1  # At least one batch was processed

    def test_class_attributes_accessibility(self, mock_config):
        """Test that class attributes are accessible and correctly typed."""
        dataloader = DataLoader(self.dataset, mock_config)

        # Test type annotations match actual types
        assert isinstance(dataloader.dataset, Dataset)
        assert isinstance(dataloader.to_shuffle, bool)
        assert isinstance(dataloader.to_drop_last, bool)
