"""Unit tests for the DoubleAugmentDataLoader class."""

from unittest.mock import patch

import pytest

from data.dataloader.base import DataLoader
from data.dataloader.double_augment import DoubleAugmentDataLoader
from data.dataset.double_augment import DoubleAugmentDataset


class TestDoubleAugmentDataLoader:
    """Test cases for the DoubleAugmentDataLoader class."""

    def test_init_basic(self, mock_config, mock_cityscapes_constructor):
        """Test DoubleAugmentDataLoader initialization with basic parameters."""
        mock_config.data.batch_size = 4
        mock_config.data.num_workers = 2

        dataloader = DoubleAugmentDataLoader(mock_config)

        assert dataloader.split == "train"  # Should always be "train"
        assert dataloader.dataset is not None
        assert isinstance(dataloader.dataset, DoubleAugmentDataset)
        assert dataloader.batch_size == 4
        assert dataloader.num_workers == 2

    def test_inheritance_from_base_dataloader(self, mock_config, mock_cityscapes_constructor):
        """Test that DoubleAugmentDataLoader properly inherits from DataLoader."""
        dataloader = DoubleAugmentDataLoader(mock_config)

        assert isinstance(dataloader, DataLoader)
        assert hasattr(dataloader, "split")
        assert hasattr(dataloader, "dataset")
        assert hasattr(dataloader, "to_shuffle")
        assert hasattr(dataloader, "to_drop_last")

    def test_always_uses_train_split(self, mock_config, mock_cityscapes_constructor):
        """Test that DoubleAugmentDataLoader always uses 'train' split regardless of args."""
        # The constructor hardcodes "train" split
        dataloader = DoubleAugmentDataLoader(mock_config)

        assert dataloader.split == "train"
        assert dataloader.dataset.split == "train"

    def test_create_dataset_method_override(self, mock_config, mock_cityscapes_constructor):
        """Test that _create_dataset method creates DoubleAugmentDataset."""
        dataloader = DoubleAugmentDataLoader(mock_config)

        # Test the overridden method
        dataset = dataloader._create_dataset(mock_config.data)

        assert isinstance(dataset, DoubleAugmentDataset)
        assert dataset.config.dataset == "CityScapes"
        assert dataset.split == "train"  # DoubleAugmentDataset always uses train

    def test_dataset_type_is_double_augment(self, mock_config, mock_cityscapes_constructor):
        """Test that the created dataset is specifically DoubleAugmentDataset."""
        dataloader = DoubleAugmentDataLoader(mock_config)

        assert isinstance(dataloader.dataset, DoubleAugmentDataset)
        # Verify it's specifically a DoubleAugmentDataset, not just the base Dataset
        assert type(dataloader.dataset).__name__ == "DoubleAugmentDataset"

    def test_double_augment_dataset_functionality(
        self, mock_config, mock_cityscapes_constructor, sample_image, sample_target
    ):
        """Test that the underlying DoubleAugmentDataset works as expected."""
        dataloader = DoubleAugmentDataLoader(mock_config)

        # Mock the underlying dataset's __getitem__ to return sample data
        with patch.object(dataloader.dataset.dataset, "__getitem__", return_value=(sample_image, sample_target)):
            result = dataloader.dataset[0]

            # DoubleAugmentDataset should return 3 items: two augmented images and one target
            assert len(result) == 3
            # First two should be the augmented images, third should be the target

    @pytest.mark.parametrize("batch_size", [2, 8, 16])
    def test_batch_size_propagation(self, mock_config, batch_size, mock_cityscapes_constructor):
        """Test that batch_size is correctly propagated from args."""
        mock_config.data.batch_size = batch_size
        dataloader = DoubleAugmentDataLoader(mock_config)
        assert dataloader.batch_size == batch_size

    @pytest.mark.parametrize("num_workers", [0, 1, 4])
    def test_num_workers_propagation(self, num_workers, mock_config, mock_cityscapes_constructor):
        """Test that num_workers is correctly propagated from args."""
        mock_config.data.num_workers = num_workers
        dataloader = DoubleAugmentDataLoader(mock_config)
        assert dataloader.num_workers == num_workers

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_seed_propagation(self, seed, mock_config, mock_cityscapes_constructor):
        """Test that seed is correctly propagated to worker_init_fn."""
        mock_config.env.seed = seed

        with patch("numpy.random.seed") as mock_numpy_seed:
            dataloader = DoubleAugmentDataLoader(mock_config)

            # Call the worker_init_fn and verify it uses the correct seed
            dataloader.worker_init_fn(0)  # worker_id
            mock_numpy_seed.assert_called_with(seed)

    @patch("torch.cuda.is_available")
    def test_pin_memory_cuda_availability(self, mock_cuda_available, mock_config, mock_cityscapes_constructor):
        """Test that pin_memory setting respects CUDA availability."""
        # Test when CUDA is available
        mock_cuda_available.return_value = True
        dataloader = DoubleAugmentDataLoader(mock_config)
        assert dataloader.pin_memory is True

        # Test when CUDA is not available
        mock_cuda_available.return_value = False
        dataloader = DoubleAugmentDataLoader(mock_config)
        assert dataloader.pin_memory is False

    def test_dataset_name_propagation(self, mock_config, mock_cityscapes_constructor):
        """Test that dataset name is correctly propagated to DoubleAugmentDataset."""
        dataloader = DoubleAugmentDataLoader(mock_config)

        assert dataloader.dataset.config.dataset == "CityScapes"

    def test_default_dataloader_settings(self, mock_config, mock_cityscapes_constructor):
        """Test that default DataLoader settings are inherited."""
        dataloader = DoubleAugmentDataLoader(mock_config)

        # These should inherit the default values from base DataLoader
        assert dataloader.to_shuffle is False
        assert dataloader.to_drop_last is False
        assert dataloader.drop_last == dataloader.to_drop_last
        assert dataloader.sampler is not None  # torch creates default sampler

    def test_torch_dataloader_functionality(self, mock_config, mock_cityscapes_constructor):
        """Test that it maintains torch DataLoader functionality."""
        dataloader = DoubleAugmentDataLoader(mock_config)

        # Should have all the essential torch DataLoader methods/attributes
        assert hasattr(dataloader, "__iter__")
        assert hasattr(dataloader, "__len__")
        assert hasattr(dataloader, "dataset")
        assert callable(dataloader.__iter__)

    def test_iteration_with_double_augment_data(
        self, mock_config, mock_cityscapes_constructor, sample_image, sample_target
    ):
        """Test iteration over DoubleAugmentDataLoader returns correct batch structure."""
        # Mock the underlying dataset to return sample data
        with patch.object(DoubleAugmentDataset, "__getitem__") as mock_getitem:
            # Mock DoubleAugmentDataset to return 3 tensors (two images, one target)
            import torch

            mock_tensor1 = torch.randn(3, 256, 512)  # Mock augmented image 1
            mock_tensor2 = torch.randn(3, 256, 512)  # Mock augmented image 2
            mock_target = torch.randint(0, 20, (256, 512))  # Mock target
            mock_getitem.return_value = (mock_tensor1, mock_tensor2, mock_target)

            with patch.object(DoubleAugmentDataset, "__len__", return_value=4):
                dataloader = DoubleAugmentDataLoader(mock_config)

                # Test iteration
                for batch in dataloader:
                    assert len(batch) == 3  # Should have 3 components
                    # Each component should be a batch tensor
                    assert batch[0].shape[0] == dataloader.batch_size  # Batch dimension
                    assert batch[1].shape[0] == dataloader.batch_size  # Batch dimension
                    assert batch[2].shape[0] == dataloader.batch_size  # Batch dimension
                    break  # Just test the first batch
