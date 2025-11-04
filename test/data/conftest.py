"""Pytest configuration and fixtures for dataset tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image
from torchvision.datasets import Cityscapes


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    # Create a 512x256 RGB image with random data
    image_data = np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8)
    return Image.fromarray(image_data, "RGB")


@pytest.fixture
def sample_target():
    """Create a sample PIL target mask for testing."""
    # Create a 512x256 grayscale image with class labels 0-19
    target_data = np.random.randint(0, 20, (256, 512), dtype=np.uint8)
    return Image.fromarray(target_data, "L")


@pytest.fixture
def mock_cityscapes_classes():
    """Create mock Cityscapes classes for testing class filtering."""
    mock_classes = []

    # Create a mix of classes, some ignored, some not (realistic Cityscapes class names)
    class_configs = [
        ("unlabeled", 0, True),
        ("road", 1, False),
        ("sidewalk", 2, False),
        ("building", 3, False),
        ("wall", 4, True),
        ("fence", 5, True),
        ("vegetation", 6, False),
        ("sky", 7, False),
    ]

    for name, class_id, ignore in class_configs:
        mock_class = MagicMock()
        mock_class.name = name
        mock_class.id = class_id
        mock_class.ignore_in_eval = ignore
        mock_classes.append(mock_class)

    return mock_classes


@pytest.fixture
def mock_cityscapes_dataset(sample_image, sample_target, mock_cityscapes_classes):
    """Mock a Cityscapes dataset with sample data."""
    mock_dataset = MagicMock(spec=Cityscapes)

    # Mock dataset properties
    mock_dataset.__len__.return_value = 10
    mock_dataset.__getitem__.return_value = (sample_image, sample_target)
    mock_dataset.classes = mock_cityscapes_classes

    return mock_dataset


@pytest.fixture
def mock_cityscapes_constructor(mock_cityscapes_dataset):
    """Mock the Cityscapes constructor to return our mock dataset."""
    with patch("data.dataset.base.Cityscapes") as mock_constructor:
        mock_constructor.return_value = mock_cityscapes_dataset
        yield mock_constructor
