import numpy as np
import pytest
import torch

from config.schema.data import DatasetType
from data.dataset.label_mapping import LabelMapping


@pytest.fixture(params=list(DatasetType))
def label_mapping_case(request):
    """Provides (transform, sample_input, expected_output) for each dataset type."""
    match request.param:
        case DatasetType.CITYSCAPES:
            transform = LabelMapping._get_cityscapes_transform()
            # Use a few known real Cityscapes indices:
            # 7 (road) -> 1, 9 (parking, ignored) -> 0, 33 (bicycle) -> 19
            sample_input = np.array([[[7, 9, 33]]], dtype=np.int64)
            expected = torch.tensor([[[1, 0, 19]]], dtype=torch.int64)
        case DatasetType.VOC_SEGMENTATION:
            transform = LabelMapping._get_voc_transform()
            # Input: (C, H, W) label map containing valid labels and the void label (255)
            sample_input = np.array([[[0, 1, 10, 20], [255, 5, 15, 255]]], dtype=np.uint8)
            # Expected: 255 (void) -> 0 (background), all other labels unchanged
            expected = torch.tensor([[[0, 1, 10, 20], [0, 5, 15, 0]]], dtype=torch.int64)
    return transform, sample_input, expected
