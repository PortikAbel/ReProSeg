"""Unit tests for the ClassFilter class."""

import numpy as np
import torch
import torchvision

from data.dataset.class_filter import ClassFilter


class TestClassFilter:
    """Test cases for the ClassFilter class."""

    def test_filter_cityscapes_classes(self, mock_cityscapes_classes):
        """Test filtering removes ignored classes but keeps unlabeled."""
        
        filtered = ClassFilter.filter_cityscapes_classes(mock_cityscapes_classes)
        
        # Should keep unlabeled (index 0) + non-ignored classes
        # Expected: unlabeled, road, sidewalk, building, vegetation, sky
        assert len(filtered) == 6
        assert filtered[0].name == "unlabeled"
        assert all(not c.ignore_in_eval for c in filtered[1:])
        
        filtered_names = [c.name for c in filtered]
        assert "wall" not in filtered_names
        assert "fence" not in filtered_names

    def test_get_cityscapes_transform_returns_compose(self, mock_cityscapes_classes):
        """Test transform creation returns a Compose object."""
        
        transform = ClassFilter.get_cityscapes_transform(mock_cityscapes_classes)
        
        assert transform is not None
        assert callable(transform)
        assert isinstance(transform, torchvision.transforms.v2.Compose)

    def test_get_cityscapes_transform_maps_classes_correctly(self, mock_cityscapes_classes):
        """Test transform correctly maps class indices."""
        
        transform = ClassFilter.get_cityscapes_transform(mock_cityscapes_classes)
        
        # Create a sample target with class indices
        sample_target = np.arange(8, dtype=np.uint8).reshape((1, 2, 4))
        
        result = transform(sample_target)
        
        # Result should be a tensor
        assert isinstance(result, torch.Tensor)
        
        # Expected mapping:
        # 0 (unlabeled) -> 0
        # 1 (road) -> 1
        # 2 (sidewalk) -> 2
        # 3 (building) -> 3
        # 4 (wall, ignored) -> 0
        # 5 (fence, ignored) -> 0
        # 6 (vegetation) -> 4
        # 7 (sky) -> 5
        expected = torch.tensor([[
            [0, 1, 2, 3],
            [0, 0, 4, 5],
        ]], dtype=torch.uint8)
        
        assert result.shape == expected.shape
        assert torch.equal(result, expected)
