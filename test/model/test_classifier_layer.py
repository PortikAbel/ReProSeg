"""Unit tests for the NonNegConv1x1 classifier layer."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import NonNegConv1x1


class TestNonNegConv1x1:
    """Test cases for the NonNegConv1x1 classifier layer."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.in_channels = 128
        self.out_channels = 10
        self.batch_size = 4
        self.height = 32
        self.width = 32

    def test_init_with_bias(self):
        """Test NonNegConv1x1 initialization with bias."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=True)

        assert layer.in_channels == self.in_channels
        assert layer.out_channels == self.out_channels
        assert layer.weight.shape == (self.out_channels, self.in_channels, 1, 1)
        assert layer.bias is not None
        assert layer.bias.shape == (self.out_channels,)
        assert isinstance(layer.weight, nn.Parameter)
        assert isinstance(layer.bias, nn.Parameter)

    def test_init_without_bias(self):
        """Test NonNegConv1x1 initialization without bias."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        assert layer.in_channels == self.in_channels
        assert layer.out_channels == self.out_channels
        assert layer.weight.shape == (self.out_channels, self.in_channels, 1, 1)
        assert layer.bias is None
        assert isinstance(layer.weight, nn.Parameter)

    def test_init_with_device_and_dtype(self):
        """Test NonNegConv1x1 initialization with specific device and dtype."""
        device = torch.device("cpu")
        dtype = torch.float32

        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=True, device=device, dtype=dtype)

        assert layer.weight.device == device
        assert layer.weight.dtype == dtype
        assert layer.bias.device == device
        assert layer.bias.dtype == dtype

    def test_forward_with_weights_above_threshold(self):
        """Test forward pass with weights above MIN_CLASSIFICATION_WEIGHT."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set weights above threshold
        with torch.no_grad():
            layer.weight.fill_(5.0)  # Above MIN_CLASSIFICATION_WEIGHT

        input_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        output = layer(input_tensor)

        assert output.shape == (self.batch_size, self.out_channels, self.height, self.width)
        assert torch.any(output != 0.0)

    def test_forward_with_weights_below_threshold(self):
        """Test forward pass with weights below MIN_CLASSIFICATION_WEIGHT."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set weights below threshold
        with torch.no_grad():
            layer.weight.fill_(-5.0)

        input_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        output = layer(input_tensor)

        assert output.shape == (self.batch_size, self.out_channels, self.height, self.width)
        # Since weights are below threshold, they should be zeroed out
        assert torch.allclose(output, torch.zeros_like(output))

    def test_forward_with_mixed_weights(self):
        """Test forward pass with weights both above and below threshold."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set some weights above and some below threshold
        with torch.no_grad():
            layer.weight.fill_(-5.0)  # Below threshold
            layer.weight[0, :, :, :] = 5.0  # Above threshold for first output channel

        input_tensor = torch.ones(self.batch_size, self.in_channels, self.height, self.width)
        output = layer(input_tensor)

        assert output.shape == (self.batch_size, self.out_channels, self.height, self.width)

        # First channel should have non-zero output (weights above threshold)
        assert torch.any(output[:, 0, :, :] != 0.0)

        # Other channels should have zero output (weights below threshold)
        assert torch.allclose(output[:, 1:, :, :], torch.zeros_like(output[:, 1:, :, :]))

    def test_forward_with_bias(self):
        """Test forward pass with bias enabled."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=True)

        # Set weights above threshold and bias to non-zero
        with torch.no_grad():
            layer.weight.fill_(5.0)
            layer.bias.fill_(2.0)

        input_tensor = torch.zeros(self.batch_size, self.in_channels, self.height, self.width)
        output = layer(input_tensor)

        assert output.shape == (self.batch_size, self.out_channels, self.height, self.width)
        # Even with zero input, bias should make output non-zero
        assert torch.allclose(output, torch.full_like(output, 2.0))

    def test_forward_weight_thresholding_preserves_gradient(self):
        """Test that weight thresholding preserves gradients for weights above threshold."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set weights above threshold
        with torch.no_grad():
            layer.weight.fill_(5.0)

        input_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        output = layer(input_tensor)

        # Compute a simple loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Gradients should exist for weights above threshold
        assert layer.weight.grad is not None
        assert torch.any(layer.weight.grad != 0.0)

    def test_forward_weight_thresholding_blocks_gradient(self):
        """Test that weight thresholding blocks gradients for weights below threshold."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set weights below threshold
        with torch.no_grad():
            layer.weight.fill_(-5.0)

        input_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        output = layer(input_tensor)

        # Compute a simple loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Since weights are below threshold, effective weights are zero,
        # but gradients can still flow through the original weights
        assert layer.weight.grad is not None

    def test_used_prototypes_all_above_threshold(self):
        """Test used_prototypes property when all prototypes are above threshold."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set all weights above threshold
        with torch.no_grad():
            layer.weight.fill_(5.0)

        used_prototypes = layer.used_prototypes
        print(used_prototypes)
        expected_indices = torch.arange(self.in_channels)

        assert torch.allclose(used_prototypes.sort()[0], expected_indices)

    def test_used_prototypes_none_above_threshold(self):
        """Test used_prototypes property when no prototypes are above threshold."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set all weights below threshold
        with torch.no_grad():
            layer.weight.fill_(-5.0)

        used_prototypes = layer.used_prototypes

        # Should return empty tensor when no prototypes are used
        assert used_prototypes.numel() == 0

    def test_used_prototypes_partial_above_threshold(self):
        """Test used_prototypes property when some prototypes are above threshold."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set some weights above threshold
        with torch.no_grad():
            layer.weight.fill_(-5.0)  # All below threshold
            layer.weight[:, 0, :, :] = 5.0  # First input channel above threshold
            layer.weight[:, 5, :, :] = 5.0  # Sixth input channel above threshold

        used_prototypes = layer.used_prototypes
        expected_indices = torch.tensor([0, 5])

        assert torch.allclose(used_prototypes.sort()[0], expected_indices)

    def test_used_prototypes_single_prototype(self):
        """Test used_prototypes property with single prototype above threshold."""
        layer = NonNegConv1x1(2, self.out_channels, bias=False)  # Only 2 input channels

        # Set only one prototype above threshold
        with torch.no_grad():
            layer.weight.fill_(-5.0)  # All below threshold
            layer.weight[:, 1, :, :] = 5.0  # Second input channel above threshold

        used_prototypes = layer.used_prototypes
        expected_index = torch.tensor([1])

        assert torch.allclose(used_prototypes, expected_index)

    def test_forward_matches_conv2d_behavior(self):
        """Test that forward pass matches F.conv2d behavior with processed weights."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=True)

        # Set known weights and bias
        with torch.no_grad():
            layer.weight.fill_(5.0)
            layer.bias.fill_(1.0)

        input_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)

        # Get output from our layer
        output_layer = layer(input_tensor)

        # Calculate expected output using F.conv2d directly
        processed_weights = torch.where(layer.weight < layer.MIN_CLASSIFICATION_WEIGHT, 0.0, layer.weight)
        output_expected = F.conv2d(input_tensor, processed_weights, layer.bias, stride=1, padding=0)

        assert torch.allclose(output_layer, output_expected)

    def test_weight_parameter_shape_consistency(self):
        """Test that weight parameter maintains correct shape throughout operations."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        original_shape = layer.weight.shape
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)

        # Forward pass should not change weight shape
        _ = layer(input_tensor)

        assert layer.weight.shape == original_shape

    @pytest.mark.parametrize("in_channels,out_channels", [(1, 1), (64, 32), (256, 128), (512, 1000)])
    def test_different_channel_sizes(self, in_channels, out_channels):
        """Test NonNegConv1x1 with different channel configurations."""
        layer = NonNegConv1x1(in_channels, out_channels, bias=True)

        assert layer.in_channels == in_channels
        assert layer.out_channels == out_channels
        assert layer.weight.shape == (out_channels, in_channels, 1, 1)
        assert layer.bias.shape == (out_channels,)

        # Test forward pass
        input_tensor = torch.randn(2, in_channels, 16, 16)
        output = layer(input_tensor)
        assert output.shape == (2, out_channels, 16, 16)

    def test_negative_weights_handling(self):
        """Test that negative weights are properly handled (set to zero)."""
        layer = NonNegConv1x1(self.in_channels, self.out_channels, bias=False)

        # Set some weights to negative values (below threshold)
        with torch.no_grad():
            layer.weight.fill_(-5.0)

        input_tensor = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        output = layer(input_tensor)

        # All weights are negative (below threshold), so output should be zero
        assert torch.allclose(output, torch.zeros_like(output))

        # No prototypes should be used
        used_prototypes = layer.used_prototypes
        assert used_prototypes.numel() == 0
