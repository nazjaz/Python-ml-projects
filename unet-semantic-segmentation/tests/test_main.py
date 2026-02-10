"""Tests for U-Net semantic segmentation module."""

import pytest
import torch

from src.main import (
    DecoderBlock,
    DoubleConv,
    EncoderBlock,
    UNet,
    generate_synthetic_batch,
    segmentation_loss,
)


class TestDoubleConv:
    """Test cases for DoubleConv."""

    def test_forward_output_shape(self):
        """Test that DoubleConv preserves spatial size and changes channels."""
        layer = DoubleConv(3, 64)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 64, 32, 32)

    def test_forward_different_sizes(self):
        """Test DoubleConv on non-square input."""
        layer = DoubleConv(8, 16)
        x = torch.randn(1, 8, 64, 48)
        out = layer(x)
        assert out.shape == (1, 16, 64, 48)


class TestEncoderBlock:
    """Test cases for EncoderBlock."""

    def test_forward_returns_pooled_and_skip(self):
        """Test that EncoderBlock returns pooled output and skip with correct shapes."""
        block = EncoderBlock(3, 64)
        x = torch.randn(2, 3, 32, 32)
        pooled, skip = block(x)
        assert pooled.shape == (2, 64, 16, 16)
        assert skip.shape == (2, 64, 32, 32)

    def test_forward_skip_spatial_larger_than_pooled(self):
        """Test that skip has twice the spatial size of pooled."""
        block = EncoderBlock(16, 32)
        x = torch.randn(1, 16, 20, 20)
        pooled, skip = block(x)
        assert skip.shape[2] == 2 * pooled.shape[2]
        assert skip.shape[3] == 2 * pooled.shape[3]


class TestDecoderBlock:
    """Test cases for DecoderBlock."""

    def test_forward_output_shape_matches_skip(self):
        """Test that DecoderBlock output matches skip spatial size."""
        block = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        out = block(x, skip)
        assert out.shape == (2, 64, 32, 32)

    def test_forward_concatenates_skip(self):
        """Test that output has correct channel count after concat and conv."""
        block = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)
        x = torch.randn(1, 64, 8, 8)
        skip = torch.randn(1, 32, 16, 16)
        out = block(x, skip)
        assert out.shape == (1, 32, 16, 16)


class TestUNet:
    """Test cases for UNet."""

    def test_forward_output_shape_same_as_input_when_divisible(self):
        """Test that output H,W match input when divisible by 2^depth."""
        net = UNet(in_channels=3, num_classes=2, base_channels=16, depth=3)
        x = torch.randn(2, 3, 64, 64)
        out = net(x)
        assert out.shape == (2, 2, 64, 64)

    def test_forward_num_classes_channels(self):
        """Test that output has num_classes channels."""
        net = UNet(in_channels=3, num_classes=5, base_channels=32, depth=2)
        x = torch.randn(1, 3, 32, 32)
        out = net(x)
        assert out.shape[0] == 1 and out.shape[1] == 5

    def test_forward_depth_4(self):
        """Test UNet with depth 4 and input size 128."""
        net = UNet(in_channels=3, num_classes=2, base_channels=64, depth=4)
        x = torch.randn(1, 3, 128, 128)
        out = net(x)
        assert out.shape == (1, 2, 128, 128)


class TestSegmentationLoss:
    """Test cases for segmentation_loss."""

    def test_loss_is_scalar(self):
        """Test that segmentation_loss returns a scalar tensor."""
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = segmentation_loss(logits, targets, num_classes=3)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_reduces_with_correct_prediction(self):
        """Test that loss is lower when logits match targets."""
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        wrong_logits = torch.randn(1, 2, 4, 4) * 0.1
        right_logits = torch.randn(1, 2, 4, 4) * 0.1
        right_logits[:, 0, :, :] = 10.0
        loss_wrong = segmentation_loss(wrong_logits, targets, num_classes=2)
        loss_right = segmentation_loss(right_logits, targets, num_classes=2)
        assert loss_right.item() < loss_wrong.item()


class TestGenerateSyntheticBatch:
    """Test cases for generate_synthetic_batch."""

    def test_synthetic_batch_shapes(self):
        """Test that synthetic batch has correct image and label shapes."""
        images, labels = generate_synthetic_batch(
            batch_size=4,
            channels=3,
            height=64,
            width=64,
            num_classes=3,
            device=torch.device("cpu"),
            seed=42,
        )
        assert images.shape == (4, 3, 64, 64)
        assert labels.shape == (4, 64, 64)

    def test_labels_in_valid_range(self):
        """Test that labels are in [0, num_classes-1]."""
        _, labels = generate_synthetic_batch(
            batch_size=2,
            channels=3,
            height=16,
            width=16,
            num_classes=5,
            device=torch.device("cpu"),
            seed=123,
        )
        assert labels.min() >= 0 and labels.max() < 5
