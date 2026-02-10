"""Tests for YOLO object detection module."""

import pytest
import torch

from src.main import (
    Backbone,
    ConvBlock,
    YOLODetectionHead,
    YOLONet,
    decode_predictions,
    generate_synthetic_batch,
    non_max_suppression,
    yolo_loss,
)


class TestConvBlock:
    """Test cases for ConvBlock."""

    def test_forward_output_shape(self):
        """Test that ConvBlock produces correct output dimensions."""
        layer = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 32, 32, 32)

    def test_forward_stride_two(self):
        """Test ConvBlock with stride 2 halves spatial size."""
        layer = ConvBlock(3, 16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 16, 16, 16)


class TestBackbone:
    """Test cases for Backbone."""

    def test_forward_output_channels(self):
        """Test that Backbone output has expected channel count."""
        backbone = Backbone(
            in_channels=3,
            channel_list=[32, 64],
            block_repeats=[1, 1],
        )
        x = torch.randn(2, 3, 64, 64)
        out = backbone(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 64
        assert backbone.out_channels == 64

    def test_forward_reduces_spatial_size(self):
        """Test that Backbone reduces spatial dimensions."""
        backbone = Backbone(
            in_channels=3,
            channel_list=[32, 64],
            block_repeats=[1, 1],
        )
        x = torch.randn(2, 3, 416, 416)
        out = backbone(x)
        assert out.shape[2] < 416 and out.shape[3] < 416


class TestYOLODetectionHead:
    """Test cases for YOLODetectionHead."""

    def test_forward_output_shape(self):
        """Test that detection head produces (N, S, S, B, 5+C)."""
        head = YOLODetectionHead(
            in_channels=64,
            grid_size=13,
            num_anchors=5,
            num_classes=20,
        )
        x = torch.randn(4, 64, 13, 13)
        out = head(x)
        assert out.shape == (4, 13, 13, 5, 25)


class TestYOLONet:
    """Test cases for YOLONet."""

    def test_forward_output_shape(self):
        """Test that YOLONet produces (N, S, S, B, 5+C)."""
        net = YOLONet(
            in_channels=3,
            grid_size=13,
            num_anchors=5,
            num_classes=20,
            backbone_channels=[32, 64],
            backbone_blocks=[1, 1],
        )
        x = torch.randn(2, 3, 224, 224)
        out = net(x)
        assert out.shape[0] == 2 and out.shape[1] == out.shape[2]
        assert out.shape[3] == 5 and out.shape[4] == 25

    def test_forward_small_grid(self):
        """Test YOLONet with smaller grid and matching backbone output."""
        net = YOLONet(
            in_channels=3,
            grid_size=7,
            num_anchors=2,
            num_classes=10,
            backbone_channels=[16, 32, 64],
            backbone_blocks=[1, 1, 1],
        )
        x = torch.randn(1, 3, 224, 224)
        out = net(x)
        assert out.shape[0] == 1 and out.shape[3] == 2 and out.shape[4] == 15
        assert out.shape[1] == out.shape[2]


class TestDecodePredictions:
    """Test cases for decode_predictions."""

    def test_decode_output_shapes(self):
        """Test that decode_predictions returns correct shapes."""
        grid_size = 13
        num_anchors = 5
        num_classes = 20
        n = 2
        raw = torch.randn(n, grid_size, grid_size, num_anchors, 5 + num_classes)
        boxes, objectness, class_scores = decode_predictions(
            raw, grid_size, (416, 416)
        )
        total_cells = grid_size * grid_size * num_anchors
        assert boxes.shape == (n, total_cells, 4)
        assert objectness.shape == (n, total_cells)
        assert class_scores.shape == (n, total_cells, num_classes)

    def test_decode_boxes_in_normalized_range(self):
        """Test that decoded boxes are in [0, 1] after clamp."""
        grid_size = 7
        num_anchors = 2
        num_classes = 5
        raw = torch.randn(1, grid_size, grid_size, num_anchors, 5 + num_classes)
        boxes, _, _ = decode_predictions(raw, grid_size, (224, 224))
        assert boxes.min() >= 0.0 and boxes.max() <= 1.0

    def test_decode_objectness_in_range(self):
        """Test that objectness is in (0, 1)."""
        grid_size = 5
        raw = torch.randn(1, grid_size, grid_size, 1, 5 + 3)
        _, objectness, _ = decode_predictions(raw, grid_size, (100, 100))
        assert objectness.min() > 0.0 and objectness.max() < 1.0

    def test_decode_class_scores_sum_to_one(self):
        """Test that class_scores rows sum to 1 (softmax)."""
        grid_size = 5
        num_classes = 4
        raw = torch.randn(1, grid_size, grid_size, 1, 5 + num_classes)
        _, _, class_scores = decode_predictions(raw, grid_size, (100, 100))
        row_sums = class_scores[0].sum(dim=-1)
        assert (row_sums - 1.0).abs().max().item() < 1e-5


class TestYOLOLoss:
    """Test cases for yolo_loss."""

    def test_loss_is_scalar(self):
        """Test that yolo_loss returns a scalar tensor."""
        grid_size = 13
        num_anchors = 5
        num_classes = 20
        pred = torch.randn(2, grid_size, grid_size, num_anchors, 5 + num_classes)
        targets = torch.zeros_like(pred)
        targets[:, 2, 3, 1, 4] = 1.0
        targets[:, 2, 3, 1, 5] = 1.0
        loss = yolo_loss(pred, targets, grid_size, num_anchors, num_classes)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_decreases_with_matching_target(self):
        """Test that loss is lower when prediction matches target."""
        grid_size = 5
        num_anchors = 2
        num_classes = 3
        targets = torch.zeros(1, grid_size, grid_size, num_anchors, 5 + num_classes)
        targets[0, 1, 1, 0, :4] = 0.5
        targets[0, 1, 1, 0, 4] = 1.0
        targets[0, 1, 1, 0, 5] = 1.0
        pred_random = torch.randn(1, grid_size, grid_size, num_anchors, 5 + num_classes) * 0.1
        pred_matching = targets.clone()
        pred_matching[..., 5:] = pred_matching[..., 5:] * 10
        loss_random = yolo_loss(
            pred_random, targets, grid_size, num_anchors, num_classes
        )
        loss_matching = yolo_loss(
            pred_matching, targets, grid_size, num_anchors, num_classes
        )
        assert loss_matching.item() < loss_random.item()


class TestNonMaxSuppression:
    """Test cases for non_max_suppression."""

    def test_nms_returns_list_of_indices(self):
        """Test that NMS returns one tensor of indices per batch item."""
        boxes = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.2, 0.2],
                    [0.1, 0.1, 0.3, 0.3],
                    [0.5, 0.5, 0.7, 0.7],
                ]
            ]
        )
        scores = torch.tensor([[0.9, 0.8, 0.7]])
        kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(kept) == 1
        assert kept[0].dtype == torch.long
        assert kept[0].dim() == 1

    def test_nms_empty_boxes(self):
        """Test NMS with empty box list."""
        boxes = torch.zeros(1, 0, 4)
        scores = torch.zeros(1, 0)
        kept = non_max_suppression(boxes, scores)
        assert len(kept) == 1
        assert kept[0].numel() == 0

    def test_nms_single_box(self):
        """Test NMS keeps single box."""
        boxes = torch.tensor([[[0.0, 0.0, 0.1, 0.1]]])
        scores = torch.tensor([[1.0]])
        kept = non_max_suppression(boxes, scores)
        assert kept[0].numel() == 1
        assert kept[0].item() == 0


class TestGenerateSyntheticBatch:
    """Test cases for generate_synthetic_batch."""

    def test_synthetic_batch_shapes(self):
        """Test that synthetic batch has correct image and target shapes."""
        images, targets = generate_synthetic_batch(
            batch_size=4,
            channels=3,
            height=416,
            width=416,
            grid_size=13,
            num_anchors=5,
            num_classes=20,
            device=torch.device("cpu"),
            seed=42,
        )
        assert images.shape == (4, 3, 416, 416)
        assert targets.shape == (4, 13, 13, 5, 25)

    def test_synthetic_targets_have_one_object(self):
        """Test that each image has exactly one cell with objectness 1."""
        _, targets = generate_synthetic_batch(
            batch_size=8,
            channels=3,
            height=64,
            width=64,
            grid_size=4,
            num_anchors=2,
            num_classes=5,
            device=torch.device("cpu"),
            seed=123,
        )
        obj = targets[..., 4]
        per_image = obj.view(8, -1).max(dim=1).values
        assert (per_image == 1.0).all()
