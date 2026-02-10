"""Tests for multi-task learning module."""

import tempfile
from pathlib import Path

import torch

from src.main import (
    MultiTaskModel,
    SharedEncoder,
    TaskHead,
    _load_config,
    generate_synthetic_multi_task_data,
    multi_task_loss,
)


class TestLoadConfig:
    """Test cases for _load_config."""

    def test_load_config_returns_dict(self) -> None:
        """Test that _load_config returns a dict with expected keys."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  input_dim: 8\n")
            path = f.name
        try:
            cfg = _load_config(path)
            assert isinstance(cfg, dict)
            assert "model" in cfg
            assert cfg["model"]["input_dim"] == 8
        finally:
            Path(path).unlink(missing_ok=True)


class TestSharedEncoder:
    """Test cases for SharedEncoder."""

    def test_forward_output_shape(self) -> None:
        """Test that SharedEncoder outputs (N, shared_dim)."""
        encoder = SharedEncoder(
            input_dim=10,
            hidden_dims=[32, 32],
            shared_dim=16,
        )
        x = torch.randn(5, 10)
        out = encoder(x)
        assert out.shape == (5, 16)


class TestTaskHead:
    """Test cases for TaskHead."""

    def test_forward_output_shape(self) -> None:
        """Test that TaskHead outputs (N, num_classes)."""
        head = TaskHead(shared_dim=16, num_classes=4)
        shared = torch.randn(5, 16)
        out = head(shared)
        assert out.shape == (5, 4)


class TestMultiTaskModel:
    """Test cases for MultiTaskModel."""

    def test_forward_returns_one_logits_per_task(self) -> None:
        """Test that forward returns list of logits with correct shapes."""
        task_configs = [
            {"type": "classification", "num_classes": 3},
            {"type": "classification", "num_classes": 2},
        ]
        model = MultiTaskModel(
            input_dim=8,
            shared_hidden=[32],
            shared_dim=16,
            task_configs=task_configs,
        )
        x = torch.randn(4, 8)
        logits_list = model(x)
        assert len(logits_list) == 2
        assert logits_list[0].shape == (4, 3)
        assert logits_list[1].shape == (4, 2)

    def test_num_tasks(self) -> None:
        """Test that num_tasks matches number of task configs."""
        task_configs = [
            {"type": "classification", "num_classes": 2},
            {"type": "classification", "num_classes": 2},
            {"type": "classification", "num_classes": 5},
        ]
        model = MultiTaskModel(
            input_dim=4,
            shared_hidden=[8],
            shared_dim=8,
            task_configs=task_configs,
        )
        assert model.num_tasks() == 3


class TestMultiTaskLoss:
    """Test cases for multi_task_loss."""

    def test_loss_is_scalar(self) -> None:
        """Test that multi_task_loss returns a scalar tensor."""
        logits_list = [
            torch.randn(6, 3),
            torch.randn(6, 2),
        ]
        labels_list = [
            torch.randint(0, 3, (6,)),
            torch.randint(0, 2, (6,)),
        ]
        loss = multi_task_loss(logits_list, labels_list)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_weights_affect_loss(self) -> None:
        """Test that different weights change the loss value."""
        logits_list = [torch.randn(4, 2), torch.randn(4, 2)]
        labels_list = [torch.randint(0, 2, (4,)), torch.randint(0, 2, (4,))]
        loss_unit = multi_task_loss(logits_list, labels_list, [1.0, 1.0])
        loss_heavy = multi_task_loss(logits_list, labels_list, [10.0, 0.1])
        assert loss_heavy.item() != loss_unit.item()


class TestGenerateSyntheticMultiTaskData:
    """Test cases for generate_synthetic_multi_task_data."""

    def test_shapes_and_label_ranges(self) -> None:
        """Test that features and labels have correct shapes and ranges."""
        device = torch.device("cpu")
        features, labels_list = generate_synthetic_multi_task_data(
            num_samples=50,
            input_dim=10,
            task_num_classes=[3, 4, 2],
            device=device,
            seed=42,
        )
        assert features.shape == (50, 10)
        assert len(labels_list) == 3
        assert labels_list[0].shape == (50,)
        assert labels_list[0].min() >= 0 and labels_list[0].max() < 3
        assert labels_list[1].min() >= 0 and labels_list[1].max() < 4
        assert labels_list[2].min() >= 0 and labels_list[2].max() < 2
