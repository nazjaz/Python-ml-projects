"""Tests for model compression deployment module."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.main import (
    MLP,
    _load_config,
    apply_pruning,
    apply_quantization,
    count_parameters,
    distillation_loss,
    evaluate_accuracy,
    generate_synthetic_data,
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


class TestMLP:
    """Test cases for MLP."""

    def test_forward_output_shape(self) -> None:
        """Test that MLP returns logits of correct shape."""
        model = MLP(input_dim=6, hidden_dims=[8, 8], num_classes=3)
        x = torch.randn(4, 6)
        out = model(x)
        assert out.shape == (4, 3)


class TestQuantization:
    """Test cases for apply_quantization."""

    def test_quantized_model_forward(self) -> None:
        """Test that quantized model runs forward and returns correct shape."""
        model = MLP(input_dim=4, hidden_dims=[8], num_classes=2)
        try:
            quant = apply_quantization(model, dtype="qint8")
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "linear_prepack" in str(e):
                pytest.skip("Quantization engine not available on this build")
            raise
        x = torch.randn(2, 4)
        out = quant(x)
        assert out.shape == (2, 2)


class TestPruning:
    """Test cases for apply_pruning."""

    def test_pruning_reduces_nonzero_weights(self) -> None:
        """Test that pruning zeroes out some weights."""
        model = MLP(input_dim=4, hidden_dims=[6], num_classes=2)
        total_before = sum((p != 0).sum().item() for p in model.parameters())
        apply_pruning(model, amount=0.5, make_permanent=True)
        total_after = sum((p != 0).sum().item() for p in model.parameters())
        assert total_after < total_before

    def test_pruned_model_forward(self) -> None:
        """Test that pruned model still runs forward."""
        model = MLP(input_dim=4, hidden_dims=[8], num_classes=2)
        apply_pruning(model, amount=0.2, make_permanent=True)
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 2)


class TestDistillationLoss:
    """Test cases for distillation_loss."""

    def test_loss_is_scalar(self) -> None:
        """Test that distillation_loss returns a scalar tensor."""
        student_logits = torch.randn(4, 3)
        teacher_logits = torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        loss = distillation_loss(
            student_logits, teacher_logits, labels,
            temperature=2.0, alpha=0.5,
        )
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestCountParameters:
    """Test cases for count_parameters."""

    def test_count_positive(self) -> None:
        """Test that parameter count is positive for non-empty model."""
        model = MLP(4, [8], 2)
        n = count_parameters(model)
        assert n > 0

    def test_student_fewer_than_teacher(self) -> None:
        """Test that smaller MLP has fewer parameters."""
        teacher = MLP(8, [32, 32], 3)
        student = MLP(8, [16], 3)
        assert count_parameters(student) < count_parameters(teacher)


class TestEvaluateAccuracy:
    """Test cases for evaluate_accuracy."""

    def test_accuracy_in_range(self) -> None:
        """Test that accuracy is between 0 and 1."""
        device = torch.device("cpu")
        model = MLP(4, [8], 2).to(device)
        x = torch.randn(20, 4, device=device)
        y = torch.randint(0, 2, (20,), device=device)
        acc = evaluate_accuracy(model, x, y, batch_size=10)
        assert 0.0 <= acc <= 1.0


class TestGenerateSyntheticData:
    """Test cases for generate_synthetic_data."""

    def test_shapes_and_label_range(self) -> None:
        """Test that synthetic data has correct shapes and label range."""
        device = torch.device("cpu")
        x, y = generate_synthetic_data(
            num_samples=50,
            input_dim=6,
            num_classes=4,
            device=device,
            seed=42,
        )
        assert x.shape == (50, 6)
        assert y.shape == (50,)
        assert y.min() >= 0 and y.max() < 4
