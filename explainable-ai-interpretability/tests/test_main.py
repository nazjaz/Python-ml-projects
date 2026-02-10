"""Tests for explainable AI interpretability module."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.main import (
    ModelWithAttention,
    SimpleClassifier,
    _load_config,
    format_attention_for_visualization,
    generate_synthetic_data,
    get_attention_weights,
    integrated_gradients,
    lime_explain,
)


class TestLoadConfig:
    """Test cases for _load_config."""

    def test_load_config_returns_dict(self) -> None:
        """Test that _load_config returns a dict with expected keys."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  input_dim: 4\n")
            path = f.name
        try:
            cfg = _load_config(path)
            assert isinstance(cfg, dict)
            assert "model" in cfg
            assert cfg["model"]["input_dim"] == 4
        finally:
            Path(path).unlink(missing_ok=True)


class TestSimpleClassifier:
    """Test cases for SimpleClassifier."""

    def test_forward_output_shape(self) -> None:
        """Test that SimpleClassifier returns logits of correct shape."""
        model = SimpleClassifier(input_dim=6, hidden_dim=8, num_classes=3)
        x = torch.randn(4, 6)
        out = model(x)
        assert out.shape == (4, 3)


class TestModelWithAttention:
    """Test cases for ModelWithAttention."""

    def test_forward_sets_last_attention(self) -> None:
        """Test that forward stores attention weights."""
        model = ModelWithAttention(
            input_dim=4,
            embed_dim=8,
            num_heads=2,
            num_classes=2,
        )
        x = torch.randn(2, 4)
        _ = model(x)
        assert model.last_attention is not None
        assert model.last_attention.dim() >= 2

    def test_forward_output_shape(self) -> None:
        """Test that forward returns logits of correct shape."""
        model = ModelWithAttention(4, 8, 2, 2)
        x = torch.randn(3, 4)
        out = model(x)
        assert out.shape == (3, 2)


class TestIntegratedGradients:
    """Test cases for integrated_gradients."""

    def test_attribution_shape(self) -> None:
        """Test that integrated_gradients returns correct attribution shape."""
        model = SimpleClassifier(input_dim=5, hidden_dim=8, num_classes=2)
        x = torch.randn(1, 5)
        attr = integrated_gradients(model, x, target_class=0, steps=10)
        assert attr.shape == (5,)

    def test_baseline_zero_gives_sensible_attribution(self) -> None:
        """Test that attribution has finite values."""
        model = SimpleClassifier(input_dim=4, hidden_dim=8, num_classes=2)
        x = torch.randn(1, 4)
        attr = integrated_gradients(model, x, target_class=0, baseline=None, steps=5)
        assert torch.isfinite(attr).all()


class TestLimeExplain:
    """Test cases for lime_explain."""

    def test_returns_feature_importance_shape(self) -> None:
        """Test that LIME returns array of length num_features."""
        model = SimpleClassifier(input_dim=6, hidden_dim=8, num_classes=2)
        x = torch.randn(1, 6)
        importance = lime_explain(
            model, x, num_samples=50, num_features=6, kernel_width=0.5
        )
        assert isinstance(importance, np.ndarray)
        assert importance.shape == (6,)
        assert np.isfinite(importance).all()


class TestGetAttentionWeights:
    """Test cases for get_attention_weights."""

    def test_returns_weights_for_attention_model(self) -> None:
        """Test that get_attention_weights returns tensor for ModelWithAttention."""
        model = ModelWithAttention(4, 8, 2, 2)
        x = torch.randn(1, 4)
        _ = model(x)
        weights = get_attention_weights(model)
        assert weights is not None
        assert isinstance(weights, torch.Tensor)

    def test_returns_none_for_simple_classifier(self) -> None:
        """Test that get_attention_weights returns None for model without attention."""
        model = SimpleClassifier(4, 8, 2)
        weights = get_attention_weights(model)
        assert weights is None


class TestFormatAttentionForVisualization:
    """Test cases for format_attention_for_visualization."""

    def test_returns_string(self) -> None:
        """Test that format returns a non-empty string."""
        attn = torch.softmax(torch.randn(2, 4, 4), dim=-1)
        s = format_attention_for_visualization(attn)
        assert isinstance(s, str)
        assert len(s) > 0


class TestGenerateSyntheticData:
    """Test cases for generate_synthetic_data."""

    def test_shapes_and_label_range(self) -> None:
        """Test that synthetic data has correct shapes and label range."""
        device = torch.device("cpu")
        x, y = generate_synthetic_data(
            num_samples=30,
            input_dim=6,
            num_classes=3,
            device=device,
            seed=42,
        )
        assert x.shape == (30, 6)
        assert y.shape == (30,)
        assert y.min() >= 0 and y.max() < 3
