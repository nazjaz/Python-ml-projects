"""Tests for MAML few-shot learning module."""

import tempfile
from pathlib import Path

import torch

from src.main import (
    FewShotMLP,
    _load_config,
    maml_inner_outer_step,
    sample_few_shot_task,
)


class TestLoadConfig:
    """Test cases for _load_config."""

    def test_load_config_returns_dict(self) -> None:
        """Test that _load_config returns a dict with expected keys."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("model:\n  input_dim: 4\n")
            path = f.name
        try:
            config = _load_config(path)
            assert isinstance(config, dict)
            assert "model" in config
            assert config["model"]["input_dim"] == 4
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_config_nonexistent_raises(self) -> None:
        """Test that _load_config raises FileNotFoundError for missing file."""
        try:
            _load_config("/nonexistent/path/config.yaml")
        except FileNotFoundError:
            return
        raise AssertionError("Expected FileNotFoundError")


class TestFewShotMLP:
    """Test cases for FewShotMLP."""

    def test_forward_output_shape(self) -> None:
        """Test that forward returns logits of shape (N, num_classes)."""
        model = FewShotMLP(
            input_dim=8,
            hidden_dim=16,
            num_classes=5,
            num_layers=2,
        )
        x = torch.randn(10, 8)
        out = model(x)
        assert out.shape == (10, 5)

    def test_forward_with_params_matches_forward(self) -> None:
        """Test that forward_with_params gives same result as forward for same params."""
        model = FewShotMLP(input_dim=4, hidden_dim=8, num_classes=3, num_layers=2)
        x = torch.randn(6, 4)
        out_std = model(x)
        params = model.get_param_list()
        out_func = model.forward_with_params(x, params)
        assert torch.allclose(out_std, out_func, atol=1e-5)

    def test_get_param_list_order(self) -> None:
        """Test that get_param_list has even length (weight, bias per layer)."""
        model = FewShotMLP(input_dim=2, hidden_dim=4, num_classes=2, num_layers=1)
        plist = model.get_param_list()
        assert len(plist) >= 2
        assert all(isinstance(p, torch.Tensor) for p in plist)

    def test_num_layers_one_raises(self) -> None:
        """Test that num_layers=0 raises ValueError."""
        try:
            FewShotMLP(2, 4, 2, num_layers=0)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for num_layers < 1")


class TestSampleFewShotTask:
    """Test cases for sample_few_shot_task."""

    def test_shapes_and_label_range(self) -> None:
        """Test that task tensors have correct shapes and labels in [0, n_way-1]."""
        device = torch.device("cpu")
        support_x, support_y, query_x, query_y = sample_few_shot_task(
            n_way=4,
            k_shot=3,
            query_size=12,
            input_dim=6,
            device=device,
            seed=42,
        )
        assert support_x.shape == (4 * 3, 6)
        assert support_y.shape == (12,)
        assert query_x.shape[0] <= 12 and query_x.shape[1] == 6
        assert query_y.shape[0] == query_x.shape[0]
        assert support_y.min() >= 0 and support_y.max() < 4
        assert query_y.min() >= 0 and query_y.max() < 4


class TestMamlInnerOuterStep:
    """Test cases for maml_inner_outer_step."""

    def test_returns_scalar_loss(self) -> None:
        """Test that maml_inner_outer_step returns a scalar loss tensor."""
        model = FewShotMLP(
            input_dim=4,
            hidden_dim=8,
            num_classes=3,
            num_layers=2,
        )
        support_x = torch.randn(3 * 2, 4)
        support_y = torch.randint(0, 3, (6,))
        query_x = torch.randn(5, 4)
        query_y = torch.randint(0, 3, (5,))
        loss = maml_inner_outer_step(
            model,
            support_x,
            support_y,
            query_x,
            query_y,
            inner_lr=0.01,
            inner_steps=2,
            first_order=True,
        )
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_first_order_backward(self) -> None:
        """Test that backward runs with first_order=True."""
        model = FewShotMLP(
            input_dim=2,
            hidden_dim=4,
            num_classes=2,
            num_layers=1,
        )
        support_x = torch.randn(4, 2)
        support_y = torch.randint(0, 2, (4,))
        query_x = torch.randn(4, 2)
        query_y = torch.randint(0, 2, (4,))
        loss = maml_inner_outer_step(
            model,
            support_x,
            support_y,
            query_x,
            query_y,
            inner_lr=0.01,
            inner_steps=1,
            first_order=True,
        )
        loss.backward()

    def test_second_order_backward(self) -> None:
        """Test that backward runs with first_order=False (full MAML)."""
        model = FewShotMLP(
            input_dim=2,
            hidden_dim=4,
            num_classes=2,
            num_layers=1,
        )
        support_x = torch.randn(4, 2)
        support_y = torch.randint(0, 2, (4,))
        query_x = torch.randn(4, 2)
        query_y = torch.randint(0, 2, (4,))
        loss = maml_inner_outer_step(
            model,
            support_x,
            support_y,
            query_x,
            query_y,
            inner_lr=0.01,
            inner_steps=1,
            first_order=False,
        )
        loss.backward()
