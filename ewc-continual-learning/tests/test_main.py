"""Tests for EWC continual learning module."""

from pathlib import Path
import tempfile

import torch

from src.main import (
    SimpleMLP,
    _load_config,
    compute_accuracy,
    estimate_fisher_diagonal,
    ewc_penalty,
    generate_task_dataset,
    snapshot_parameters,
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


class TestSimpleMLP:
    """Test cases for SimpleMLP."""

    def test_forward_output_shape(self) -> None:
        """Test that forward returns logits with correct shape."""
        model = SimpleMLP(input_dim=10, hidden_dim=8, num_classes=3)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 3)


class TestGenerateTaskDataset:
    """Test cases for generate_task_dataset."""

    def test_dataset_shapes_and_ranges(self) -> None:
        """Test that dataset has correct shapes and label range."""
        device = torch.device("cpu")
        x, y = generate_task_dataset(
            task_id=0,
            num_samples=20,
            input_dim=6,
            num_classes=4,
            device=device,
            seed=123,
        )
        assert x.shape == (20, 6)
        assert y.shape == (20,)
        assert y.min() >= 0 and y.max() < 4


class TestComputeAccuracy:
    """Test cases for compute_accuracy."""

    def test_accuracy_between_zero_and_one(self) -> None:
        """Test that computed accuracy is within [0, 1]."""
        device = torch.device("cpu")
        model = SimpleMLP(input_dim=4, hidden_dim=8, num_classes=2).to(device)
        x = torch.randn(30, 4, device=device)
        y = torch.randint(0, 2, (30,), device=device)
        acc = compute_accuracy(model, x, y, batch_size=10)
        assert 0.0 <= acc <= 1.0


class TestFisherAndEWC:
    """Test cases for Fisher estimation and EWC penalty."""

    def test_fisher_shapes_match_params(self) -> None:
        """Test that Fisher diagonals match model parameter shapes."""
        device = torch.device("cpu")
        model = SimpleMLP(input_dim=5, hidden_dim=7, num_classes=3).to(device)
        x = torch.randn(40, 5, device=device)
        y = torch.randint(0, 3, (40,), device=device)
        fisher = estimate_fisher_diagonal(model, x, y, batch_size=10)
        params = list(model.parameters())
        assert len(fisher) == len(params)
        for f, p in zip(fisher, params):
            assert f.shape == p.shape

    def test_ewc_penalty_zero_when_params_equal_snapshot(self) -> None:
        """Test that EWC penalty is zero when params equal previous snapshot."""
        device = torch.device("cpu")
        model = SimpleMLP(input_dim=3, hidden_dim=4, num_classes=2).to(device)
        x = torch.randn(20, 3, device=device)
        y = torch.randint(0, 2, (20,), device=device)
        fisher = estimate_fisher_diagonal(model, x, y, batch_size=5)
        prev = snapshot_parameters(model)
        penalty = ewc_penalty(model, fisher, prev, lambda_ewc=10.0)
        assert torch.allclose(penalty, torch.tensor(0.0, device=device), atol=1e-6)

    def test_ewc_penalty_positive_when_params_move(self) -> None:
        """Test that EWC penalty increases when parameters deviate from snapshot."""
        device = torch.device("cpu")
        model = SimpleMLP(input_dim=3, hidden_dim=4, num_classes=2).to(device)
        x = torch.randn(20, 3, device=device)
        y = torch.randint(0, 2, (20,), device=device)
        fisher = estimate_fisher_diagonal(model, x, y, batch_size=5)
        prev = snapshot_parameters(model)
        # Perturb parameters
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * torch.randn_like(p))
        penalty = ewc_penalty(model, fisher, prev, lambda_ewc=10.0)
        assert penalty.item() > 0.0

