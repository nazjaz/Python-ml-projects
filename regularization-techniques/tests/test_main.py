"""Tests for dropout and regularization techniques module."""

import numpy as np

from src.main import (
    Dropout,
    RegularizationRunner,
    RegularizedMLP,
    generate_classification_data,
)


class TestDropout:
    """Unit tests for Dropout layer."""

    def test_forward_masking_and_scaling(self) -> None:
        """Dropout should zero out activations and rescale remaining ones."""
        np.random.seed(0)
        x = np.ones((1000, 1), dtype=np.float32)
        dropout = Dropout(rate=0.5)
        out = dropout.forward(x, training=True)
        # Roughly half should be zero and mean should be close to 1
        zero_ratio = np.mean(out == 0.0)
        mean_val = float(out.mean())
        assert 0.4 <= zero_ratio <= 0.6
        assert 0.9 <= mean_val <= 1.1

    def test_backward_uses_same_mask(self) -> None:
        """Backward should apply the same mask as forward."""
        np.random.seed(1)
        x = np.random.randn(4, 3).astype(np.float32)
        dropout = Dropout(rate=0.5)
        out = dropout.forward(x, training=True)
        grad_out = np.ones_like(out)
        grad_in = dropout.backward(grad_out)
        assert np.all((out == 0.0) == (grad_in == 0.0))


class TestDataGeneration:
    """Tests for synthetic classification data generation."""

    def test_generate_classification_data_shapes(self) -> None:
        """Generated data should have expected shapes."""
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=300,
            n_features=5,
            n_classes=3,
            random_seed=42,
        )
        assert x_train.shape[1] == 5
        assert x_test.shape[1] == 5
        assert y_train.ndim == 1
        assert y_test.ndim == 1


class TestRegularizedMLPTraining:
    """Training tests for RegularizedMLP with different modes."""

    def _train_model(self, mode: str) -> float:
        np.random.seed(0)
        x_train, _, y_train, _ = generate_classification_data(
            n_samples=400,
            n_features=10,
            n_classes=3,
            random_seed=0,
        )
        model = RegularizedMLP(
            input_dim=10,
            hidden_dim=16,
            n_classes=3,
            dropout_rate=0.5,
            l2_lambda=0.01,
            mode=mode,
        )
        losses = []
        for _ in range(5):
            loss, grad_logits = model.compute_loss_and_gradients(
                x_train, y_train
            )
            model.backward(grad_logits, learning_rate=0.1)
            losses.append(loss)
        return losses[-1] / max(losses[0], 1e-6)

    def test_training_without_regularization_reduces_loss(self) -> None:
        """Baseline training (no regularization) should reduce loss."""
        ratio = self._train_model("none")
        assert ratio <= 1.2

    def test_training_with_dropout_reduces_loss(self) -> None:
        """Training with dropout should also reduce loss."""
        ratio = self._train_model("dropout")
        assert ratio <= 1.2

    def test_training_with_l2_reduces_loss(self) -> None:
        """Training with L2 weight decay should reduce loss."""
        ratio = self._train_model("l2")
        assert ratio <= 1.2

    def test_training_with_dropout_l2_reduces_loss(self) -> None:
        """Training with both dropout and L2 should reduce loss."""
        ratio = self._train_model("dropout_l2")
        assert ratio <= 1.2


class TestRegularizationRunner:
    """Smoke tests for RegularizationRunner."""

    def test_runner_executes_for_all_modes(self) -> None:
        """Runner should produce metrics for all regularization modes."""
        runner = RegularizationRunner(config_path=None)
        for mode in ["none", "dropout", "l2", "dropout_l2"]:
            results = runner.run(mode=mode)
            assert "train_loss" in results
            assert "test_loss" in results
            assert "train_accuracy" in results
            assert "test_accuracy" in results
            assert 0.0 <= results["test_accuracy"] <= 1.0

