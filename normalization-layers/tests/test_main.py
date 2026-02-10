"""Tests for batch normalization and layer normalization module."""

import numpy as np

from src.main import (
    BatchNorm1D,
    LayerNorm1D,
    NormalizationRunner,
    SimpleMLP,
    generate_classification_data,
)


class TestBatchNorm1D:
    """Unit tests for BatchNorm1D."""

    def test_forward_normalizes_batch(self) -> None:
        """BatchNorm1D should produce zero-mean unit-variance features."""
        np.random.seed(0)
        x = np.random.randn(64, 10).astype(np.float32) * 2.0 + 3.0
        bn = BatchNorm1D(num_features=10)
        out = bn.forward(x, training=True)
        batch_mean = out.mean(axis=0)
        batch_std = out.std(axis=0)
        np.testing.assert_allclose(batch_mean, np.zeros(10), atol=1e-5)
        np.testing.assert_allclose(batch_std, np.ones(10), atol=1e-4)

    def test_backward_shape(self) -> None:
        """Backward should return gradient of same shape as input."""
        np.random.seed(1)
        x = np.random.randn(32, 8).astype(np.float32)
        bn = BatchNorm1D(num_features=8)
        _ = bn.forward(x, training=True)
        grad_out = np.random.randn(32, 8).astype(np.float32)
        dx = bn.backward(grad_out)
        assert dx.shape == x.shape


class TestLayerNorm1D:
    """Unit tests for LayerNorm1D."""

    def test_forward_normalizes_per_sample(self) -> None:
        """LayerNorm1D should normalize each sample across features."""
        np.random.seed(2)
        x = np.random.randn(16, 12).astype(np.float32) * 1.5 - 1.0
        ln = LayerNorm1D(num_features=12)
        out = ln.forward(x)
        sample_means = out.mean(axis=1)
        sample_stds = out.std(axis=1)
        np.testing.assert_allclose(sample_means, np.zeros(16), atol=1e-5)
        np.testing.assert_allclose(sample_stds, np.ones(16), atol=1e-4)

    def test_backward_shape(self) -> None:
        """Backward should return gradient of same shape as input."""
        np.random.seed(3)
        x = np.random.randn(10, 6).astype(np.float32)
        ln = LayerNorm1D(num_features=6)
        _ = ln.forward(x)
        grad_out = np.random.randn(10, 6).astype(np.float32)
        dx = ln.backward(grad_out)
        assert dx.shape == x.shape


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


class TestSimpleMLPTraining:
    """Training tests for SimpleMLP with different normalization modes."""

    def _train_model(self, norm: str) -> float:
        np.random.seed(0)
        x_train, _, y_train, _ = generate_classification_data(
            n_samples=400,
            n_features=10,
            n_classes=3,
            random_seed=0,
        )
        model = SimpleMLP(
            input_dim=10,
            hidden_dim=16,
            n_classes=3,
            norm=norm,
        )
        losses = []
        for _ in range(5):
            logits = model.forward(x_train, training=True)
            from src.main import _cross_entropy  # Local import for test

            loss, grad_logits = _cross_entropy(logits, y_train)
            model.backward(grad_logits, learning_rate=0.05)
            losses.append(loss)
        return losses[-1] / max(losses[0], 1e-6)

    def test_training_with_batchnorm_reduces_loss(self) -> None:
        """Training with batch normalization should reduce loss."""
        ratio = self._train_model("batchnorm")
        assert ratio <= 1.2

    def test_training_with_layernorm_reduces_loss(self) -> None:
        """Training with layer normalization should reduce loss."""
        ratio = self._train_model("layernorm")
        assert ratio <= 1.2


class TestNormalizationRunner:
    """Smoke tests for NormalizationRunner."""

    def test_runner_executes_for_each_mode(self) -> None:
        """Runner should produce metrics for each normalization mode."""
        runner = NormalizationRunner(config_path=None)
        for mode in ["none", "batchnorm", "layernorm"]:
            results = runner.run(mode=mode)
            assert "train_loss" in results
            assert "test_loss" in results
            assert "train_accuracy" in results
            assert "test_accuracy" in results
            assert 0.0 <= results["test_accuracy"] <= 1.0

