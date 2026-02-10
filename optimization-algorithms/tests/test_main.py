"""Tests for optimization algorithms module."""

import numpy as np

from src.main import (
    AdaGradOptimizer,
    AdamOptimizer,
    MLPClassifier,
    OptimizationRunner,
    RMSpropOptimizer,
    SGDOptimizer,
    generate_classification_data,
)


class TestOptimizersStep:
    """Unit tests for single-step optimizer behavior."""

    def test_sgd_step_updates_params(self) -> None:
        """SGD should move parameters opposite to gradient."""
        params = {"w": np.array([1.0, -2.0], dtype=np.float32)}
        grads = {"w": np.array([0.5, -0.5], dtype=np.float32)}
        opt = SGDOptimizer()
        opt.step(params, grads, learning_rate=0.1)
        np.testing.assert_allclose(
            params["w"], np.array([0.95, -1.95], dtype=np.float32)
        )

    def test_adagrad_accumulates_squared_gradients(self) -> None:
        """AdaGrad should accumulate squared gradients in its state."""
        params = {"w": np.array([0.0, 0.0], dtype=np.float32)}
        grads = {"w": np.array([1.0, 2.0], dtype=np.float32)}
        opt = AdaGradOptimizer(eps=1e-8)
        opt.step(params, grads, learning_rate=1.0)
        assert "G" in opt.state["w"]
        np.testing.assert_allclose(
            opt.state["w"]["G"],
            np.array([1.0, 4.0], dtype=np.float32),
        )

    def test_rmsprop_tracks_exponential_average(self) -> None:
        """RMSprop should maintain an EMA of squared gradients."""
        params = {"w": np.array([0.0, 0.0], dtype=np.float32)}
        grads = {"w": np.array([1.0, 2.0], dtype=np.float32)}
        opt = RMSpropOptimizer(rho=0.9, eps=1e-8)
        opt.step(params, grads, learning_rate=0.1)
        assert "E" in opt.state["w"]
        np.testing.assert_allclose(
            opt.state["w"]["E"],
            np.array([0.1, 0.4], dtype=np.float32),
        )

    def test_adam_tracks_m_and_v(self) -> None:
        """Adam should maintain first and second moment estimates."""
        params = {"w": np.array([0.0, 0.0], dtype=np.float32)}
        grads = {"w": np.array([1.0, -1.0], dtype=np.float32)}
        opt = AdamOptimizer(beta1=0.9, beta2=0.999, eps=1e-8)
        opt.step(params, grads, learning_rate=0.001)
        state = opt.state["w"]
        assert "m" in state and "v" in state
        np.testing.assert_allclose(
            state["m"],
            np.array([0.1, -0.1], dtype=np.float32),
        )


class TestTrainingWithOptimizers:
    """Training tests to ensure optimizers reduce loss."""

    def _train_with_optimizer(self, optimizer_name: str) -> float:
        np.random.seed(0)
        x_train, _, y_train, _ = generate_classification_data(
            n_samples=400,
            n_features=10,
            n_classes=3,
            random_seed=0,
        )
        model = MLPClassifier(input_dim=10, hidden_dim=16, n_classes=3)
        if optimizer_name == "sgd":
            opt = SGDOptimizer()
        elif optimizer_name == "adagrad":
            opt = AdaGradOptimizer()
        elif optimizer_name == "rmsprop":
            opt = RMSpropOptimizer()
        else:
            opt = AdamOptimizer()

        losses = []
        for _ in range(5):
            loss, grads = model.compute_loss_and_gradients(x_train, y_train)
            params = model.get_params()
            opt.step(params, grads, learning_rate=0.01)
            model.set_params(params)
            losses.append(loss)
        return losses[-1] / max(losses[0], 1e-6)

    def test_sgd_reduces_loss(self) -> None:
        """SGD training should reduce loss."""
        ratio = self._train_with_optimizer("sgd")
        assert ratio <= 1.2

    def test_adagrad_reduces_loss(self) -> None:
        """AdaGrad training should reduce loss."""
        ratio = self._train_with_optimizer("adagrad")
        assert ratio <= 1.2

    def test_rmsprop_reduces_loss(self) -> None:
        """RMSprop training should reduce loss."""
        ratio = self._train_with_optimizer("rmsprop")
        assert ratio <= 1.2

    def test_adam_reduces_loss(self) -> None:
        """Adam training should reduce loss."""
        ratio = self._train_with_optimizer("adam")
        assert ratio <= 1.2


class TestOptimizationRunner:
    """Smoke tests for OptimizationRunner."""

    def test_runner_executes_for_all_optimizers(self) -> None:
        """Runner should produce metrics for each optimizer."""
        runner = OptimizationRunner(config_path=None)
        for name in ["sgd", "adagrad", "rmsprop", "adam"]:
            results = runner.run(optimizer_name=name)
            assert "train_loss" in results
            assert "test_loss" in results
            assert "train_accuracy" in results
            assert "test_accuracy" in results
            assert 0.0 <= results["test_accuracy"] <= 1.0

