"""Tests for custom loss functions module."""

import numpy as np
import pytest

from src.main import ClassificationLoss, LossFunctionEvaluator, RegressionLoss


class TestRegressionLoss:
    """Test cases for regression loss functions."""

    def test_mean_squared_error_basic(self):
        """Test MSE with simple inputs."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        loss, gradient = RegressionLoss.mean_squared_error(y_true, y_pred)

        expected_loss = np.mean((y_true - y_pred) ** 2)
        assert abs(loss - expected_loss) < 1e-10
        assert gradient.shape == y_true.shape

    def test_mean_squared_error_shape_mismatch(self):
        """Test MSE raises error on shape mismatch."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            RegressionLoss.mean_squared_error(y_true, y_pred)

    def test_mean_absolute_error_basic(self):
        """Test MAE with simple inputs."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        loss, gradient = RegressionLoss.mean_absolute_error(y_true, y_pred)

        expected_loss = np.mean(np.abs(y_true - y_pred))
        assert abs(loss - expected_loss) < 1e-10
        assert gradient.shape == y_true.shape

    def test_huber_loss_basic(self):
        """Test Huber loss with simple inputs."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        loss, gradient = RegressionLoss.huber_loss(y_true, y_pred, delta=1.0)

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_huber_loss_invalid_delta(self):
        """Test Huber loss raises error for invalid delta."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="delta must be positive"):
            RegressionLoss.huber_loss(y_true, y_pred, delta=-1.0)

    def test_smooth_l1_loss_basic(self):
        """Test Smooth L1 loss with simple inputs."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        loss, gradient = RegressionLoss.smooth_l1_loss(y_true, y_pred, beta=1.0)

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_smooth_l1_loss_invalid_beta(self):
        """Test Smooth L1 loss raises error for invalid beta."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="beta must be positive"):
            RegressionLoss.smooth_l1_loss(y_true, y_pred, beta=0.0)

    def test_log_cosh_loss_basic(self):
        """Test Log-Cosh loss with simple inputs."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        loss, gradient = RegressionLoss.log_cosh_loss(y_true, y_pred)

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_regression_losses_perfect_prediction(self):
        """Test all regression losses with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()

        mse_loss, _ = RegressionLoss.mean_squared_error(y_true, y_pred)
        mae_loss, _ = RegressionLoss.mean_absolute_error(y_true, y_pred)
        huber_loss, _ = RegressionLoss.huber_loss(y_true, y_pred)
        log_cosh_loss, _ = RegressionLoss.log_cosh_loss(y_true, y_pred)

        assert mse_loss == 0.0
        assert mae_loss == 0.0
        assert huber_loss == 0.0
        assert log_cosh_loss == 0.0


class TestClassificationLoss:
    """Test cases for classification loss functions."""

    def test_binary_cross_entropy_basic(self):
        """Test binary cross-entropy with simple inputs."""
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])
        loss, gradient = ClassificationLoss.binary_cross_entropy(y_true, y_pred)

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_binary_cross_entropy_invalid_labels(self):
        """Test binary cross-entropy raises error for invalid labels."""
        y_true = np.array([0.5, 1.0])
        y_pred = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="y_true must contain only 0 and 1"):
            ClassificationLoss.binary_cross_entropy(y_true, y_pred)

    def test_binary_cross_entropy_invalid_probabilities(self):
        """Test binary cross-entropy raises error for invalid probabilities."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.5, 0.5])
        with pytest.raises(ValueError, match="y_pred must be in"):
            ClassificationLoss.binary_cross_entropy(y_true, y_pred)

    def test_categorical_cross_entropy_basic(self):
        """Test categorical cross-entropy with simple inputs."""
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        loss, gradient = ClassificationLoss.categorical_cross_entropy(
            y_true, y_pred
        )

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_categorical_cross_entropy_not_one_hot(self):
        """Test categorical cross-entropy raises error for non-one-hot encoding."""
        y_true = np.array([[1.0, 0.0], [0.5, 0.5]])
        y_pred = np.array([[0.9, 0.1], [0.5, 0.5]])
        with pytest.raises(ValueError, match="y_true must be one-hot encoded"):
            ClassificationLoss.categorical_cross_entropy(y_true, y_pred)

    def test_focal_loss_basic(self):
        """Test focal loss with simple inputs."""
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])
        loss, gradient = ClassificationLoss.focal_loss(
            y_true, y_pred, alpha=1.0, gamma=2.0
        )

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_focal_loss_invalid_gamma(self):
        """Test focal loss raises error for invalid gamma."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            ClassificationLoss.focal_loss(y_true, y_pred, gamma=-1.0)

    def test_hinge_loss_basic(self):
        """Test hinge loss with simple inputs."""
        y_true = np.array([-1.0, 1.0, 1.0, -1.0])
        y_pred = np.array([-0.5, 0.8, 1.2, -1.5])
        loss, gradient = ClassificationLoss.hinge_loss(y_true, y_pred)

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_hinge_loss_invalid_labels(self):
        """Test hinge loss raises error for invalid labels."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="y_true must contain only -1 and 1"):
            ClassificationLoss.hinge_loss(y_true, y_pred)

    def test_kl_divergence_basic(self):
        """Test KL divergence with simple inputs."""
        y_true = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        loss, gradient = ClassificationLoss.kullback_leibler_divergence(
            y_true, y_pred
        )

        assert loss >= 0
        assert gradient.shape == y_true.shape

    def test_classification_losses_perfect_prediction(self):
        """Test classification losses with perfect predictions."""
        y_true_binary = np.array([0.0, 1.0, 1.0])
        y_pred_binary = y_true_binary.copy()

        bce_loss, _ = ClassificationLoss.binary_cross_entropy(
            y_true_binary, y_pred_binary
        )
        assert bce_loss < 1e-10

        y_true_cat = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred_cat = y_true_cat.copy()
        cce_loss, _ = ClassificationLoss.categorical_cross_entropy(
            y_true_cat, y_pred_cat
        )
        assert cce_loss < 1e-10


class TestLossFunctionEvaluator:
    """Test cases for LossFunctionEvaluator."""

    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        evaluator = LossFunctionEvaluator()
        assert evaluator is not None

    def test_evaluate_regression_losses(self):
        """Test regression loss evaluation."""
        evaluator = LossFunctionEvaluator()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])

        results = evaluator.evaluate_regression_losses(y_true, y_pred)

        assert "mse" in results
        assert "mae" in results
        assert "huber" in results
        assert "smooth_l1" in results
        assert "log_cosh" in results

        for name, result in results.items():
            if "error" not in result:
                assert "loss" in result
                assert "gradient" in result
                assert "gradient_norm" in result

    def test_evaluate_classification_losses_binary(self):
        """Test binary classification loss evaluation."""
        evaluator = LossFunctionEvaluator()
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])

        results = evaluator.evaluate_classification_losses(
            y_true, y_pred, loss_type="binary"
        )

        assert "binary_cross_entropy" in results
        assert "focal_loss" in results

    def test_evaluate_classification_losses_multiclass(self):
        """Test multiclass classification loss evaluation."""
        evaluator = LossFunctionEvaluator()
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])

        results = evaluator.evaluate_classification_losses(
            y_true, y_pred, loss_type="multiclass"
        )

        assert "categorical_cross_entropy" in results
        assert "kl_divergence" in results


class TestGradientComputation:
    """Test gradient computation accuracy using numerical differentiation."""

    def test_mse_gradient_numerical(self):
        """Test MSE gradient using numerical differentiation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        epsilon = 1e-7

        _, analytical_grad = RegressionLoss.mean_squared_error(y_true, y_pred)

        numerical_grad = np.zeros_like(y_pred)
        for i in range(len(y_pred)):
            y_pred_plus = y_pred.copy()
            y_pred_plus[i] += epsilon
            loss_plus, _ = RegressionLoss.mean_squared_error(y_true, y_pred_plus)

            y_pred_minus = y_pred.copy()
            y_pred_minus[i] -= epsilon
            loss_minus, _ = RegressionLoss.mean_squared_error(y_true, y_pred_minus)

            numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        np.testing.assert_allclose(
            analytical_grad, numerical_grad, rtol=1e-5, atol=1e-7
        )

    def test_bce_gradient_numerical(self):
        """Test binary cross-entropy gradient using numerical differentiation."""
        y_true = np.array([0.0, 1.0, 1.0])
        y_pred = np.array([0.2, 0.8, 0.9])
        epsilon = 1e-7

        _, analytical_grad = ClassificationLoss.binary_cross_entropy(
            y_true, y_pred
        )

        numerical_grad = np.zeros_like(y_pred)
        for i in range(len(y_pred)):
            y_pred_plus = np.clip(y_pred.copy(), 1e-15, 1 - 1e-15)
            y_pred_plus[i] = np.clip(y_pred_plus[i] + epsilon, 1e-15, 1 - 1e-15)
            loss_plus, _ = ClassificationLoss.binary_cross_entropy(
                y_true, y_pred_plus
            )

            y_pred_minus = np.clip(y_pred.copy(), 1e-15, 1 - 1e-15)
            y_pred_minus[i] = np.clip(y_pred_minus[i] - epsilon, 1e-15, 1 - 1e-15)
            loss_minus, _ = ClassificationLoss.binary_cross_entropy(
                y_true, y_pred_minus
            )

            numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        np.testing.assert_allclose(
            analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6
        )
