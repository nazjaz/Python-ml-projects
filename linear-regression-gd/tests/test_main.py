"""Unit tests for linear regression with gradient descent implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import LinearRegression, LearningRateScheduler


class TestLearningRateScheduler:
    """Test LearningRateScheduler functionality."""

    def test_constant_scheduler(self):
        """Test constant learning rate scheduler."""
        scheduler = LearningRateScheduler.constant(initial_lr=0.01)
        assert scheduler(0) == 0.01
        assert scheduler(100) == 0.01
        assert scheduler(1000) == 0.01

    def test_exponential_decay_scheduler(self):
        """Test exponential decay learning rate scheduler."""
        scheduler = LearningRateScheduler.exponential_decay(
            initial_lr=0.01, decay_rate=0.95
        )
        assert abs(scheduler(0) - 0.01) < 1e-6
        assert scheduler(1) < scheduler(0)
        assert scheduler(10) < scheduler(1)

    def test_step_decay_scheduler(self):
        """Test step decay learning rate scheduler."""
        scheduler = LearningRateScheduler.step_decay(
            initial_lr=0.01, drop_rate=0.5, epochs_drop=10
        )
        assert abs(scheduler(0) - 0.01) < 1e-6
        assert abs(scheduler(9) - 0.01) < 1e-6
        assert abs(scheduler(10) - 0.005) < 1e-6
        assert abs(scheduler(20) - 0.0025) < 1e-6

    def test_polynomial_decay_scheduler(self):
        """Test polynomial decay learning rate scheduler."""
        scheduler = LearningRateScheduler.polynomial_decay(
            initial_lr=0.01, end_lr=0.001, max_epochs=100
        )
        assert abs(scheduler(0) - 0.01) < 1e-6
        assert scheduler(50) < scheduler(0)
        assert abs(scheduler(100) - 0.001) < 1e-3


class TestLinearRegression:
    """Test LinearRegression functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = LinearRegression()
        assert model.learning_rate == 0.01
        assert model.max_iterations == 1000
        assert model.weights is None
        assert model.intercept is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        model = LinearRegression(
            learning_rate=0.05,
            max_iterations=500,
            scheduler="exponential",
            scheduler_params={"decay_rate": 0.9},
        )
        assert model.learning_rate == 0.05
        assert model.max_iterations == 500
        assert model.scheduler_type == "exponential"

    def test_fit_simple_linear(self):
        """Test fitting simple linear relationship."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        model.fit(X, y)

        assert model.weights is not None
        assert model.intercept is not None
        assert len(model.cost_history) > 0

    def test_fit_without_intercept(self):
        """Test fitting without intercept."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(fit_intercept=False, max_iterations=1000)
        model.fit(X, y)

        assert model.intercept == 0.0
        assert model.weights is not None

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(pred >= 0 for pred in predictions)

    def test_score(self):
        """Test R-squared score calculation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_cost_decreases(self):
        """Test that cost decreases during training."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(learning_rate=0.01, max_iterations=100)
        model.fit(X, y)

        assert len(model.cost_history) > 1
        assert model.cost_history[-1] <= model.cost_history[0]

    def test_convergence(self):
        """Test that model converges with tolerance."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(
            learning_rate=0.01, max_iterations=1000, tolerance=1e-6
        )
        model.fit(X, y)

        assert len(model.cost_history) <= model.max_iterations

    def test_multiple_features(self):
        """Test fitting with multiple features."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])
        model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        model.fit(X, y)

        assert model.weights is not None
        assert len(model.weights) == 2

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([5, 7, 9])
        model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        model.fit(df, y)

        predictions = model.predict(df)
        assert len(predictions) == len(df)

    def test_exponential_scheduler(self):
        """Test training with exponential decay scheduler."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            scheduler="exponential",
            scheduler_params={"decay_rate": 0.95},
        )
        model.fit(X, y)

        assert len(model.lr_history) > 0
        assert model.lr_history[-1] < model.lr_history[0]

    def test_step_scheduler(self):
        """Test training with step decay scheduler."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            scheduler="step",
            scheduler_params={"drop_rate": 0.5, "epochs_drop": 10},
        )
        model.fit(X, y)

        assert len(model.lr_history) > 0

    def test_polynomial_scheduler(self):
        """Test training with polynomial decay scheduler."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            scheduler="polynomial",
            scheduler_params={"end_lr": 0.001, "max_epochs": 100},
        )
        model.fit(X, y)

        assert len(model.lr_history) > 0

    def test_get_cost_history(self):
        """Test getting cost history."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(learning_rate=0.01, max_iterations=100)
        model.fit(X, y)

        cost_history = model.get_cost_history()
        assert len(cost_history) > 0
        assert cost_history == model.cost_history

    def test_get_lr_history(self):
        """Test getting learning rate history."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            scheduler="exponential",
        )
        model.fit(X, y)

        lr_history = model.get_lr_history()
        assert len(lr_history) > 0

    def test_plot_training_history_no_exception(self):
        """Test that plot_training_history doesn't raise exceptions."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(learning_rate=0.01, max_iterations=100)
        model.fit(X, y)

        model.plot_training_history(show=False)

    def test_plot_training_history_save(self):
        """Test saving training history plot."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(learning_rate=0.01, max_iterations=100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                model.plot_training_history(save_path=save_path, show=False)
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_fit_empty_data(self):
        """Test that fitting with empty data raises error."""
        model = LinearRegression()
        X = np.array([]).reshape(0, 1)
        y = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        model = LinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="Length mismatch"):
            model.fit(X, y)

    def test_unknown_scheduler(self):
        """Test that unknown scheduler defaults to constant."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            scheduler="unknown",
        )
        model.fit(X, y)

        assert model.scheduler_type == "constant"
