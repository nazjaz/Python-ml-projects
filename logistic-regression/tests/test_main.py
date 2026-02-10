"""Unit tests for logistic regression implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import LogisticRegression


class TestLogisticRegression:
    """Test LogisticRegression functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = LogisticRegression()
        assert model.learning_rate == 0.01
        assert model.max_iterations == 1000
        assert model.weights is None
        assert model.intercept is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        model = LogisticRegression(
            learning_rate=0.05, max_iterations=500, scale_features=False
        )
        assert model.learning_rate == 0.05
        assert model.max_iterations == 500
        assert model.scale_features is False

    def test_sigmoid(self):
        """Test sigmoid activation function."""
        model = LogisticRegression()
        z = np.array([0, 1, -1, 10, -10])
        sigmoid_output = model._sigmoid(z)

        assert len(sigmoid_output) == len(z)
        assert np.all(sigmoid_output >= 0)
        assert np.all(sigmoid_output <= 1)
        assert abs(sigmoid_output[0] - 0.5) < 1e-6

    def test_sigmoid_extreme_values(self):
        """Test sigmoid with extreme values."""
        model = LogisticRegression()
        z = np.array([-500, 500])
        sigmoid_output = model._sigmoid(z)

        assert sigmoid_output[0] > 0
        assert sigmoid_output[1] < 1

    def test_fit_binary_classification(self):
        """Test fitting binary classification model."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.intercept is not None
        assert len(model.cost_history) > 0

    def test_fit_multiple_features(self):
        """Test fitting with multiple features."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        assert model.weights is not None
        assert len(model.weights) == 2

    def test_fit_with_scaling(self):
        """Test fitting with feature scaling."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=True
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.scale_params is not None

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = LogisticRegression()
        X = np.array([[1], [2], [3]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        probabilities = model.predict_proba(X)
        assert len(probabilities) == len(X)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_predict_with_threshold(self):
        """Test prediction with custom threshold."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        predictions_low = model.predict(X, threshold=0.3)
        predictions_high = model.predict(X, threshold=0.7)

        assert len(predictions_low) == len(X)
        assert len(predictions_high) == len(X)

    def test_score(self):
        """Test accuracy score calculation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_cost_decreases(self):
        """Test that cost decreases during training."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        assert len(model.cost_history) > 1
        assert model.cost_history[-1] <= model.cost_history[0]

    def test_cost_function(self):
        """Test logistic cost function computation."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 1])
        model = LogisticRegression()
        X_with_intercept = model._add_intercept(X)
        theta = np.array([0.5, 0.5])

        cost = model._compute_cost(X_with_intercept, y, theta)
        assert cost >= 0
        assert isinstance(cost, float)

    def test_gradient_computation(self):
        """Test gradient computation."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 1])
        model = LogisticRegression()
        X_with_intercept = model._add_intercept(X)
        theta = np.array([0.5, 0.5])

        gradient = model._compute_gradient(X_with_intercept, y, theta)
        assert gradient.shape == theta.shape

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 1]}
        )
        X = df[["feature1", "feature2"]]
        y = df["target"]
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(df)

    def test_get_cost_history(self):
        """Test getting cost history."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        cost_history = model.get_cost_history()
        assert len(cost_history) > 0
        assert cost_history == model.cost_history

    def test_plot_training_history_no_exception(self):
        """Test that plot_training_history doesn't raise exceptions."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        model.plot_training_history(show=False)

    def test_plot_training_history_save(self):
        """Test saving training history plot."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
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
        model = LogisticRegression()
        X = np.array([]).reshape(0, 1)
        y = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        model = LogisticRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="Length mismatch"):
            model.fit(X, y)

    def test_fit_invalid_labels(self):
        """Test that invalid labels raise error."""
        model = LogisticRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            model.fit(X, y)

    def test_predict_proba_range(self):
        """Test that predicted probabilities are in [0, 1] range."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        probabilities = model.predict_proba(X)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_convergence(self):
        """Test that model converges with tolerance."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, tolerance=1e-6, scale_features=False
        )
        model.fit(X, y)

        assert len(model.cost_history) <= model.max_iterations
