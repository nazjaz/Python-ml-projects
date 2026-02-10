"""Unit tests for multinomial logistic regression implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import MultinomialLogisticRegression


class TestMultinomialLogisticRegression:
    """Test MultinomialLogisticRegression functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = MultinomialLogisticRegression()
        assert model.learning_rate == 0.01
        assert model.max_iterations == 1000
        assert model.weights is None
        assert model.intercept is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        model = MultinomialLogisticRegression(
            learning_rate=0.05, max_iterations=500, scale_features=False
        )
        assert model.learning_rate == 0.05
        assert model.max_iterations == 500
        assert model.scale_features is False

    def test_softmax(self):
        """Test softmax activation function."""
        model = MultinomialLogisticRegression()
        z = np.array([[1, 2, 3], [1, 1, 1]])
        softmax_output = model._softmax(z)

        assert softmax_output.shape == z.shape
        assert np.all(softmax_output >= 0)
        assert np.all(softmax_output <= 1)
        assert np.allclose(np.sum(softmax_output, axis=1), 1.0)

    def test_softmax_sums_to_one(self):
        """Test that softmax probabilities sum to one."""
        model = MultinomialLogisticRegression()
        z = np.array([[1, 2, 3], [0, 0, 0], [-1, 0, 1]])
        softmax_output = model._softmax(z)

        for row in softmax_output:
            assert abs(np.sum(row) - 1.0) < 1e-10

    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        model = MultinomialLogisticRegression()
        model.classes = np.array([0, 1, 2])
        y = np.array([0, 1, 2, 0, 1])
        one_hot = model._one_hot_encode(y)

        assert one_hot.shape == (5, 3)
        assert np.all(np.sum(one_hot, axis=1) == 1)
        assert one_hot[0, 0] == 1
        assert one_hot[1, 1] == 1
        assert one_hot[2, 2] == 1

    def test_fit_multiclass(self):
        """Test fitting multiclass classification model."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.intercept is not None
        assert len(model.cost_history) > 0
        assert len(model.classes) == 3

    def test_fit_multiple_features(self):
        """Test fitting with multiple features."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.weights.shape[0] == 2
        assert model.weights.shape[1] == 3

    def test_fit_with_scaling(self):
        """Test fitting with feature scaling."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=True
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.scale_params is not None

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = MultinomialLogisticRegression()
        X = np.array([[1], [2], [3]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        probabilities = model.predict_proba(X)
        assert probabilities.shape[0] == len(X)
        assert probabilities.shape[1] == len(model.classes)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, model.classes))

    def test_score(self):
        """Test accuracy score calculation."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_cost_decreases(self):
        """Test that cost decreases during training."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        assert len(model.cost_history) > 1
        assert model.cost_history[-1] <= model.cost_history[0]

    def test_cost_function(self):
        """Test cross-entropy cost function computation."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 2])
        model = MultinomialLogisticRegression()
        model.classes = np.array([0, 1, 2])
        X_with_intercept = model._add_intercept(X)
        y_one_hot = model._one_hot_encode(y)
        theta = np.zeros((X_with_intercept.shape[1], 3))

        cost = model._compute_cost(X_with_intercept, y_one_hot, theta)
        assert cost >= 0
        assert isinstance(cost, float)

    def test_gradient_computation(self):
        """Test gradient computation."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 2])
        model = MultinomialLogisticRegression()
        model.classes = np.array([0, 1, 2])
        X_with_intercept = model._add_intercept(X)
        y_one_hot = model._one_hot_encode(y)
        theta = np.zeros((X_with_intercept.shape[1], 3))

        gradient = model._compute_gradient(X_with_intercept, y_one_hot, theta)
        assert gradient.shape == theta.shape

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [4, 5, 6, 7, 8, 9],
                "target": [0, 0, 1, 1, 2, 2],
            }
        )
        X = df[["feature1", "feature2"]]
        y = df["target"]
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(df)

    def test_get_cost_history(self):
        """Test getting cost history."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        cost_history = model.get_cost_history()
        assert len(cost_history) > 0
        assert cost_history == model.cost_history

    def test_plot_training_history_no_exception(self):
        """Test that plot_training_history doesn't raise exceptions."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        model.plot_training_history(show=False)

    def test_plot_training_history_save(self):
        """Test saving training history plot."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
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
        model = MultinomialLogisticRegression()
        X = np.array([]).reshape(0, 1)
        y = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        model = MultinomialLogisticRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="Length mismatch"):
            model.fit(X, y)

    def test_fit_single_class(self):
        """Test that single class raises error."""
        model = MultinomialLogisticRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="At least 2 classes"):
            model.fit(X, y)

    def test_predict_proba_sums_to_one(self):
        """Test that predicted probabilities sum to one."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        probabilities = model.predict_proba(X)
        for row in probabilities:
            assert abs(np.sum(row) - 1.0) < 1e-10

    def test_convergence(self):
        """Test that model converges with tolerance."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        model = MultinomialLogisticRegression(
            learning_rate=0.01,
            max_iterations=1000,
            tolerance=1e-6,
            scale_features=False,
        )
        model.fit(X, y)

        assert len(model.cost_history) <= model.max_iterations
