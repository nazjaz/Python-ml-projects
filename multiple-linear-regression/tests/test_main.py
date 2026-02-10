"""Unit tests for multiple linear regression implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import MultipleLinearRegression, FeatureScaler


class TestFeatureScaler:
    """Test FeatureScaler functionality."""

    def test_standardize(self):
        """Test feature standardization."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaled_X, mean, std = FeatureScaler.standardize(X)

        assert scaled_X.shape == X.shape
        assert np.allclose(np.mean(scaled_X, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(scaled_X, axis=0), 1, atol=1e-10)

    def test_normalize(self):
        """Test feature normalization."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaled_X, min_val, max_val = FeatureScaler.normalize(X)

        assert scaled_X.shape == X.shape
        assert np.all(scaled_X >= 0)
        assert np.all(scaled_X <= 1)

    def test_apply_standardization(self):
        """Test applying standardization with pre-computed parameters."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        _, mean, std = FeatureScaler.standardize(X)

        X_new = np.array([[2, 3], [4, 5]])
        scaled_X_new = FeatureScaler.apply_standardization(X_new, mean, std)

        assert scaled_X_new.shape == X_new.shape

    def test_apply_normalization(self):
        """Test applying normalization with pre-computed parameters."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        _, min_val, max_val = FeatureScaler.normalize(X)

        X_new = np.array([[2, 3], [4, 5]])
        scaled_X_new = FeatureScaler.apply_normalization(X_new, min_val, max_val)

        assert scaled_X_new.shape == X_new.shape


class TestMultipleLinearRegression:
    """Test MultipleLinearRegression functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = MultipleLinearRegression()
        assert model.learning_rate == 0.01
        assert model.max_iterations == 1000
        assert model.weights is None
        assert model.intercept is None

    def test_initialization_with_regularization(self):
        """Test model initialization with regularization."""
        model = MultipleLinearRegression(regularization="ridge", alpha=0.1)
        assert model.regularization == "ridge"
        assert model.alpha == 0.1

    def test_fit_simple_linear(self):
        """Test fitting simple linear relationship."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.intercept is not None
        assert len(model.cost_history) > 0

    def test_fit_multiple_features(self):
        """Test fitting with multiple features."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        assert model.weights is not None
        assert len(model.weights) == 2

    def test_fit_with_standardization(self):
        """Test fitting with feature standardization."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])
        model = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=1000,
            scale_features=True,
            scaling_method="standardize",
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.scale_params is not None
        assert model.scale_params["method"] == "standardize"

    def test_fit_with_normalization(self):
        """Test fitting with feature normalization."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])
        model = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=1000,
            scale_features=True,
            scaling_method="normalize",
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.scale_params is not None
        assert model.scale_params["method"] == "normalize"

    def test_fit_with_ridge_regularization(self):
        """Test fitting with Ridge regularization."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])
        model = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=1000,
            regularization="ridge",
            alpha=0.1,
            scale_features=False,
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.regularization == "ridge"

    def test_fit_with_lasso_regularization(self):
        """Test fitting with Lasso regularization."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])
        model = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=1000,
            regularization="lasso",
            alpha=0.1,
            scale_features=False,
        )
        model.fit(X, y)

        assert model.weights is not None
        assert model.regularization == "lasso"

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        model = MultipleLinearRegression()
        X = np.array([[1], [2], [3]])
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)

    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_predict_with_scaling(self):
        """Test prediction with feature scaling."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])
        model = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=1000,
            scale_features=True,
            scaling_method="standardize",
        )
        model.fit(X, y)

        X_test = np.array([[2, 3], [4, 5]])
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_score(self):
        """Test R-squared score calculation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_cost_decreases(self):
        """Test that cost decreases during training."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        assert len(model.cost_history) > 1
        assert model.cost_history[-1] <= model.cost_history[0]

    def test_ridge_regularization_penalty(self):
        """Test that Ridge regularization adds penalty to cost."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])

        model_no_reg = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model_ridge = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            regularization="ridge",
            alpha=0.1,
            scale_features=False,
        )

        model_no_reg.fit(X, y)
        model_ridge.fit(X, y)

        assert model_ridge.weights is not None
        assert model_no_reg.weights is not None

    def test_lasso_regularization_penalty(self):
        """Test that Lasso regularization adds penalty to cost."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([5, 8, 11, 14, 17])

        model_lasso = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            regularization="lasso",
            alpha=0.1,
            scale_features=False,
        )

        model_lasso.fit(X, y)
        assert model_lasso.weights is not None

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [5, 7, 9]}
        )
        X = df[["feature1", "feature2"]]
        y = df["target"]
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=1000, scale_features=False
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(df)

    def test_get_cost_history(self):
        """Test getting cost history."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        cost_history = model.get_cost_history()
        assert len(cost_history) > 0
        assert cost_history == model.cost_history

    def test_plot_training_history_no_exception(self):
        """Test that plot_training_history doesn't raise exceptions."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
            learning_rate=0.01, max_iterations=100, scale_features=False
        )
        model.fit(X, y)

        model.plot_training_history(show=False)

    def test_plot_training_history_save(self):
        """Test saving training history plot."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
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
        model = MultipleLinearRegression()
        X = np.array([]).reshape(0, 1)
        y = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(X, y)

    def test_fit_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        model = MultipleLinearRegression()
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="Length mismatch"):
            model.fit(X, y)

    def test_unknown_scaling_method(self):
        """Test that unknown scaling method raises error."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = MultipleLinearRegression(
            learning_rate=0.01,
            max_iterations=100,
            scale_features=True,
            scaling_method="unknown",
        )
        with pytest.raises(ValueError, match="Unknown scaling method"):
            model.fit(X, y)
