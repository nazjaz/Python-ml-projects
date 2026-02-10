"""Unit tests for Gradient Boosting Classifier implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import GradientBoostingClassifier


class TestGradientBoostingClassifier:
    """Test Gradient Boosting Classifier functionality."""

    def test_initialization(self):
        """Test model initialization."""
        gb = GradientBoostingClassifier(n_estimators=10)
        assert gb.n_estimators == 10
        assert gb.learning_rate == 0.1
        assert len(gb.estimators_) == 0

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        gb = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
        )
        assert gb.n_estimators == 50
        assert gb.learning_rate == 0.05
        assert gb.max_depth == 5
        assert gb.subsample == 0.8

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(X, y)

        assert len(gb.estimators_) == 10
        assert gb.n_features_ == 2
        assert gb.classes_ is not None
        assert gb.init_score_ is not None

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])
        gb = GradientBoostingClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="same length"):
            gb.fit(X, y)

    def test_fit_multiclass_error(self):
        """Test that multiclass raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 1, 1, 2, 2])
        gb = GradientBoostingClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="binary classification"):
            gb.fit(X, y)

    def test_sigmoid(self):
        """Test sigmoid function."""
        gb = GradientBoostingClassifier()
        x = np.array([0, 1, -1, 10, -10])
        sigmoid_values = gb._sigmoid(x)
        assert np.all(sigmoid_values >= 0)
        assert np.all(sigmoid_values <= 1)

    def test_log_loss_gradient(self):
        """Test log loss gradient calculation."""
        gb = GradientBoostingClassifier()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.3, 0.7, 0.2, 0.8])
        gradients = gb._log_loss_gradient(y_true, y_pred)
        assert len(gradients) == len(y_true)

    def test_initial_prediction(self):
        """Test initial prediction calculation."""
        gb = GradientBoostingClassifier()
        y = np.array([0, 0, 1, 1])
        init_score = gb._initial_prediction(y)
        assert isinstance(init_score, (int, float))

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        gb = GradientBoostingClassifier()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            gb.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(X, y)

        predictions = gb.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in gb.classes_ for pred in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(X, y)

        proba = gb.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_score(self):
        """Test classification accuracy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(X, y)

        accuracy = gb.score(X, y)
        assert 0 <= accuracy <= 1

    def test_get_feature_importances(self):
        """Test getting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(X, y)

        importances = gb.get_feature_importances()
        assert isinstance(importances, dict)
        assert len(importances) == 2
        assert abs(sum(importances.values()) - 1.0) < 0.01

    def test_get_feature_importances_before_fit(self):
        """Test that getting feature importance before fitting raises error."""
        gb = GradientBoostingClassifier()
        with pytest.raises(ValueError, match="must be fitted"):
            gb.get_feature_importances()

    def test_learning_rate(self):
        """Test different learning rates."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        for lr in [0.01, 0.1, 0.5]:
            gb = GradientBoostingClassifier(n_estimators=10, learning_rate=lr)
            gb.fit(X, y)
            assert len(gb.estimators_) == 10

    def test_max_depth(self):
        """Test different max depths."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        for depth in [1, 3, 5]:
            gb = GradientBoostingClassifier(n_estimators=10, max_depth=depth)
            gb.fit(X, y)
            assert len(gb.estimators_) == 10

    def test_subsample(self):
        """Test subsample parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        gb = GradientBoostingClassifier(n_estimators=10, subsample=0.8)
        gb.fit(X, y)
        assert len(gb.estimators_) == 10

    def test_different_n_estimators(self):
        """Test with different number of estimators."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        for n in [5, 10, 20]:
            gb = GradientBoostingClassifier(n_estimators=n)
            gb.fit(X, y)
            assert len(gb.estimators_) == n

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 1, 1]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(X, y)

        predictions = gb.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 0, 1, 1, 1]

        gb = GradientBoostingClassifier(n_estimators=10)
        gb.fit(X, y)

        predictions = gb.predict(X)
        assert len(predictions) == len(X)

    def test_plot_feature_importance(self):
        """Test plotting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        gb = GradientBoostingClassifier(n_estimators=10)
        gb.feature_names_ = ["feature1", "feature2"]
        gb.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.png"
            gb.plot_feature_importance(save_path=str(save_path), show=False)
            assert save_path.exists()
