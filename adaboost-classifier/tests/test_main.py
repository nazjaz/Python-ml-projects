"""Unit tests for AdaBoost Classifier implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import AdaBoostClassifier, DecisionStump


class TestDecisionStump:
    """Test Decision Stump functionality."""

    def test_initialization(self):
        """Test stump initialization."""
        stump = DecisionStump()
        assert stump.feature_index is None
        assert stump.threshold is None
        assert stump.alpha == 0.0

    def test_fit(self):
        """Test fitting stump."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([-1, -1, 1, 1])
        sample_weights = np.array([0.25, 0.25, 0.25, 0.25])
        stump = DecisionStump()
        stump.fit(X, y, sample_weights)

        assert stump.feature_index is not None
        assert stump.threshold is not None

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([-1, -1, 1, 1])
        sample_weights = np.array([0.25, 0.25, 0.25, 0.25])
        stump = DecisionStump()
        stump.fit(X, y, sample_weights)

        predictions = stump.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [-1, 1] for pred in predictions)


class TestAdaBoostClassifier:
    """Test AdaBoost Classifier functionality."""

    def test_initialization(self):
        """Test model initialization."""
        adaboost = AdaBoostClassifier(n_estimators=10)
        assert adaboost.n_estimators == 10
        assert adaboost.learning_rate == 1.0
        assert len(adaboost.estimators_) == 0

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        adaboost = AdaBoostClassifier(
            n_estimators=50, learning_rate=0.5, random_state=42
        )
        assert adaboost.n_estimators == 50
        assert adaboost.learning_rate == 0.5

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        assert len(adaboost.estimators_) > 0
        assert adaboost.n_features_ == 2
        assert adaboost.classes_ is not None
        assert adaboost.feature_importances_ is not None

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])
        adaboost = AdaBoostClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="same length"):
            adaboost.fit(X, y)

    def test_fit_multiclass_error(self):
        """Test that multiclass raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 1, 1, 2, 2])
        adaboost = AdaBoostClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="binary classification"):
            adaboost.fit(X, y)

    def test_calculate_alpha(self):
        """Test alpha calculation."""
        adaboost = AdaBoostClassifier()
        alpha = adaboost._calculate_alpha(0.3)
        assert alpha > 0

    def test_update_weights(self):
        """Test weight update."""
        adaboost = AdaBoostClassifier()
        sample_weights = np.array([0.25, 0.25, 0.25, 0.25])
        y = np.array([-1, -1, 1, 1])
        predictions = np.array([-1, 1, 1, 1])
        alpha = 0.5

        updated_weights = adaboost._update_weights(
            sample_weights, y, predictions, alpha
        )

        assert len(updated_weights) == len(sample_weights)
        assert np.allclose(np.sum(updated_weights), 1.0)

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        adaboost = AdaBoostClassifier()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            adaboost.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        predictions = adaboost.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in adaboost.classes_ for pred in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        proba = adaboost.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_score(self):
        """Test classification accuracy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        accuracy = adaboost.score(X, y)
        assert 0 <= accuracy <= 1

    def test_get_feature_importances(self):
        """Test getting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        importances = adaboost.get_feature_importances()
        assert isinstance(importances, dict)
        assert len(importances) == 2
        assert abs(sum(importances.values()) - 1.0) < 0.01

    def test_get_feature_importances_before_fit(self):
        """Test that getting feature importance before fitting raises error."""
        adaboost = AdaBoostClassifier()
        with pytest.raises(ValueError, match="must be fitted"):
            adaboost.get_feature_importances()

    def test_different_n_estimators(self):
        """Test with different number of estimators."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        for n in [5, 10, 20]:
            adaboost = AdaBoostClassifier(n_estimators=n)
            adaboost.fit(X, y)
            assert len(adaboost.estimators_) > 0

    def test_different_learning_rates(self):
        """Test with different learning rates."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        for lr in [0.5, 1.0, 1.5]:
            adaboost = AdaBoostClassifier(n_estimators=10, learning_rate=lr)
            adaboost.fit(X, y)
            assert len(adaboost.estimators_) > 0

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 1, 1]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        predictions = adaboost.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 0, 1, 1, 1]

        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        predictions = adaboost.predict(X)
        assert len(predictions) == len(X)

    def test_get_estimator_errors(self):
        """Test getting estimator errors."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.fit(X, y)

        errors = adaboost.get_estimator_errors()
        assert len(errors) == len(adaboost.estimators_)
        assert all(0 <= error <= 1 for error in errors)

    def test_plot_feature_importance(self):
        """Test plotting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        adaboost = AdaBoostClassifier(n_estimators=10)
        adaboost.feature_names_ = ["feature1", "feature2"]
        adaboost.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.png"
            adaboost.plot_feature_importance(save_path=str(save_path), show=False)
            assert save_path.exists()
