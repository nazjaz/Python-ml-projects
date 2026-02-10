"""Unit tests for XGBoost Classifier implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import XGBoostClassifier, XGBoostTree


class TestXGBoostTree:
    """Test XGBoost Tree functionality."""

    def test_initialization(self):
        """Test tree initialization."""
        tree = XGBoostTree(max_depth=3)
        assert tree.max_depth == 3
        assert tree.reg_lambda == 1.0
        assert tree.root is None

    def test_calculate_leaf_value(self):
        """Test leaf value calculation."""
        tree = XGBoostTree(reg_lambda=1.0)
        gradients = np.array([-0.5, -0.3, 0.2, 0.4])
        hessians = np.array([0.25, 0.21, 0.16, 0.24])
        value = tree._calculate_leaf_value(gradients, hessians)
        assert isinstance(value, (int, float))

    def test_calculate_gain(self):
        """Test gain calculation."""
        tree = XGBoostTree(reg_lambda=1.0, gamma=0.0)
        gain = tree._calculate_gain(10.0, 5.0, 8.0, 4.0, 18.0, 9.0)
        assert isinstance(gain, (int, float))

    def test_fit(self):
        """Test fitting tree."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        gradients = np.array([-0.5, -0.3, 0.2, 0.4])
        hessians = np.array([0.25, 0.21, 0.16, 0.24])
        tree = XGBoostTree(max_depth=2)
        tree.fit(X, gradients, hessians)

        assert tree.root is not None
        assert tree.n_features_ == 2

    def test_predict(self):
        """Test prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        gradients = np.array([-0.5, -0.3, 0.2, 0.4])
        hessians = np.array([0.25, 0.21, 0.16, 0.24])
        tree = XGBoostTree(max_depth=2)
        tree.fit(X, gradients, hessians)

        predictions = tree.predict(X)
        assert len(predictions) == len(X)


class TestXGBoostClassifier:
    """Test XGBoost Classifier functionality."""

    def test_initialization(self):
        """Test model initialization."""
        xgb = XGBoostClassifier(n_estimators=10)
        assert xgb.n_estimators == 10
        assert xgb.learning_rate == 0.1
        assert len(xgb.estimators_) == 0

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        xgb = XGBoostClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5,
            reg_lambda=2.0,
            reg_alpha=0.5,
            random_state=42,
        )
        assert xgb.n_estimators == 50
        assert xgb.learning_rate == 0.05
        assert xgb.reg_lambda == 2.0
        assert xgb.reg_alpha == 0.5

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        xgb = XGBoostClassifier(n_estimators=10)
        xgb.fit(X, y)

        assert len(xgb.estimators_) == 10
        assert xgb.n_features_ == 2
        assert xgb.classes_ is not None
        assert xgb.init_score_ is not None

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])
        xgb = XGBoostClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="same length"):
            xgb.fit(X, y)

    def test_fit_multiclass_error(self):
        """Test that multiclass raises error."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 1, 1, 2, 2])
        xgb = XGBoostClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="binary classification"):
            xgb.fit(X, y)

    def test_gradient_hessian(self):
        """Test gradient and hessian calculation."""
        xgb = XGBoostClassifier()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([-1.0, 1.0, -0.5, 0.5])
        gradients, hessians = xgb._gradient_hessian(y_true, y_pred)

        assert len(gradients) == len(y_true)
        assert len(hessians) == len(y_true)
        assert np.all(hessians >= 0)

    def test_sigmoid(self):
        """Test sigmoid function."""
        xgb = XGBoostClassifier()
        x = np.array([0, 1, -1, 10, -10])
        sigmoid_values = xgb._sigmoid(x)
        assert np.all(sigmoid_values >= 0)
        assert np.all(sigmoid_values <= 1)

    def test_initial_prediction(self):
        """Test initial prediction calculation."""
        xgb = XGBoostClassifier()
        y = np.array([0, 0, 1, 1])
        init_score = xgb._initial_prediction(y)
        assert isinstance(init_score, (int, float))

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        xgb = XGBoostClassifier()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            xgb.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        xgb = XGBoostClassifier(n_estimators=10)
        xgb.fit(X, y)

        predictions = xgb.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in xgb.classes_ for pred in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        xgb = XGBoostClassifier(n_estimators=10)
        xgb.fit(X, y)

        proba = xgb.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_score(self):
        """Test classification accuracy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        xgb = XGBoostClassifier(n_estimators=10)
        xgb.fit(X, y)

        accuracy = xgb.score(X, y)
        assert 0 <= accuracy <= 1

    def test_get_feature_importances(self):
        """Test getting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        xgb = XGBoostClassifier(n_estimators=10)
        xgb.fit(X, y)

        importances = xgb.get_feature_importances()
        assert isinstance(importances, dict)
        assert len(importances) == 2
        assert abs(sum(importances.values()) - 1.0) < 0.01

    def test_get_feature_importances_before_fit(self):
        """Test that getting feature importance before fitting raises error."""
        xgb = XGBoostClassifier()
        with pytest.raises(ValueError, match="must be fitted"):
            xgb.get_feature_importances()

    def test_regularization(self):
        """Test L1 and L2 regularization."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        xgb_l2 = XGBoostClassifier(n_estimators=10, reg_lambda=2.0)
        xgb_l2.fit(X, y)

        xgb_l1 = XGBoostClassifier(n_estimators=10, reg_alpha=0.5)
        xgb_l1.fit(X, y)

        assert len(xgb_l2.estimators_) == 10
        assert len(xgb_l1.estimators_) == 10

    def test_early_stopping(self):
        """Test early stopping."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        X_val = np.array([[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]])
        y_val = np.array([0, 0, 1])

        xgb = XGBoostClassifier(n_estimators=50, early_stopping_rounds=5)
        xgb.fit(X, y, eval_set=(X_val, y_val))

        assert len(xgb.estimators_) <= 50
        assert xgb.best_iteration_ is not None or len(xgb.estimators_) < 50

    def test_subsample(self):
        """Test subsample parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        xgb = XGBoostClassifier(n_estimators=10, subsample=0.8)
        xgb.fit(X, y)

        assert len(xgb.estimators_) == 10

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 1, 1]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        xgb = XGBoostClassifier(n_estimators=10)
        xgb.fit(X, y)

        predictions = xgb.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 0, 1, 1, 1]

        xgb = XGBoostClassifier(n_estimators=10)
        xgb.fit(X, y)

        predictions = xgb.predict(X)
        assert len(predictions) == len(X)

    def test_plot_feature_importance(self):
        """Test plotting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        xgb = XGBoostClassifier(n_estimators=10)
        xgb.feature_names_ = ["feature1", "feature2"]
        xgb.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.png"
            xgb.plot_feature_importance(save_path=str(save_path), show=False)
            assert save_path.exists()
