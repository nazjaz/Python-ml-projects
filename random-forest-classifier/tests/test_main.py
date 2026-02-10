"""Unit tests for Random Forest Classifier implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.main import RandomForestClassifier


class TestRandomForestClassifier:
    """Test Random Forest Classifier functionality."""

    def test_initialization(self):
        """Test model initialization."""
        forest = RandomForestClassifier(n_estimators=10)
        assert forest.n_estimators == 10
        assert forest.bootstrap is True
        assert len(forest.estimators_) == 0

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        forest = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=3,
            bootstrap=False,
            random_state=42,
        )
        assert forest.n_estimators == 50
        assert forest.max_depth == 5
        assert forest.bootstrap is False

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        assert len(forest.estimators_) == 10
        assert forest.n_features_ == 2
        assert forest.classes_ is not None
        assert forest.feature_importances_ is not None

    def test_fit_mismatched_lengths(self):
        """Test that mismatched X and y lengths raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])
        forest = RandomForestClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="same length"):
            forest.fit(X, y)

    def test_bootstrap_sampling(self):
        """Test bootstrap sampling."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        forest_bootstrap = RandomForestClassifier(n_estimators=5, bootstrap=True)
        forest_bootstrap.fit(X, y)
        
        forest_no_bootstrap = RandomForestClassifier(n_estimators=5, bootstrap=False)
        forest_no_bootstrap.fit(X, y)
        
        assert len(forest_bootstrap.estimators_) == 5
        assert len(forest_no_bootstrap.estimators_) == 5

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        forest = RandomForestClassifier()
        X = np.array([[1, 2], [2, 3]])
        with pytest.raises(ValueError, match="must be fitted"):
            forest.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        predictions = forest.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in forest.classes_ for pred in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        proba = forest.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == len(forest.classes_)
        assert np.allclose(np.sum(proba, axis=1), 1.0)

    def test_score(self):
        """Test classification accuracy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        accuracy = forest.score(X, y)
        assert 0 <= accuracy <= 1

    def test_get_feature_importances(self):
        """Test getting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        importances = forest.get_feature_importances()
        assert isinstance(importances, dict)
        assert len(importances) == 2
        assert abs(sum(importances.values()) - 1.0) < 0.01

    def test_get_feature_importances_before_fit(self):
        """Test that getting feature importance before fitting raises error."""
        forest = RandomForestClassifier()
        with pytest.raises(ValueError, match="must be fitted"):
            forest.get_feature_importances()

    def test_max_features_sqrt(self):
        """Test max_features='sqrt'."""
        X = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
        y = np.array([0, 0, 1, 1])
        forest = RandomForestClassifier(n_estimators=5, max_features="sqrt")
        forest.fit(X, y)

        assert len(forest.estimators_) == 5

    def test_max_features_log2(self):
        """Test max_features='log2'."""
        X = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
        y = np.array([0, 0, 1, 1])
        forest = RandomForestClassifier(n_estimators=5, max_features="log2")
        forest.fit(X, y)

        assert len(forest.estimators_) == 5

    def test_max_features_int(self):
        """Test max_features as integer."""
        X = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
        y = np.array([0, 0, 1, 1])
        forest = RandomForestClassifier(n_estimators=5, max_features=2)
        forest.fit(X, y)

        assert len(forest.estimators_) == 5

    def test_max_features_float(self):
        """Test max_features as float."""
        X = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
        y = np.array([0, 0, 1, 1])
        forest = RandomForestClassifier(n_estimators=5, max_features=0.5)
        forest.fit(X, y)

        assert len(forest.estimators_) == 5

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [2, 3, 4, 5, 6, 7],
            "target": [0, 0, 0, 1, 1, 1]
        })
        X = df[["feature1", "feature2"]]
        y = df["target"]

        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        predictions = forest.predict(X)
        assert len(predictions) == len(df)

    def test_with_list_input(self):
        """Test with list input."""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
        y = [0, 0, 0, 1, 1, 1]

        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        predictions = forest.predict(X)
        assert len(predictions) == len(X)

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X = np.array([
            [1, 2], [2, 3], [3, 4],
            [4, 5], [5, 6], [6, 7],
            [7, 8], [8, 9], [9, 10]
        ])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        forest = RandomForestClassifier(n_estimators=10)
        forest.fit(X, y)

        assert len(forest.classes_) == 3
        predictions = forest.predict(X)
        assert len(predictions) == len(X)

    def test_different_n_estimators(self):
        """Test with different number of estimators."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])

        for n in [5, 10, 20]:
            forest = RandomForestClassifier(n_estimators=n)
            forest.fit(X, y)
            assert len(forest.estimators_) == n

    def test_plot_feature_importance(self):
        """Test plotting feature importance."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        y = np.array([0, 0, 0, 1, 1, 1])
        forest = RandomForestClassifier(n_estimators=10)
        forest.feature_names_ = ["feature1", "feature2"]
        forest.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.png"
            forest.plot_feature_importance(save_path=str(save_path), show=False)
            assert save_path.exists()
