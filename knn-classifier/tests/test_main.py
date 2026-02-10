"""Unit tests for KNN classifier implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import KNNClassifier, KNNOptimizer, DistanceMetrics


class TestDistanceMetrics:
    """Test distance metric implementations."""

    def test_euclidean(self):
        """Test Euclidean distance."""
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        distance = DistanceMetrics.euclidean(x1, x2)
        assert abs(distance - 5.0) < 1e-10

    def test_manhattan(self):
        """Test Manhattan distance."""
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        distance = DistanceMetrics.manhattan(x1, x2)
        assert abs(distance - 7.0) < 1e-10

    def test_minkowski(self):
        """Test Minkowski distance."""
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        distance_p2 = DistanceMetrics.minkowski(x1, x2, p=2.0)
        assert abs(distance_p2 - 5.0) < 1e-10

        distance_p1 = DistanceMetrics.minkowski(x1, x2, p=1.0)
        assert abs(distance_p1 - 7.0) < 1e-10

    def test_hamming(self):
        """Test Hamming distance."""
        x1 = np.array([0, 1, 0, 1])
        x2 = np.array([1, 1, 0, 0])
        distance = DistanceMetrics.hamming(x1, x2)
        assert abs(distance - 0.5) < 1e-10

    def test_cosine(self):
        """Test cosine distance."""
        x1 = np.array([1, 0])
        x2 = np.array([0, 1])
        distance = DistanceMetrics.cosine(x1, x2)
        assert abs(distance - 1.0) < 1e-10

    def test_cosine_similar_vectors(self):
        """Test cosine distance for similar vectors."""
        x1 = np.array([1, 1])
        x2 = np.array([2, 2])
        distance = DistanceMetrics.cosine(x1, x2)
        assert abs(distance) < 1e-10


class TestKNNClassifier:
    """Test KNNClassifier functionality."""

    def test_initialization(self):
        """Test model initialization."""
        knn = KNNClassifier(k=5)
        assert knn.k == 5
        assert knn.distance_metric == "euclidean"
        assert knn.X_train is None

    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        knn = KNNClassifier(
            k=3, distance_metric="manhattan", scale_features=False
        )
        assert knn.k == 3
        assert knn.distance_metric == "manhattan"
        assert knn.scale_features is False

    def test_fit(self):
        """Test fitting the model."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        knn = KNNClassifier(k=3, scale_features=False)
        knn.fit(X, y)

        assert knn.X_train is not None
        assert knn.y_train is not None
        assert len(knn.X_train) == len(X)

    def test_fit_with_scaling(self):
        """Test fitting with feature scaling."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        knn = KNNClassifier(k=3, scale_features=True)
        knn.fit(X, y)

        assert knn.scale_params is not None

    def test_fit_invalid_k(self):
        """Test that invalid k raises error."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 0])
        knn = KNNClassifier(k=5)
        with pytest.raises(ValueError, match="cannot be greater"):
            knn.fit(X, y)

    def test_fit_k_zero(self):
        """Test that k=0 raises error."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 0])
        knn = KNNClassifier(k=0)
        with pytest.raises(ValueError, match="must be at least 1"):
            knn.fit(X, y)

    def test_predict_before_fit(self):
        """Test that prediction before fitting raises error."""
        knn = KNNClassifier(k=3)
        X = np.array([[1], [2]])
        with pytest.raises(ValueError, match="must be fitted"):
            knn.predict(X)

    def test_predict(self):
        """Test class prediction."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        knn = KNNClassifier(k=3, scale_features=False)
        knn.fit(X, y)

        X_test = np.array([[1.5], [3.5]])
        predictions = knn.predict(X_test)
        assert len(predictions) == len(X_test)
        assert np.all(np.isin(predictions, y))

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        knn = KNNClassifier(k=3, scale_features=False)
        knn.fit(X, y)

        X_test = np.array([[1.5], [3.5]])
        probabilities = knn.predict_proba(X_test)
        assert probabilities.shape[0] == len(X_test)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    def test_score(self):
        """Test accuracy score calculation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        knn = KNNClassifier(k=3, scale_features=False)
        knn.fit(X, y)

        score = knn.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_different_distance_metrics(self):
        """Test different distance metrics."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])

        for metric in ["euclidean", "manhattan", "cosine"]:
            knn = KNNClassifier(k=3, distance_metric=metric, scale_features=False)
            knn.fit(X, y)
            predictions = knn.predict(X)
            assert len(predictions) == len(X)

    def test_minkowski_distance(self):
        """Test Minkowski distance with custom p."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        knn = KNNClassifier(
            k=3,
            distance_metric="minkowski",
            metric_params={"p": 3.0},
            scale_features=False,
        )
        knn.fit(X, y)
        predictions = knn.predict(X)
        assert len(predictions) == len(X)

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "target": [0, 0, 1, 1, 1],
            }
        )
        X = df[["feature1", "feature2"]]
        y = df["target"]
        knn = KNNClassifier(k=3, scale_features=False)
        knn.fit(X, y)

        predictions = knn.predict(X)
        assert len(predictions) == len(df)

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        knn = KNNClassifier(k=3, scale_features=False)
        knn.fit(X, y)

        predictions = knn.predict(X)
        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, y))


class TestKNNOptimizer:
    """Test KNNOptimizer functionality."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = KNNOptimizer(k_range=[1, 3, 5])
        assert optimizer.k_range == [1, 3, 5]
        assert optimizer.cv_folds == 5

    def test_optimize(self):
        """Test k-value optimization."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        optimizer = KNNOptimizer(k_range=[1, 3, 5], cv_folds=3, scale_features=False)
        results = optimizer.optimize(X, y)

        assert "best_k" in results
        assert "best_score" in results
        assert "results" in results
        assert results["best_k"] in optimizer.k_range

    def test_optimize_results_structure(self):
        """Test optimization results structure."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        optimizer = KNNOptimizer(k_range=[1, 3, 5], cv_folds=3, scale_features=False)
        results = optimizer.optimize(X, y)

        for k, k_result in results["results"].items():
            assert "mean" in k_result
            assert "std" in k_result
            assert "scores" in k_result
            assert 0.0 <= k_result["mean"] <= 1.0

    def test_plot_optimization_results_no_exception(self):
        """Test that plot doesn't raise exceptions."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        optimizer = KNNOptimizer(k_range=[1, 3, 5], cv_folds=3, scale_features=False)
        optimizer.optimize(X, y)
        optimizer.plot_optimization_results(show=False)

    def test_plot_optimization_results_save(self):
        """Test saving optimization plot."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        optimizer = KNNOptimizer(k_range=[1, 3, 5], cv_folds=3, scale_features=False)
        optimizer.optimize(X, y)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                optimizer.plot_optimization_results(save_path=save_path, show=False)
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_optimize_with_different_metrics(self):
        """Test optimization with different distance metrics."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        for metric in ["euclidean", "manhattan"]:
            optimizer = KNNOptimizer(
                k_range=[1, 3, 5],
                distance_metric=metric,
                cv_folds=3,
                scale_features=False,
            )
            results = optimizer.optimize(X, y)
            assert "best_k" in results
