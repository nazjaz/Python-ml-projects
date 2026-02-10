"""Tests for anomaly detection module."""

import numpy as np
import pytest

from src.main import (
    AnomalyDetector,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
)


class TestIsolationForestDetector:
    """Test cases for Isolation Forest detector."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(
            n_estimators=50, contamination=0.1, random_state=42
        )
        assert detector.n_estimators == 50
        assert detector.contamination == 0.1

    def test_fit_basic(self):
        """Test basic model fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = IsolationForestDetector(random_state=42)
        detector.fit(X)

        assert detector.model is not None

    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        X = np.array([[1, 2]])

        detector = IsolationForestDetector()
        with pytest.raises(ValueError, match="X must have at least 2"):
            detector.fit(X)

    def test_fit_invalid_shape(self):
        """Test fitting with invalid shape."""
        X = np.array([1, 2, 3])

        detector = IsolationForestDetector()
        with pytest.raises(ValueError, match="X must be 2D array"):
            detector.fit(X)

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = IsolationForestDetector(random_state=42)
        detector.fit(X)

        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [-1, 1] for pred in predictions)

    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        detector = IsolationForestDetector()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(X)

    def test_decision_function(self):
        """Test decision function."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = IsolationForestDetector(random_state=42)
        detector.fit(X)

        scores = detector.decision_function(X)
        assert len(scores) == len(X)
        assert isinstance(scores[0], (float, np.floating))

    def test_decision_function_not_fitted(self):
        """Test decision function without fitting."""
        detector = IsolationForestDetector()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.decision_function(X)


class TestOneClassSVMDetector:
    """Test cases for One-Class SVM detector."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = OneClassSVMDetector(kernel="rbf", nu=0.1)
        assert detector.kernel == "rbf"
        assert detector.nu == 0.1

    def test_fit_basic(self):
        """Test basic model fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = OneClassSVMDetector()
        detector.fit(X)

        assert detector.model is not None

    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        X = np.array([[1, 2]])

        detector = OneClassSVMDetector()
        with pytest.raises(ValueError, match="X must have at least 2"):
            detector.fit(X)

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = OneClassSVMDetector()
        detector.fit(X)

        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [-1, 1] for pred in predictions)

    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        detector = OneClassSVMDetector()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(X)

    def test_decision_function(self):
        """Test decision function."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = OneClassSVMDetector()
        detector.fit(X)

        scores = detector.decision_function(X)
        assert len(scores) == len(X)
        assert isinstance(scores[0], (float, np.floating))


class TestLocalOutlierFactorDetector:
    """Test cases for Local Outlier Factor detector."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = LocalOutlierFactorDetector(n_neighbors=10, contamination=0.1)
        assert detector.n_neighbors == 10
        assert detector.contamination == 0.1

    def test_fit_basic(self):
        """Test basic model fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = LocalOutlierFactorDetector(n_neighbors=10)
        detector.fit(X)

        assert detector.model is not None

    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        X = np.random.randn(10, 2)

        detector = LocalOutlierFactorDetector(n_neighbors=20)
        with pytest.raises(ValueError, match="X must have at least"):
            detector.fit(X)

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = LocalOutlierFactorDetector(n_neighbors=10)
        detector.fit(X)

        predictions = detector.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [-1, 1] for pred in predictions)

    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        detector = LocalOutlierFactorDetector()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(X)

    def test_decision_function(self):
        """Test decision function."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = LocalOutlierFactorDetector(n_neighbors=10)
        detector.fit(X)

        scores = detector.decision_function(X)
        assert len(scores) == len(X)
        assert isinstance(scores[0], (float, np.floating))


class TestAnomalyDetector:
    """Test cases for main AnomalyDetector class."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector()
        assert detector.detectors == {}
        assert detector.results == {}

    def test_fit_isolation_forest(self):
        """Test fitting Isolation Forest."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_isolation_forest(X)

        assert "isolation_forest" in detector.detectors

    def test_fit_one_class_svm(self):
        """Test fitting One-Class SVM."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_one_class_svm(X)

        assert "one_class_svm" in detector.detectors

    def test_fit_local_outlier_factor(self):
        """Test fitting Local Outlier Factor."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_local_outlier_factor(X)

        assert "local_outlier_factor" in detector.detectors

    def test_fit_all(self):
        """Test fitting all detectors."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_all(X)

        assert "isolation_forest" in detector.detectors
        assert "one_class_svm" in detector.detectors
        assert "local_outlier_factor" in detector.detectors

    def test_predict_all(self):
        """Test prediction with all methods."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_all(X)

        predictions = detector.predict(X)

        assert isinstance(predictions, dict)
        assert "isolation_forest" in predictions
        assert "one_class_svm" in predictions
        assert "local_outlier_factor" in predictions

    def test_predict_single_method(self):
        """Test prediction with single method."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_isolation_forest(X)

        predictions = detector.predict(X, method="isolation_forest")

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_invalid_method(self):
        """Test prediction with invalid method."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_isolation_forest(X)

        with pytest.raises(ValueError, match="Method 'invalid' not fitted"):
            detector.predict(X, method="invalid")

    def test_get_scores_all(self):
        """Test getting scores from all methods."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_all(X)

        scores = detector.get_scores(X)

        assert isinstance(scores, dict)
        assert "isolation_forest" in scores
        assert "one_class_svm" in scores
        assert "local_outlier_factor" in scores

    def test_get_scores_single_method(self):
        """Test getting scores from single method."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_isolation_forest(X)

        scores = detector.get_scores(X, method="isolation_forest")

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(X)

    def test_evaluate_with_labels(self):
        """Test evaluation with true labels."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y_true = np.ones(100)
        y_true[:10] = -1

        detector = AnomalyDetector()
        detector.fit_all(X)

        evaluation = detector.evaluate(X, y_true)

        assert "isolation_forest" in evaluation
        assert "one_class_svm" in evaluation
        assert "local_outlier_factor" in evaluation

        for method, metrics in evaluation.items():
            assert "n_anomalies" in metrics
            assert "n_normal" in metrics
            assert "anomaly_rate" in metrics
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics

    def test_evaluate_without_labels(self):
        """Test evaluation without true labels."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_all(X)

        evaluation = detector.evaluate(X)

        for method, metrics in evaluation.items():
            assert "n_anomalies" in metrics
            assert "n_normal" in metrics
            assert "anomaly_rate" in metrics
            assert "accuracy" not in metrics

    def test_plot_results(self):
        """Test plotting results."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector = AnomalyDetector()
        detector.fit_all(X)

        predictions = detector.predict(X)
        detector.plot_results(X, predictions)

    def test_plot_results_high_dim(self):
        """Test plotting with high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        detector = AnomalyDetector()
        detector.fit_all(X)

        predictions = detector.predict(X)
        detector.plot_results(X, predictions)


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self):
        """Test complete anomaly detection workflow."""
        np.random.seed(42)
        X_normal = np.random.randn(90, 2)
        X_anomaly = np.random.randn(10, 2) * 3 + np.array([5, 5])
        X = np.vstack([X_normal, X_anomaly])

        detector = AnomalyDetector()
        detector.fit_all(X)

        predictions = detector.predict(X)
        scores = detector.get_scores(X)
        evaluation = detector.evaluate(X)

        assert predictions is not None
        assert scores is not None
        assert evaluation is not None

        for method in ["isolation_forest", "one_class_svm", "local_outlier_factor"]:
            assert method in predictions
            assert method in scores
            assert method in evaluation

    def test_different_contamination_levels(self):
        """Test with different contamination levels."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        detector1 = IsolationForestDetector(contamination=0.05, random_state=42)
        detector1.fit(X)

        detector2 = IsolationForestDetector(contamination=0.2, random_state=42)
        detector2.fit(X)

        pred1 = detector1.predict(X)
        pred2 = detector2.predict(X)

        n_anomalies1 = np.sum(pred1 == -1)
        n_anomalies2 = np.sum(pred2 == -1)

        assert n_anomalies2 > n_anomalies1
