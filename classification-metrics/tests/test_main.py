"""Unit tests for classification metrics implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import ClassificationMetrics


class TestClassificationMetrics:
    """Test ClassificationMetrics functionality."""

    def create_temp_config(self, config_dict: dict) -> str:
        """Create temporary config file for testing.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Path to temporary config file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def test_initialization_with_default_config(self):
        """Test initialization with default config file."""
        metrics = ClassificationMetrics()
        assert metrics.config is not None
        assert "logging" in metrics.config

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "logging": {"level": "WARNING", "file": "logs/test.log"}
        }
        config_path = self.create_temp_config(config)
        try:
            metrics = ClassificationMetrics(config_path=config_path)
            assert metrics.config["logging"]["level"] == "WARNING"
        finally:
            Path(config_path).unlink()

    def test_validate_inputs_mismatched_lengths(self):
        """Test that mismatched input lengths raise ValueError."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1]
        y_pred = [0, 1]
        with pytest.raises(ValueError, match="Length mismatch"):
            metrics._validate_inputs(y_true, y_pred)

    def test_validate_inputs_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        metrics = ClassificationMetrics()
        y_true = []
        y_pred = []
        with pytest.raises(ValueError, match="cannot be empty"):
            metrics._validate_inputs(y_true, y_pred)

    def test_accuracy_perfect_prediction(self):
        """Test accuracy calculation with perfect predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        assert metrics.accuracy(y_true, y_pred) == 1.0

    def test_accuracy_partial_correct(self):
        """Test accuracy calculation with partial correctness."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        assert metrics.accuracy(y_true, y_pred) == 0.8

    def test_accuracy_all_wrong(self):
        """Test accuracy calculation with all wrong predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1]
        y_pred = [1, 0, 0]
        assert metrics.accuracy(y_true, y_pred) == 0.0

    def test_accuracy_with_numpy_arrays(self):
        """Test accuracy calculation with numpy arrays."""
        metrics = ClassificationMetrics()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        assert metrics.accuracy(y_true, y_pred) == 0.75

    def test_accuracy_with_pandas_series(self):
        """Test accuracy calculation with pandas Series."""
        metrics = ClassificationMetrics()
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 0])
        assert metrics.accuracy(y_true, y_pred) == 0.75

    def test_precision_binary_perfect(self):
        """Test precision calculation with perfect binary predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        assert metrics.precision(y_true, y_pred) == 1.0

    def test_precision_binary_with_false_positives(self):
        """Test precision calculation with false positives."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [1, 1, 1, 0, 1]
        precision = metrics.precision(y_true, y_pred)
        assert precision == 0.75

    def test_precision_binary_no_positives(self):
        """Test precision calculation when no positive predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0]
        assert metrics.precision(y_true, y_pred) == 0.0

    def test_precision_multiclass_macro(self):
        """Test precision calculation with macro averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        precision = metrics.precision(y_true, y_pred, average="macro")
        assert isinstance(precision, float)
        assert 0.0 <= precision <= 1.0

    def test_precision_multiclass_micro(self):
        """Test precision calculation with micro averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        precision = metrics.precision(y_true, y_pred, average="micro")
        assert isinstance(precision, float)
        assert 0.0 <= precision <= 1.0

    def test_precision_multiclass_weighted(self):
        """Test precision calculation with weighted averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        precision = metrics.precision(y_true, y_pred, average="weighted")
        assert isinstance(precision, float)
        assert 0.0 <= precision <= 1.0

    def test_precision_multiclass_per_class(self):
        """Test precision calculation per class."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        precision = metrics.precision(y_true, y_pred, average=None)
        assert isinstance(precision, dict)
        assert len(precision) > 0

    def test_recall_binary_perfect(self):
        """Test recall calculation with perfect binary predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        assert metrics.recall(y_true, y_pred) == 1.0

    def test_recall_binary_with_false_negatives(self):
        """Test recall calculation with false negatives."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        recall = metrics.recall(y_true, y_pred)
        assert abs(recall - 2.0 / 3.0) < 1e-6

    def test_recall_binary_no_positives(self):
        """Test recall calculation when no positive predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0]
        assert metrics.recall(y_true, y_pred) == 0.0

    def test_recall_multiclass_macro(self):
        """Test recall calculation with macro averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        recall = metrics.recall(y_true, y_pred, average="macro")
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    def test_recall_multiclass_micro(self):
        """Test recall calculation with micro averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        recall = metrics.recall(y_true, y_pred, average="micro")
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    def test_recall_multiclass_weighted(self):
        """Test recall calculation with weighted averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        recall = metrics.recall(y_true, y_pred, average="weighted")
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    def test_recall_multiclass_per_class(self):
        """Test recall calculation per class."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        recall = metrics.recall(y_true, y_pred, average=None)
        assert isinstance(recall, dict)
        assert len(recall) > 0

    def test_f1_score_binary_perfect(self):
        """Test F1-score calculation with perfect binary predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        assert metrics.f1_score(y_true, y_pred) == 1.0

    def test_f1_score_binary_imperfect(self):
        """Test F1-score calculation with imperfect predictions."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        f1 = metrics.f1_score(y_true, y_pred)
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_f1_score_binary_zero_precision(self):
        """Test F1-score when precision is zero."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0]
        assert metrics.f1_score(y_true, y_pred) == 0.0

    def test_f1_score_multiclass_macro(self):
        """Test F1-score calculation with macro averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        f1 = metrics.f1_score(y_true, y_pred, average="macro")
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_f1_score_multiclass_micro(self):
        """Test F1-score calculation with micro averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        f1 = metrics.f1_score(y_true, y_pred, average="micro")
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_f1_score_multiclass_weighted(self):
        """Test F1-score calculation with weighted averaging."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        f1 = metrics.f1_score(y_true, y_pred, average="weighted")
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_f1_score_multiclass_per_class(self):
        """Test F1-score calculation per class."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        f1 = metrics.f1_score(y_true, y_pred, average=None)
        assert isinstance(f1, dict)
        assert len(f1) > 0

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics at once."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        results = metrics.calculate_all_metrics(y_true, y_pred)

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results

        assert isinstance(results["accuracy"], float)
        assert isinstance(results["precision"], float)
        assert isinstance(results["recall"], float)
        assert isinstance(results["f1_score"], float)

        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["precision"] <= 1.0
        assert 0.0 <= results["recall"] <= 1.0
        assert 0.0 <= results["f1_score"] <= 1.0

    def test_confusion_matrix_binary(self):
        """Test confusion matrix calculation for binary classification."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        cm = metrics.confusion_matrix(y_true, y_pred)

        assert isinstance(cm, dict)
        assert "0" in cm
        assert "1" in cm
        assert isinstance(cm["0"], dict)
        assert isinstance(cm["1"], dict)

    def test_confusion_matrix_multiclass(self):
        """Test confusion matrix calculation for multiclass classification."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        cm = metrics.confusion_matrix(y_true, y_pred)

        assert isinstance(cm, dict)
        assert len(cm) == 3
        for true_label in cm:
            assert isinstance(cm[true_label], dict)
            assert len(cm[true_label]) == 3

    def test_precision_invalid_average(self):
        """Test that invalid average parameter raises ValueError."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        with pytest.raises(ValueError, match="Invalid average parameter"):
            metrics.precision(y_true, y_pred, average="invalid")

    def test_recall_invalid_average(self):
        """Test that invalid average parameter raises ValueError."""
        metrics = ClassificationMetrics()
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        with pytest.raises(ValueError, match="Invalid average parameter"):
            metrics.recall(y_true, y_pred, average="invalid")
