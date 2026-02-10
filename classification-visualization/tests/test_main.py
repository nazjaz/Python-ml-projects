"""Unit tests for classification visualization implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import ClassificationVisualizer


class TestClassificationVisualizer:
    """Test ClassificationVisualizer functionality."""

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
        viz = ClassificationVisualizer()
        assert viz.config is not None
        assert "logging" in viz.config
        assert "visualization" in viz.config

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "logging": {"level": "WARNING", "file": "logs/test.log"},
            "visualization": {"figsize": [12, 10], "dpi": 150},
        }
        config_path = self.create_temp_config(config)
        try:
            viz = ClassificationVisualizer(config_path=config_path)
            assert viz.config["logging"]["level"] == "WARNING"
            assert viz.figsize == (12, 10)
            assert viz.dpi == 150
        finally:
            Path(config_path).unlink()

    def test_validate_inputs_mismatched_lengths(self):
        """Test that mismatched input lengths raise ValueError."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1]
        y_pred = [0, 1]
        with pytest.raises(ValueError, match="Length mismatch"):
            viz._validate_inputs(y_true, y_pred)

    def test_validate_inputs_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        viz = ClassificationVisualizer()
        y_true = []
        y_pred = []
        with pytest.raises(ValueError, match="cannot be empty"):
            viz._validate_inputs(y_true, y_pred)

    def test_confusion_matrix_binary(self):
        """Test confusion matrix calculation for binary classification."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        cm = viz.confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2
        assert cm[0, 1] == 0
        assert cm[1, 0] == 1
        assert cm[1, 1] == 2

    def test_confusion_matrix_multiclass(self):
        """Test confusion matrix calculation for multiclass classification."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        cm = viz.confusion_matrix(y_true, y_pred)

        assert cm.shape == (3, 3)
        assert np.sum(cm) == 5

    def test_confusion_matrix_with_labels(self):
        """Test confusion matrix with specified labels."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        cm = viz.confusion_matrix(y_true, y_pred, labels=[0, 1])

        assert cm.shape == (2, 2)

    def test_confusion_matrix_with_numpy_arrays(self):
        """Test confusion matrix with numpy arrays."""
        viz = ClassificationVisualizer()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        cm = viz.confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)

    def test_confusion_matrix_with_pandas_series(self):
        """Test confusion matrix with pandas Series."""
        viz = ClassificationVisualizer()
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 0])
        cm = viz.confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)

    def test_classification_report_binary(self):
        """Test classification report for binary classification."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        report = viz.classification_report(y_true, y_pred)

        assert "per_class" in report
        assert "accuracy" in report
        assert "macro_avg" in report
        assert "weighted_avg" in report

        assert isinstance(report["accuracy"], float)
        assert 0.0 <= report["accuracy"] <= 1.0

    def test_classification_report_multiclass(self):
        """Test classification report for multiclass classification."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        report = viz.classification_report(y_true, y_pred)

        assert len(report["per_class"]) == 3
        assert "precision" in report["per_class"]["0"]
        assert "recall" in report["per_class"]["0"]
        assert "f1_score" in report["per_class"]["0"]
        assert "support" in report["per_class"]["0"]

    def test_classification_report_with_target_names(self):
        """Test classification report with custom target names."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        report = viz.classification_report(
            y_true, y_pred, target_names=["Negative", "Positive"]
        )

        assert "Negative" in report["per_class"]
        assert "Positive" in report["per_class"]

    def test_classification_report_perfect_prediction(self):
        """Test classification report with perfect predictions."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        report = viz.classification_report(y_true, y_pred)

        assert report["accuracy"] == 1.0
        for class_metrics in report["per_class"].values():
            assert class_metrics["precision"] == 1.0
            assert class_metrics["recall"] == 1.0
            assert class_metrics["f1_score"] == 1.0

    def test_plot_confusion_matrix_no_exception(self):
        """Test that plot_confusion_matrix doesn't raise exceptions."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        viz.plot_confusion_matrix(y_true, y_pred, show=False)

    def test_plot_confusion_matrix_normalized(self):
        """Test normalized confusion matrix plot."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        viz.plot_confusion_matrix(
            y_true, y_pred, normalize=True, show=False
        )

    def test_plot_confusion_matrix_save(self):
        """Test saving confusion matrix plot."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                viz.plot_confusion_matrix(
                    y_true, y_pred, save_path=save_path, show=False
                )
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_plot_classification_report_no_exception(self):
        """Test that plot_classification_report doesn't raise exceptions."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        viz.plot_classification_report(y_true, y_pred, show=False)

    def test_plot_classification_report_save(self):
        """Test saving classification report plot."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                viz.plot_classification_report(
                    y_true, y_pred, save_path=save_path, show=False
                )
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_print_classification_report_no_exception(self):
        """Test that print_classification_report doesn't raise exceptions."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        viz.print_classification_report(y_true, y_pred)

    def test_save_report(self):
        """Test saving classification report to JSON."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name
            try:
                viz.save_report(y_true, y_pred, output_path=save_path)
                assert Path(save_path).exists()

                import json

                with open(save_path) as f:
                    data = json.load(f)
                    assert "classification_report" in data
                    assert "confusion_matrix" in data
            finally:
                Path(save_path).unlink()

    def test_metrics_consistency(self):
        """Test that metrics are consistent across methods."""
        viz = ClassificationVisualizer()
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        cm = viz.confusion_matrix(y_true, y_pred)
        report = viz.classification_report(y_true, y_pred)

        accuracy_from_cm = np.sum(np.diag(cm)) / np.sum(cm)
        assert abs(accuracy_from_cm - report["accuracy"]) < 1e-6
