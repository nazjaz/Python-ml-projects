"""Unit tests for ROC curve and AUC calculation implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import ROCCalculator


class TestROCCalculator:
    """Test ROCCalculator functionality."""

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
        calc = ROCCalculator()
        assert calc.config is not None
        assert "logging" in calc.config
        assert "visualization" in calc.config

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "logging": {"level": "WARNING", "file": "logs/test.log"},
            "visualization": {"figsize": [12, 10], "dpi": 150},
        }
        config_path = self.create_temp_config(config)
        try:
            calc = ROCCalculator(config_path=config_path)
            assert calc.config["logging"]["level"] == "WARNING"
            assert calc.figsize == (12, 10)
            assert calc.dpi == 150
        finally:
            Path(config_path).unlink()

    def test_validate_inputs_mismatched_lengths(self):
        """Test that mismatched input lengths raise ValueError."""
        calc = ROCCalculator()
        y_true = [0, 1, 1]
        y_scores = [0.5, 0.7]
        with pytest.raises(ValueError, match="Length mismatch"):
            calc._validate_inputs(y_true, y_scores)

    def test_validate_inputs_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        calc = ROCCalculator()
        y_true = []
        y_scores = []
        with pytest.raises(ValueError, match="cannot be empty"):
            calc._validate_inputs(y_true, y_scores)

    def test_validate_inputs_multiclass_error(self):
        """Test that multiclass labels raise ValueError."""
        calc = ROCCalculator()
        y_true = [0, 1, 2]
        y_scores = [0.3, 0.5, 0.7]
        with pytest.raises(ValueError, match="binary classification"):
            calc._validate_inputs(y_true, y_scores)

    def test_validate_inputs_invalid_labels(self):
        """Test that invalid labels raise ValueError."""
        calc = ROCCalculator()
        y_true = [0, 1, 2, 3]
        y_scores = [0.3, 0.5, 0.7, 0.9]
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            calc._validate_inputs(y_true, y_scores)

    def test_roc_curve_perfect_classifier(self):
        """Test ROC curve calculation with perfect classifier."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)

        assert len(fpr) == len(tpr)
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0

    def test_roc_curve_random_classifier(self):
        """Test ROC curve calculation with random classifier."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.5, 0.5, 0.5, 0.5]
        fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)

        assert len(fpr) == len(tpr)
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0

    def test_roc_curve_basic(self):
        """Test basic ROC curve calculation."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)

        assert len(fpr) > 0
        assert len(tpr) > 0
        assert len(thresholds) > 0
        assert all(0.0 <= x <= 1.0 for x in fpr)
        assert all(0.0 <= x <= 1.0 for x in tpr)

    def test_roc_curve_with_numpy_arrays(self):
        """Test ROC curve calculation with numpy arrays."""
        calc = ROCCalculator()
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)

        assert len(fpr) > 0
        assert len(tpr) > 0

    def test_roc_curve_with_pandas_series(self):
        """Test ROC curve calculation with pandas Series."""
        calc = ROCCalculator()
        y_true = pd.Series([0, 0, 1, 1])
        y_scores = pd.Series([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)

        assert len(fpr) > 0
        assert len(tpr) > 0

    def test_auc_perfect_classifier(self):
        """Test AUC calculation with perfect classifier."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.2, 0.8, 0.9]
        auc_score = calc.auc(y_true, y_scores)

        assert abs(auc_score - 1.0) < 1e-6

    def test_auc_random_classifier(self):
        """Test AUC calculation with random classifier."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.5, 0.5, 0.5, 0.5]
        auc_score = calc.auc(y_true, y_scores)

        assert 0.0 <= auc_score <= 1.0

    def test_auc_basic(self):
        """Test basic AUC calculation."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        auc_score = calc.auc(y_true, y_scores)

        assert 0.0 <= auc_score <= 1.0
        assert isinstance(auc_score, float)

    def test_auc_range(self):
        """Test that AUC is always between 0 and 1."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        auc_score = calc.auc(y_true, y_scores)

        assert 0.0 <= auc_score <= 1.0

    def test_auc_worse_than_random(self):
        """Test AUC calculation when classifier is worse than random."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.9, 0.8, 0.2, 0.1]
        auc_score = calc.auc(y_true, y_scores)

        assert 0.0 <= auc_score <= 1.0

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics at once."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        results = calc.calculate_all_metrics(y_true, y_scores)

        assert "auc" in results
        assert "roc_curve" in results
        assert "fpr" in results["roc_curve"]
        assert "tpr" in results["roc_curve"]
        assert "thresholds" in results["roc_curve"]

        assert isinstance(results["auc"], float)
        assert 0.0 <= results["auc"] <= 1.0

    def test_plot_roc_curve_no_exception(self):
        """Test that plot_roc_curve doesn't raise exceptions."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        calc.plot_roc_curve(y_true, y_scores, show=False)

    def test_plot_roc_curve_save(self):
        """Test saving ROC curve plot."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name
            try:
                calc.plot_roc_curve(
                    y_true, y_scores, save_path=save_path, show=False
                )
                assert Path(save_path).exists()
            finally:
                Path(save_path).unlink()

    def test_print_report_no_exception(self):
        """Test that print_report doesn't raise exceptions."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        calc.print_report(y_true, y_scores)

    def test_save_report(self):
        """Test saving ROC and AUC report to JSON."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name
            try:
                calc.save_report(y_true, y_scores, output_path=save_path)
                assert Path(save_path).exists()

                import json

                with open(save_path) as f:
                    data = json.load(f)
                    assert "auc" in data
                    assert "roc_curve" in data
            finally:
                Path(save_path).unlink()

    def test_roc_curve_monotonic(self):
        """Test that ROC curve is monotonic."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = [0.1, 0.3, 0.5, 0.7, 0.2, 0.9]
        fpr, tpr, _ = calc.roc_curve(y_true, y_scores)

        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        for i in range(1, len(tpr_sorted)):
            assert tpr_sorted[i] >= tpr_sorted[i - 1] - 1e-6

    def test_auc_consistency(self):
        """Test that AUC is consistent with ROC curve."""
        calc = ROCCalculator()
        y_true = [0, 0, 1, 1]
        y_scores = [0.1, 0.4, 0.35, 0.8]

        fpr, tpr, _ = calc.roc_curve(y_true, y_scores)
        auc_from_curve = np.trapz(tpr, fpr)
        auc_direct = calc.auc(y_true, y_scores)

        assert abs(auc_from_curve - auc_direct) < 1e-6

    def test_pos_label_parameter(self):
        """Test ROC curve with different positive label."""
        calc = ROCCalculator()
        y_true = [1, 1, 0, 0]
        y_scores = [0.1, 0.4, 0.35, 0.8]

        fpr, tpr, _ = calc.roc_curve(y_true, y_scores, pos_label=0)
        auc_score = calc.auc(y_true, y_scores, pos_label=0)

        assert 0.0 <= auc_score <= 1.0
        assert len(fpr) > 0
        assert len(tpr) > 0
