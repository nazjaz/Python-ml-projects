"""Unit tests for regression metrics implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import RegressionMetrics


class TestRegressionMetrics:
    """Test RegressionMetrics functionality."""

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
        metrics = RegressionMetrics()
        assert metrics.config is not None
        assert "logging" in metrics.config

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "logging": {"level": "WARNING", "file": "logs/test.log"}
        }
        config_path = self.create_temp_config(config)
        try:
            metrics = RegressionMetrics(config_path=config_path)
            assert metrics.config["logging"]["level"] == "WARNING"
        finally:
            Path(config_path).unlink()

    def test_validate_inputs_mismatched_lengths(self):
        """Test that mismatched input lengths raise ValueError."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0]
        with pytest.raises(ValueError, match="Length mismatch"):
            metrics._validate_inputs(y_true, y_pred)

    def test_validate_inputs_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        metrics = RegressionMetrics()
        y_true = []
        y_pred = []
        with pytest.raises(ValueError, match="cannot be empty"):
            metrics._validate_inputs(y_true, y_pred)

    def test_mae_perfect_prediction(self):
        """Test MAE calculation with perfect predictions."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        assert metrics.mae(y_true, y_pred) == 0.0

    def test_mae_imperfect_prediction(self):
        """Test MAE calculation with imperfect predictions."""
        metrics = RegressionMetrics()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        mae = metrics.mae(y_true, y_pred)
        assert abs(mae - 0.5) < 1e-6

    def test_mae_with_numpy_arrays(self):
        """Test MAE calculation with numpy arrays."""
        metrics = RegressionMetrics()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        mae = metrics.mae(y_true, y_pred)
        assert abs(mae - 0.5) < 1e-6

    def test_mae_with_pandas_series(self):
        """Test MAE calculation with pandas Series."""
        metrics = RegressionMetrics()
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.5, 2.5, 2.5])
        mae = metrics.mae(y_true, y_pred)
        assert abs(mae - 0.5) < 1e-6

    def test_mse_perfect_prediction(self):
        """Test MSE calculation with perfect predictions."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        assert metrics.mse(y_true, y_pred) == 0.0

    def test_mse_imperfect_prediction(self):
        """Test MSE calculation with imperfect predictions."""
        metrics = RegressionMetrics()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        mse = metrics.mse(y_true, y_pred)
        assert abs(mse - 0.375) < 1e-6

    def test_mse_penalizes_large_errors(self):
        """Test that MSE penalizes large errors more than MAE."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0]
        y_pred = [2.0, 2.0, 2.0]
        mae = metrics.mae(y_true, y_pred)
        mse = metrics.mse(y_true, y_pred)
        assert mse > mae

    def test_rmse_perfect_prediction(self):
        """Test RMSE calculation with perfect predictions."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        assert metrics.rmse(y_true, y_pred) == 0.0

    def test_rmse_imperfect_prediction(self):
        """Test RMSE calculation with imperfect predictions."""
        metrics = RegressionMetrics()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        rmse = metrics.rmse(y_true, y_pred)
        expected = np.sqrt(0.375)
        assert abs(rmse - expected) < 1e-6

    def test_rmse_is_sqrt_of_mse(self):
        """Test that RMSE is the square root of MSE."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.5, 2.5, 3.5, 4.5]
        mse = metrics.mse(y_true, y_pred)
        rmse = metrics.rmse(y_true, y_pred)
        assert abs(rmse - np.sqrt(mse)) < 1e-6

    def test_r_squared_perfect_prediction(self):
        """Test R-squared calculation with perfect predictions."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        r2 = metrics.r_squared(y_true, y_pred)
        assert abs(r2 - 1.0) < 1e-6

    def test_r_squared_imperfect_prediction(self):
        """Test R-squared calculation with imperfect predictions."""
        metrics = RegressionMetrics()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        r2 = metrics.r_squared(y_true, y_pred)
        assert isinstance(r2, float)
        assert r2 <= 1.0

    def test_r_squared_negative_for_poor_model(self):
        """Test that R-squared can be negative for poor models."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [10.0, 10.0, 10.0, 10.0]
        r2 = metrics.r_squared(y_true, y_pred)
        assert r2 < 0.0

    def test_r_squared_with_constant_true_values(self):
        """Test R-squared when true values are constant."""
        metrics = RegressionMetrics()
        y_true = [2.0, 2.0, 2.0, 2.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        r2 = metrics.r_squared(y_true, y_pred)
        assert r2 == 0.0

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics at once."""
        metrics = RegressionMetrics()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        results = metrics.calculate_all_metrics(y_true, y_pred)

        assert "mae" in results
        assert "mse" in results
        assert "rmse" in results
        assert "r_squared" in results

        assert isinstance(results["mae"], float)
        assert isinstance(results["mse"], float)
        assert isinstance(results["rmse"], float)
        assert isinstance(results["r_squared"], float)

        assert results["mae"] >= 0.0
        assert results["mse"] >= 0.0
        assert results["rmse"] >= 0.0

    def test_generate_detailed_report(self):
        """Test detailed report generation."""
        metrics = RegressionMetrics()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        report = metrics.generate_detailed_report(y_true, y_pred)

        assert "metrics" in report
        assert "statistics" in report
        assert "residuals" in report
        assert "sample_size" in report

        assert "mae" in report["metrics"]
        assert "mse" in report["metrics"]
        assert "rmse" in report["metrics"]
        assert "r_squared" in report["metrics"]

        assert "mean_residual" in report["statistics"]
        assert "std_residual" in report["statistics"]
        assert "min_residual" in report["statistics"]
        assert "max_residual" in report["statistics"]
        assert "median_residual" in report["statistics"]

        assert "mean" in report["residuals"]
        assert "std" in report["residuals"]
        assert "min" in report["residuals"]
        assert "max" in report["residuals"]
        assert "median" in report["residuals"]
        assert "q25" in report["residuals"]
        assert "q75" in report["residuals"]

        assert report["sample_size"] == len(y_true)

    def test_print_report_no_exception(self):
        """Test that print_report doesn't raise exceptions."""
        metrics = RegressionMetrics()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        metrics.print_report(y_true, y_pred)

    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other."""
        metrics = RegressionMetrics()
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.5, 2.5, 3.5, 4.5, 5.5]

        mae = metrics.mae(y_true, y_pred)
        mse = metrics.mse(y_true, y_pred)
        rmse = metrics.rmse(y_true, y_pred)

        assert rmse >= mae
        assert mse >= mae * mae
        assert abs(rmse - np.sqrt(mse)) < 1e-6

    def test_negative_values(self):
        """Test metrics with negative values."""
        metrics = RegressionMetrics()
        y_true = [-1.0, -2.0, -3.0]
        y_pred = [-1.5, -2.5, -2.5]
        mae = metrics.mae(y_true, y_pred)
        assert mae >= 0.0

    def test_large_values(self):
        """Test metrics with large values."""
        metrics = RegressionMetrics()
        y_true = [1000.0, 2000.0, 3000.0]
        y_pred = [1001.0, 1999.0, 3001.0]
        mae = metrics.mae(y_true, y_pred)
        assert mae >= 0.0

    def test_single_value(self):
        """Test metrics with single value."""
        metrics = RegressionMetrics()
        y_true = [5.0]
        y_pred = [4.0]
        mae = metrics.mae(y_true, y_pred)
        assert mae == 1.0
