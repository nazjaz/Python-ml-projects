"""Regression Metrics Calculator.

This module provides functionality to calculate evaluation metrics
for regression tasks including MAE, MSE, RMSE, and R-squared.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """Calculate regression evaluation metrics."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize RegressionMetrics with configuration.

        Args:
            config_path: Path to configuration YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dictionary containing configuration settings.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError("Configuration file is empty")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {e}")
            raise

    def _setup_logging(self) -> None:
        """Configure logging based on configuration settings."""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_file = self.config.get("logging", {}).get("file", "logs/app.log")
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - " "%(message)s"
        )

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=10485760, backupCount=5
                ),
                logging.StreamHandler(),
            ],
        )

    def _validate_inputs(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> tuple:
        """Validate and convert inputs to numpy arrays.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Tuple of (y_true_array, y_pred_array) as numpy arrays.

        Raises:
            ValueError: If inputs are invalid or have mismatched lengths.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true has {len(y_true)} "
                f"samples, y_pred has {len(y_pred)} samples"
            )

        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            logger.warning("Input contains NaN values, results may be invalid")

        return y_true, y_pred

    def mae(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate Mean Absolute Error (MAE).

        MAE measures the average absolute difference between predicted
        and actual values. Lower values indicate better performance.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MAE score (non-negative float).

        Example:
            >>> metrics = RegressionMetrics()
            >>> y_true = [3.0, -0.5, 2.0, 7.0]
            >>> y_pred = [2.5, 0.0, 2.0, 8.0]
            >>> metrics.mae(y_true, y_pred)
            0.5
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        mae_score = np.mean(np.abs(y_true - y_pred))

        logger.debug(f"MAE: {mae_score:.4f}")
        return float(mae_score)

    def mse(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate Mean Squared Error (MSE).

        MSE measures the average squared difference between predicted
        and actual values. It penalizes larger errors more than MAE.
        Lower values indicate better performance.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            MSE score (non-negative float).

        Example:
            >>> metrics = RegressionMetrics()
            >>> y_true = [3.0, -0.5, 2.0, 7.0]
            >>> y_pred = [2.5, 0.0, 2.0, 8.0]
            >>> metrics.mse(y_true, y_pred)
            0.375
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        mse_score = np.mean((y_true - y_pred) ** 2)

        logger.debug(f"MSE: {mse_score:.4f}")
        return float(mse_score)

    def rmse(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate Root Mean Squared Error (RMSE).

        RMSE is the square root of MSE, providing error in the same
        units as the target variable. Lower values indicate better
        performance.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            RMSE score (non-negative float).

        Example:
            >>> metrics = RegressionMetrics()
            >>> y_true = [3.0, -0.5, 2.0, 7.0]
            >>> y_pred = [2.5, 0.0, 2.0, 8.0]
            >>> metrics.rmse(y_true, y_pred)
            0.6123724356957945
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        rmse_score = np.sqrt(self.mse(y_true, y_pred))

        logger.debug(f"RMSE: {rmse_score:.4f}")
        return float(rmse_score)

    def r_squared(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate R-squared (Coefficient of Determination).

        R-squared measures the proportion of variance in the dependent
        variable that is predictable from the independent variable(s).
        Higher values (closer to 1.0) indicate better fit. Can be
        negative for poor models.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            R-squared score (can be negative for poor models).

        Example:
            >>> metrics = RegressionMetrics()
            >>> y_true = [3.0, -0.5, 2.0, 7.0]
            >>> y_pred = [2.5, 0.0, 2.0, 8.0]
            >>> metrics.r_squared(y_true, y_pred)
            0.9486081370449679
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            logger.warning(
                "Total sum of squares is zero, R-squared is undefined"
            )
            return 0.0

        r2_score = 1 - (ss_res / ss_tot)

        logger.debug(f"R-squared: {r2_score:.4f}")
        return float(r2_score)

    def calculate_all_metrics(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """Calculate all regression metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Dictionary containing all calculated metrics:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r_squared: R-squared score

        Example:
            >>> metrics = RegressionMetrics()
            >>> y_true = [3.0, -0.5, 2.0, 7.0]
            >>> y_pred = [2.5, 0.0, 2.0, 8.0]
            >>> results = metrics.calculate_all_metrics(y_true, y_pred)
            >>> print(results['mae'])
            0.5
        """
        logger.info("Calculating all regression metrics")
        results = {
            "mae": self.mae(y_true, y_pred),
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "r_squared": self.r_squared(y_true, y_pred),
        }

        logger.info("Metrics calculation complete")
        return results

    def generate_detailed_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Generate detailed regression metrics report.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Dictionary containing:
            - metrics: All calculated metrics
            - statistics: Statistical summary of errors
            - residuals: Residual statistics

        Example:
            >>> metrics = RegressionMetrics()
            >>> y_true = [3.0, -0.5, 2.0, 7.0]
            >>> y_pred = [2.5, 0.0, 2.0, 8.0]
            >>> report = metrics.generate_detailed_report(y_true, y_pred)
            >>> print(report['metrics']['mae'])
            0.5
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)

        metrics = self.calculate_all_metrics(y_true, y_pred)
        residuals = y_true - y_pred

        statistics = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "min_residual": float(np.min(residuals)),
            "max_residual": float(np.max(residuals)),
            "median_residual": float(np.median(residuals)),
        }

        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
            "q25": float(np.percentile(residuals, 25)),
            "q75": float(np.percentile(residuals, 75)),
        }

        report = {
            "metrics": metrics,
            "statistics": statistics,
            "residuals": residual_stats,
            "sample_size": len(y_true),
        }

        logger.info("Detailed report generated")
        return report

    def print_report(
        self,
        y_true: Union[List, np.ndarray, pd.Series],
        y_pred: Union[List, np.ndarray, pd.Series],
    ) -> None:
        """Print formatted detailed report to console.

        Args:
            y_true: True values.
            y_pred: Predicted values.
        """
        report = self.generate_detailed_report(y_true, y_pred)

        print("\n" + "=" * 60)
        print("REGRESSION METRICS REPORT")
        print("=" * 60)

        print("\n--- Performance Metrics ---")
        print(f"Mean Absolute Error (MAE):     {report['metrics']['mae']:.6f}")
        print(f"Mean Squared Error (MSE):      {report['metrics']['mse']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {report['metrics']['rmse']:.6f}")
        print(f"R-squared (R²):                {report['metrics']['r_squared']:.6f}")

        print("\n--- Residual Statistics ---")
        residuals = report["residuals"]
        print(f"Mean Residual:                 {residuals['mean']:.6f}")
        print(f"Std Residual:                  {residuals['std']:.6f}")
        print(f"Min Residual:                  {residuals['min']:.6f}")
        print(f"Max Residual:                  {residuals['max']:.6f}")
        print(f"Median Residual:               {residuals['median']:.6f}")
        print(f"25th Percentile:               {residuals['q25']:.6f}")
        print(f"75th Percentile:               {residuals['q75']:.6f}")

        print(f"\n--- Dataset Information ---")
        print(f"Sample Size:                    {report['sample_size']}")

        print("\n" + "=" * 60)


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regression Metrics Calculator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--y-true",
        type=str,
        required=True,
        help="Path to CSV file with true values or comma-separated values",
    )
    parser.add_argument(
        "--y-pred",
        type=str,
        required=True,
        help="Path to CSV file with predicted values or comma-separated values",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Column name if input is CSV file (default: first column)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed report as JSON (optional)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed formatted report to console",
    )

    args = parser.parse_args()

    metrics = RegressionMetrics(config_path=args.config)

    try:
        # Load true values
        if args.y_true.endswith(".csv"):
            df_true = pd.read_csv(args.y_true)
            column = args.column if args.column else df_true.columns[0]
            y_true = df_true[column].values
        else:
            y_true = [float(x.strip()) for x in args.y_true.split(",")]

        # Load predicted values
        if args.y_pred.endswith(".csv"):
            df_pred = pd.read_csv(args.y_pred)
            column = args.column if args.column else df_pred.columns[0]
            y_pred = df_pred[column].values
        else:
            y_pred = [float(x.strip()) for x in args.y_pred.split(",")]

        print("\n=== Regression Metrics ===")
        print(f"True values: {len(y_true)} samples")
        print(f"Predicted values: {len(y_pred)} samples")

        if args.report:
            metrics.print_report(y_true, y_pred)
        else:
            results = metrics.calculate_all_metrics(y_true, y_pred)

            print("\n=== Results ===")
            print(f"MAE:       {results['mae']:.6f}")
            print(f"MSE:       {results['mse']:.6f}")
            print(f"RMSE:      {results['rmse']:.6f}")
            print(f"R-squared: {results['r_squared']:.6f}")

        if args.output:
            report = metrics.generate_detailed_report(y_true, y_pred)
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nDetailed report saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


if __name__ == "__main__":
    main()
