"""Time Series Data Preprocessing Tool.

This module provides functionality to perform time series data preprocessing
including resampling, interpolation, and trend removal.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Performs preprocessing operations on time series data."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize TimeSeriesPreprocessor with configuration.

        Args:
            config_path: Path to configuration YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_parameters()
        self.data: Optional[pd.DataFrame] = None
        self.datetime_column: Optional[str] = None

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

    def _initialize_parameters(self) -> None:
        """Initialize algorithm parameters from configuration."""
        preprocess_config = self.config.get("preprocessing", {})
        self.default_resample_freq = preprocess_config.get(
            "default_resample_freq", "D"
        )
        self.default_interpolation_method = preprocess_config.get(
            "default_interpolation_method", "linear"
        )
        self.default_trend_removal_method = preprocess_config.get(
            "default_trend_removal_method", "linear"
        )

    def load_data(
        self,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        datetime_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load time series data from file or use provided DataFrame.

        Args:
            file_path: Path to CSV file (optional).
            dataframe: Pandas DataFrame (optional).
            datetime_column: Name of datetime column (optional, auto-detect if None).

        Returns:
            Loaded DataFrame with datetime index.

        Raises:
            ValueError: If neither file_path nor dataframe provided.
            FileNotFoundError: If file doesn't exist.
        """
        if dataframe is not None:
            self.data = dataframe.copy()
            logger.info(f"Loaded DataFrame with shape {self.data.shape}")
        elif file_path is not None:
            try:
                self.data = pd.read_csv(file_path)
                logger.info(
                    f"Loaded CSV file: {file_path}, " f"shape: {self.data.shape}"
                )
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
        else:
            raise ValueError("Either file_path or dataframe must be provided")

        if datetime_column is None:
            datetime_column = self._detect_datetime_column()

        if datetime_column:
            self.datetime_column = datetime_column
            if datetime_column in self.data.columns:
                self.data[datetime_column] = pd.to_datetime(
                    self.data[datetime_column]
                )
                self.data = self.data.set_index(datetime_column)
                logger.info(f"Set datetime index: {datetime_column}")
            else:
                logger.warning(f"Datetime column '{datetime_column}' not found")
        else:
            if not isinstance(self.data.index, pd.DatetimeIndex):
                logger.warning("No datetime column detected, using default index")

        return self.data

    def _detect_datetime_column(self) -> Optional[str]:
        """Detect datetime column in the dataset.

        Returns:
            Name of datetime column or None if not found.
        """
        for col in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                return col
            try:
                pd.to_datetime(self.data[col].head(5))
                return col
            except (ValueError, TypeError):
                continue
        return None

    def resample(
        self,
        frequency: str,
        method: str = "mean",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Resample time series data to specified frequency.

        Args:
            frequency: Resampling frequency (e.g., 'D', 'H', 'W', 'M').
            method: Aggregation method (mean, sum, min, max, median).
            columns: List of columns to resample (None for all).

        Returns:
            Resampled DataFrame.

        Raises:
            ValueError: If no data loaded or invalid method.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index for resampling")

        if columns is None:
            columns = self.data.columns.tolist()

        method_map = {
            "mean": "mean",
            "sum": "sum",
            "min": "min",
            "max": "max",
            "median": "median",
            "first": "first",
            "last": "last",
        }

        if method not in method_map:
            raise ValueError(
                f"Invalid method: {method}. "
                f"Use one of: {list(method_map.keys())}"
            )

        resampled = self.data[columns].resample(frequency).agg(method_map[method])

        logger.info(
            f"Resampled data to frequency '{frequency}' "
            f"using method '{method}'"
        )

        return resampled

    def interpolate_missing(
        self,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Interpolate missing values in time series data.

        Args:
            method: Interpolation method (linear, polynomial, spline, time).
            columns: List of columns to interpolate (None for all numeric).
            limit: Maximum number of consecutive NaNs to fill.

        Returns:
            DataFrame with interpolated values.

        Raises:
            ValueError: If no data loaded or invalid method.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        method = method or self.default_interpolation_method

        if columns is None:
            columns = self.data.select_dtypes(include=["number"]).columns.tolist()

        if not columns:
            logger.warning("No numerical columns found for interpolation")
            return self.data.copy()

        result = self.data.copy()

        valid_methods = ["linear", "polynomial", "spline", "time", "quadratic", "cubic"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. " f"Use one of: {valid_methods}"
            )

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue

            if method == "time" and isinstance(result.index, pd.DatetimeIndex):
                result[col] = result[col].interpolate(
                    method="time", limit=limit, limit_direction="both"
                )
            elif method == "polynomial":
                result[col] = result[col].interpolate(
                    method="polynomial", order=2, limit=limit, limit_direction="both"
                )
            elif method == "spline":
                result[col] = result[col].interpolate(
                    method="spline", order=2, limit=limit, limit_direction="both"
                )
            else:
                result[col] = result[col].interpolate(
                    method=method, limit=limit, limit_direction="both"
                )

            logger.info(f"Interpolated '{col}' using method '{method}'")

        return result

    def remove_trend(
        self,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
        order: int = 1,
    ) -> pd.DataFrame:
        """Remove trend from time series data.

        Args:
            method: Trend removal method (linear, polynomial, moving_average).
            columns: List of columns to detrend (None for all numeric).
            order: Polynomial order (for polynomial method).

        Returns:
            DataFrame with trend removed.

        Raises:
            ValueError: If no data loaded or invalid method.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        method = method or self.default_trend_removal_method

        if columns is None:
            columns = self.data.select_dtypes(include=["number"]).columns.tolist()

        if not columns:
            logger.warning("No numerical columns found for trend removal")
            return self.data.copy()

        result = self.data.copy()

        valid_methods = ["linear", "polynomial", "moving_average"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. " f"Use one of: {valid_methods}"
            )

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue

            values = result[col].values
            indices = np.arange(len(values))

            if method == "linear":
                coeffs = np.polyfit(indices, values, 1)
                trend = np.polyval(coeffs, indices)
                result[col] = values - trend
            elif method == "polynomial":
                coeffs = np.polyfit(indices, values, order)
                trend = np.polyval(coeffs, indices)
                result[col] = values - trend
            elif method == "moving_average":
                window_size = min(30, len(values) // 10) if len(values) > 10 else len(values)
                trend = pd.Series(values).rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                result[col] = values - trend.values

            logger.info(f"Removed trend from '{col}' using method '{method}'")

        return result

    def get_time_series_info(self) -> Dict[str, any]:
        """Get information about the time series data.

        Returns:
            Dictionary with time series information.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        info = {
            "shape": self.data.shape,
            "n_samples": len(self.data),
            "n_features": len(self.data.columns),
            "columns": list(self.data.columns),
            "has_datetime_index": isinstance(self.data.index, pd.DatetimeIndex),
        }

        if isinstance(self.data.index, pd.DatetimeIndex):
            info["start_date"] = str(self.data.index.min())
            info["end_date"] = str(self.data.index.max())
            info["frequency"] = str(pd.infer_freq(self.data.index))
            info["date_range_days"] = (
                (self.data.index.max() - self.data.index.min()).days
            )

        info["missing_values"] = self.data.isnull().sum().to_dict()
        info["numeric_columns"] = (
            self.data.select_dtypes(include=["number"]).columns.tolist()
        )

        return info

    def save_preprocessed_data(self, output_path: str) -> None:
        """Save preprocessed data to CSV file.

        Args:
            output_path: Path to output CSV file.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        output_data = self.data.copy()
        if isinstance(output_data.index, pd.DatetimeIndex):
            output_data = output_data.reset_index()

        output_data.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed data to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Preprocessing Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--datetime-column",
        type=str,
        default=None,
        help="Name of datetime column (auto-detect if not specified)",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default=None,
        help="Resampling frequency (e.g., 'D', 'H', 'W')",
    )
    parser.add_argument(
        "--resample-method",
        type=str,
        default="mean",
        help="Resampling aggregation method",
    )
    parser.add_argument(
        "--interpolate",
        type=str,
        default=None,
        help="Interpolation method (linear, polynomial, spline, time)",
    )
    parser.add_argument(
        "--remove-trend",
        type=str,
        default=None,
        help="Trend removal method (linear, polynomial, moving_average)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )

    args = parser.parse_args()

    preprocessor = TimeSeriesPreprocessor(config_path=args.config)

    try:
        preprocessor.load_data(
            file_path=args.input, datetime_column=args.datetime_column
        )

        print("\n=== Time Series Information ===")
        info = preprocessor.get_time_series_info()
        print(f"Shape: {info['shape']}")
        print(f"Samples: {info['n_samples']}")
        print(f"Features: {info['n_features']}")
        print(f"Has datetime index: {info['has_datetime_index']}")
        if info["has_datetime_index"]:
            print(f"Start date: {info['start_date']}")
            print(f"End date: {info['end_date']}")
            print(f"Frequency: {info['frequency']}")

        if args.resample:
            print(f"\n=== Resampling to '{args.resample}' ===")
            preprocessor.data = preprocessor.resample(
                frequency=args.resample, method=args.resample_method
            )
            print(f"New shape: {preprocessor.data.shape}")

        if args.interpolate:
            print(f"\n=== Interpolating missing values ===")
            preprocessor.data = preprocessor.interpolate_missing(
                method=args.interpolate
            )
            print("Interpolation completed")

        if args.remove_trend:
            print(f"\n=== Removing trend ===")
            preprocessor.data = preprocessor.remove_trend(method=args.remove_trend)
            print("Trend removal completed")

        if args.output:
            preprocessor.save_preprocessed_data(args.output)
            print(f"\nPreprocessed data saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error preprocessing time series: {e}")
        raise


if __name__ == "__main__":
    main()
