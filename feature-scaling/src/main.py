"""Feature Scaling Tool.

This module provides functionality to normalize and standardize numerical
features using multiple feature scaling techniques.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FeatureScaler:
    """Handles feature scaling using multiple scaling techniques."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize FeatureScaler with configuration.

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
        self.scaling_params: Dict[str, Dict[str, Any]] = {}

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
        scaling_config = self.config.get("scaling", {})
        self.min_max_range = tuple(
            scaling_config.get("min_max_range", [0, 1])
        )
        self.inplace = scaling_config.get("inplace", False)

        robust_config = scaling_config.get("robust", {})
        self.robust_quantile_range = tuple(
            robust_config.get("quantile_range", [25.0, 75.0])
        )
        self.robust_with_centering = robust_config.get("with_centering", True)
        self.robust_with_scaling = robust_config.get("with_scaling", True)
        self.robust_unit_variance = robust_config.get("unit_variance", False)

        quantile_config = scaling_config.get("quantile", {})
        self.quantile_n_quantiles = int(quantile_config.get("n_quantiles", 1000))
        self.quantile_output_distribution = str(
            quantile_config.get("output_distribution", "uniform")
        )
        self.quantile_subsample = int(quantile_config.get("subsample", 10000))
        self.quantile_random_state = quantile_config.get("random_state", 42)

        power_config = scaling_config.get("power", {})
        self.power_method = str(power_config.get("method", "yeo-johnson"))
        self.power_standardize = bool(power_config.get("standardize", True))

    def load_data(
        self,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load data from file or use provided DataFrame.

        Args:
            file_path: Path to CSV file (optional).
            dataframe: Pandas DataFrame (optional).

        Returns:
            Loaded DataFrame.

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
                    f"Loaded CSV file: {file_path}, "
                    f"shape: {self.data.shape}"
                )
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
        else:
            raise ValueError("Either file_path or dataframe must be provided")

        return self.data

    def get_numeric_columns(self) -> List[str]:
        """Get list of numerical columns in the dataset.

        Returns:
            List of numerical column names.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        numeric_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        logger.info(f"Found {len(numeric_cols)} numerical columns")
        return numeric_cols

    def min_max_scale(
        self,
        columns: Optional[List[str]] = None,
        feature_range: Optional[Tuple[float, float]] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Apply min-max scaling to numerical columns.

        Transforms features to a specified range (default [0, 1]).
        Formula: (x - min) / (max - min) * (max_range - min_range) + min_range

        Args:
            columns: List of column names to scale (None for all numeric).
            feature_range: Tuple of (min, max) for output range (default from config).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with scaled values.

        Raises:
            ValueError: If no data loaded, columns invalid, or range invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()
        feature_range = feature_range or self.min_max_range

        if len(feature_range) != 2:
            raise ValueError("feature_range must be a tuple of (min, max)")
        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range min must be less than max")

        min_range, max_range = feature_range

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for scaling")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping min-max scaling"
                )
                continue

            col_min = result[col].min()
            col_max = result[col].max()
            col_range = col_max - col_min

            if col_range == 0:
                logger.warning(
                    f"Column '{col}' has zero range, setting to {min_range}"
                )
                result[col] = min_range
                self.scaling_params[col] = {
                    "method": "min_max",
                    "min": col_min,
                    "max": col_max,
                    "feature_range": feature_range,
                }
            else:
                result[col] = (
                    (result[col] - col_min) / col_range * (max_range - min_range)
                    + min_range
                )
                self.scaling_params[col] = {
                    "method": "min_max",
                    "min": col_min,
                    "max": col_max,
                    "feature_range": feature_range,
                }

            logger.info(
                f"Min-max scaled '{col}': range [{col_min:.4f}, {col_max:.4f}] "
                f"-> [{min_range}, {max_range}]"
            )

        if not inplace:
            self.data = result

        return result

    def z_score_normalize(
        self,
        columns: Optional[List[str]] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Apply z-score normalization (standardization) to numerical columns.

        Transforms features to have mean=0 and std=1.
        Formula: (x - mean) / std

        Args:
            columns: List of column names to normalize (None for all numeric).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with normalized values.

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for normalization")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping z-score normalization"
                )
                continue

            col_mean = result[col].mean()
            col_std = result[col].std()

            if col_std == 0:
                logger.warning(
                    f"Column '{col}' has zero standard deviation, "
                    f"setting to mean (0)"
                )
                result[col] = 0.0
                self.scaling_params[col] = {
                    "method": "z_score",
                    "mean": col_mean,
                    "std": col_std,
                }
            else:
                result[col] = (result[col] - col_mean) / col_std
                self.scaling_params[col] = {
                    "method": "z_score",
                    "mean": col_mean,
                    "std": col_std,
                }

            logger.info(
                f"Z-score normalized '{col}': mean={col_mean:.4f}, "
                f"std={col_std:.4f} -> mean=0, std=1"
            )

        if not inplace:
            self.data = result

        return result

    def robust_scale(
        self,
        columns: Optional[List[str]] = None,
        quantile_range: Optional[Tuple[float, float]] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Apply robust scaling to numerical columns.

        Robust scaling uses the median and the interquartile range (IQR). It is
        less sensitive to outliers compared to min-max scaling and z-score
        normalization.

        Args:
            columns: List of column names to scale (None for all numeric).
            quantile_range: Quantile range used to calculate IQR as (q_min, q_max).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with robust scaled values.

        Raises:
            ValueError: If no data loaded, columns invalid, or quantile range invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        quantile_range = quantile_range or self.robust_quantile_range
        if len(quantile_range) != 2:
            raise ValueError("quantile_range must be a tuple of (q_min, q_max)")
        if quantile_range[0] >= quantile_range[1]:
            raise ValueError("quantile_range q_min must be less than q_max")
        if quantile_range[0] < 0 or quantile_range[1] > 100:
            raise ValueError("quantile_range values must be between 0 and 100")

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for robust scaling")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping robust scaling"
                )
                continue

            transformer = RobustScaler(
                with_centering=self.robust_with_centering,
                with_scaling=self.robust_with_scaling,
                quantile_range=quantile_range,
                unit_variance=self.robust_unit_variance,
            )
            scaled = transformer.fit_transform(
                result[[col]].to_numpy(dtype=float)
            ).reshape(-1)
            result[col] = scaled

            self.scaling_params[col] = {
                "method": "robust",
                "transformer": transformer,
                "quantile_range": quantile_range,
                "with_centering": self.robust_with_centering,
                "with_scaling": self.robust_with_scaling,
                "unit_variance": self.robust_unit_variance,
            }
            logger.info(
                f"Robust scaled '{col}' using quantile_range={quantile_range}"
            )

        if not inplace:
            self.data = result

        return result

    def quantile_transform(
        self,
        columns: Optional[List[str]] = None,
        n_quantiles: Optional[int] = None,
        output_distribution: Optional[str] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Apply quantile transformation to numerical columns.

        Quantile transformation maps the data to a uniform or normal distribution
        using the empirical cumulative distribution function (CDF). This can
        reduce the impact of outliers and make distributions more Gaussian-like.

        Args:
            columns: List of column names to transform (None for all numeric).
            n_quantiles: Number of quantiles to compute (capped at n_samples).
            output_distribution: "uniform" or "normal".
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with quantile transformed values.

        Raises:
            ValueError: If no data loaded, columns invalid, or parameters invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for quantile transformation")
            return result

        output_distribution = (
            output_distribution or self.quantile_output_distribution
        ).lower()
        if output_distribution not in {"uniform", "normal"}:
            raise ValueError("output_distribution must be 'uniform' or 'normal'")

        n_quantiles_val = int(n_quantiles or self.quantile_n_quantiles)
        if n_quantiles_val < 2:
            raise ValueError("n_quantiles must be at least 2")

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping quantile transform"
                )
                continue

            x = result[[col]].to_numpy(dtype=float)
            n_samples = x.shape[0]
            effective_n_quantiles = min(n_quantiles_val, n_samples)

            transformer = QuantileTransformer(
                n_quantiles=effective_n_quantiles,
                output_distribution=output_distribution,
                subsample=self.quantile_subsample,
                random_state=self.quantile_random_state,
                copy=True,
            )
            transformed = transformer.fit_transform(x).reshape(-1)
            result[col] = transformed

            self.scaling_params[col] = {
                "method": "quantile",
                "transformer": transformer,
                "n_quantiles": effective_n_quantiles,
                "output_distribution": output_distribution,
            }
            logger.info(
                f"Quantile transformed '{col}' using "
                f"output_distribution={output_distribution}, "
                f"n_quantiles={effective_n_quantiles}"
            )

        if not inplace:
            self.data = result

        return result

    def power_transform(
        self,
        columns: Optional[List[str]] = None,
        method: Optional[str] = None,
        standardize: Optional[bool] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Apply power transformation to numerical columns.

        Power transformation stabilizes variance and can make the data more
        Gaussian-like. Supported methods are "yeo-johnson" and "box-cox".
        Note: "box-cox" requires strictly positive values.

        Args:
            columns: List of column names to transform (None for all numeric).
            method: "yeo-johnson" or "box-cox" (default from config).
            standardize: Whether to standardize the output (default from config).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with power transformed values.

        Raises:
            ValueError: If no data loaded, columns invalid, or parameters invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for power transformation")
            return result

        method_val = (method or self.power_method).lower()
        if method_val not in {"yeo-johnson", "box-cox"}:
            raise ValueError("method must be 'yeo-johnson' or 'box-cox'")

        standardize_val = (
            self.power_standardize if standardize is None else bool(standardize)
        )

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping power transform"
                )
                continue

            x = result[[col]].to_numpy(dtype=float)
            if method_val == "box-cox" and np.any(x <= 0):
                raise ValueError(
                    f"Power transform 'box-cox' requires strictly positive values "
                    f"in column '{col}'"
                )

            transformer = PowerTransformer(
                method=method_val, standardize=standardize_val, copy=True
            )
            transformed = transformer.fit_transform(x).reshape(-1)
            result[col] = transformed

            self.scaling_params[col] = {
                "method": "power",
                "transformer": transformer,
                "power_method": method_val,
                "standardize": standardize_val,
            }
            logger.info(
                f"Power transformed '{col}' using method={method_val}, "
                f"standardize={standardize_val}"
            )

        if not inplace:
            self.data = result

        return result

    def inverse_transform(
        self,
        scaled_data: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Inverse transform scaled data back to original scale.

        Args:
            scaled_data: Scaled DataFrame (None uses internal data).
            columns: List of columns to inverse transform (None for all scaled).

        Returns:
            DataFrame with original scale values.

        Raises:
            ValueError: If no scaling parameters or invalid columns.
        """
        if not self.scaling_params:
            raise ValueError("No scaling parameters found. Scale data first.")

        if scaled_data is None:
            if self.data is None:
                raise ValueError("No data available for inverse transform")
            data = self.data.copy()
        else:
            data = scaled_data.copy()

        if columns is None:
            columns = list(self.scaling_params.keys())

        for col in columns:
            if col not in self.scaling_params:
                logger.warning(
                    f"Column '{col}' not in scaling parameters, skipping"
                )
                continue
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

            params = self.scaling_params[col]
            method = params["method"]

            if method == "min_max":
                col_min = params["min"]
                col_max = params["max"]
                min_range, max_range = params["feature_range"]
                col_range = col_max - col_min
                range_size = max_range - min_range

                if col_range == 0:
                    data[col] = col_min
                else:
                    data[col] = (
                        (data[col] - min_range) / range_size * col_range + col_min
                    )
            elif method == "z_score":
                col_mean = params["mean"]
                col_std = params["std"]

                if col_std == 0:
                    data[col] = col_mean
                else:
                    data[col] = data[col] * col_std + col_mean
            elif method in {"robust", "quantile", "power"}:
                transformer = params.get("transformer")
                if transformer is None:
                    raise ValueError(
                        f"Missing transformer for inverse transform of '{col}'"
                    )
                inverse = transformer.inverse_transform(
                    data[[col]].to_numpy(dtype=float)
                ).reshape(-1)
                data[col] = inverse

            logger.info(f"Inverse transformed column '{col}'")

        return data

    def get_scaling_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of scaling parameters used.

        Returns:
            Dictionary mapping column names to scaling parameters.
        """
        return self.scaling_params.copy()

    def save_scaled_data(self, output_path: str) -> None:
        """Save scaled data to CSV file.

        Args:
            output_path: Path to output CSV file.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved scaled data to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Scaling Tool")
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
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["min_max", "z_score", "robust", "quantile", "power", "both"],
        default="both",
        help="Scaling method to apply",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Specific columns to scale (optional)",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Feature range for min-max scaling (default: 0 1)",
    )

    args = parser.parse_args()

    scaler = FeatureScaler(config_path=args.config)

    try:
        scaler.load_data(file_path=args.input)

        print("\n=== Numerical Columns ===")
        numeric_cols = scaler.get_numeric_columns()
        print(f"Found {len(numeric_cols)} numerical columns: {numeric_cols}")

        if args.method in ["min_max", "both"]:
            print("\n=== Applying Min-Max Scaling ===")
            feature_range = tuple(args.range) if args.range else None
            scaler.min_max_scale(columns=args.columns, feature_range=feature_range)

        if args.method in ["z_score", "both"]:
            print("\n=== Applying Z-Score Normalization ===")
            scaler.z_score_normalize(columns=args.columns)

        if args.method == "robust":
            print("\n=== Applying Robust Scaling ===")
            scaler.robust_scale(columns=args.columns)

        if args.method == "quantile":
            print("\n=== Applying Quantile Transformation ===")
            scaler.quantile_transform(columns=args.columns)

        if args.method == "power":
            print("\n=== Applying Power Transformation ===")
            scaler.power_transform(columns=args.columns)

        print("\n=== Scaling Summary ===")
        summary = scaler.get_scaling_summary()
        for col, params in summary.items():
            method = params["method"]
            if method == "min_max":
                print(
                    f"  {col}: {method} "
                    f"(range: [{params['min']:.4f}, {params['max']:.4f}] "
                    f"-> {params['feature_range']})"
                )
            elif method == "z_score":
                print(
                    f"  {col}: {method} "
                    f"(mean: {params['mean']:.4f}, std: {params['std']:.4f})"
                )
            elif method == "robust":
                print(
                    f"  {col}: {method} "
                    f"(quantile_range: {params['quantile_range']})"
                )
            elif method == "quantile":
                print(
                    f"  {col}: {method} "
                    f"(output_distribution: {params['output_distribution']}, "
                    f"n_quantiles: {params['n_quantiles']})"
                )
            elif method == "power":
                print(
                    f"  {col}: {method} "
                    f"(method: {params['power_method']}, "
                    f"standardize: {params['standardize']})"
                )

        if args.output:
            scaler.save_scaled_data(args.output)
            print(f"\nScaled data saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
