"""Outlier Detection and Handling Tool.

This module provides functionality to detect and handle outliers using
IQR method, Z-score, and isolation forest techniques.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class OutlierDetector:
    """Detects and handles outliers using multiple methods."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize OutlierDetector with configuration.

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
        self.outlier_masks: Dict[str, pd.Series] = {}

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
        detection_config = self.config.get("outlier_detection", {})
        self.iqr_multiplier = detection_config.get("iqr_multiplier", 1.5)
        self.z_score_threshold = detection_config.get("z_score_threshold", 3.0)
        self.isolation_contamination = detection_config.get(
            "isolation_contamination", 0.1
        )
        self.isolation_random_state = detection_config.get(
            "isolation_random_state", 42
        )

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

    def detect_iqr(
        self,
        columns: Optional[List[str]] = None,
        multiplier: Optional[float] = None,
    ) -> pd.Series:
        """Detect outliers using IQR (Interquartile Range) method.

        Args:
            columns: List of column names to analyze (None for all numeric).
            multiplier: IQR multiplier (default from config).

        Returns:
            Boolean Series indicating outliers (True = outlier).

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        multiplier = multiplier or self.iqr_multiplier

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for IQR detection")
            return pd.Series(dtype=bool)

        outlier_mask = pd.Series(False, index=self.data.index)

        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping IQR detection"
                )
                continue

            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            col_outliers = (self.data[col] < lower_bound) | (
                self.data[col] > upper_bound
            )
            outlier_mask |= col_outliers

            outlier_count = col_outliers.sum()
            logger.info(
                f"IQR detected {outlier_count} outliers in '{col}' "
                f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
            )

        self.outlier_masks["iqr"] = outlier_mask
        total_outliers = outlier_mask.sum()
        logger.info(f"IQR method detected {total_outliers} total outliers")
        return outlier_mask

    def detect_zscore(
        self,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> pd.Series:
        """Detect outliers using Z-score method.

        Args:
            columns: List of column names to analyze (None for all numeric).
            threshold: Z-score threshold (default from config).

        Returns:
            Boolean Series indicating outliers (True = outlier).

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        threshold = threshold or self.z_score_threshold

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for Z-score detection")
            return pd.Series(dtype=bool)

        outlier_mask = pd.Series(False, index=self.data.index)

        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping Z-score detection"
                )
                continue

            z_scores = np.abs(
                (self.data[col] - self.data[col].mean()) / self.data[col].std()
            )
            col_outliers = z_scores > threshold
            outlier_mask |= col_outliers

            outlier_count = col_outliers.sum()
            logger.info(
                f"Z-score detected {outlier_count} outliers in '{col}' "
                f"(threshold: {threshold})"
            )

        self.outlier_masks["zscore"] = outlier_mask
        total_outliers = outlier_mask.sum()
        logger.info(f"Z-score method detected {total_outliers} total outliers")
        return outlier_mask

    def detect_isolation_forest(
        self,
        columns: Optional[List[str]] = None,
        contamination: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> pd.Series:
        """Detect outliers using Isolation Forest method.

        Args:
            columns: List of column names to analyze (None for all numeric).
            contamination: Expected proportion of outliers (default from config).
            random_state: Random seed (default from config).

        Returns:
            Boolean Series indicating outliers (True = outlier).

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        contamination = contamination or self.isolation_contamination
        random_state = random_state or self.isolation_random_state

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for Isolation Forest")
            return pd.Series(dtype=bool)

        numeric_data = self.data[columns].dropna()

        if len(numeric_data) == 0:
            logger.warning("No valid data for Isolation Forest")
            return pd.Series(False, index=self.data.index)

        iso_forest = IsolationForest(
            contamination=contamination, random_state=random_state
        )
        predictions = iso_forest.fit_predict(numeric_data)

        outlier_mask = pd.Series(False, index=self.data.index)
        outlier_mask.loc[numeric_data.index] = predictions == -1

        total_outliers = outlier_mask.sum()
        logger.info(
            f"Isolation Forest detected {total_outliers} outliers "
            f"(contamination: {contamination})"
        )

        self.outlier_masks["isolation_forest"] = outlier_mask
        return outlier_mask

    def get_outlier_summary(
        self, outlier_mask: Optional[pd.Series] = None
    ) -> Dict[str, any]:
        """Get summary of detected outliers.

        Args:
            outlier_mask: Boolean Series indicating outliers (None uses all methods).

        Returns:
            Dictionary with outlier summary statistics.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if outlier_mask is None:
            if not self.outlier_masks:
                return {"message": "No outliers detected yet"}
            outlier_mask = pd.Series(False, index=self.data.index)
            for mask in self.outlier_masks.values():
                outlier_mask |= mask

        total_samples = len(self.data)
        outlier_count = outlier_mask.sum()
        outlier_percentage = (outlier_count / total_samples) * 100

        summary = {
            "total_samples": int(total_samples),
            "outlier_count": int(outlier_count),
            "outlier_percentage": float(outlier_percentage),
            "normal_count": int(total_samples - outlier_count),
        }

        if outlier_count > 0:
            numeric_cols = self.get_numeric_columns()
            outlier_data = self.data.loc[outlier_mask, numeric_cols]
            summary["outlier_statistics"] = {
                col: {
                    "mean": float(outlier_data[col].mean()),
                    "std": float(outlier_data[col].std()),
                    "min": float(outlier_data[col].min()),
                    "max": float(outlier_data[col].max()),
                }
                for col in numeric_cols
                if col in outlier_data.columns
            }

        return summary

    def remove_outliers(
        self, outlier_mask: Optional[pd.Series] = None, inplace: bool = False
    ) -> pd.DataFrame:
        """Remove outliers from dataset.

        Args:
            outlier_mask: Boolean Series indicating outliers (None uses all methods).
            inplace: Whether to modify in place.

        Returns:
            DataFrame with outliers removed.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if outlier_mask is None:
            if not self.outlier_masks:
                raise ValueError("No outliers detected. Run detection methods first.")
            outlier_mask = pd.Series(False, index=self.data.index)
            for mask in self.outlier_masks.values():
                outlier_mask |= mask

        result = self.data if inplace else self.data.copy()
        result = result[~outlier_mask]

        removed_count = outlier_mask.sum()
        logger.info(f"Removed {removed_count} outliers from dataset")

        if not inplace:
            return result

    def cap_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Cap outliers to boundary values.

        Args:
            columns: List of columns to cap (None for all numeric).
            method: Detection method to use for boundaries (iqr or zscore).
            inplace: Whether to modify in place.

        Returns:
            DataFrame with outliers capped.

        Raises:
            ValueError: If no data loaded or invalid method.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns is None:
            columns = self.get_numeric_columns()

        if not columns:
            logger.warning("No numerical columns found for capping")
            return self.data if inplace else self.data.copy()

        result = self.data if inplace else self.data.copy()

        for col in columns:
            if col not in result.columns:
                continue
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue

            if method == "iqr":
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
            elif method == "zscore":
                mean = result[col].mean()
                std = result[col].std()
                lower_bound = mean - self.z_score_threshold * std
                upper_bound = mean + self.z_score_threshold * std
            else:
                raise ValueError(f"Invalid method: {method}. Use 'iqr' or 'zscore'")

            capped_count = (
                (result[col] < lower_bound) | (result[col] > upper_bound)
            ).sum()
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

            logger.info(
                f"Capped {capped_count} outliers in '{col}' "
                f"to bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
            )

        if not inplace:
            return result


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Outlier Detection and Handling Tool")
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
        "--method",
        type=str,
        choices=["iqr", "zscore", "isolation_forest", "all"],
        default="all",
        help="Outlier detection method",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Specific columns to analyze (optional)",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["detect", "remove", "cap"],
        default="detect",
        help="Action to take on outliers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )

    args = parser.parse_args()

    detector = OutlierDetector(config_path=args.config)

    try:
        detector.load_data(file_path=args.input)

        print("\n=== Numerical Columns ===")
        numeric_cols = detector.get_numeric_columns()
        print(f"Found {len(numeric_cols)} numerical columns: {numeric_cols}")

        print("\n=== Outlier Detection ===")
        if args.method in ["iqr", "all"]:
            iqr_mask = detector.detect_iqr(columns=args.columns)
            print(f"IQR method: {iqr_mask.sum()} outliers detected")

        if args.method in ["zscore", "all"]:
            zscore_mask = detector.detect_zscore(columns=args.columns)
            print(f"Z-score method: {zscore_mask.sum()} outliers detected")

        if args.method in ["isolation_forest", "all"]:
            iso_mask = detector.detect_isolation_forest(columns=args.columns)
            print(f"Isolation Forest: {iso_mask.sum()} outliers detected")

        print("\n=== Outlier Summary ===")
        summary = detector.get_outlier_summary()
        print(f"Total samples: {summary['total_samples']}")
        print(f"Outliers: {summary['outlier_count']} ({summary['outlier_percentage']:.2f}%)")
        print(f"Normal: {summary['normal_count']}")

        if args.action == "remove":
            print("\n=== Removing Outliers ===")
            cleaned_data = detector.remove_outliers()
            print(f"Cleaned dataset shape: {cleaned_data.shape}")
            if args.output:
                cleaned_data.to_csv(args.output, index=False)
                print(f"Saved to: {args.output}")
        elif args.action == "cap":
            print("\n=== Capping Outliers ===")
            capped_data = detector.cap_outliers(columns=args.columns)
            print(f"Capped dataset shape: {capped_data.shape}")
            if args.output:
                capped_data.to_csv(args.output, index=False)
                print(f"Saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
