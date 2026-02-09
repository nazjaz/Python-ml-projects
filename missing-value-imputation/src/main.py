"""Missing Value Imputation Tool.

This module provides functionality to clean datasets by handling missing
values using mean, median, and mode imputation methods.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MissingValueImputer:
    """Handles missing value imputation using various strategies."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize MissingValueImputer with configuration.

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
        self.imputation_strategy: Dict[str, str] = {}

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
        imputation_config = self.config.get("imputation", {})
        self.default_numeric_strategy = imputation_config.get(
            "default_numeric_strategy", "mean"
        )
        self.default_categorical_strategy = imputation_config.get(
            "default_categorical_strategy", "mode"
        )
        self.inplace = imputation_config.get("inplace", False)

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

    def analyze_missing_values(self) -> Dict[str, any]:
        """Analyze missing values in the dataset.

        Returns:
            Dictionary with missing value analysis.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        missing_count = self.data.isnull().sum()
        missing_percent = (missing_count / len(self.data)) * 100

        analysis = {
            "total_missing": int(missing_count.sum()),
            "total_cells": len(self.data) * len(self.data.columns),
            "missing_percentage": float(
                (missing_count.sum() / (len(self.data) * len(self.data.columns)))
                * 100
            ),
            "columns_with_missing": {},
            "columns_without_missing": [],
            "numeric_columns_with_missing": [],
            "categorical_columns_with_missing": [],
        }

        for col in self.data.columns:
            count = int(missing_count[col])
            percent = float(missing_percent[col])
            if count > 0:
                analysis["columns_with_missing"][col] = {
                    "count": count,
                    "percentage": percent,
                    "dtype": str(self.data[col].dtype),
                }
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    analysis["numeric_columns_with_missing"].append(col)
                else:
                    analysis["categorical_columns_with_missing"].append(col)
            else:
                analysis["columns_without_missing"].append(col)

        logger.info(
            f"Missing values analysis: {analysis['total_missing']} "
            f"missing ({analysis['missing_percentage']:.2f}%)"
        )

        return analysis

    def impute_mean(
        self, columns: Optional[List[str]] = None, inplace: Optional[bool] = None
    ) -> pd.DataFrame:
        """Impute missing values using mean for numerical columns.

        Args:
            columns: List of column names to impute (None for all numeric).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with imputed values.

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        if columns is None:
            numeric_cols = result.select_dtypes(include=["number"]).columns
            columns = [col for col in numeric_cols if result[col].isnull().any()]

        if not columns:
            logger.warning("No columns with missing values to impute")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping mean imputation"
                )
                continue

            mean_value = result[col].mean()
            missing_count = result[col].isnull().sum()
            result[col].fillna(mean_value, inplace=True)
            self.imputation_strategy[col] = f"mean ({mean_value:.4f})"

            logger.info(
                f"Imputed {missing_count} missing values in '{col}' "
                f"with mean: {mean_value:.4f}"
            )

        if not inplace:
            self.data = result

        return result

    def impute_median(
        self, columns: Optional[List[str]] = None, inplace: Optional[bool] = None
    ) -> pd.DataFrame:
        """Impute missing values using median for numerical columns.

        Args:
            columns: List of column names to impute (None for all numeric).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with imputed values.

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        if columns is None:
            numeric_cols = result.select_dtypes(include=["number"]).columns
            columns = [col for col in numeric_cols if result[col].isnull().any()]

        if not columns:
            logger.warning("No columns with missing values to impute")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not numeric, skipping median imputation"
                )
                continue

            median_value = result[col].median()
            missing_count = result[col].isnull().sum()
            result[col].fillna(median_value, inplace=True)
            self.imputation_strategy[col] = f"median ({median_value:.4f})"

            logger.info(
                f"Imputed {missing_count} missing values in '{col}' "
                f"with median: {median_value:.4f}"
            )

        if not inplace:
            self.data = result

        return result

    def impute_mode(
        self, columns: Optional[List[str]] = None, inplace: Optional[bool] = None
    ) -> pd.DataFrame:
        """Impute missing values using mode for categorical columns.

        Args:
            columns: List of column names to impute (None for all categorical).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with imputed values.

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        if columns is None:
            categorical_cols = result.select_dtypes(
                include=["object", "category"]
            ).columns
            columns = [
                col for col in categorical_cols if result[col].isnull().any()
            ]

        if not columns:
            logger.warning("No columns with missing values to impute")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")

            mode_values = result[col].mode()
            if len(mode_values) == 0:
                logger.warning(
                    f"Column '{col}' has no mode, cannot impute with mode"
                )
                continue

            mode_value = mode_values.iloc[0]
            missing_count = result[col].isnull().sum()
            result[col].fillna(mode_value, inplace=True)
            self.imputation_strategy[col] = f"mode ({mode_value})"

            logger.info(
                f"Imputed {missing_count} missing values in '{col}' "
                f"with mode: {mode_value}"
            )

        if not inplace:
            self.data = result

        return result

    def impute_all(
        self,
        numeric_strategy: Optional[str] = None,
        categorical_strategy: Optional[str] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Impute all missing values using specified strategies.

        Args:
            numeric_strategy: Strategy for numeric columns (mean, median).
            categorical_strategy: Strategy for categorical columns (mode).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with all missing values imputed.

        Raises:
            ValueError: If no data loaded or invalid strategy.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        numeric_strategy = (
            numeric_strategy or self.default_numeric_strategy
        ).lower()
        categorical_strategy = (
            categorical_strategy or self.default_categorical_strategy
        ).lower()
        inplace = inplace if inplace is not None else self.inplace

        result = self.data if inplace else self.data.copy()

        numeric_cols = result.select_dtypes(include=["number"]).columns
        categorical_cols = result.select_dtypes(
            include=["object", "category"]
        ).columns

        numeric_missing = [
            col for col in numeric_cols if result[col].isnull().any()
        ]
        categorical_missing = [
            col for col in categorical_cols if result[col].isnull().any()
        ]

        if numeric_strategy == "mean":
            if numeric_missing:
                self.impute_mean(columns=numeric_missing, inplace=True)
        elif numeric_strategy == "median":
            if numeric_missing:
                self.impute_median(columns=numeric_missing, inplace=True)
        else:
            raise ValueError(
                f"Invalid numeric strategy: {numeric_strategy}. "
                f"Use 'mean' or 'median'"
            )

        if categorical_strategy == "mode":
            if categorical_missing:
                self.impute_mode(columns=categorical_missing, inplace=True)
        else:
            raise ValueError(
                f"Invalid categorical strategy: {categorical_strategy}. "
                f"Use 'mode'"
            )

        if not inplace:
            self.data = result

        logger.info("Completed imputation for all missing values")
        return result

    def get_imputation_summary(self) -> Dict[str, str]:
        """Get summary of imputation strategies used.

        Returns:
            Dictionary mapping column names to imputation strategies.
        """
        return self.imputation_strategy.copy()

    def save_cleaned_data(self, output_path: str) -> None:
        """Save cleaned data to CSV file.

        Args:
            output_path: Path to output CSV file.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Missing Value Imputation Tool"
    )
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
        "--strategy",
        type=str,
        choices=["mean", "median", "mode", "auto"],
        default="auto",
        help="Imputation strategy (auto uses config defaults)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Specific columns to impute (optional)",
    )

    args = parser.parse_args()

    imputer = MissingValueImputer(config_path=args.config)

    try:
        imputer.load_data(file_path=args.input)

        print("\n=== Missing Value Analysis ===")
        analysis = imputer.analyze_missing_values()
        print(f"Total missing values: {analysis['total_missing']}")
        print(
            f"Missing percentage: {analysis['missing_percentage']:.2f}%"
        )
        print(
            f"Columns with missing values: "
            f"{len(analysis['columns_with_missing'])}"
        )

        if analysis["total_missing"] > 0:
            print("\n=== Performing Imputation ===")
            if args.strategy == "auto":
                imputer.impute_all()
            elif args.strategy == "mean":
                imputer.impute_mean(columns=args.columns)
            elif args.strategy == "median":
                imputer.impute_median(columns=args.columns)
            elif args.strategy == "mode":
                imputer.impute_mode(columns=args.columns)

            print("\n=== Imputation Summary ===")
            summary = imputer.get_imputation_summary()
            for col, strategy in summary.items():
                print(f"  {col}: {strategy}")

            if args.output:
                imputer.save_cleaned_data(args.output)
                print(f"\nCleaned data saved to: {args.output}")
        else:
            print("\nNo missing values found. Data is clean!")

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
