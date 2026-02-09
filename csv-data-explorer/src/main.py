"""CSV Dataset Explorer.

This module provides functionality to load and explore CSV datasets
with basic statistics, data type analysis, and missing value detection.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CSVDataExplorer:
    """Explores CSV datasets with statistics and analysis."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize CSVDataExplorer with configuration.

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
        explorer_config = self.config.get("csv_explorer", {})
        self.separator = explorer_config.get("separator", ",")
        self.encoding = explorer_config.get("encoding", "utf-8")
        self.max_rows_preview = explorer_config.get("max_rows_preview", 10)

    def load_csv(
        self,
        file_path: str,
        separator: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load CSV file into pandas DataFrame.

        Args:
            file_path: Path to CSV file.
            separator: Column separator (default from config).
            encoding: File encoding (default from config).

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If file doesn't exist.
            pd.errors.EmptyDataError: If file is empty.
            pd.errors.ParserError: If file cannot be parsed.
        """
        separator = separator or self.separator
        encoding = encoding or self.encoding

        try:
            logger.info(f"Loading CSV file: {file_path}")
            self.data = pd.read_csv(
                file_path, sep=separator, encoding=encoding
            )
            logger.info(
                f"Successfully loaded {len(self.data)} rows and "
                f"{len(self.data.columns)} columns"
            )
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {file_path}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise

    def get_basic_info(self) -> Dict[str, any]:
        """Get basic information about the dataset.

        Returns:
            Dictionary with basic dataset information.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        info = {
            "shape": self.data.shape,
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "column_names": list(self.data.columns),
            "memory_usage_bytes": self.data.memory_usage(deep=True).sum(),
        }

        logger.info(f"Dataset shape: {info['shape']}")
        logger.info(f"Memory usage: {info['memory_usage_bytes']} bytes")

        return info

    def get_data_types(self) -> Dict[str, str]:
        """Get data types for each column.

        Returns:
            Dictionary mapping column names to data types.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        dtypes = self.data.dtypes.to_dict()
        dtypes_str = {col: str(dtype) for col, dtype in dtypes.items()}

        logger.info("Data types:")
        for col, dtype in dtypes_str.items():
            logger.info(f"  {col}: {dtype}")

        return dtypes_str

    def get_basic_statistics(self) -> pd.DataFrame:
        """Get basic statistical summary for numerical columns.

        Returns:
            DataFrame with statistical summary.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        numeric_cols = self.data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            logger.warning("No numerical columns found for statistics")
            return pd.DataFrame()

        stats = self.data[numeric_cols].describe()
        logger.info(f"Statistical summary for {len(numeric_cols)} numerical columns")

        return stats

    def get_missing_value_analysis(self) -> Dict[str, any]:
        """Analyze missing values in the dataset.

        Returns:
            Dictionary with missing value analysis.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        missing_count = self.data.isnull().sum()
        missing_percent = (missing_count / len(self.data)) * 100

        analysis = {
            "total_missing": int(missing_count.sum()),
            "total_rows": len(self.data),
            "missing_percentage": float(
                (missing_count.sum() / (len(self.data) * len(self.data.columns)))
                * 100
            ),
            "columns_with_missing": {},
            "columns_without_missing": [],
        }

        for col in self.data.columns:
            count = int(missing_count[col])
            percent = float(missing_percent[col])
            if count > 0:
                analysis["columns_with_missing"][col] = {
                    "count": count,
                    "percentage": percent,
                }
            else:
                analysis["columns_without_missing"].append(col)

        logger.info(f"Total missing values: {analysis['total_missing']}")
        logger.info(
            f"Missing percentage: {analysis['missing_percentage']:.2f}%"
        )
        logger.info(
            f"Columns with missing values: "
            f"{len(analysis['columns_with_missing'])}"
        )

        return analysis

    def get_categorical_summary(self) -> Dict[str, any]:
        """Get summary for categorical columns.

        Returns:
            Dictionary with categorical column summaries.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        categorical_cols = self.data.select_dtypes(
            include=["object", "category"]
        ).columns

        summary = {}
        for col in categorical_cols:
            summary[col] = {
                "unique_count": int(self.data[col].nunique()),
                "most_frequent": self.data[col].mode().iloc[0]
                if len(self.data[col].mode()) > 0
                else None,
                "most_frequent_count": int(
                    self.data[col].value_counts().iloc[0]
                )
                if len(self.data[col].value_counts()) > 0
                else 0,
            }

        logger.info(f"Categorical summary for {len(categorical_cols)} columns")
        return summary

    def generate_report(self) -> str:
        """Generate comprehensive exploration report.

        Returns:
            Formatted report string.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CSV Dataset Exploration Report")
        report_lines.append("=" * 80)
        report_lines.append("")

        info = self.get_basic_info()
        report_lines.append("BASIC INFORMATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Shape: {info['shape']}")
        report_lines.append(f"Rows: {info['rows']}")
        report_lines.append(f"Columns: {info['columns']}")
        report_lines.append(
            f"Memory Usage: {info['memory_usage_bytes']:,} bytes "
            f"({info['memory_usage_bytes'] / 1024 / 1024:.2f} MB)"
        )
        report_lines.append("")

        dtypes = self.get_data_types()
        report_lines.append("DATA TYPES")
        report_lines.append("-" * 80)
        for col, dtype in dtypes.items():
            report_lines.append(f"  {col}: {dtype}")
        report_lines.append("")

        missing_analysis = self.get_missing_value_analysis()
        report_lines.append("MISSING VALUE ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Missing Values: {missing_analysis['total_missing']}")
        report_lines.append(
            f"Missing Percentage: {missing_analysis['missing_percentage']:.2f}%"
        )
        report_lines.append(
            f"Columns with Missing Values: "
            f"{len(missing_analysis['columns_with_missing'])}"
        )
        if missing_analysis["columns_with_missing"]:
            report_lines.append("")
            report_lines.append("Missing Values by Column:")
            for col, info in missing_analysis["columns_with_missing"].items():
                report_lines.append(
                    f"  {col}: {info['count']} ({info['percentage']:.2f}%)"
                )
        report_lines.append("")

        stats = self.get_basic_statistics()
        if not stats.empty:
            report_lines.append("BASIC STATISTICS (Numerical Columns)")
            report_lines.append("-" * 80)
            report_lines.append(stats.to_string())
            report_lines.append("")

        categorical = self.get_categorical_summary()
        if categorical:
            report_lines.append("CATEGORICAL SUMMARY")
            report_lines.append("-" * 80)
            for col, info in categorical.items():
                report_lines.append(f"  {col}:")
                report_lines.append(f"    Unique values: {info['unique_count']}")
                if info["most_frequent"] is not None:
                    report_lines.append(
                        f"    Most frequent: {info['most_frequent']} "
                        f"({info['most_frequent_count']} times)"
                    )
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)
        logger.info("Generated exploration report")
        return report

    def preview_data(self, n_rows: Optional[int] = None) -> pd.DataFrame:
        """Preview first n rows of the dataset.

        Args:
            n_rows: Number of rows to preview (default from config).

        Returns:
            DataFrame with preview rows.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        n_rows = n_rows or self.max_rows_preview
        preview = self.data.head(n_rows)
        logger.info(f"Previewing first {len(preview)} rows")
        return preview


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CSV Dataset Explorer with Statistics and Analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to CSV file to explore",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate and print comprehensive report",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=None,
        help="Number of rows to preview",
    )

    args = parser.parse_args()

    explorer = CSVDataExplorer(config_path=args.config)

    try:
        explorer.load_csv(args.file)

        if args.report:
            report = explorer.generate_report()
            print(report)
        else:
            info = explorer.get_basic_info()
            print(f"\nDataset Shape: {info['shape']}")
            print(f"Rows: {info['rows']}, Columns: {info['columns']}")

            print("\nData Types:")
            dtypes = explorer.get_data_types()
            for col, dtype in dtypes.items():
                print(f"  {col}: {dtype}")

            print("\nMissing Value Analysis:")
            missing = explorer.get_missing_value_analysis()
            print(f"  Total Missing: {missing['total_missing']}")
            print(f"  Missing Percentage: {missing['missing_percentage']:.2f}%")

            if args.preview:
                print(f"\nPreview (first {args.preview} rows):")
                print(explorer.preview_data(args.preview))
    except Exception as e:
        logger.error(f"Error exploring dataset: {e}")
        raise


if __name__ == "__main__":
    main()
