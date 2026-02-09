"""Categorical Variable Encoding Tool.

This module provides functionality to encode categorical variables using
one-hot encoding and label encoding with comparison analysis.
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


class CategoricalEncoder:
    """Handles categorical variable encoding using one-hot and label encoding."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize CategoricalEncoder with configuration.

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
        self.encoding_mappings: Dict[str, Dict] = {}
        self.one_hot_columns: List[str] = []
        self.label_encodings: Dict[str, Dict[str, int]] = {}

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
        encoding_config = self.config.get("encoding", {})
        self.drop_first = encoding_config.get("drop_first", False)
        self.prefix = encoding_config.get("prefix", None)
        self.prefix_sep = encoding_config.get("prefix_sep", "_")
        self.inplace = encoding_config.get("inplace", False)

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

    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns in the dataset.

        Returns:
            List of categorical column names.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        categorical_cols = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        logger.info(f"Found {len(categorical_cols)} categorical columns")
        return categorical_cols

    def one_hot_encode(
        self,
        columns: Optional[List[str]] = None,
        drop_first: Optional[bool] = None,
        prefix: Optional[str] = None,
        prefix_sep: Optional[str] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Apply one-hot encoding to categorical columns.

        Creates binary columns for each category value.

        Args:
            columns: List of column names to encode (None for all categorical).
            drop_first: Whether to drop first category (default from config).
            prefix: Prefix for new column names (default from config).
            prefix_sep: Separator for prefix (default from config).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with one-hot encoded columns.

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()
        drop_first = drop_first if drop_first is not None else self.drop_first
        prefix = prefix if prefix is not None else self.prefix
        prefix_sep = prefix_sep if prefix_sep is not None else self.prefix_sep

        if columns is None:
            columns = self.get_categorical_columns()

        if not columns:
            logger.warning("No categorical columns found for encoding")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_categorical_dtype(
                result[col]
            ) and not pd.api.types.is_object_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not categorical, skipping one-hot encoding"
                )
                continue

            unique_values = result[col].unique()
            encoded = pd.get_dummies(
                result[col],
                prefix=prefix or col,
                prefix_sep=prefix_sep,
                drop_first=drop_first,
                dtype=int,
            )

            result = pd.concat([result.drop(columns=[col]), encoded], axis=1)
            self.one_hot_columns.extend(encoded.columns.tolist())
            self.encoding_mappings[col] = {
                "method": "one_hot",
                "unique_values": unique_values.tolist(),
                "encoded_columns": encoded.columns.tolist(),
                "drop_first": drop_first,
            }

            logger.info(
                f"One-hot encoded '{col}': {len(unique_values)} categories -> "
                f"{len(encoded.columns)} columns"
            )

        if not inplace:
            self.data = result

        return result

    def label_encode(
        self,
        columns: Optional[List[str]] = None,
        inplace: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Apply label encoding to categorical columns.

        Maps each category to a unique integer.

        Args:
            columns: List of column names to encode (None for all categorical).
            inplace: Whether to modify in place (default from config).

        Returns:
            DataFrame with label encoded columns.

        Raises:
            ValueError: If no data loaded or columns invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        inplace = inplace if inplace is not None else self.inplace
        result = self.data if inplace else self.data.copy()

        if columns is None:
            columns = self.get_categorical_columns()

        if not columns:
            logger.warning("No categorical columns found for encoding")
            return result

        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_categorical_dtype(
                result[col]
            ) and not pd.api.types.is_object_dtype(result[col]):
                logger.warning(
                    f"Column '{col}' is not categorical, skipping label encoding"
                )
                continue

            unique_values = sorted(result[col].dropna().unique())
            label_mapping = {val: idx for idx, val in enumerate(unique_values)}

            result[col] = result[col].map(label_mapping)
            result[col] = result[col].astype("Int64")

            self.label_encodings[col] = label_mapping
            self.encoding_mappings[col] = {
                "method": "label",
                "unique_values": unique_values,
                "label_mapping": label_mapping,
            }

            logger.info(
                f"Label encoded '{col}': {len(unique_values)} categories "
                f"mapped to integers 0-{len(unique_values)-1}"
            )

        if not inplace:
            self.data = result

        return result

    def inverse_label_encode(
        self,
        encoded_data: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Inverse transform label encoded columns back to original categories.

        Args:
            encoded_data: Encoded DataFrame (None uses internal data).
            columns: List of columns to inverse transform (None for all encoded).

        Returns:
            DataFrame with original categorical values.

        Raises:
            ValueError: If no label encodings or invalid columns.
        """
        if not self.label_encodings:
            raise ValueError("No label encodings found. Encode data first.")

        if encoded_data is None:
            if self.data is None:
                raise ValueError("No data available for inverse transform")
            data = self.data.copy()
        else:
            data = encoded_data.copy()

        if columns is None:
            columns = list(self.label_encodings.keys())

        for col in columns:
            if col not in self.label_encodings:
                logger.warning(
                    f"Column '{col}' not in label encodings, skipping"
                )
                continue
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

            reverse_mapping = {
                v: k for k, v in self.label_encodings[col].items()
            }
            data[col] = data[col].map(reverse_mapping)

            logger.info(f"Inverse transformed column '{col}'")

        return data

    def compare_encodings(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """Compare one-hot and label encoding for specified columns.

        Args:
            columns: List of columns to compare (None for all categorical).

        Returns:
            Dictionary with comparison analysis.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if columns is None:
            columns = self.get_categorical_columns()

        if not columns:
            return {"message": "No categorical columns found"}

        comparison = {}

        for col in columns:
            if col not in self.data.columns:
                continue
            if not pd.api.types.is_categorical_dtype(
                self.data[col]
            ) and not pd.api.types.is_object_dtype(self.data[col]):
                continue

            unique_count = self.data[col].nunique()
            total_count = len(self.data[col])

            col_comparison = {
                "column": col,
                "unique_values": int(unique_count),
                "total_values": int(total_count),
                "one_hot": {
                    "columns_created": unique_count if not self.drop_first else unique_count - 1,
                    "sparsity": 1.0 - (1.0 / unique_count),
                    "memory_increase": "High",
                    "use_case": "Nominal data, no ordinal relationship",
                },
                "label": {
                    "columns_created": 1,
                    "sparsity": 0.0,
                    "memory_increase": "Low",
                    "use_case": "Ordinal data, many categories",
                },
            }

            if unique_count > 10:
                col_comparison["recommendation"] = "label"
                col_comparison["reason"] = "Too many categories for one-hot"
            elif unique_count <= 3:
                col_comparison["recommendation"] = "one_hot"
                col_comparison["reason"] = "Few categories, one-hot preferred"
            else:
                col_comparison["recommendation"] = "depends"
                col_comparison["reason"] = "Consider data type and model"

            comparison[col] = col_comparison

        logger.info(f"Generated comparison for {len(comparison)} columns")
        return comparison

    def get_encoding_summary(self) -> Dict[str, Dict]:
        """Get summary of encoding operations performed.

        Returns:
            Dictionary mapping column names to encoding information.
        """
        return self.encoding_mappings.copy()

    def save_encoded_data(self, output_path: str) -> None:
        """Save encoded data to CSV file.

        Args:
            output_path: Path to output CSV file.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.data.to_csv(output_path, index=False)
        logger.info(f"Saved encoded data to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Categorical Variable Encoding Tool"
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
        "--method",
        type=str,
        choices=["one_hot", "label", "both", "compare"],
        default="compare",
        help="Encoding method to apply",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Specific columns to encode (optional)",
    )

    args = parser.parse_args()

    encoder = CategoricalEncoder(config_path=args.config)

    try:
        encoder.load_data(file_path=args.input)

        print("\n=== Categorical Columns ===")
        categorical_cols = encoder.get_categorical_columns()
        print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")

        if args.method == "compare":
            print("\n=== Encoding Comparison ===")
            comparison = encoder.compare_encodings(columns=args.columns)
            for col, info in comparison.items():
                if isinstance(info, dict) and "column" in info:
                    print(f"\nColumn: {info['column']}")
                    print(f"  Unique values: {info['unique_values']}")
                    print(f"  One-hot: {info['one_hot']['columns_created']} columns")
                    print(f"  Label: {info['label']['columns_created']} column")
                    print(f"  Recommendation: {info.get('recommendation', 'N/A')}")
                    print(f"  Reason: {info.get('reason', 'N/A')}")

        if args.method in ["one_hot", "both"]:
            print("\n=== Applying One-Hot Encoding ===")
            encoder.one_hot_encode(columns=args.columns)

        if args.method in ["label", "both"]:
            print("\n=== Applying Label Encoding ===")
            encoder.label_encode(columns=args.columns)

        if args.method != "compare":
            print("\n=== Encoding Summary ===")
            summary = encoder.get_encoding_summary()
            for col, info in summary.items():
                method = info["method"]
                if method == "one_hot":
                    print(
                        f"  {col}: {method} -> "
                        f"{len(info['encoded_columns'])} columns"
                    )
                else:
                    print(
                        f"  {col}: {method} -> "
                        f"{len(info['unique_values'])} unique values mapped"
                    )

            if args.output:
                encoder.save_encoded_data(args.output)
                print(f"\nEncoded data saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
