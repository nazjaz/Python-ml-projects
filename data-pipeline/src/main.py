"""Data Pipeline with Custom Transformers.

This module provides functionality to create data pipelines with preprocessing
steps chained together using custom transformers.
"""

import logging
import logging.handlers
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseTransformer(ABC):
    """Base class for all custom transformers."""

    def __init__(self) -> None:
        """Initialize base transformer."""
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseTransformer":
        """Fit the transformer on the data.

        Args:
            X: Input DataFrame.
            y: Target Series (optional).

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If transformer not fitted.
        """
        pass

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit transformer and transform data.

        Args:
            X: Input DataFrame.
            y: Target Series (optional).

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)


class StandardScalerTransformer(BaseTransformer):
    """Standard scaler transformer (z-score normalization)."""

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        """Initialize standard scaler transformer.

        Args:
            columns: List of columns to scale (None for all numeric).
        """
        super().__init__()
        self.columns = columns
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "StandardScalerTransformer":
        """Fit the scaler on the data.

        Args:
            X: Input DataFrame.
            y: Target Series (optional, ignored).

        Returns:
            Self for method chaining.
        """
        if self.columns is None:
            self.columns = X.select_dtypes(include=["number"]).columns.tolist()

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"Column '{col}' is not numeric")

            self.means[col] = float(X[col].mean())
            self.stds[col] = float(X[col].std())
            if self.stds[col] == 0:
                logger.warning(f"Column '{col}' has zero standard deviation")

        self.is_fitted = True
        logger.info(f"Fitted StandardScalerTransformer on {len(self.columns)} columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If transformer not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X_transformed = X.copy()

        for col in self.columns:
            if col not in X_transformed.columns:
                raise ValueError(f"Column '{col}' not found in data")

            if self.stds[col] != 0:
                X_transformed[col] = (X_transformed[col] - self.means[col]) / self.stds[col]
            else:
                X_transformed[col] = 0.0

        logger.info("Applied StandardScalerTransformer transformation")
        return X_transformed


class MinMaxScalerTransformer(BaseTransformer):
    """Min-max scaler transformer (normalization to [0, 1])."""

    def __init__(
        self, columns: Optional[List[str]] = None, feature_range: Tuple[float, float] = (0, 1)
    ) -> None:
        """Initialize min-max scaler transformer.

        Args:
            columns: List of columns to scale (None for all numeric).
            feature_range: Desired range of transformed data.
        """
        super().__init__()
        self.columns = columns
        self.feature_range = feature_range
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MinMaxScalerTransformer":
        """Fit the scaler on the data.

        Args:
            X: Input DataFrame.
            y: Target Series (optional, ignored).

        Returns:
            Self for method chaining.
        """
        if self.columns is None:
            self.columns = X.select_dtypes(include=["number"]).columns.tolist()

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"Column '{col}' is not numeric")

            self.mins[col] = float(X[col].min())
            self.maxs[col] = float(X[col].max())

        self.is_fitted = True
        logger.info(f"Fitted MinMaxScalerTransformer on {len(self.columns)} columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If transformer not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X_transformed = X.copy()
        min_val, max_val = self.feature_range

        for col in self.columns:
            if col not in X_transformed.columns:
                raise ValueError(f"Column '{col}' not found in data")

            range_diff = self.maxs[col] - self.mins[col]
            if range_diff != 0:
                X_transformed[col] = (
                    (X_transformed[col] - self.mins[col]) / range_diff
                ) * (max_val - min_val) + min_val
            else:
                X_transformed[col] = min_val

        logger.info("Applied MinMaxScalerTransformer transformation")
        return X_transformed


class ImputerTransformer(BaseTransformer):
    """Imputer transformer for handling missing values."""

    def __init__(
        self,
        strategy: str = "mean",
        columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize imputer transformer.

        Args:
            strategy: Imputation strategy (mean, median, mode, constant).
            columns: List of columns to impute (None for all numeric).
        """
        super().__init__()
        self.strategy = strategy
        self.columns = columns
        self.imputation_values: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ImputerTransformer":
        """Fit the imputer on the data.

        Args:
            X: Input DataFrame.
            y: Target Series (optional, ignored).

        Returns:
            Self for method chaining.
        """
        if self.columns is None:
            self.columns = X.select_dtypes(include=["number"]).columns.tolist()

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in data")

            if self.strategy == "mean":
                self.imputation_values[col] = float(X[col].mean())
            elif self.strategy == "median":
                self.imputation_values[col] = float(X[col].median())
            elif self.strategy == "mode":
                mode_values = X[col].mode()
                self.imputation_values[col] = (
                    mode_values.iloc[0] if len(mode_values) > 0 else 0
                )
            elif self.strategy == "constant":
                self.imputation_values[col] = 0
            else:
                raise ValueError(
                    f"Invalid strategy: {self.strategy}. "
                    f"Use 'mean', 'median', 'mode', or 'constant'"
                )

        self.is_fitted = True
        logger.info(
            f"Fitted ImputerTransformer with strategy '{self.strategy}' "
            f"on {len(self.columns)} columns"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If transformer not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X_transformed = X.copy()

        for col in self.columns:
            if col not in X_transformed.columns:
                raise ValueError(f"Column '{col}' not found in data")

            X_transformed[col] = X_transformed[col].fillna(self.imputation_values[col])

        logger.info("Applied ImputerTransformer transformation")
        return X_transformed


class OneHotEncoderTransformer(BaseTransformer):
    """One-hot encoder transformer for categorical variables."""

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        """Initialize one-hot encoder transformer.

        Args:
            columns: List of columns to encode (None for all categorical).
        """
        super().__init__()
        self.columns = columns
        self.categories: Dict[str, List[Any]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OneHotEncoderTransformer":
        """Fit the encoder on the data.

        Args:
            X: Input DataFrame.
            y: Target Series (optional, ignored).

        Returns:
            Self for method chaining.
        """
        if self.columns is None:
            self.columns = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in data")

            self.categories[col] = sorted(X[col].dropna().unique().tolist())

        self.is_fitted = True
        logger.info(f"Fitted OneHotEncoderTransformer on {len(self.columns)} columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If transformer not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X_transformed = X.copy()

        for col in self.columns:
            if col not in X_transformed.columns:
                raise ValueError(f"Column '{col}' not found in data")

            for category in self.categories[col]:
                new_col_name = f"{col}_{category}"
                X_transformed[new_col_name] = (X_transformed[col] == category).astype(int)

            X_transformed = X_transformed.drop(columns=[col])

        logger.info("Applied OneHotEncoderTransformer transformation")
        return X_transformed


class DataPipeline:
    """Pipeline for chaining multiple transformers together."""

    def __init__(self, transformers: List[BaseTransformer]) -> None:
        """Initialize data pipeline.

        Args:
            transformers: List of transformers to chain together.

        Raises:
            ValueError: If transformers list is empty.
        """
        if not transformers:
            raise ValueError("Transformers list cannot be empty")

        self.transformers = transformers
        self.is_fitted = False
        logger.info(f"Initialized DataPipeline with {len(transformers)} transformers")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPipeline":
        """Fit all transformers in the pipeline.

        Args:
            X: Input DataFrame.
            y: Target Series (optional).

        Returns:
            Self for method chaining.
        """
        X_current = X.copy()

        for i, transformer in enumerate(self.transformers):
            logger.info(f"Fitting transformer {i+1}/{len(self.transformers)}")
            transformer.fit(X_current, y)
            X_current = transformer.transform(X_current)

        self.is_fitted = True
        logger.info("Fitted all transformers in pipeline")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all transformers in the pipeline.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If pipeline not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X_transformed = X.copy()

        for i, transformer in enumerate(self.transformers):
            logger.info(f"Transforming with transformer {i+1}/{len(self.transformers)}")
            X_transformed = transformer.transform(X_transformed)

        logger.info("Completed pipeline transformation")
        return X_transformed

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit pipeline and transform data.

        Args:
            X: Input DataFrame.
            y: Target Series (optional).

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.

        Returns:
            Dictionary with pipeline information.
        """
        return {
            "n_transformers": len(self.transformers),
            "transformer_types": [
                type(transformer).__name__ for transformer in self.transformers
            ],
            "is_fitted": self.is_fitted,
        }


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Data Pipeline Tool")
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

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        print(f"\n=== Loaded Data ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        print("\n=== Creating Pipeline ===")
        pipeline = DataPipeline(
            transformers=[
                ImputerTransformer(strategy="mean"),
                StandardScalerTransformer(),
            ]
        )

        print("\n=== Fitting and Transforming ===")
        transformed_df = pipeline.fit_transform(df)

        print(f"\n=== Transformed Data ===")
        print(f"Shape: {transformed_df.shape}")
        print(f"Columns: {list(transformed_df.columns)}")

        print("\n=== Pipeline Info ===")
        info = pipeline.get_pipeline_info()
        print(f"Number of transformers: {info['n_transformers']}")
        print(f"Transformer types: {info['transformer_types']}")
        print(f"Is fitted: {info['is_fitted']}")

        if args.output:
            transformed_df.to_csv(args.output, index=False)
            print(f"\nTransformed data saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
