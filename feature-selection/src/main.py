"""Feature Selection Tool.

This module provides functionality to perform feature selection using
variance threshold, correlation analysis, and univariate statistics.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    variance_threshold,
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Performs feature selection using multiple methods."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize FeatureSelector with configuration.

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
        self.target: Optional[pd.Series] = None
        self.selected_features: List[str] = []

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
        selection_config = self.config.get("feature_selection", {})
        self.variance_threshold = selection_config.get("variance_threshold", 0.0)
        self.correlation_threshold = selection_config.get(
            "correlation_threshold", 0.95
        )
        self.univariate_k = selection_config.get("univariate_k", 10)
        self.univariate_score_func = selection_config.get(
            "univariate_score_func", "f_classif"
        )

    def load_data(
        self,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Load data from file or use provided DataFrame.

        Args:
            file_path: Path to CSV file (optional).
            dataframe: Pandas DataFrame (optional).
            target_column: Name of target column (optional).

        Returns:
            Tuple of (features DataFrame, target Series).

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

        if target_column:
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            self.target = self.data[target_column]
            self.data = self.data.drop(columns=[target_column])
            logger.info(f"Set target column: {target_column}")
        else:
            self.target = None

        return self.data, self.target

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

    def select_variance_threshold(
        self, threshold: Optional[float] = None
    ) -> List[str]:
        """Select features using variance threshold.

        Removes features with variance below threshold.

        Args:
            threshold: Variance threshold (default from config).

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        threshold = threshold or self.variance_threshold

        numeric_cols = self.get_numeric_columns()
        if not numeric_cols:
            logger.warning("No numerical columns found for variance threshold")
            return []

        numeric_data = self.data[numeric_cols]

        selector = variance_threshold(threshold=threshold)
        selector.fit(numeric_data)

        selected_mask = selector.get_support()
        selected_features = [
            col for col, selected in zip(numeric_cols, selected_mask) if selected
        ]

        removed_count = len(numeric_cols) - len(selected_features)
        logger.info(
            f"Variance threshold: {len(selected_features)} features selected, "
            f"{removed_count} removed (threshold: {threshold})"
        )

        self.selected_features = selected_features
        return selected_features

    def select_correlation(
        self, threshold: Optional[float] = None
    ) -> List[str]:
        """Select features using correlation analysis.

        Removes highly correlated features.

        Args:
            threshold: Correlation threshold (default from config).

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        threshold = threshold or self.correlation_threshold

        numeric_cols = self.get_numeric_columns()
        if not numeric_cols:
            logger.warning("No numerical columns found for correlation analysis")
            return []

        numeric_data = self.data[numeric_cols]
        corr_matrix = numeric_data.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        selected_features = [col for col in numeric_cols if col not in to_drop]

        logger.info(
            f"Correlation analysis: {len(selected_features)} features selected, "
            f"{len(to_drop)} removed (threshold: {threshold})"
        )

        self.selected_features = selected_features
        return selected_features

    def select_univariate(
        self,
        k: Optional[int] = None,
        score_func: Optional[str] = None,
    ) -> List[str]:
        """Select features using univariate statistical tests.

        Args:
            k: Number of top features to select (default from config).
            score_func: Score function (f_classif, f_regression, chi2,
                       mutual_info_classif, mutual_info_regression).

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If no data loaded, no target, or invalid score function.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.target is None:
            raise ValueError("Target column required for univariate selection")

        k = k or self.univariate_k
        score_func = score_func or self.univariate_score_func

        numeric_cols = self.get_numeric_columns()
        if not numeric_cols:
            logger.warning("No numerical columns found for univariate selection")
            return []

        numeric_data = self.data[numeric_cols]

        score_func_map = {
            "f_classif": f_classif,
            "f_regression": f_regression,
            "chi2": chi2,
            "mutual_info_classif": mutual_info_classif,
            "mutual_info_regression": mutual_info_regression,
        }

        if score_func not in score_func_map:
            raise ValueError(
                f"Invalid score function: {score_func}. "
                f"Use one of {list(score_func_map.keys())}"
            )

        selector = SelectKBest(
            score_func=score_func_map[score_func], k=min(k, len(numeric_cols))
        )

        selector.fit(numeric_data, self.target)

        selected_mask = selector.get_support()
        selected_features = [
            col for col, selected in zip(numeric_cols, selected_mask) if selected
        ]

        scores = selector.scores_
        feature_scores = {
            col: float(score)
            for col, score in zip(numeric_cols, scores)
            if col in selected_features
        }

        logger.info(
            f"Univariate selection: {len(selected_features)} features selected "
            f"(method: {score_func}, k={k})"
        )

        self.selected_features = selected_features
        return selected_features

    def select_all(
        self,
        variance_threshold: Optional[float] = None,
        correlation_threshold: Optional[float] = None,
        univariate_k: Optional[int] = None,
        univariate_score_func: Optional[str] = None,
    ) -> List[str]:
        """Apply all feature selection methods sequentially.

        Args:
            variance_threshold: Variance threshold (default from config).
            correlation_threshold: Correlation threshold (default from config).
            univariate_k: Number of top features (default from config).
            univariate_score_func: Score function (default from config).

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If no data loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Applying all feature selection methods sequentially")

        original_data = self.data.copy()

        variance_features = self.select_variance_threshold(
            threshold=variance_threshold
        )
        if not variance_features:
            variance_features = self.get_numeric_columns()

        self.data = original_data[variance_features].copy()

        correlation_features = self.select_correlation(
            threshold=correlation_threshold
        )
        if not correlation_features:
            correlation_features = variance_features

        common_features = [
            f for f in variance_features if f in correlation_features
        ]

        if self.target is not None:
            self.data = original_data[common_features].copy()
            univariate_features = self.select_univariate(
                k=univariate_k, score_func=univariate_score_func
            )
            final_features = univariate_features
        else:
            final_features = common_features

        self.data = original_data

        logger.info(f"Final selection: {len(final_features)} features")
        self.selected_features = final_features
        return final_features

    def get_selection_summary(self) -> Dict[str, any]:
        """Get summary of feature selection results.

        Returns:
            Dictionary with selection summary.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        summary = {
            "original_features": len(self.data.columns),
            "selected_features": len(self.selected_features),
            "removed_features": len(self.data.columns) - len(self.selected_features),
            "selected_feature_names": self.selected_features,
        }

        if self.selected_features:
            removed = [
                col for col in self.data.columns if col not in self.selected_features
            ]
            summary["removed_feature_names"] = removed

        return summary

    def apply_selection(
        self, features: Optional[List[str]] = None, inplace: bool = False
    ) -> pd.DataFrame:
        """Apply feature selection to data.

        Args:
            features: List of features to select (None uses selected_features).
            inplace: Whether to modify in place.

        Returns:
            DataFrame with selected features.

        Raises:
            ValueError: If no data loaded or features invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        features = features or self.selected_features

        if not features:
            raise ValueError("No features selected. Run selection methods first.")

        invalid_features = [f for f in features if f not in self.data.columns]
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}")

        result = self.data if inplace else self.data.copy()
        result = result[features]

        logger.info(f"Applied feature selection: {len(features)} features")

        if not inplace:
            return result

    def save_selected_data(self, output_path: str) -> None:
        """Save selected features to CSV file.

        Args:
            output_path: Path to output CSV file.

        Raises:
            ValueError: If no data loaded or no features selected.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if not self.selected_features:
            raise ValueError("No features selected. Run selection methods first.")

        selected_data = self.apply_selection()
        selected_data.to_csv(output_path, index=False)
        logger.info(f"Saved selected features to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Selection Tool")
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
        "--target",
        type=str,
        default=None,
        help="Name of target column (optional, required for univariate)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["variance", "correlation", "univariate", "all"],
        default="all",
        help="Feature selection method",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=None,
        help="Variance threshold (default from config)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=None,
        help="Correlation threshold (default from config)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of top features for univariate (default from config)",
    )
    parser.add_argument(
        "--score-func",
        type=str,
        choices=["f_classif", "f_regression", "chi2", "mutual_info_classif", "mutual_info_regression"],
        default=None,
        help="Score function for univariate (default from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )

    args = parser.parse_args()

    selector = FeatureSelector(config_path=args.config)

    try:
        selector.load_data(file_path=args.input, target_column=args.target)

        print("\n=== Dataset Information ===")
        print(f"Shape: {selector.data.shape}")
        print(f"Features: {len(selector.data.columns)}")
        if selector.target is not None:
            print(f"Target: {args.target}")

        print("\n=== Feature Selection ===")
        if args.method == "variance":
            selected = selector.select_variance_threshold(
                threshold=args.variance_threshold
            )
        elif args.method == "correlation":
            selected = selector.select_correlation(
                threshold=args.correlation_threshold
            )
        elif args.method == "univariate":
            if selector.target is None:
                print("Error: Target column required for univariate selection")
                return
            selected = selector.select_univariate(
                k=args.k, score_func=args.score_func
            )
        elif args.method == "all":
            selected = selector.select_all(
                variance_threshold=args.variance_threshold,
                correlation_threshold=args.correlation_threshold,
                univariate_k=args.k,
                univariate_score_func=args.score_func,
            )

        print("\n=== Selection Summary ===")
        summary = selector.get_selection_summary()
        print(f"Original features: {summary['original_features']}")
        print(f"Selected features: {summary['selected_features']}")
        print(f"Removed features: {summary['removed_features']}")
        print(f"\nSelected features: {summary['selected_feature_names']}")

        if args.output:
            selector.save_selected_data(args.output)
            print(f"\nSelected features saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error performing feature selection: {e}")
        raise


if __name__ == "__main__":
    main()
