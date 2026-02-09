"""Feature Engineering Utilities.

This module provides functionality to create polynomial features and
interaction terms for machine learning models.
"""

import itertools
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates polynomial features and interaction terms."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize FeatureEngineer with configuration.

        Args:
            config_path: Path to configuration YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_parameters()
        self.feature_names: List[str] = []

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
        feature_config = self.config.get("feature_engineering", {})
        self.default_polynomial_degree = feature_config.get(
            "default_polynomial_degree", 2
        )
        self.include_bias = feature_config.get("include_bias", False)

    def create_polynomial_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        degree: Optional[int] = None,
        include_bias: Optional[bool] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create polynomial features from input data.

        Args:
            X: Input DataFrame or numpy array.
            degree: Degree of polynomial features (default from config).
            include_bias: Whether to include bias term (default from config).
            columns: List of column names for DataFrame (optional).

        Returns:
            DataFrame with polynomial features.

        Raises:
            ValueError: If degree is invalid or columns don't match.
        """
        degree = degree if degree is not None else self.default_polynomial_degree
        include_bias = (
            include_bias if include_bias is not None else self.include_bias
        )

        if degree < 1:
            raise ValueError("Degree must be >= 1")

        if isinstance(X, pd.DataFrame):
            data = X.values
            original_columns = X.columns.tolist()
        else:
            data = np.asarray(X)
            original_columns = (
                columns
                if columns
                else [f"feature_{i}" for i in range(data.shape[1])]
            )

        n_samples, n_features = data.shape

        polynomial_features = []
        feature_names = []

        if include_bias:
            polynomial_features.append(np.ones((n_samples, 1)))
            feature_names.append("bias")

        for d in range(1, degree + 1):
            for feature_indices in itertools.combinations_with_replacement(
                range(n_features), d
            ):
                feature_product = np.ones((n_samples, 1))
                feature_name_parts = []

                for idx in feature_indices:
                    feature_product *= data[:, idx:idx+1]
                    feature_name_parts.append(original_columns[idx])

                polynomial_features.append(feature_product)

                if len(feature_name_parts) == 1:
                    if d == 1:
                        feature_names.append(feature_name_parts[0])
                    else:
                        feature_names.append(f"{feature_name_parts[0]}^{d}")
                else:
                    feature_name = " * ".join(feature_name_parts)
                    feature_names.append(feature_name)

        result_array = np.hstack(polynomial_features)
        result_df = pd.DataFrame(result_array, columns=feature_names)

        self.feature_names = feature_names

        logger.info(
            f"Created polynomial features: {n_features} features -> "
            f"{len(feature_names)} polynomial features (degree={degree})"
        )

        return result_df

    def create_interaction_terms(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[List[str]] = None,
        max_interactions: Optional[int] = None,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> pd.DataFrame:
        """Create interaction terms between features.

        Args:
            X: Input DataFrame or numpy array.
            columns: List of column names for DataFrame (optional).
            max_interactions: Maximum number of interactions to create (None for all).
            feature_pairs: Specific pairs of features to create interactions for (optional).

        Returns:
            DataFrame with original features and interaction terms.

        Raises:
            ValueError: If columns don't match or feature pairs invalid.
        """
        if isinstance(X, pd.DataFrame):
            data = X.values
            original_columns = X.columns.tolist()
            result_df = X.copy()
        else:
            data = np.asarray(X)
            original_columns = (
                columns
                if columns
                else [f"feature_{i}" for i in range(data.shape[1])]
            )
            result_df = pd.DataFrame(data, columns=original_columns)

        n_samples, n_features = data.shape
        interaction_features = []
        interaction_names = []

        if feature_pairs:
            pairs_to_create = []
            for pair in feature_pairs:
                if len(pair) != 2:
                    raise ValueError("Each feature pair must contain exactly 2 features")
                col1, col2 = pair
                if col1 not in original_columns or col2 not in original_columns:
                    raise ValueError(
                        f"Feature pair ({col1}, {col2}) contains invalid column names"
                    )
                pairs_to_create.append(
                    (original_columns.index(col1), original_columns.index(col2))
                )
        else:
            pairs_to_create = list(itertools.combinations(range(n_features), 2))

        if max_interactions is not None:
            pairs_to_create = pairs_to_create[:max_interactions]

        for idx1, idx2 in pairs_to_create:
            interaction = data[:, idx1] * data[:, idx2]
            interaction_features.append(interaction)
            interaction_name = (
                f"{original_columns[idx1]} * {original_columns[idx2]}"
            )
            interaction_names.append(interaction_name)

        if interaction_features:
            interaction_array = np.column_stack(interaction_features)
            interaction_df = pd.DataFrame(interaction_array, columns=interaction_names)
            result_df = pd.concat([result_df, interaction_df], axis=1)

            logger.info(
                f"Created {len(interaction_names)} interaction terms "
                f"from {n_features} features"
            )
        else:
            logger.warning("No interaction terms created")

        return result_df

    def create_polynomial_and_interactions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        polynomial_degree: Optional[int] = None,
        include_interactions: bool = True,
        max_interactions: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create both polynomial features and interaction terms.

        Args:
            X: Input DataFrame or numpy array.
            polynomial_degree: Degree of polynomial features (default from config).
            include_interactions: Whether to include interaction terms.
            max_interactions: Maximum number of interactions to create.
            columns: List of column names for DataFrame (optional).

        Returns:
            DataFrame with polynomial features and interaction terms.
        """
        result_df = self.create_polynomial_features(
            X, degree=polynomial_degree, columns=columns
        )

        if include_interactions:
            if isinstance(X, pd.DataFrame):
                original_df = X
            else:
                n_features = X.shape[1] if hasattr(X, 'shape') else len(columns) if columns else len(result_df.columns)
                original_df = pd.DataFrame(
                    X,
                    columns=columns if columns else [f"feature_{i}" for i in range(n_features)]
                )
            interaction_df = self.create_interaction_terms(
                original_df, max_interactions=max_interactions
            )

            interaction_cols = [
                col
                for col in interaction_df.columns
                if col not in original_df.columns
            ]
            if interaction_cols:
                result_df = pd.concat(
                    [result_df, interaction_df[interaction_cols]], axis=1
                )

        logger.info(
            f"Created polynomial features and interactions: "
            f"shape {result_df.shape}"
        )

        return result_df

    def get_feature_info(self) -> Dict[str, any]:
        """Get information about created features.

        Returns:
            Dictionary with feature information.
        """
        return {
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names.copy(),
        }


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Tool")
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
        "--polynomial-degree",
        type=int,
        default=None,
        help="Degree of polynomial features (default from config)",
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        help="Create interaction terms",
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=None,
        help="Maximum number of interactions to create",
    )

    args = parser.parse_args()

    engineer = FeatureEngineer(config_path=args.config)

    try:
        df = pd.read_csv(args.input)

        print("\n=== Feature Engineering ===")
        print(f"Input file: {args.input}")
        print(f"Original shape: {df.shape}")

        if args.polynomial_degree or args.interactions:
            if args.polynomial_degree:
                print(f"\nCreating polynomial features (degree={args.polynomial_degree})")
                df_engineered = engineer.create_polynomial_features(
                    df, degree=args.polynomial_degree
                )
            else:
                df_engineered = df.copy()

            if args.interactions:
                print("\nCreating interaction terms")
                df_engineered = engineer.create_interaction_terms(
                    df_engineered, max_interactions=args.max_interactions
                )
        else:
            print("\nCreating polynomial features and interactions")
            df_engineered = engineer.create_polynomial_and_interactions(
                df,
                polynomial_degree=args.polynomial_degree,
                include_interactions=True,
                max_interactions=args.max_interactions,
            )

        print(f"\nEngineered shape: {df_engineered.shape}")
        print(f"Features added: {df_engineered.shape[1] - df.shape[1]}")

        print("\n=== Feature Information ===")
        info = engineer.get_feature_info()
        print(f"Total features: {info['n_features']}")
        print(f"Sample feature names: {info['feature_names'][:5]}")

        if args.output:
            df_engineered.to_csv(args.output, index=False)
            print(f"\nEngineered features saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        raise


if __name__ == "__main__":
    main()
