"""Synthetic Dataset Generator.

This module provides functionality to generate synthetic datasets for
classification and regression tasks with configurable parameters.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.datasets import make_classification, make_regression

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """Generates synthetic datasets for classification and regression."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize SyntheticDatasetGenerator with configuration.

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
        gen_config = self.config.get("generation", {})
        self.n_samples = gen_config.get("n_samples", 1000)
        self.n_features = gen_config.get("n_features", 10)
        self.random_state = gen_config.get("random_state", 42)
        self.noise = gen_config.get("noise", 0.1)

    def generate_classification(
        self,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        n_informative: Optional[int] = None,
        n_redundant: Optional[int] = None,
        n_repeated: Optional[int] = None,
        n_classes: Optional[int] = None,
        n_clusters_per_class: Optional[int] = None,
        weights: Optional[List[float]] = None,
        flip_y: Optional[float] = None,
        class_sep: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic classification dataset.

        Args:
            n_samples: Number of samples (default from config).
            n_features: Number of features (default from config).
            n_informative: Number of informative features.
            n_redundant: Number of redundant features.
            n_repeated: Number of repeated features.
            n_classes: Number of classes.
            n_clusters_per_class: Number of clusters per class.
            weights: Class weights (proportions).
            flip_y: Fraction of samples with flipped labels.
            class_sep: Class separation factor.
            random_state: Random seed (default from config).

        Returns:
            Tuple of (features DataFrame, target Series).

        Raises:
            ValueError: If parameters invalid.
        """
        n_samples = n_samples or self.n_samples
        n_features = n_features or self.n_features
        random_state = random_state or self.random_state

        n_informative = n_informative or max(2, n_features // 2)
        n_redundant = n_redundant or max(0, n_features // 4)
        n_repeated = n_repeated or 0
        n_classes = n_classes or 2
        n_clusters_per_class = n_clusters_per_class or 1
        flip_y = flip_y or 0.01
        class_sep = class_sep or 1.0

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            weights=weights,
            flip_y=flip_y,
            class_sep=class_sep,
            random_state=random_state,
        )

        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        self.data = pd.DataFrame(X, columns=feature_names)
        self.target = pd.Series(y, name="target")

        logger.info(
            f"Generated classification dataset: {n_samples} samples, "
            f"{n_features} features, {n_classes} classes"
        )

        return self.data, self.target

    def generate_regression(
        self,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        n_informative: Optional[int] = None,
        noise: Optional[float] = None,
        bias: Optional[float] = None,
        effective_rank: Optional[int] = None,
        tail_strength: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic regression dataset.

        Args:
            n_samples: Number of samples (default from config).
            n_features: Number of features (default from config).
            n_informative: Number of informative features.
            noise: Standard deviation of gaussian noise.
            bias: Bias term in linear model.
            effective_rank: Approximate number of singular vectors.
            tail_strength: Strength of tail in singular values.
            random_state: Random seed (default from config).

        Returns:
            Tuple of (features DataFrame, target Series).

        Raises:
            ValueError: If parameters invalid.
        """
        n_samples = n_samples or self.n_samples
        n_features = n_features or self.n_features
        noise = noise if noise is not None else self.noise
        random_state = random_state or self.random_state

        n_informative = n_informative or max(10, n_features // 2)
        bias = bias or 0.0
        effective_rank = effective_rank or None
        tail_strength = tail_strength or 0.5

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            bias=bias,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            random_state=random_state,
        )

        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        self.data = pd.DataFrame(X, columns=feature_names)
        self.target = pd.Series(y, name="target")

        logger.info(
            f"Generated regression dataset: {n_samples} samples, "
            f"{n_features} features, noise={noise}"
        )

        return self.data, self.target

    def generate_custom_classification(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int = 2,
        class_distribution: Optional[List[float]] = None,
        feature_ranges: Optional[List[Tuple[float, float]]] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate custom classification dataset with specified distributions.

        Args:
            n_samples: Number of samples.
            n_features: Number of features.
            n_classes: Number of classes.
            class_distribution: Distribution of classes (must sum to 1.0).
            feature_ranges: List of (min, max) tuples for each feature.
            random_state: Random seed (default from config).

        Returns:
            Tuple of (features DataFrame, target Series).

        Raises:
            ValueError: If parameters invalid.
        """
        random_state = random_state or self.random_state
        np.random.seed(random_state)

        if class_distribution is None:
            class_distribution = [1.0 / n_classes] * n_classes
        elif abs(sum(class_distribution) - 1.0) > 1e-6:
            raise ValueError("class_distribution must sum to 1.0")

        if feature_ranges is None:
            feature_ranges = [(-10.0, 10.0)] * n_features
        elif len(feature_ranges) != n_features:
            raise ValueError(
                f"feature_ranges length ({len(feature_ranges)}) "
                f"must match n_features ({n_features})"
            )

        class_sizes = [
            int(n_samples * dist) for dist in class_distribution[:-1]
        ]
        class_sizes.append(n_samples - sum(class_sizes))

        X_list = []
        y_list = []

        for class_idx, class_size in enumerate(class_sizes):
            class_X = []
            for feature_idx, (min_val, max_val) in enumerate(feature_ranges):
                mean = (min_val + max_val) / 2 + class_idx * 2.0
                std = (max_val - min_val) / 4
                values = np.random.normal(mean, std, class_size)
                values = np.clip(values, min_val, max_val)
                class_X.append(values)

            class_X = np.column_stack(class_X)
            class_y = np.full(class_size, class_idx)

            X_list.append(class_X)
            y_list.append(class_y)

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        self.data = pd.DataFrame(X, columns=feature_names)
        self.target = pd.Series(y, name="target")

        logger.info(
            f"Generated custom classification dataset: {n_samples} samples, "
            f"{n_features} features, {n_classes} classes"
        )

        return self.data, self.target

    def get_dataset_info(self) -> Dict[str, any]:
        """Get information about generated dataset.

        Returns:
            Dictionary with dataset information.

        Raises:
            ValueError: If no data generated.
        """
        if self.data is None:
            raise ValueError("No dataset generated. Call generation method first.")

        info = {
            "shape": self.data.shape,
            "n_samples": len(self.data),
            "n_features": len(self.data.columns),
            "feature_names": list(self.data.columns),
        }

        if self.target is not None:
            info["has_target"] = True
            info["target_name"] = self.target.name

            if pd.api.types.is_numeric_dtype(self.target):
                info["task_type"] = "regression"
                info["target_stats"] = {
                    "mean": float(self.target.mean()),
                    "std": float(self.target.std()),
                    "min": float(self.target.min()),
                    "max": float(self.target.max()),
                }
            else:
                info["task_type"] = "classification"
                info["n_classes"] = int(self.target.nunique())
                info["class_distribution"] = self.target.value_counts().to_dict()
        else:
            info["has_target"] = False

        return info

    def save_dataset(
        self,
        output_path: str,
        include_target: bool = True,
    ) -> None:
        """Save generated dataset to CSV file.

        Args:
            output_path: Path to output CSV file.
            include_target: Whether to include target column.

        Raises:
            ValueError: If no data generated.
        """
        if self.data is None:
            raise ValueError("No dataset generated. Call generation method first.")

        if include_target and self.target is not None:
            output_data = pd.concat([self.data, self.target], axis=1)
        else:
            output_data = self.data

        output_data.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to: {output_path}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Synthetic Dataset Generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression", "custom_classification"],
        required=True,
        help="Type of dataset to generate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples (default from config)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Number of features (default from config)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=None,
        help="Number of classes (for classification)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed (default from config)",
    )

    args = parser.parse_args()

    generator = SyntheticDatasetGenerator(config_path=args.config)

    try:
        if args.task == "classification":
            data, target = generator.generate_classification(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_classes=args.n_classes,
                random_state=args.random_state,
            )
        elif args.task == "regression":
            data, target = generator.generate_regression(
                n_samples=args.n_samples,
                n_features=args.n_features,
                random_state=args.random_state,
            )
        elif args.task == "custom_classification":
            data, target = generator.generate_custom_classification(
                n_samples=args.n_samples or 1000,
                n_features=args.n_features or 10,
                n_classes=args.n_classes or 2,
                random_state=args.random_state,
            )

        print("\n=== Dataset Information ===")
        info = generator.get_dataset_info()
        print(f"Shape: {info['shape']}")
        print(f"Samples: {info['n_samples']}")
        print(f"Features: {info['n_features']}")
        if info["has_target"]:
            print(f"Task type: {info['task_type']}")
            if info["task_type"] == "classification":
                print(f"Classes: {info['n_classes']}")
                print(f"Class distribution: {info['class_distribution']}")
            else:
                print(f"Target stats: {info['target_stats']}")

        generator.save_dataset(args.output)
        print(f"\nDataset saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise


if __name__ == "__main__":
    main()
