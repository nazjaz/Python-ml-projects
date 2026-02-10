"""Imbalanced Dataset Handling with SMOTE, Undersampling, and Class Weighting.

This module provides implementations of various techniques for handling
imbalanced datasets in machine learning, including SMOTE oversampling,
multiple undersampling strategies, and class weighting methods.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SMOTE:
    """Synthetic Minority Oversampling Technique (SMOTE).

    Generates synthetic samples for minority class by interpolating
    between existing minority class samples and their nearest neighbors.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
        sampling_strategy: float = 1.0,
    ):
        """Initialize SMOTE.

        Args:
            k_neighbors: Number of nearest neighbors to use (default: 5)
            random_state: Random seed for reproducibility (default: None)
            sampling_strategy: Desired ratio of minority to majority class
                after oversampling. 1.0 means equal classes (default: 1.0)
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.nn_model = None

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit SMOTE and resample the dataset.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)

        Returns:
            Tuple containing:
                - X_resampled: Resampled feature matrix
                - y_resampled: Resampled target labels

        Raises:
            ValueError: If input data is invalid
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        X = np.array(X)
        y = np.array(y)

        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError("SMOTE requires at least 2 classes")

        majority_class = unique_classes[np.argmax(class_counts)]
        minority_classes = unique_classes[unique_classes != majority_class]

        X_resampled = X.copy()
        y_resampled = y.copy()

        n_majority = class_counts[np.argmax(class_counts)]

        for minority_class in minority_classes:
            minority_indices = np.where(y == minority_class)[0]
            X_minority = X[minority_indices]
            n_minority = len(minority_indices)

            target_count = int(n_majority * self.sampling_strategy)
            n_synthetic = target_count - n_minority

            if n_synthetic <= 0:
                logger.info(
                    f"Minority class {minority_class} already has enough samples"
                )
                continue

            logger.info(
                f"Generating {n_synthetic} synthetic samples for class {minority_class}"
            )

            self.nn_model = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, n_minority)
            )
            self.nn_model.fit(X_minority)

            synthetic_samples = self._generate_synthetic_samples(
                X_minority, n_synthetic
            )

            X_resampled = np.vstack([X_resampled, synthetic_samples])
            y_resampled = np.hstack(
                [y_resampled, np.full(n_synthetic, minority_class)]
            )

        return X_resampled, y_resampled

    def _generate_synthetic_samples(
        self, X_minority: np.ndarray, n_synthetic: int
    ) -> np.ndarray:
        """Generate synthetic samples using SMOTE algorithm.

        Args:
            X_minority: Minority class samples, shape (n_minority, n_features)
            n_synthetic: Number of synthetic samples to generate

        Returns:
            Synthetic samples, shape (n_synthetic, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_minority = X_minority.shape[0]
        synthetic_samples = []

        for _ in range(n_synthetic):
            random_idx = np.random.randint(0, n_minority)
            sample = X_minority[random_idx]

            distances, indices = self.nn_model.kneighbors(
                sample.reshape(1, -1), return_distance=True
            )

            if len(indices[0]) > 1:
                nn_idx = np.random.choice(indices[0][1:])
            else:
                nn_idx = indices[0][0]

            neighbor = X_minority[nn_idx]
            diff = neighbor - sample
            gap = np.random.random()
            synthetic = sample + gap * diff
            synthetic_samples.append(synthetic)

        return np.array(synthetic_samples)


class Undersampler:
    """Undersampling techniques for handling imbalanced datasets."""

    @staticmethod
    def random_undersample(
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random undersampling of majority class.

        Randomly removes samples from majority class to balance dataset.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
            sampling_strategy: Desired ratio of minority to majority class
                after undersampling (default: 1.0)
            random_state: Random seed for reproducibility (default: None)

        Returns:
            Tuple containing:
                - X_resampled: Undersampled feature matrix
                - y_resampled: Undersampled target labels

        Raises:
            ValueError: If input data is invalid
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        X = np.array(X)
        y = np.array(y)

        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError("Undersampling requires at least 2 classes")

        majority_class = unique_classes[np.argmax(class_counts)]
        minority_classes = unique_classes[unique_classes != majority_class]

        min_minority_count = min(
            [class_counts[unique_classes == c][0] for c in minority_classes]
        )
        target_majority_count = int(min_minority_count / sampling_strategy)

        majority_indices = np.where(y == majority_class)[0]
        n_majority = len(majority_indices)

        if target_majority_count >= n_majority:
            logger.info("Majority class already balanced, no undersampling needed")
            return X, y

        np.random.seed(random_state)
        selected_indices = np.random.choice(
            majority_indices, size=target_majority_count, replace=False
        )

        all_indices = np.concatenate(
            [
                selected_indices,
                *[np.where(y == c)[0] for c in minority_classes],
            ]
        )

        X_resampled = X[all_indices]
        y_resampled = y[all_indices]

        return X_resampled, y_resampled

    @staticmethod
    def tomek_links_undersample(
        X: np.ndarray, y: np.ndarray, n_neighbors: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tomek Links undersampling.

        Removes Tomek links (pairs of samples from different classes
        that are each other's nearest neighbors).

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
            n_neighbors: Number of neighbors to consider (default: 3)

        Returns:
            Tuple containing:
                - X_resampled: Undersampled feature matrix
                - y_resampled: Undersampled target labels

        Raises:
            ValueError: If input data is invalid
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        X = np.array(X)
        y = np.array(y)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("Tomek Links requires at least 2 classes")

        nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn_model.fit(X)

        indices_to_remove = set()

        for i in range(len(X)):
            distances, indices = nn_model.kneighbors(
                X[i].reshape(1, -1), return_distance=True
            )

            for j in indices[0][1:]:
                if y[i] != y[j]:
                    if i not in indices_to_remove and j not in indices_to_remove:
                        majority_class = unique_classes[
                            np.argmax([np.sum(y == c) for c in unique_classes])
                        ]
                        if y[i] == majority_class:
                            indices_to_remove.add(i)
                        elif y[j] == majority_class:
                            indices_to_remove.add(j)

        all_indices = np.array([i for i in range(len(X)) if i not in indices_to_remove])
        X_resampled = X[all_indices]
        y_resampled = y[all_indices]

        logger.info(f"Removed {len(indices_to_remove)} Tomek links")

        return X_resampled, y_resampled

    @staticmethod
    def edited_nearest_neighbours_undersample(
        X: np.ndarray, y: np.ndarray, n_neighbors: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Edited Nearest Neighbours (ENN) undersampling.

        Removes samples whose class label differs from the majority
        of its k nearest neighbors.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
            n_neighbors: Number of neighbors to consider (default: 3)

        Returns:
            Tuple containing:
                - X_resampled: Undersampled feature matrix
                - y_resampled: Undersampled target labels

        Raises:
            ValueError: If input data is invalid
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        X = np.array(X)
        y = np.array(y)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("ENN requires at least 2 classes")

        nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn_model.fit(X)

        indices_to_remove = set()

        for i in range(len(X)):
            distances, indices = nn_model.kneighbors(
                X[i].reshape(1, -1), return_distance=True
            )

            neighbor_labels = y[indices[0][1:]]
            neighbor_majority = np.bincount(neighbor_labels).argmax()

            if y[i] != neighbor_majority:
                majority_class = unique_classes[
                    np.argmax([np.sum(y == c) for c in unique_classes])
                ]
                if y[i] == majority_class:
                    indices_to_remove.add(i)

        all_indices = np.array([i for i in range(len(X)) if i not in indices_to_remove])
        X_resampled = X[all_indices]
        y_resampled = y[all_indices]

        logger.info(f"Removed {len(indices_to_remove)} samples using ENN")

        return X_resampled, y_resampled


class ClassWeightCalculator:
    """Calculate class weights for imbalanced datasets."""

    @staticmethod
    def balanced_weights(y: np.ndarray) -> Dict[int, float]:
        """Calculate balanced class weights.

        Weights are inversely proportional to class frequencies.

        Args:
            y: Target labels, shape (n_samples,)

        Returns:
            Dictionary mapping class labels to weights

        Raises:
            ValueError: If input data is invalid
        """
        y = np.array(y)
        unique_classes, class_counts = np.unique(y, return_counts=True)

        if len(unique_classes) < 2:
            raise ValueError("Class weights require at least 2 classes")

        n_samples = len(y)
        n_classes = len(unique_classes)

        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[int(cls)] = n_samples / (n_classes * count)

        return weights

    @staticmethod
    def compute_class_weight(
        y: np.ndarray, method: str = "balanced"
    ) -> Dict[int, float]:
        """Compute class weights using specified method.

        Args:
            y: Target labels, shape (n_samples,)
            method: Weighting method - "balanced" or "inverse" (default: "balanced")

        Returns:
            Dictionary mapping class labels to weights

        Raises:
            ValueError: If input data or method is invalid
        """
        if method not in ["balanced", "inverse"]:
            raise ValueError(f"Unknown method: {method}. Use 'balanced' or 'inverse'")

        y = np.array(y)
        unique_classes, class_counts = np.unique(y, return_counts=True)

        if len(unique_classes) < 2:
            raise ValueError("Class weights require at least 2 classes")

        weights = {}

        if method == "balanced":
            n_samples = len(y)
            n_classes = len(unique_classes)
            for cls, count in zip(unique_classes, class_counts):
                weights[int(cls)] = n_samples / (n_classes * count)
        else:
            max_count = max(class_counts)
            for cls, count in zip(unique_classes, class_counts):
                weights[int(cls)] = max_count / count

        return weights

    @staticmethod
    def custom_weights(y: np.ndarray, weight_dict: Dict[int, float]) -> Dict[int, float]:
        """Apply custom class weights.

        Args:
            y: Target labels, shape (n_samples,)
            weight_dict: Dictionary mapping class labels to custom weights

        Returns:
            Dictionary mapping class labels to weights

        Raises:
            ValueError: If input data is invalid
        """
        y = np.array(y)
        unique_classes = np.unique(y)

        weights = {}
        for cls in unique_classes:
            if int(cls) in weight_dict:
                weights[int(cls)] = weight_dict[int(cls)]
            else:
                logger.warning(
                    f"Class {cls} not in weight_dict, using default weight 1.0"
                )
                weights[int(cls)] = 1.0

        return weights


class ImbalancedDatasetHandler:
    """Main handler for imbalanced dataset techniques."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the imbalanced dataset handler.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.scaler = StandardScaler()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/app.log")

        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: Optional[int] = None,
        sampling_strategy: Optional[float] = None,
        scale_features: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to dataset.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
            k_neighbors: Number of nearest neighbors (default: from config)
            sampling_strategy: Desired ratio after oversampling (default: from config)
            scale_features: Whether to scale features before SMOTE (default: True)

        Returns:
            Tuple containing resampled X and y
        """
        smote_config = self.config.get("smote", {})
        k_neighbors = k_neighbors or smote_config.get("k_neighbors", 5)
        sampling_strategy = sampling_strategy or smote_config.get(
            "sampling_strategy", 1.0
        )

        X_scaled = X.copy()
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)

        smote = SMOTE(
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=self.config.get("random_state", None),
        )

        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        if scale_features:
            X_resampled = self.scaler.inverse_transform(X_resampled)

        logger.info(
            f"SMOTE: Original shape {X.shape} -> Resampled shape {X_resampled.shape}"
        )

        return X_resampled, y_resampled

    def apply_undersampling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "random",
        sampling_strategy: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply undersampling to dataset.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
            method: Undersampling method - "random", "tomek", or "enn" (default: "random")
            sampling_strategy: Desired ratio after undersampling (default: from config)

        Returns:
            Tuple containing resampled X and y

        Raises:
            ValueError: If method is invalid
        """
        if method not in ["random", "tomek", "enn"]:
            raise ValueError(f"Unknown method: {method}. Use 'random', 'tomek', or 'enn'")

        undersample_config = self.config.get("undersampling", {})
        sampling_strategy = sampling_strategy or undersample_config.get(
            "sampling_strategy", 1.0
        )

        if method == "random":
            X_resampled, y_resampled = Undersampler.random_undersample(
                X,
                y,
                sampling_strategy=sampling_strategy,
                random_state=self.config.get("random_state", None),
            )
        elif method == "tomek":
            X_resampled, y_resampled = Undersampler.tomek_links_undersample(X, y)
        else:
            X_resampled, y_resampled = (
                Undersampler.edited_nearest_neighbours_undersample(X, y)
            )

        logger.info(
            f"Undersampling ({method}): Original shape {X.shape} -> "
            f"Resampled shape {X_resampled.shape}"
        )

        return X_resampled, y_resampled

    def compute_class_weights(
        self, y: np.ndarray, method: str = "balanced"
    ) -> Dict[int, float]:
        """Compute class weights for imbalanced dataset.

        Args:
            y: Target labels, shape (n_samples,)
            method: Weighting method - "balanced", "inverse", or "custom" (default: "balanced")

        Returns:
            Dictionary mapping class labels to weights
        """
        if method == "balanced":
            weights = ClassWeightCalculator.balanced_weights(y)
        elif method == "inverse":
            weights = ClassWeightCalculator.compute_class_weight(y, method="inverse")
        elif method == "custom":
            custom_weights_dict = self.config.get("class_weights", {}).get(
                "custom_weights", {}
            )
            weights = ClassWeightCalculator.custom_weights(y, custom_weights_dict)
        else:
            weights = ClassWeightCalculator.compute_class_weight(y, method="balanced")

        logger.info(f"Computed class weights ({method}): {weights}")

        return weights

    def get_class_distribution(self, y: np.ndarray) -> Dict:
        """Get class distribution statistics.

        Args:
            y: Target labels, shape (n_samples,)

        Returns:
            Dictionary with class distribution information
        """
        y = np.array(y)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total = len(y)

        distribution = {
            "total_samples": int(total),
            "n_classes": len(unique_classes),
            "class_counts": {
                int(cls): int(count) for cls, count in zip(unique_classes, class_counts)
            },
            "class_percentages": {
                int(cls): float(count / total * 100)
                for cls, count in zip(unique_classes, class_counts)
            },
        }

        if len(unique_classes) == 2:
            minority_class = unique_classes[np.argmin(class_counts)]
            majority_class = unique_classes[np.argmax(class_counts)]
            imbalance_ratio = class_counts[np.argmax(class_counts)] / class_counts[
                np.argmin(class_counts)
            ]
            distribution["imbalance_ratio"] = float(imbalance_ratio)
            distribution["minority_class"] = int(minority_class)
            distribution["majority_class"] = int(majority_class)

        return distribution


def main():
    """Main entry point for the imbalanced dataset handler."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Handle imbalanced datasets using SMOTE, undersampling, and class weighting"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Column name for target variable",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["smote", "undersample", "class_weights", "all"],
        default="all",
        help="Method to apply (default: all)",
    )
    parser.add_argument(
        "--undersample-method",
        type=str,
        choices=["random", "tomek", "enn"],
        default="random",
        help="Undersampling method if method=undersample (default: random)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV file for resampled data",
    )
    parser.add_argument(
        "--weights-output",
        type=str,
        help="Path to output JSON file for class weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale features before resampling",
    )

    args = parser.parse_args()

    handler = ImbalancedDatasetHandler(
        config_path=Path(args.config) if args.config else None
    )

    df = pd.read_csv(args.input)
    target_col = args.target_col

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    logger.info("Original dataset distribution:")
    distribution = handler.get_class_distribution(y)
    print("\nOriginal Class Distribution:")
    print(json.dumps(distribution, indent=2))

    results = {}

    if args.method in ["smote", "all"]:
        X_resampled, y_resampled = handler.apply_smote(
            X, y, scale_features=args.scale
        )
        results["smote"] = {
            "X": X_resampled,
            "y": y_resampled,
            "distribution": handler.get_class_distribution(y_resampled),
        }

    if args.method in ["undersample", "all"]:
        X_resampled, y_resampled = handler.apply_undersampling(
            X, y, method=args.undersample_method
        )
        results["undersample"] = {
            "X": X_resampled,
            "y": y_resampled,
            "distribution": handler.get_class_distribution(y_resampled),
        }

    if args.method in ["class_weights", "all"]:
        weights = handler.compute_class_weights(y, method="balanced")
        results["class_weights"] = weights

    if args.output:
        if args.method == "smote" or (args.method == "all" and "smote" in results):
            result_key = "smote"
        elif args.method == "undersample" or (
            args.method == "all" and "undersample" in results
        ):
            result_key = "undersample"
        else:
            result_key = None

        if result_key and result_key in results:
            X_resampled = results[result_key]["X"]
            y_resampled = results[result_key]["y"]

            output_df = pd.DataFrame(X_resampled, columns=feature_cols)
            output_df[target_col] = y_resampled
            output_df.to_csv(args.output, index=False)
            logger.info(f"Resampled data saved to {args.output}")

    if args.weights_output and "class_weights" in results:
        with open(args.weights_output, "w") as f:
            json.dump(results["class_weights"], f, indent=2)
        logger.info(f"Class weights saved to {args.weights_output}")

    print("\nResults Summary:")
    print("=" * 50)
    for method, result in results.items():
        if method in ["smote", "undersample"]:
            print(f"\n{method.upper()} Distribution:")
            print(json.dumps(result["distribution"], indent=2))
        elif method == "class_weights":
            print(f"\nCLASS_WEIGHTS:")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
