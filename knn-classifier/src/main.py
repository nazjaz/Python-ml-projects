"""K-Nearest Neighbors (KNN) Classifier.

This module provides functionality to implement k-nearest neighbors
algorithm from scratch with different distance metrics and k-value
optimization.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DistanceMetrics:
    """Distance metric implementations."""

    @staticmethod
    def euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Euclidean distance.
        """
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))

    @staticmethod
    def manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan (L1) distance.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Manhattan distance.
        """
        return float(np.sum(np.abs(x1 - x2)))

    @staticmethod
    def minkowski(x1: np.ndarray, x2: np.ndarray, p: float = 2.0) -> float:
        """Calculate Minkowski distance.

        Args:
            x1: First feature vector.
            x2: Second feature vector.
            p: Power parameter (p=1: Manhattan, p=2: Euclidean).

        Returns:
            Minkowski distance.
        """
        return float(np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1 / p))

    @staticmethod
    def hamming(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Hamming distance (for categorical data).

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Hamming distance (proportion of differing elements).
        """
        return float(np.mean(x1 != x2))

    @staticmethod
    def cosine(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate cosine distance.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Cosine distance (1 - cosine similarity).
        """
        dot_product = np.dot(x1, x2)
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        cosine_similarity = dot_product / (norm1 * norm2)
        return float(1 - cosine_similarity)


class KNNClassifier:
    """K-Nearest Neighbors classifier."""

    def __init__(
        self,
        k: int = 5,
        distance_metric: str = "euclidean",
        metric_params: Optional[Dict] = None,
        scale_features: bool = True,
    ) -> None:
        """Initialize KNNClassifier.

        Args:
            k: Number of neighbors to consider.
            distance_metric: Distance metric to use. Options:
                'euclidean', 'manhattan', 'minkowski', 'hamming', 'cosine'.
            metric_params: Additional parameters for distance metric
                (e.g., {'p': 3} for Minkowski).
            scale_features: Whether to scale features.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.metric_params = metric_params or {}
        self.scale_features = scale_features

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.scale_params: Optional[Dict] = None
        self._setup_distance_function()

    def _setup_distance_function(self) -> None:
        """Setup distance function based on metric name."""
        metric_map = {
            "euclidean": DistanceMetrics.euclidean,
            "manhattan": DistanceMetrics.manhattan,
            "minkowski": lambda x1, x2: DistanceMetrics.minkowski(
                x1, x2, self.metric_params.get("p", 2.0)
            ),
            "hamming": DistanceMetrics.hamming,
            "cosine": DistanceMetrics.cosine,
        }

        if self.distance_metric not in metric_map:
            logger.warning(
                f"Unknown metric '{self.distance_metric}', using euclidean"
            )
            self.distance_metric = "euclidean"

        self.distance_func = metric_map[self.distance_metric]

    def _standardize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize features (zero mean, unit variance).

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (scaled_X, mean, std).
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        scaled_X = (X - mean) / std
        return scaled_X, mean, std

    def _apply_standardization(
        self, X: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Apply standardization using pre-computed mean and std.

        Args:
            X: Feature matrix.
            mean: Mean values.
            std: Standard deviation values.

        Returns:
            Scaled feature matrix.
        """
        return (X - mean) / std

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> "KNNClassifier":
        """Fit KNN classifier (stores training data).

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples, "
                f"y has {len(y)} samples"
            )

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if self.k > len(X):
            raise ValueError(
                f"k ({self.k}) cannot be greater than number of samples ({len(X)})"
            )

        if self.k < 1:
            raise ValueError("k must be at least 1")

        if self.scale_features:
            X, mean, std = self._standardize(X)
            self.scale_params = {"mean": mean, "std": std}
            logger.info("Features standardized")

        self.X_train = X
        self.y_train = y

        logger.info(
            f"KNN fitted: k={self.k}, metric={self.distance_metric}, "
            f"samples={len(X)}"
        )

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute distances from test samples to all training samples.

        Args:
            X: Test feature matrix.

        Returns:
            Distance matrix (shape: [n_test, n_train]).
        """
        n_test = len(X)
        n_train = len(self.X_train)
        distances = np.zeros((n_test, n_train))

        for i in range(n_test):
            for j in range(n_train):
                distances[i, j] = self.distance_func(X[i], self.X_train[j])

        return distances

    def predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class labels for test samples.

        Args:
            X: Test feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.scale_features and self.scale_params:
            X = self._apply_standardization(
                X, self.scale_params["mean"], self.scale_params["std"]
            )

        distances = self._compute_distances(X)
        predictions = []

        for i in range(len(X)):
            k_indices = np.argsort(distances[i])[: self.k]
            k_nearest_labels = self.y_train[k_indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]
            predictions.append(most_common_label)

        return np.array(predictions)

    def predict_proba(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class probabilities for test samples.

        Args:
            X: Test feature matrix.

        Returns:
            Probability matrix (shape: [n_samples, n_classes]).

        Raises:
            ValueError: If model not fitted.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.scale_features and self.scale_params:
            X = self._apply_standardization(
                X, self.scale_params["mean"], self.scale_params["std"]
            )

        distances = self._compute_distances(X)
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        probabilities = np.zeros((len(X), n_classes))

        for i in range(len(X)):
            k_indices = np.argsort(distances[i])[: self.k]
            k_nearest_labels = self.y_train[k_indices]

            for j, cls in enumerate(classes):
                probabilities[i, j] = np.sum(k_nearest_labels == cls) / self.k

        return probabilities

    def score(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate accuracy score.

        Args:
            X: Feature matrix.
            y: True target labels.

        Returns:
            Accuracy score (between 0 and 1).
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        return float(accuracy)


class KNNOptimizer:
    """Optimize k-value for KNN classifier using cross-validation."""

    def __init__(
        self,
        k_range: List[int],
        distance_metric: str = "euclidean",
        metric_params: Optional[Dict] = None,
        cv_folds: int = 5,
        scale_features: bool = True,
    ) -> None:
        """Initialize KNNOptimizer.

        Args:
            k_range: List of k values to test.
            distance_metric: Distance metric to use.
            metric_params: Additional parameters for distance metric.
            cv_folds: Number of cross-validation folds.
            scale_features: Whether to scale features.
        """
        self.k_range = k_range
        self.distance_metric = distance_metric
        self.metric_params = metric_params
        self.cv_folds = cv_folds
        self.scale_features = scale_features

        self.results: Optional[Dict] = None

    def _k_fold_split(
        self, X: np.ndarray, y: np.ndarray, n_splits: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Perform k-fold cross-validation split.

        Args:
            X: Feature matrix.
            y: Target labels.
            n_splits: Number of folds.

        Returns:
            List of (X_train, X_test, y_train, y_test) tuples.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_size = n_samples // n_splits
        splits = []

        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else n_samples

            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            splits.append((X_train, X_test, y_train, y_test))

        return splits

    def optimize(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> Dict[str, Union[int, float, Dict]]:
        """Optimize k-value using cross-validation.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Dictionary containing:
            - best_k: Optimal k value
            - best_score: Best cross-validation score
            - results: Dictionary with scores for each k value
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        k_scores = {}

        logger.info(
            f"Optimizing k from {min(self.k_range)} to {max(self.k_range)} "
            f"using {self.cv_folds}-fold CV"
        )

        for k in self.k_range:
            if k > len(X):
                logger.warning(f"Skipping k={k} (exceeds sample size)")
                continue

            fold_scores = []

            splits = self._k_fold_split(X, y, self.cv_folds)

            for X_train, X_test, y_train, y_test in splits:
                knn = KNNClassifier(
                    k=k,
                    distance_metric=self.distance_metric,
                    metric_params=self.metric_params,
                    scale_features=self.scale_features,
                )
                knn.fit(X_train, y_train)
                score = knn.score(X_test, y_test)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            k_scores[k] = {
                "mean": float(avg_score),
                "std": float(np.std(fold_scores)),
                "scores": [float(s) for s in fold_scores],
            }

            logger.debug(f"k={k}: CV score={avg_score:.4f}")

        best_k = max(k_scores.keys(), key=lambda k: k_scores[k]["mean"])
        best_score = k_scores[best_k]["mean"]

        self.results = {
            "best_k": int(best_k),
            "best_score": float(best_score),
            "results": k_scores,
        }

        logger.info(f"Best k: {best_k} with score: {best_score:.4f}")

        return self.results

    def plot_optimization_results(
        self, save_path: Optional[str] = None, show: bool = True
    ) -> None:
        """Plot k-value optimization results.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
        """
        if self.results is None:
            logger.warning("No optimization results available")
            return

        k_values = sorted(self.results["results"].keys())
        mean_scores = [self.results["results"][k]["mean"] for k in k_values]
        std_scores = [self.results["results"][k]["std"] for k in k_values]

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            k_values, mean_scores, yerr=std_scores, marker="o", capsize=5
        )
        plt.axvline(
            x=self.results["best_k"],
            color="r",
            linestyle="--",
            label=f"Best k={self.results['best_k']}",
        )
        plt.xlabel("k Value", fontsize=12)
        plt.ylabel("Cross-Validation Score", fontsize=12)
        plt.title("K-Value Optimization Results", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Optimization plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="K-Nearest Neighbors Classifier")
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
        help="Path to CSV file with data",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of target column",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of feature columns (default: all except target)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of neighbors (default: from config or optimize)",
    )
    parser.add_argument(
        "--distance",
        type=str,
        choices=["euclidean", "manhattan", "minkowski", "hamming", "cosine"],
        default=None,
        help="Distance metric (default: from config)",
    )
    parser.add_argument(
        "--optimize-k",
        action="store_true",
        help="Optimize k-value using cross-validation",
    )
    parser.add_argument(
        "--k-range",
        type=str,
        default=None,
        help="Range of k values to test (e.g., '1,3,5,7,9' or '1:10')",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of cross-validation folds (default: from config)",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Don't scale features",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot optimization results",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save optimization plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save model predictions as CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        k = (
            args.k
            if args.k is not None
            else model_config.get("k", 5)
        )
        distance = (
            args.distance
            if args.distance is not None
            else model_config.get("distance_metric", "euclidean")
        )
        scale_features = (
            not args.no_scale
            if args.no_scale
            else model_config.get("scale_features", True)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== K-Nearest Neighbors Classifier ===")
        print(f"Data shape: {df.shape}")

        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found")

        if args.features:
            feature_cols = [col.strip() for col in args.features.split(",")]
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
        else:
            feature_cols = [col for col in df.columns if col != args.target]

        X = df[feature_cols].values
        y = df[args.target].values

        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        unique_classes = np.unique(y)
        print(f"Classes: {len(unique_classes)} {unique_classes}")

        if args.optimize_k:
            if args.k_range:
                if ":" in args.k_range:
                    start, end = map(int, args.k_range.split(":"))
                    k_range = list(range(start, end + 1))
                else:
                    k_range = [int(x.strip()) for x in args.k_range.split(",")]
            else:
                k_range = model_config.get("k_range", list(range(1, 21)))

            cv_folds = (
                args.cv_folds
                if args.cv_folds is not None
                else model_config.get("cv_folds", 5)
            )

            print(f"\nOptimizing k-value...")
            print(f"K range: {k_range}")
            print(f"CV folds: {cv_folds}")

            optimizer = KNNOptimizer(
                k_range=k_range,
                distance_metric=distance,
                cv_folds=cv_folds,
                scale_features=scale_features,
            )

            results = optimizer.optimize(X, y)

            print(f"\n=== Optimization Results ===")
            print(f"Best k: {results['best_k']}")
            print(f"Best CV score: {results['best_score']:.4f}")
            print(f"\nAll results:")
            for k_val, k_result in sorted(results["results"].items()):
                print(
                    f"  k={k_val}: {k_result['mean']:.4f} "
                    f"(±{k_result['std']:.4f})"
                )

            k = results["best_k"]

            if args.plot or args.save_plot:
                optimizer.plot_optimization_results(
                    save_path=args.save_plot, show=args.plot
                )

        print(f"\nTraining KNN with k={k}...")
        print(f"Distance metric: {distance}")
        print(f"Feature scaling: {scale_features}")

        knn = KNNClassifier(
            k=k, distance_metric=distance, scale_features=scale_features
        )
        knn.fit(X, y)

        print(f"\n=== Model Performance ===")
        print(f"Accuracy: {knn.score(X, y):.6f}")

        if args.output:
            predictions = knn.predict(X)
            probabilities = knn.predict_proba(X)
            output_df = pd.DataFrame({"actual": y, "predicted": predictions})
            for i, cls in enumerate(unique_classes):
                output_df[f"prob_class_{cls}"] = probabilities[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


if __name__ == "__main__":
    main()
