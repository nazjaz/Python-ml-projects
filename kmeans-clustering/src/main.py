"""K-Means Clustering Algorithm.

This module provides functionality to implement k-means clustering
from scratch with elbow method for optimal cluster number selection.
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


class KMeans:
    """K-Means clustering algorithm."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iterations: int = 300,
        tolerance: float = 1e-4,
        init: str = "random",
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize KMeans.

        Args:
            n_clusters: Number of clusters.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.
            init: Initialization method. Options: 'random', 'k-means++'.
            random_state: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init = init
        self.random_state = random_state

        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.inertia: Optional[float] = None
        self.n_iterations: int = 0

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance.

        Args:
            x1: First point.
            x2: Second point.

        Returns:
            Euclidean distance.
        """
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))

    def _initialize_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids randomly.

        Args:
            X: Feature matrix.

        Returns:
            Initial centroids.
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        for i in range(self.n_clusters):
            centroids[i] = X[np.random.randint(0, n_samples)]

        return centroids

    def _initialize_centroids_kmeans_plusplus(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++.

        Args:
            X: Feature matrix.

        Returns:
            Initial centroids.
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]

        for i in range(1, self.n_clusters):
            distances = np.zeros(n_samples)

            for j in range(n_samples):
                min_dist = float("inf")
                for k in range(i):
                    dist = self._euclidean_distance(X[j], centroids[k])
                    min_dist = min(min_dist, dist)
                distances[j] = min_dist

            probabilities = distances**2 / np.sum(distances**2)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            idx = np.searchsorted(cumulative_probs, r)
            centroids[i] = X[idx]

        return centroids

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign samples to nearest centroids.

        Args:
            X: Feature matrix.
            centroids: Cluster centroids.

        Returns:
            Cluster labels for each sample.
        """
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            distances = [
                self._euclidean_distance(X[i], centroid) for centroid in centroids
            ]
            labels[i] = np.argmin(distances)

        return labels

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Update centroids based on cluster assignments.

        Args:
            X: Feature matrix.
            labels: Cluster labels.

        Returns:
            Updated centroids.
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))

        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                centroids[i] = X[np.random.randint(0, len(X))]

        return centroids

    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia).

        Args:
            X: Feature matrix.
            labels: Cluster labels.

        Returns:
            Inertia value.
        """
        inertia = 0.0

        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                for point in cluster_points:
                    inertia += self._euclidean_distance(point, centroid) ** 2

        return inertia

    def fit(self, X: Union[List, np.ndarray, pd.DataFrame]) -> "KMeans":
        """Fit k-means clustering model.

        Args:
            X: Feature matrix.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if self.n_clusters > len(X):
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot be greater than "
                f"number of samples ({len(X)})"
            )

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")

        if self.init == "random":
            centroids = self._initialize_centroids_random(X)
        elif self.init == "k-means++":
            centroids = self._initialize_centroids_kmeans_plusplus(X)
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")

        logger.info(
            f"Starting k-means: {self.n_clusters} clusters, "
            f"max_iterations={self.max_iterations}, init={self.init}"
        )

        for iteration in range(self.max_iterations):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            centroid_shift = np.sum(
                [
                    self._euclidean_distance(centroids[i], new_centroids[i])
                    for i in range(self.n_clusters)
                ]
            )

            centroids = new_centroids

            if centroid_shift < self.tolerance:
                logger.info(
                    f"Converged at iteration {iteration + 1} "
                    f"with shift={centroid_shift:.6f}"
                )
                break

            if (iteration + 1) % 50 == 0:
                logger.debug(f"Iteration {iteration + 1}: shift={centroid_shift:.6f}")

        self.centroids = centroids
        self.labels = self._assign_clusters(X, centroids)
        self.inertia = self._calculate_inertia(X, self.labels)
        self.n_iterations = iteration + 1

        logger.info(
            f"K-means complete: iterations={self.n_iterations}, "
            f"inertia={self.inertia:.6f}"
        )

        return self

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict cluster labels for new samples.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels.

        Raises:
            ValueError: If model not fitted.
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        labels = self._assign_clusters(X, self.centroids)
        return labels

    def fit_predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Fit model and predict cluster labels.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels.
        """
        self.fit(X)
        return self.labels


class ElbowMethod:
    """Elbow method for optimal cluster number selection."""

    def __init__(
        self,
        k_range: List[int],
        max_iterations: int = 300,
        tolerance: float = 1e-4,
        init: str = "random",
        random_state: Optional[int] = None,
        n_runs: int = 1,
    ) -> None:
        """Initialize ElbowMethod.

        Args:
            k_range: List of k values to test.
            max_iterations: Maximum iterations for each k-means run.
            tolerance: Convergence tolerance.
            init: Initialization method.
            random_state: Random seed for reproducibility.
            n_runs: Number of runs per k value (for averaging).
        """
        self.k_range = k_range
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init = init
        self.random_state = random_state
        self.n_runs = n_runs

        self.inertias: Optional[List[float]] = None
        self.results: Optional[Dict] = None

    def fit(self, X: Union[List, np.ndarray, pd.DataFrame]) -> Dict:
        """Find optimal k using elbow method.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary containing:
            - k_range: List of tested k values
            - inertias: List of inertia values
            - optimal_k: Optimal k value (if elbow detected)
            - results: Detailed results for each k
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        inertias = []
        results = {}

        logger.info(
            f"Elbow method: testing k values {self.k_range}, "
            f"n_runs={self.n_runs} per k"
        )

        for k in self.k_range:
            k_inertias = []

            for run in range(self.n_runs):
                kmeans = KMeans(
                    n_clusters=k,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                    init=self.init,
                    random_state=(
                        self.random_state + run
                        if self.random_state is not None
                        else None
                    ),
                )
                kmeans.fit(X)
                k_inertias.append(kmeans.inertia)

            avg_inertia = np.mean(k_inertias)
            std_inertia = np.std(k_inertias)
            inertias.append(avg_inertia)

            results[k] = {
                "inertia": float(avg_inertia),
                "std": float(std_inertia),
                "runs": [float(i) for i in k_inertias],
            }

            logger.debug(f"k={k}: inertia={avg_inertia:.4f} (±{std_inertia:.4f})")

        self.inertias = inertias
        self.results = results

        optimal_k = self._find_elbow()
        if optimal_k:
            logger.info(f"Optimal k detected: {optimal_k}")
        else:
            logger.info("No clear elbow detected")

        return {
            "k_range": self.k_range,
            "inertias": [float(i) for i in inertias],
            "optimal_k": optimal_k,
            "results": results,
        }

    def _find_elbow(self) -> Optional[int]:
        """Find elbow point using rate of change.

        Returns:
            Optimal k value if elbow detected, None otherwise.
        """
        if len(self.inertias) < 3:
            return None

        k_values = np.array(self.k_range)
        inertias = np.array(self.inertias)

        rates_of_change = []
        for i in range(1, len(inertias)):
            rate = (inertias[i - 1] - inertias[i]) / (k_values[i] - k_values[i - 1])
            rates_of_change.append(rate)

        if len(rates_of_change) < 2:
            return None

        second_derivatives = []
        for i in range(1, len(rates_of_change)):
            second_deriv = rates_of_change[i - 1] - rates_of_change[i]
            second_derivatives.append(second_deriv)

        if len(second_derivatives) == 0:
            return None

        elbow_idx = np.argmax(second_derivatives) + 1
        optimal_k = self.k_range[elbow_idx]

        return int(optimal_k)

    def plot_elbow(
        self, save_path: Optional[str] = None, show: bool = True
    ) -> None:
        """Plot elbow curve.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
        """
        if self.inertias is None:
            logger.warning("No elbow method results available")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.k_range, self.inertias, "bo-", linewidth=2, markersize=8)

        if self.results and self._find_elbow():
            optimal_k = self._find_elbow()
            optimal_inertia = self.results[optimal_k]["inertia"]
            plt.axvline(
                x=optimal_k,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Optimal k={optimal_k}",
            )
            plt.plot(optimal_k, optimal_inertia, "ro", markersize=12)

        plt.xlabel("Number of Clusters (k)", fontsize=12)
        plt.ylabel("Inertia (Within-cluster Sum of Squares)", fontsize=12)
        plt.title("Elbow Method for Optimal k Selection", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Elbow plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="K-Means Clustering")
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
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of feature columns (default: all columns)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: from config or use elbow method)",
    )
    parser.add_argument(
        "--elbow",
        action="store_true",
        help="Use elbow method to find optimal k",
    )
    parser.add_argument(
        "--k-range",
        type=str,
        default=None,
        help="Range of k values to test (e.g., '2:10' or '2,3,4,5')",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations (default: from config)",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["random", "k-means++"],
        default=None,
        help="Initialization method (default: from config)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot clusters and/or elbow curve",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save cluster assignments as CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        n_clusters = (
            args.n_clusters
            if args.n_clusters is not None
            else model_config.get("n_clusters", 3)
        )
        max_iter = (
            args.max_iterations
            if args.max_iterations is not None
            else model_config.get("max_iterations", 300)
        )
        init = (
            args.init
            if args.init is not None
            else model_config.get("init", "random")
        )

        df = pd.read_csv(args.input)
        print(f"\n=== K-Means Clustering ===")
        print(f"Data shape: {df.shape}")

        if args.features:
            feature_cols = [col.strip() for col in args.features.split(",")]
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
        else:
            feature_cols = list(df.columns)

        X = df[feature_cols].values

        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")

        if args.elbow:
            if args.k_range:
                if ":" in args.k_range:
                    start, end = map(int, args.k_range.split(":"))
                    k_range = list(range(start, end + 1))
                else:
                    k_range = [int(x.strip()) for x in args.k_range.split(",")]
            else:
                k_range = model_config.get("k_range", list(range(2, 11)))

            print(f"\nFinding optimal k using elbow method...")
            print(f"K range: {k_range}")

            elbow_method = ElbowMethod(
                k_range=k_range,
                max_iterations=max_iter,
                init=init,
            )
            results = elbow_method.fit(X)

            print(f"\n=== Elbow Method Results ===")
            for k, k_result in sorted(results["results"].items()):
                print(
                    f"k={k}: inertia={k_result['inertia']:.4f} "
                    f"(±{k_result['std']:.4f})"
                )

            if results["optimal_k"]:
                print(f"\nOptimal k: {results['optimal_k']}")
                n_clusters = results["optimal_k"]
            else:
                print("\nNo clear elbow detected, using minimum k")
                n_clusters = min(k_range)

            if args.plot or args.save_plot:
                elbow_method.plot_elbow(save_path=args.save_plot, show=args.plot)

        print(f"\nClustering with k={n_clusters}...")
        print(f"Initialization: {init}")
        print(f"Max iterations: {max_iter}")

        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iterations=max_iter,
            init=init,
        )
        kmeans.fit(X)

        print(f"\n=== Clustering Results ===")
        print(f"Iterations: {kmeans.n_iterations}")
        print(f"Inertia: {kmeans.inertia:.6f}")
        print(f"\nCluster sizes:")
        unique, counts = np.unique(kmeans.labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} samples")

        if args.plot or args.save_plot:
            if X.shape[1] == 2:
                plt.figure(figsize=(10, 8))
                for i in range(n_clusters):
                    cluster_points = X[kmeans.labels == i]
                    plt.scatter(
                        cluster_points[:, 0],
                        cluster_points[:, 1],
                        label=f"Cluster {i}",
                        alpha=0.6,
                    )
                plt.scatter(
                    kmeans.centroids[:, 0],
                    kmeans.centroids[:, 1],
                    c="black",
                    marker="x",
                    s=200,
                    linewidths=3,
                    label="Centroids",
                )
                plt.xlabel(feature_cols[0], fontsize=12)
                plt.ylabel(feature_cols[1], fontsize=12)
                plt.title(f"K-Means Clustering (k={n_clusters})", fontsize=14, fontweight="bold")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                if args.save_plot:
                    plt.savefig(args.save_plot, dpi=100, bbox_inches="tight")
                    print(f"\nPlot saved to: {args.save_plot}")

                if args.plot:
                    plt.show()
                else:
                    plt.close()

        if args.output:
            output_df = pd.DataFrame(X, columns=feature_cols)
            output_df["cluster"] = kmeans.labels
            output_df.to_csv(args.output, index=False)
            print(f"\nCluster assignments saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error running k-means: {e}")
        raise


if __name__ == "__main__":
    main()
