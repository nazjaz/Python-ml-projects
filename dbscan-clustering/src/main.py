"""DBSCAN Clustering Algorithm.

This module provides functionality to implement DBSCAN (Density-Based
Spatial Clustering of Applications with Noise) from scratch for
density-based clustering with noise detection.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DBSCAN:
    """DBSCAN clustering algorithm for density-based clustering."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        distance_metric: str = "euclidean",
    ) -> None:
        """Initialize DBSCAN.

        Args:
            eps: Maximum distance between two samples for one to be
                considered in the neighborhood of the other.
            min_samples: Minimum number of samples in a neighborhood
                for a point to be considered a core point.
            distance_metric: Distance metric. Currently only 'euclidean'
                is supported.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric

        self.X: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.core_samples: Optional[np.ndarray] = None
        self.noise_samples: Optional[np.ndarray] = None

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance.

        Args:
            x1: First point.
            x2: Second point.

        Returns:
            Euclidean distance.
        """
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))

    def _get_neighbors(self, point_idx: int, X: np.ndarray) -> List[int]:
        """Get indices of neighbors within eps distance.

        Args:
            point_idx: Index of the point.
            X: Feature matrix.

        Returns:
            List of neighbor indices.
        """
        neighbors = []
        point = X[point_idx]

        for i in range(len(X)):
            if i != point_idx:
                dist = self._euclidean_distance(point, X[i])
                if dist <= self.eps:
                    neighbors.append(i)

        return neighbors

    def _expand_cluster(
        self,
        point_idx: int,
        neighbors: List[int],
        cluster_id: int,
        X: np.ndarray,
        labels: np.ndarray,
        visited: Set[int],
    ) -> bool:
        """Expand cluster from a core point.

        Args:
            point_idx: Index of the core point.
            neighbors: List of neighbor indices.
            cluster_id: Current cluster ID.
            X: Feature matrix.
            labels: Cluster labels array.
            visited: Set of visited point indices.

        Returns:
            True if cluster was expanded, False otherwise.
        """
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                neighbor_neighbors = self._get_neighbors(neighbor_idx, X)

                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend(neighbor_neighbors)

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            i += 1

        return True

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> "DBSCAN":
        """Fit DBSCAN clustering model.

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

        if self.eps <= 0:
            raise ValueError("eps must be greater than 0")

        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        self.X = X
        n_samples = len(X)

        labels = np.full(n_samples, -1, dtype=int)
        visited = set()
        cluster_id = 0

        logger.info(
            f"Starting DBSCAN: eps={self.eps}, min_samples={self.min_samples}, "
            f"samples={n_samples}"
        )

        for point_idx in range(n_samples):
            if point_idx in visited:
                continue

            visited.add(point_idx)
            neighbors = self._get_neighbors(point_idx, X)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                self._expand_cluster(
                    point_idx, neighbors, cluster_id, X, labels, visited
                )
                cluster_id += 1

                if cluster_id % 10 == 0:
                    logger.debug(f"Created {cluster_id} clusters so far")

        self.labels = labels

        core_samples = []
        noise_samples = []

        for i in range(n_samples):
            neighbors = self._get_neighbors(i, X)
            if len(neighbors) >= self.min_samples:
                core_samples.append(i)
            elif labels[i] == -1:
                noise_samples.append(i)

        self.core_samples = np.array(core_samples)
        self.noise_samples = np.array(noise_samples)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        logger.info(
            f"DBSCAN complete: {n_clusters} clusters, {n_noise} noise points, "
            f"{len(self.core_samples)} core points"
        )

        return self

    def fit_predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Fit model and predict cluster labels.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels (-1 for noise).
        """
        self.fit(X)
        return self.labels

    def get_core_samples(self) -> np.ndarray:
        """Get indices of core samples.

        Returns:
            Array of core sample indices.
        """
        if self.core_samples is None:
            raise ValueError("Model must be fitted before getting core samples")
        return self.core_samples.copy()

    def get_noise_samples(self) -> np.ndarray:
        """Get indices of noise samples.

        Returns:
            Array of noise sample indices.
        """
        if self.noise_samples is None:
            raise ValueError("Model must be fitted before getting noise samples")
        return self.noise_samples.copy()

    def get_cluster_info(self) -> Dict:
        """Get information about clusters.

        Returns:
            Dictionary with cluster information.
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before getting cluster info")

        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.labels == -1)

        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(self.labels == label))

        return {
            "n_clusters": n_clusters,
            "n_noise": int(n_noise),
            "n_core_samples": len(self.core_samples) if self.core_samples is not None else 0,
            "cluster_sizes": cluster_sizes,
            "labels": self.labels.copy(),
        }

    def plot_clusters(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Plot clusters and noise points.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            title: Optional plot title.
        """
        if self.labels is None or self.X is None:
            logger.warning("Model must be fitted before plotting")
            return

        if self.X.shape[1] != 2:
            logger.warning("Plotting only supported for 2D data")
            return

        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        plt.figure(figsize=(12, 8))

        colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

        for k, col in zip(range(n_clusters), colors):
            if k == -1:
                col = "black"

            class_member_mask = self.labels == k
            xy = self.X[class_member_mask]
            plt.scatter(
                xy[:, 0],
                xy[:, 1],
                c=[col],
                s=50,
                alpha=0.6,
                label=f"Cluster {k}" if k != -1 else "Noise",
            )

        if -1 in unique_labels:
            noise_mask = self.labels == -1
            noise_xy = self.X[noise_mask]
            plt.scatter(
                noise_xy[:, 0],
                noise_xy[:, 1],
                c="black",
                marker="x",
                s=100,
                linewidths=2,
                label="Noise",
                alpha=0.8,
            )

        if self.core_samples is not None and len(self.core_samples) > 0:
            core_xy = self.X[self.core_samples]
            plt.scatter(
                core_xy[:, 0],
                core_xy[:, 1],
                c="red",
                marker="o",
                s=200,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                label="Core Points",
                alpha=0.8,
            )

        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        if title is None:
            title = f"DBSCAN Clustering (eps={self.eps}, min_samples={self.min_samples})"
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Cluster plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="DBSCAN Clustering")
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
        "--eps",
        type=float,
        default=None,
        help="Maximum distance for neighborhood (default: from config)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Minimum samples in neighborhood (default: from config)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot clusters",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save cluster plot",
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
        eps = (
            args.eps
            if args.eps is not None
            else model_config.get("eps", 0.5)
        )
        min_samples = (
            args.min_samples
            if args.min_samples is not None
            else model_config.get("min_samples", 5)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== DBSCAN Clustering ===")
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

        print(f"\nClustering with DBSCAN...")
        print(f"eps: {eps}")
        print(f"min_samples: {min_samples}")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)

        info = dbscan.get_cluster_info()

        print(f"\n=== Clustering Results ===")
        print(f"Number of clusters: {info['n_clusters']}")
        print(f"Number of noise points: {info['n_noise']}")
        print(f"Number of core points: {info['n_core_samples']}")
        print(f"\nCluster sizes:")
        for cluster_id, size in sorted(info["cluster_sizes"].items()):
            print(f"  Cluster {cluster_id}: {size} samples")

        if args.plot or args.save_plot:
            dbscan.plot_clusters(save_path=args.save_plot, show=args.plot)

        if args.output:
            output_df = pd.DataFrame(X, columns=feature_cols)
            output_df["cluster"] = dbscan.labels
            output_df["is_core"] = False
            output_df.loc[dbscan.core_samples, "is_core"] = True
            output_df["is_noise"] = dbscan.labels == -1
            output_df.to_csv(args.output, index=False)
            print(f"\nCluster assignments saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error running DBSCAN: {e}")
        raise


if __name__ == "__main__":
    main()
