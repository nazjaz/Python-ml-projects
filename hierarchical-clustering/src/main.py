"""Hierarchical Clustering Algorithm.

This module provides functionality to implement hierarchical clustering
from scratch with single, complete, and average linkage methods and
dendrogram visualization.
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


class HierarchicalClustering:
    """Hierarchical clustering with different linkage methods."""

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        linkage: str = "average",
        distance_metric: str = "euclidean",
    ) -> None:
        """Initialize HierarchicalClustering.

        Args:
            n_clusters: Number of clusters to form. If None, returns full
                dendrogram without cutting.
            linkage: Linkage method. Options: 'single', 'complete', 'average'.
            distance_metric: Distance metric. Currently only 'euclidean'
                is supported.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric

        self.X: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.dendrogram_data: Optional[List[Dict]] = None

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance.

        Args:
            x1: First point.
            x2: Second point.

        Returns:
            Euclidean distance.
        """
        return float(np.sqrt(np.sum((x1 - x2) ** 2)))

    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix.

        Args:
            X: Feature matrix.

        Returns:
            Distance matrix.
        """
        n_samples = len(X)
        distance_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self._euclidean_distance(X[i], X[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _single_linkage(
        self, cluster1: set, cluster2: set, distance_matrix: np.ndarray
    ) -> float:
        """Calculate single linkage (minimum distance).

        Args:
            cluster1: First cluster (set of indices).
            cluster2: Second cluster (set of indices).
            distance_matrix: Pairwise distance matrix.

        Returns:
            Minimum distance between clusters.
        """
        min_dist = float("inf")
        for i in cluster1:
            for j in cluster2:
                dist = distance_matrix[i, j]
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _complete_linkage(
        self, cluster1: set, cluster2: set, distance_matrix: np.ndarray
    ) -> float:
        """Calculate complete linkage (maximum distance).

        Args:
            cluster1: First cluster (set of indices).
            cluster2: Second cluster (set of indices).
            distance_matrix: Pairwise distance matrix.

        Returns:
            Maximum distance between clusters.
        """
        max_dist = 0.0
        for i in cluster1:
            for j in cluster2:
                dist = distance_matrix[i, j]
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def _average_linkage(
        self, cluster1: set, cluster2: set, distance_matrix: np.ndarray
    ) -> float:
        """Calculate average linkage (average distance).

        Args:
            cluster1: First cluster (set of indices).
            cluster2: Second cluster (set of indices).
            distance_matrix: Pairwise distance matrix.

        Returns:
            Average distance between clusters.
        """
        total_dist = 0.0
        count = 0
        for i in cluster1:
            for j in cluster2:
                total_dist += distance_matrix[i, j]
                count += 1
        return total_dist / count if count > 0 else 0.0

    def _calculate_linkage_distance(
        self, cluster1: set, cluster2: set, distance_matrix: np.ndarray
    ) -> float:
        """Calculate linkage distance based on method.

        Args:
            cluster1: First cluster (set of indices).
            cluster2: Second cluster (set of indices).
            distance_matrix: Pairwise distance matrix.

        Returns:
            Linkage distance.
        """
        if self.linkage == "single":
            return self._single_linkage(cluster1, cluster2, distance_matrix)
        elif self.linkage == "complete":
            return self._complete_linkage(cluster1, cluster2, distance_matrix)
        elif self.linkage == "average":
            return self._average_linkage(cluster1, cluster2, distance_matrix)
        else:
            raise ValueError(f"Unknown linkage method: {self.linkage}")

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> "HierarchicalClustering":
        """Fit hierarchical clustering model.

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

        if len(X) < 2:
            raise ValueError("Need at least 2 samples for clustering")

        self.X = X
        n_samples = len(X)

        distance_matrix = self._compute_distance_matrix(X)

        clusters = [{i} for i in range(n_samples)]
        cluster_ids = list(range(n_samples))
        next_cluster_id = n_samples

        linkage_matrix = []
        dendrogram_data = []

        logger.info(
            f"Starting hierarchical clustering: {self.linkage} linkage, "
            f"{n_samples} samples"
        )

        while len(clusters) > 1:
            min_dist = float("inf")
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._calculate_linkage_distance(
                        clusters[i], clusters[j], distance_matrix
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            if merge_i == -1 or merge_j == -1:
                break

            cluster_i_id = cluster_ids[merge_i]
            cluster_j_id = cluster_ids[merge_j]

            new_cluster = clusters[merge_i].union(clusters[merge_j])
            new_cluster_id = next_cluster_id
            next_cluster_id += 1

            linkage_matrix.append(
                [cluster_i_id, cluster_j_id, min_dist, len(new_cluster)]
            )

            dendrogram_data.append(
                {
                    "cluster1": cluster_i_id,
                    "cluster2": cluster_j_id,
                    "distance": min_dist,
                    "size": len(new_cluster),
                }
            )

            clusters[merge_i] = new_cluster
            cluster_ids[merge_i] = new_cluster_id
            del clusters[merge_j]
            del cluster_ids[merge_j]

            if (n_samples - len(clusters)) % 10 == 0:
                logger.debug(
                    f"Merged clusters: {len(clusters)} remaining, "
                    f"distance={min_dist:.4f}"
                )

        self.linkage_matrix = np.array(linkage_matrix)
        self.dendrogram_data = dendrogram_data

        if self.n_clusters is not None:
            self.labels = self._get_cluster_labels(n_samples)

        logger.info(f"Hierarchical clustering complete: {len(linkage_matrix)} merges")

        return self

    def _get_cluster_labels(self, n_samples: int) -> np.ndarray:
        """Get cluster labels by cutting dendrogram at n_clusters.

        Args:
            n_samples: Number of samples.

        Returns:
            Cluster labels.
        """
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted before getting labels")

        labels = np.zeros(n_samples, dtype=int)
        clusters = [{i} for i in range(n_samples)]

        n_merges_to_keep = n_samples - self.n_clusters

        for i in range(n_merges_to_keep):
            cluster1_idx = int(self.linkage_matrix[i, 0])
            cluster2_idx = int(self.linkage_matrix[i, 1])

            cluster1 = None
            cluster2 = None
            cluster1_pos = None
            cluster2_pos = None

            for pos, cluster in enumerate(clusters):
                if cluster1_idx in cluster:
                    cluster1 = cluster
                    cluster1_pos = pos
                if cluster2_idx in cluster:
                    cluster2 = cluster
                    cluster2_pos = pos

            if cluster1 is not None and cluster2 is not None:
                new_cluster = cluster1.union(cluster2)
                clusters[cluster1_pos] = new_cluster
                if cluster2_pos is not None and cluster2_pos != cluster1_pos:
                    del clusters[cluster2_pos]

        for cluster_id, cluster in enumerate(clusters):
            for sample_idx in cluster:
                if sample_idx < n_samples:
                    labels[sample_idx] = cluster_id

        return labels

    def fit_predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Fit model and predict cluster labels.

        Args:
            X: Feature matrix.

        Returns:
            Cluster labels.

        Raises:
            ValueError: If n_clusters not set.
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be set for fit_predict")

        self.fit(X)
        return self.labels

    def plot_dendrogram(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        truncate_mode: Optional[str] = None,
        p: Optional[int] = None,
    ) -> None:
        """Plot dendrogram.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            truncate_mode: Truncation mode (not implemented yet).
            p: Number of levels to show if truncate_mode is set.
        """
        if self.linkage_matrix is None:
            logger.warning("No linkage matrix available")
            return

        n_samples = len(self.X)
        x_positions = {}
        x_counter = [0]

        def get_x_position(cluster_id: int) -> float:
            """Get x position for a cluster."""
            if cluster_id < n_samples:
                if cluster_id not in x_positions:
                    x_positions[cluster_id] = x_counter[0]
                    x_counter[0] += 1
                return x_positions[cluster_id]
            else:
                merge_idx = cluster_id - n_samples
                if merge_idx >= len(self.linkage_matrix):
                    return 0.0
                cluster1_id = int(self.linkage_matrix[merge_idx, 0])
                cluster2_id = int(self.linkage_matrix[merge_idx, 1])
                x1 = get_x_position(cluster1_id)
                x2 = get_x_position(cluster2_id)
                x_center = (x1 + x2) / 2
                x_positions[cluster_id] = x_center
                return x_center

        fig, ax = plt.subplots(figsize=(14, 8))

        for i, merge in enumerate(self.linkage_matrix):
            cluster1_id = int(merge[0])
            cluster2_id = int(merge[1])
            distance = merge[2]

            x1 = get_x_position(cluster1_id)
            x2 = get_x_position(cluster2_id)
            x_center = (x1 + x2) / 2
            new_cluster_id = n_samples + i
            x_positions[new_cluster_id] = x_center

            y1 = 0.0 if cluster1_id < n_samples else self.linkage_matrix[
                cluster1_id - n_samples, 2
            ]
            y2 = 0.0 if cluster2_id < n_samples else self.linkage_matrix[
                cluster2_id - n_samples, 2
            ]

            ax.plot([x1, x1], [y1, distance], "b-", linewidth=1.5)
            ax.plot([x2, x2], [y2, distance], "b-", linewidth=1.5)
            ax.plot([x1, x2], [distance, distance], "b-", linewidth=1.5)

        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Distance", fontsize=12)
        title = f"Dendrogram ({self.linkage.capitalize()} Linkage)"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(range(n_samples))
        ax.set_xticklabels(range(n_samples))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Dendrogram saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical Clustering")
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
        help="Number of clusters (default: from config or None for full dendrogram)",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        choices=["single", "complete", "average"],
        default=None,
        help="Linkage method (default: from config)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot dendrogram",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save dendrogram plot",
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
            else model_config.get("n_clusters")
        )
        linkage = (
            args.linkage
            if args.linkage is not None
            else model_config.get("linkage", "average")
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Hierarchical Clustering ({linkage.capitalize()} Linkage) ===")
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

        model = HierarchicalClustering(n_clusters=n_clusters, linkage=linkage)
        model.fit(X)

        print(f"\n=== Clustering Results ===")
        print(f"Linkage method: {linkage}")
        print(f"Number of merges: {len(model.linkage_matrix)}")

        if n_clusters is not None:
            print(f"Number of clusters: {n_clusters}")
            unique, counts = np.unique(model.labels, return_counts=True)
            print(f"\nCluster sizes:")
            for cluster_id, count in zip(unique, counts):
                print(f"  Cluster {cluster_id}: {count} samples")

        if args.plot or args.save_plot:
            model.plot_dendrogram(save_path=args.save_plot, show=args.plot)

        if args.output and n_clusters is not None:
            output_df = pd.DataFrame(X, columns=feature_cols)
            output_df["cluster"] = model.labels
            output_df.to_csv(args.output, index=False)
            print(f"\nCluster assignments saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error running hierarchical clustering: {e}")
        raise


if __name__ == "__main__":
    main()
