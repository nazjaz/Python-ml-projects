"""Principal Component Analysis (PCA) for Dimensionality Reduction.

This module provides functionality to implement PCA from scratch for
dimensionality reduction with explained variance analysis.
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


class PCA:
    """Principal Component Analysis for dimensionality reduction."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
    ) -> None:
        """Initialize PCA.

        Args:
            n_components: Number of components to keep. If None, keeps all
                components. If int, keeps top n_components. If float between
                0 and 1, keeps components that explain at least that variance.
            whiten: Whether to whiten the components (default: False).
        """
        self.n_components = n_components
        self.whiten = whiten

        self.components: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

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

    def _center(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Center features (zero mean).

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (centered_X, mean).
        """
        mean = np.mean(X, axis=0)
        centered_X = X - mean
        return centered_X, mean

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> "PCA":
        """Fit PCA model.

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
            raise ValueError("Need at least 2 samples for PCA")

        n_samples, n_features = X.shape

        X_centered, self.mean = self._center(X)

        covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if self.n_components is None:
            n_components = n_features
        elif isinstance(self.n_components, float):
            if not 0 < self.n_components <= 1:
                raise ValueError(
                    "n_components must be between 0 and 1 when float"
                )
            cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            n_components = np.searchsorted(cumulative_variance, self.n_components) + 1
        else:
            n_components = min(self.n_components, n_features)

        self.n_components_ = n_components
        self.components = eigenvectors[:, :n_components].T

        if self.whiten:
            self.components = self.components / np.sqrt(eigenvalues[:n_components])[:, np.newaxis]

        self.explained_variance = eigenvalues[:n_components]
        self.explained_variance_ratio = (
            self.explained_variance / np.sum(eigenvalues)
        )

        logger.info(
            f"PCA fitted: {n_features} features -> {n_components} components, "
            f"explained variance: {np.sum(self.explained_variance_ratio):.4f}"
        )

        return self

    def transform(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Transform data to lower-dimensional space.

        Args:
            X: Feature matrix.

        Returns:
            Transformed data.

        Raises:
            ValueError: If model not fitted.
        """
        if self.components is None:
            raise ValueError("Model must be fitted before transformation")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_centered = X - self.mean
        X_transformed = X_centered @ self.components.T

        return X_transformed

    def fit_transform(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Fit model and transform data.

        Args:
            X: Feature matrix.

        Returns:
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(
        self, X_transformed: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Transform data back to original space.

        Args:
            X_transformed: Transformed feature matrix.

        Returns:
            Data in original space.

        Raises:
            ValueError: If model not fitted.
        """
        if self.components is None:
            raise ValueError("Model must be fitted before inverse transformation")

        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)

        if self.whiten:
            X_reconstructed = X_transformed @ (
                self.components * np.sqrt(self.explained_variance)[:, np.newaxis]
            )
        else:
            X_reconstructed = X_transformed @ self.components

        X_original = X_reconstructed + self.mean

        return X_original

    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance for each component.

        Returns:
            Explained variance array.

        Raises:
            ValueError: If model not fitted.
        """
        if self.explained_variance is None:
            raise ValueError("Model must be fitted before getting explained variance")
        return self.explained_variance.copy()

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component.

        Returns:
            Explained variance ratio array.

        Raises:
            ValueError: If model not fitted.
        """
        if self.explained_variance_ratio is None:
            raise ValueError(
                "Model must be fitted before getting explained variance ratio"
            )
        return self.explained_variance_ratio.copy()

    def get_cumulative_variance(self) -> np.ndarray:
        """Get cumulative explained variance ratio.

        Returns:
            Cumulative explained variance ratio array.

        Raises:
            ValueError: If model not fitted.
        """
        if self.explained_variance_ratio is None:
            raise ValueError(
                "Model must be fitted before getting cumulative variance"
            )
        return np.cumsum(self.explained_variance_ratio)

    def plot_explained_variance(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        cumulative: bool = False,
    ) -> None:
        """Plot explained variance.

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            cumulative: Whether to plot cumulative variance (default: False).
        """
        if self.explained_variance_ratio is None:
            logger.warning("No explained variance data available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if cumulative:
            variance_data = self.get_cumulative_variance()
            ylabel = "Cumulative Explained Variance Ratio"
            title = "Cumulative Explained Variance"
        else:
            variance_data = self.explained_variance_ratio
            ylabel = "Explained Variance Ratio"
            title = "Explained Variance by Component"

        components = range(1, len(variance_data) + 1)
        ax.bar(components, variance_data, alpha=0.7, color="steelblue")
        ax.plot(components, variance_data, "ro-", linewidth=2, markersize=8)

        ax.set_xlabel("Principal Component", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(components)

        if cumulative:
            ax.axhline(y=0.95, color="r", linestyle="--", label="95% Variance", alpha=0.7)
            ax.axhline(y=0.99, color="g", linestyle="--", label="99% Variance", alpha=0.7)
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Explained variance plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_components(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        n_components: Optional[int] = None,
    ) -> None:
        """Plot principal components (for 2D original data).

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
            n_components: Number of components to plot (default: all).
        """
        if self.components is None or self.mean is None:
            logger.warning("Model must be fitted before plotting components")
            return

        if self.components.shape[1] != 2:
            logger.warning("Component plotting only supported for 2D original data")
            return

        n_plot = n_components if n_components is not None else self.n_components_
        n_plot = min(n_plot, self.n_components_)

        fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 5))
        if n_plot == 1:
            axes = [axes]

        for i in range(n_plot):
            ax = axes[i]
            component = self.components[i]

            ax.arrow(
                0,
                0,
                component[0],
                component[1],
                head_width=0.1,
                head_length=0.1,
                fc="red",
                ec="red",
                linewidth=2,
            )
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_xlabel("Feature 1", fontsize=10)
            ax.set_ylabel("Feature 2", fontsize=10)
            ax.set_title(
                f"PC{i+1} (Var: {self.explained_variance_ratio[i]:.3f})",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
            ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Components plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Principal Component Analysis")
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
        "--n-components",
        type=str,
        default=None,
        help="Number of components (int) or variance to retain (float 0-1) (default: from config)",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="Whiten components",
    )
    parser.add_argument(
        "--plot-variance",
        action="store_true",
        help="Plot explained variance",
    )
    parser.add_argument(
        "--plot-cumulative",
        action="store_true",
        help="Plot cumulative explained variance",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save variance plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save transformed data as CSV",
    )
    parser.add_argument(
        "--output-original",
        type=str,
        default=None,
        help="Path to save reconstructed original data as CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        n_components = (
            args.n_components
            if args.n_components is not None
            else model_config.get("n_components")
        )

        if n_components is not None:
            try:
                n_components = int(n_components)
            except ValueError:
                try:
                    n_components = float(n_components)
                except ValueError:
                    raise ValueError(
                        "n_components must be int or float between 0 and 1"
                    )

        whiten = (
            args.whiten
            if args.whiten
            else model_config.get("whiten", False)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Principal Component Analysis ===")
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

        print(f"\nFitting PCA...")
        if n_components:
            print(f"n_components: {n_components}")
        else:
            print("n_components: All components")
        print(f"Whiten: {whiten}")

        pca = PCA(n_components=n_components, whiten=whiten)
        pca.fit(X)

        print(f"\n=== PCA Results ===")
        print(f"Number of components: {pca.n_components_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio):.6f}")

        print(f"\nExplained variance by component:")
        for i, (var, ratio) in enumerate(
            zip(pca.explained_variance, pca.explained_variance_ratio)
        ):
            print(f"  PC{i+1}: variance={var:.6f}, ratio={ratio:.6f}")

        cumulative = pca.get_cumulative_variance()
        print(f"\nCumulative explained variance:")
        for i, cum_var in enumerate(cumulative):
            print(f"  PC{i+1}: {cum_var:.6f}")

        X_transformed = pca.transform(X)
        print(f"\nTransformed data shape: {X_transformed.shape}")

        if args.plot_variance or args.plot_cumulative or args.save_plot:
            pca.plot_explained_variance(
                save_path=args.save_plot,
                show=args.plot_variance or args.plot_cumulative,
                cumulative=args.plot_cumulative,
            )

        if args.output:
            output_df = pd.DataFrame(
                X_transformed,
                columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            )
            output_df.to_csv(args.output, index=False)
            print(f"\nTransformed data saved to: {args.output}")

        if args.output_original:
            X_reconstructed = pca.inverse_transform(X_transformed)
            output_df = pd.DataFrame(X_reconstructed, columns=feature_cols)
            output_df.to_csv(args.output_original, index=False)
            print(f"Reconstructed data saved to: {args.output_original}")

    except Exception as e:
        logger.error(f"Error running PCA: {e}")
        raise


if __name__ == "__main__":
    main()
