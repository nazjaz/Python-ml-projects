"""Linear Discriminant Analysis (LDA) for Dimensionality Reduction and Classification.

This module provides functionality to implement LDA from scratch for
dimensionality reduction and classification with class separation analysis.
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


class LDA:
    """Linear Discriminant Analysis for dimensionality reduction and classification."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        solver: str = "eigen",
        shrinkage: Optional[float] = None,
    ) -> None:
        """Initialize LDA.

        Args:
            n_components: Number of components to keep. If None, keeps
                min(n_features, n_classes - 1) components.
            solver: Solver to use. Options: "eigen" (default), "svd".
            shrinkage: Shrinkage parameter for regularization (0-1).
                If None, no shrinkage is applied.
        """
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage

        self.scalings: Optional[np.ndarray] = None
        self.xbar_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def _compute_scatter_matrices(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute within-class and between-class scatter matrices.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Tuple of (within-class scatter, between-class scatter).
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        overall_mean = np.mean(X, axis=0)
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            n_c = len(X_c)
            mean_c = np.mean(X_c, axis=0)

            X_c_centered = X_c - mean_c
            Sw += X_c_centered.T @ X_c_centered

            mean_diff = mean_c - overall_mean
            Sb += n_c * np.outer(mean_diff, mean_diff)

        Sw /= n_samples
        return Sw, Sb

    def _solve_eigen(
        self, Sw: np.ndarray, Sb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve LDA using eigenvalue decomposition.

        Args:
            Sw: Within-class scatter matrix.
            Sb: Between-class scatter matrix.

        Returns:
            Tuple of (eigenvalues, eigenvectors).
        """
        if self.shrinkage is not None:
            if not 0 <= self.shrinkage <= 1:
                raise ValueError("shrinkage must be between 0 and 1")
            n_samples = Sw.shape[0]
            Sw = (1 - self.shrinkage) * Sw + self.shrinkage * np.eye(n_samples) * np.trace(Sw) / n_samples

        try:
            Sw_inv = np.linalg.inv(Sw)
        except np.linalg.LinAlgError:
            logger.warning("Sw is singular, using pseudo-inverse")
            Sw_inv = np.linalg.pinv(Sw)

        matrix = Sw_inv @ Sb
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def _solve_svd(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve LDA using SVD.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Tuple of (singular values, right singular vectors).
        """
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape

        overall_mean = np.mean(X, axis=0)
        X_centered = X - overall_mean

        means = []
        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            means.append(mean_c)

        means = np.array(means)
        means_centered = means - overall_mean

        priors = np.array([np.sum(y == c) / n_samples for c in classes])
        scaling = np.sqrt(priors[:, np.newaxis])

        X_star = X_centered / np.sqrt(n_samples)
        means_star = scaling * means_centered

        X_combined = np.vstack([X_star, means_star])
        _, s, Vt = np.linalg.svd(X_combined, full_matrices=False)

        return s, Vt.T

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "LDA":
        """Fit LDA model.

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

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("LDA requires at least 2 classes")

        max_components = min(n_features, n_classes - 1)
        if self.n_components is None:
            n_components = max_components
        else:
            n_components = min(self.n_components, max_components)

        self.n_components_ = n_components

        self.xbar_ = np.mean(X, axis=0)
        self.priors_ = np.array([np.sum(y == c) / n_samples for c in self.classes_])

        self.means_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes_])

        if self.solver == "eigen":
            Sw, Sb = self._compute_scatter_matrices(X, y)
            eigenvalues, eigenvectors = self._solve_eigen(Sw, Sb)

            self.scalings = eigenvectors[:, :n_components]
            self.explained_variance_ratio_ = eigenvalues[:n_components] / np.sum(eigenvalues[:max_components])

            self.covariance_ = Sw

        elif self.solver == "svd":
            s, Vt = self._solve_svd(X, y)
            self.scalings = Vt[:, :n_components]
            self.explained_variance_ratio_ = s[:n_components] ** 2 / np.sum(s[:max_components] ** 2)

        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        logger.info(
            f"LDA fitted: {n_features} features -> {n_components} components, "
            f"{n_classes} classes"
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
        if self.scalings is None:
            raise ValueError("Model must be fitted before transformation")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_centered = X - self.xbar_
        X_transformed = X_centered @ self.scalings

        return X_transformed

    def fit_transform(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> np.ndarray:
        """Fit model and transform data.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        if self.scalings is None or self.means_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_transformed = self.transform(X)
        means_transformed = self.transform(self.means_)

        distances = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            diff = X_transformed - means_transformed[i]
            distances[:, i] = np.sum(diff ** 2, axis=1)

        predictions = self.classes_[np.argmin(distances, axis=1)]
        return predictions

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities for each sample.

        Raises:
            ValueError: If model not fitted.
        """
        if self.scalings is None or self.means_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_transformed = self.transform(X)
        means_transformed = self.transform(self.means_)

        if self.covariance_ is None:
            Sw, _ = self._compute_scatter_matrices(X, X)
            covariance_transformed = self.scalings.T @ Sw @ self.scalings
        else:
            covariance_transformed = self.scalings.T @ self.covariance_ @ self.scalings

        try:
            cov_inv = np.linalg.inv(covariance_transformed)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(covariance_transformed)

        log_probs = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            diff = X_transformed - means_transformed[i]
            log_probs[:, i] = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            log_probs[:, i] += np.log(self.priors_[i] + 1e-10)

        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)

        return probs

    def score(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> float:
        """Calculate classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Classification accuracy.
        """
        predictions = self.predict(X)
        y = np.asarray(y)
        accuracy = np.mean(predictions == y)
        return accuracy

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component.

        Returns:
            Explained variance ratio array.

        Raises:
            ValueError: If model not fitted.
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError(
                "Model must be fitted before getting explained variance ratio"
            )
        return self.explained_variance_ratio_.copy()

    def plot_components(
        self,
        X: Optional[Union[List, np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot LDA components with data visualization.

        Args:
            X: Optional feature matrix for visualization.
            y: Optional target labels for visualization.
            save_path: Optional path to save figure.
            show: Whether to display plot.
        """
        if self.scalings is None:
            logger.warning("Model must be fitted before plotting components")
            return

        if X is not None and y is not None:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_transformed = self.transform(X)

            if self.n_components_ >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    X_transformed[:, 0],
                    X_transformed[:, 1],
                    c=y,
                    cmap="viridis",
                    alpha=0.6,
                    edgecolors="k",
                    linewidth=0.5,
                )
                ax.set_xlabel("LD1", fontsize=12)
                ax.set_ylabel("LD2", fontsize=12)
                ax.set_title("LDA: First Two Components", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label="Class")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(
                    X_transformed[:, 0],
                    np.zeros(len(X_transformed)),
                    c=y,
                    cmap="viridis",
                    alpha=0.6,
                    edgecolors="k",
                    linewidth=0.5,
                )
                ax.set_xlabel("LD1", fontsize=12)
                ax.set_ylabel("", fontsize=12)
                ax.set_title("LDA: First Component", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="y")
        else:
            if self.n_components_ >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.set_xlabel("LD1", fontsize=12)
                ax.set_ylabel("LD2", fontsize=12)
                ax.set_title("LDA Components", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_xlabel("LD1", fontsize=12)
                ax.set_title("LDA Component", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)

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

    parser = argparse.ArgumentParser(description="Linear Discriminant Analysis")
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
        "--n-components",
        type=int,
        default=None,
        help="Number of components (default: from config)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=["eigen", "svd"],
        help="Solver to use (default: from config)",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=None,
        help="Shrinkage parameter (0-1) (default: from config)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot LDA components",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save components plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save transformed data as CSV",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to CSV file for prediction",
    )
    parser.add_argument(
        "--output-predictions",
        type=str,
        default=None,
        help="Path to save predictions as CSV",
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
        solver = (
            args.solver
            if args.solver is not None
            else model_config.get("solver", "eigen")
        )
        shrinkage = (
            args.shrinkage
            if args.shrinkage is not None
            else model_config.get("shrinkage")
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Linear Discriminant Analysis ===")
        print(f"Data shape: {df.shape}")

        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in data")

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
        print(f"Classes: {len(np.unique(y))}")

        print(f"\nFitting LDA...")
        if n_components:
            print(f"n_components: {n_components}")
        else:
            print("n_components: Auto (min(n_features, n_classes - 1))")
        print(f"Solver: {solver}")
        if shrinkage:
            print(f"Shrinkage: {shrinkage}")

        lda = LDA(n_components=n_components, solver=solver, shrinkage=shrinkage)
        lda.fit(X, y)

        print(f"\n=== LDA Results ===")
        print(f"Number of components: {lda.n_components_}")
        print(f"Classes: {lda.classes_}")

        if lda.explained_variance_ratio_ is not None:
            print(f"\nExplained variance ratio by component:")
            for i, ratio in enumerate(lda.explained_variance_ratio_):
                print(f"  LD{i+1}: {ratio:.6f}")

        X_transformed = lda.transform(X)
        print(f"\nTransformed data shape: {X_transformed.shape}")

        accuracy = lda.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.6f}")

        if args.plot or args.save_plot:
            lda.plot_components(
                X=X,
                y=y,
                save_path=args.save_plot,
                show=args.plot,
            )

        if args.output:
            output_df = pd.DataFrame(
                X_transformed,
                columns=[f"LD{i+1}" for i in range(lda.n_components_)],
            )
            output_df[args.target] = y
            output_df.to_csv(args.output, index=False)
            print(f"\nTransformed data saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = lda.predict(X_predict)
            proba = lda.predict_proba(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame(
                    {
                        "prediction": predictions,
                    }
                )
                for i, c in enumerate(lda.classes_):
                    output_df[f"prob_class_{c}"] = proba[:, i]
                output_df.to_csv(args.output_predictions, index=False)
                print(f"Predictions saved to: {args.output_predictions}")
            else:
                print(f"\nPredictions:")
                for i, pred in enumerate(predictions[:10]):
                    print(f"  Sample {i+1}: {pred}")
                if len(predictions) > 10:
                    print(f"  ... and {len(predictions) - 10} more")

    except Exception as e:
        logger.error(f"Error running LDA: {e}")
        raise


if __name__ == "__main__":
    main()
