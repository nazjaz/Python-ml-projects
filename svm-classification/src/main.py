"""Support Vector Machine (SVM) for Classification.

This module provides functionality to implement SVM from scratch with support
for linear, polynomial, and RBF kernels.
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


class SVM:
    """Support Vector Machine for classification."""

    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        degree: int = 3,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ) -> None:
        """Initialize SVM.

        Args:
            kernel: Kernel type. Options: "linear", "poly", "rbf" (default: "linear").
            C: Regularization parameter (default: 1.0).
            degree: Degree for polynomial kernel (default: 3).
            gamma: Kernel coefficient for RBF and polynomial. If None, uses 1/n_features (default: None).
            coef0: Independent term for polynomial kernel (default: 0.0).
            tol: Tolerance for stopping criterion (default: 1e-3).
            max_iter: Maximum number of iterations (default: 1000).
        """
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        self.support_vectors_: Optional[np.ndarray] = None
        self.support_vector_labels_: Optional[np.ndarray] = None
        self.support_vector_alphas_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute linear kernel.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Kernel matrix.
        """
        return X1 @ X2.T

    def _polynomial_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute polynomial kernel.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Kernel matrix.
        """
        return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF (Gaussian) kernel.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Kernel matrix.
        """
        pairwise_sq_dists = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + np.sum(
            X2 ** 2, axis=1
        ) - 2 * X1 @ X2.T
        return np.exp(-self.gamma * pairwise_sq_dists)

    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix based on kernel type.

        Args:
            X1: First feature matrix.
            X2: Second feature matrix.

        Returns:
            Kernel matrix.
        """
        if self.kernel == "linear":
            return self._linear_kernel(X1, X2)
        elif self.kernel == "poly":
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == "rbf":
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_bias(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alphas: np.ndarray,
        support_mask: np.ndarray,
    ) -> float:
        """Compute bias term.

        Args:
            X: Feature matrix.
            y: Target labels.
            alphas: Lagrange multipliers.
            support_mask: Boolean mask for support vectors.

        Returns:
            Bias term.
        """
        support_vectors = X[support_mask]
        support_labels = y[support_mask]
        support_alphas = alphas[support_mask]

        K = self._kernel_function(support_vectors, support_vectors)
        predictions = (support_alphas * support_labels) @ K

        bias = np.mean(support_labels - predictions)
        return bias

    def _smo_step(
        self,
        i: int,
        j: int,
        X: np.ndarray,
        y: np.ndarray,
        alphas: np.ndarray,
        bias: float,
    ) -> Tuple[float, float, float]:
        """Perform one SMO optimization step.

        Args:
            i: First index.
            j: Second index.
            X: Feature matrix.
            y: Target labels.
            alphas: Lagrange multipliers.
            bias: Current bias.

        Returns:
            Tuple of (new_alpha_i, new_alpha_j, new_bias).
        """
        if i == j:
            return alphas[i], alphas[j], bias

        K = self._kernel_function(X, X)
        Ei = (alphas * y) @ K[:, i] + bias - y[i]
        Ej = (alphas * y) @ K[:, j] + bias - y[j]

        old_alpha_i = alphas[i]
        old_alpha_j = alphas[j]

        if y[i] != y[j]:
            L = max(0, old_alpha_j - old_alpha_i)
            H = min(self.C, self.C + old_alpha_j - old_alpha_i)
        else:
            L = max(0, old_alpha_i + old_alpha_j - self.C)
            H = min(self.C, old_alpha_i + old_alpha_j)

        if L == H:
            return alphas[i], alphas[j], bias

        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        if eta <= 0:
            return alphas[i], alphas[j], bias

        new_alpha_j = old_alpha_j + y[j] * (Ei - Ej) / eta
        new_alpha_j = max(L, min(H, new_alpha_j))

        if abs(new_alpha_j - old_alpha_j) < 1e-5:
            return alphas[i], alphas[j], bias

        new_alpha_i = old_alpha_i + y[i] * y[j] * (old_alpha_j - new_alpha_j)

        new_bias_i = bias - Ei - y[i] * (new_alpha_i - old_alpha_i) * K[i, i] - y[
            j
        ] * (new_alpha_j - old_alpha_j) * K[i, j]
        new_bias_j = bias - Ej - y[i] * (new_alpha_i - old_alpha_i) * K[i, j] - y[
            j
        ] * (new_alpha_j - old_alpha_j) * K[j, j]

        if 0 < new_alpha_i < self.C:
            new_bias = new_bias_i
        elif 0 < new_alpha_j < self.C:
            new_bias = new_bias_j
        else:
            new_bias = (new_bias_i + new_bias_j) / 2

        return new_alpha_i, new_alpha_j, new_bias

    def fit(
        self, X: Union[List, np.ndarray, pd.DataFrame], y: Union[List, np.ndarray, pd.Series]
    ) -> "SVM":
        """Fit SVM model.

        Args:
            X: Feature matrix.
            y: Target labels (binary: -1, 1 or 0, 1).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM currently supports binary classification only")

        y_binary = y.copy()
        y_binary[y_binary == self.classes_[0]] = -1
        y_binary[y_binary == self.classes_[1]] = 1

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if self.gamma is None:
            self.gamma = 1.0 / n_features

        alphas = np.zeros(n_samples)
        bias = 0.0

        K = self._kernel_function(X, X)

        for iteration in range(self.max_iter):
            alpha_prev = alphas.copy()

            for i in range(n_samples):
                Ei = (alphas * y_binary) @ K[:, i] + bias - y_binary[i]

                if (
                    (y_binary[i] * Ei < -self.tol and alphas[i] < self.C)
                    or (y_binary[i] * Ei > self.tol and alphas[i] > 0)
                ):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    alphas[i], alphas[j], bias = self._smo_step(
                        i, j, X, y_binary, alphas, bias
                    )

            if np.linalg.norm(alphas - alpha_prev) < self.tol:
                break

        support_mask = alphas > 1e-5
        self.support_vectors_ = X[support_mask]
        self.support_vector_labels_ = y_binary[support_mask]
        self.support_vector_alphas_ = alphas[support_mask]

        self.bias_ = self._compute_bias(X, y_binary, alphas, support_mask)
        self.X_train_ = X
        self.y_train_ = y_binary

        logger.info(
            f"SVM fitted: kernel={self.kernel}, C={self.C}, "
            f"support_vectors={len(self.support_vectors_)}"
        )

        return self

    def _decision_function(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compute decision function values.

        Args:
            X: Feature matrix.

        Returns:
            Decision function values.
        """
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        K = self._kernel_function(self.support_vectors_, X)
        decisions = (self.support_vector_alphas_ * self.support_vector_labels_) @ K + self.bias_

        return decisions

    def predict(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        decisions = self._decision_function(X)
        predictions = np.where(decisions >= 0, self.classes_[1], self.classes_[0])
        return predictions

    def predict_proba(self, X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities (using decision function).

        Args:
            X: Feature matrix.

        Returns:
            Class probabilities.

        Raises:
            ValueError: If model not fitted.
        """
        decisions = self._decision_function(X)
        proba = 1 / (1 + np.exp(-decisions))
        proba = np.clip(proba, 0, 1)
        proba = np.column_stack([1 - proba, proba])
        return proba

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

    def plot_decision_boundary(
        self,
        X: Optional[Union[List, np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[List, np.ndarray, pd.Series]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot decision boundary (for 2D features only).

        Args:
            X: Optional feature matrix for visualization.
            y: Optional target labels for visualization.
            save_path: Optional path to save figure.
            show: Whether to display plot.
        """
        if self.support_vectors_ is None:
            logger.warning("Model must be fitted before plotting")
            return

        if self.n_features_ != 2:
            logger.warning("Decision boundary plotting only supported for 2D features")
            return

        if X is None:
            X = self.X_train_
            y = self.y_train_

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="RdYlBu",
            edgecolors="k",
            linewidth=1,
            s=50,
        )

        if self.support_vectors_ is not None:
            ax.scatter(
                self.support_vectors_[:, 0],
                self.support_vectors_[:, 1],
                s=200,
                facecolors="none",
                edgecolors="k",
                linewidth=2,
                label="Support Vectors",
            )

        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        ax.set_title(
            f"SVM Decision Boundary (kernel={self.kernel}, C={self.C})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Decision boundary plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Support Vector Machine")
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
        "--kernel",
        type=str,
        default=None,
        choices=["linear", "poly", "rbf"],
        help="Kernel type (default: from config)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=None,
        help="Regularization parameter (default: from config)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Degree for polynomial kernel (default: from config)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Kernel coefficient (default: from config or auto)",
    )
    parser.add_argument(
        "--coef0",
        type=float,
        default=None,
        help="Independent term for polynomial kernel (default: from config)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot decision boundary (2D features only)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save decision boundary plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions as CSV",
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

        kernel = (
            args.kernel
            if args.kernel is not None
            else model_config.get("kernel", "linear")
        )
        C = (
            args.C if args.C is not None else model_config.get("C", 1.0)
        )
        degree = (
            args.degree
            if args.degree is not None
            else model_config.get("degree", 3)
        )
        gamma = (
            args.gamma
            if args.gamma is not None
            else model_config.get("gamma")
        )
        coef0 = (
            args.coef0
            if args.coef0 is not None
            else model_config.get("coef0", 0.0)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Support Vector Machine ===")
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

        print(f"\nFitting SVM...")
        print(f"Kernel: {kernel}")
        print(f"C: {C}")
        if kernel == "poly":
            print(f"Degree: {degree}")
            print(f"Coef0: {coef0}")
        if kernel in ["poly", "rbf"]:
            print(f"Gamma: {gamma if gamma else 'auto (1/n_features)'}")

        svm = SVM(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0)
        svm.fit(X, y)

        print(f"\n=== SVM Results ===")
        print(f"Support vectors: {len(svm.support_vectors_)}")
        print(f"Classes: {svm.classes_}")

        accuracy = svm.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.6f}")

        if args.plot or args.save_plot:
            if X.shape[1] == 2:
                svm.plot_decision_boundary(
                    X=X,
                    y=y,
                    save_path=args.save_plot,
                    show=args.plot,
                )
            else:
                print("\nDecision boundary plotting only supported for 2D features")

        if args.output:
            predictions = svm.predict(X)
            proba = svm.predict_proba(X)
            output_df = pd.DataFrame({
                "prediction": predictions,
                args.target: y,
                "prob_class_0": proba[:, 0],
                "prob_class_1": proba[:, 1],
            })
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

        if args.predict:
            df_predict = pd.read_csv(args.predict)
            X_predict = df_predict[feature_cols].values
            predictions = svm.predict(X_predict)
            proba = svm.predict_proba(X_predict)

            if args.output_predictions:
                output_df = pd.DataFrame({
                    "prediction": predictions,
                    "prob_class_0": proba[:, 0],
                    "prob_class_1": proba[:, 1],
                })
                output_df.to_csv(args.output_predictions, index=False)
                print(f"Predictions saved to: {args.output_predictions}")
            else:
                print(f"\nPredictions:")
                for i, pred in enumerate(predictions[:10]):
                    print(f"  Sample {i+1}: {pred}")
                if len(predictions) > 10:
                    print(f"  ... and {len(predictions) - 10} more")

    except Exception as e:
        logger.error(f"Error running SVM: {e}")
        raise


if __name__ == "__main__":
    main()
