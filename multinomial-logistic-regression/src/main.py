"""Multinomial Logistic Regression for Multi-Class Classification.

This module provides functionality to implement multinomial logistic
regression from scratch for multi-class classification with softmax
activation and cost function optimization.
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


class MultinomialLogisticRegression:
    """Multinomial logistic regression for multi-class classification."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        scale_features: bool = True,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize MultinomialLogisticRegression.

        Args:
            learning_rate: Initial learning rate.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.
            scale_features: Whether to scale features.
            fit_intercept: Whether to fit intercept term.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.scale_features = scale_features
        self.fit_intercept = fit_intercept

        self.weights: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None
        self.scale_params: Optional[Dict] = None
        self.classes: Optional[np.ndarray] = None
        self.cost_history: List[float] = []

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute softmax activation function.

        Args:
            z: Input values (shape: [n_samples, n_classes]).

        Returns:
            Softmax output (probabilities for each class, shape: [n_samples, n_classes]).

        Example:
            >>> model = MultinomialLogisticRegression()
            >>> z = np.array([[1, 2, 3], [1, 1, 1]])
            >>> probs = model._softmax(z)
            >>> np.sum(probs, axis=1)
            array([1., 1.])
        """
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term to features.

        Args:
            X: Feature matrix.

        Returns:
            Feature matrix with intercept column.
        """
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate([intercept, X], axis=1)
        return X

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

    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """Convert class labels to one-hot encoding.

        Args:
            y: Class labels.

        Returns:
            One-hot encoded matrix (shape: [n_samples, n_classes]).
        """
        n_classes = len(self.classes)
        n_samples = len(y)
        one_hot = np.zeros((n_samples, n_classes))

        for i, cls in enumerate(self.classes):
            one_hot[y == cls, i] = 1

        return one_hot

    def _compute_cost(
        self, X: np.ndarray, y_one_hot: np.ndarray, theta: np.ndarray
    ) -> float:
        """Compute cross-entropy cost function.

        Args:
            X: Feature matrix (with intercept if applicable).
            y_one_hot: One-hot encoded target labels.
            theta: Model parameters (shape: [n_features, n_classes]).

        Returns:
            Cross-entropy cost.
        """
        m = len(y_one_hot)
        z = X @ theta
        h = self._softmax(z)

        cost = -(1 / m) * np.sum(y_one_hot * np.log(h + 1e-15))
        return float(cost)

    def _compute_gradient(
        self, X: np.ndarray, y_one_hot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of cross-entropy cost function.

        Args:
            X: Feature matrix (with intercept if applicable).
            y_one_hot: One-hot encoded target labels.
            theta: Model parameters (shape: [n_features, n_classes]).

        Returns:
            Gradient matrix (shape: [n_features, n_classes]).
        """
        m = len(y_one_hot)
        z = X @ theta
        h = self._softmax(z)

        gradient = (1 / m) * (X.T @ (h - y_one_hot))
        return gradient

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> "MultinomialLogisticRegression":
        """Fit multinomial logistic regression model using gradient descent.

        Args:
            X: Feature matrix.
            y: Target class labels.

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

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        if n_classes < 2:
            raise ValueError("At least 2 classes required for classification")

        original_X = X.copy()

        if self.scale_features:
            X, mean, std = self._standardize(X)
            self.scale_params = {"mean": mean, "std": std}
            logger.info("Features standardized")

        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]

        y_one_hot = self._one_hot_encode(y)

        theta = np.zeros((n_features, n_classes))
        self.cost_history = []

        logger.info(
            f"Starting gradient descent: {self.max_iterations} "
            f"iterations, LR={self.learning_rate}, "
            f"Classes={n_classes}"
        )

        for iteration in range(self.max_iterations):
            gradient = self._compute_gradient(X_with_intercept, y_one_hot, theta)
            theta = theta - self.learning_rate * gradient

            cost = self._compute_cost(X_with_intercept, y_one_hot, theta)
            self.cost_history.append(cost)

            if iteration > 0:
                cost_change = abs(self.cost_history[-2] - cost)
                if cost_change < self.tolerance:
                    logger.info(
                        f"Converged at iteration {iteration + 1} "
                        f"with cost={cost:.6f}"
                    )
                    break

            if (iteration + 1) % 100 == 0:
                logger.debug(f"Iteration {iteration + 1}: cost={cost:.6f}")

        if self.fit_intercept:
            self.intercept = theta[0, :]
            self.weights = theta[1:, :]
        else:
            self.intercept = np.zeros(n_classes)
            self.weights = theta

        logger.info(
            f"Training complete: final cost={self.cost_history[-1]:.6f}, "
            f"iterations={len(self.cost_history)}"
        )

        return self

    def predict_proba(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix (shape: [n_samples, n_classes]).

        Raises:
            ValueError: If model not fitted.
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.scale_features and self.scale_params:
            X = self._apply_standardization(
                X, self.scale_params["mean"], self.scale_params["std"]
            )

        X_with_intercept = self._add_intercept(X)
        theta = np.vstack([self.intercept.reshape(1, -1), self.weights])

        z = X_with_intercept @ theta
        probabilities = self._softmax(z)
        return probabilities

    def predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model not fitted.
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes[class_indices]
        return predictions

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

    def get_cost_history(self) -> List[float]:
        """Get cost history from training.

        Returns:
            List of cost values per iteration.
        """
        return self.cost_history.copy()

    def plot_training_history(
        self, save_path: Optional[str] = None, show: bool = True
    ) -> None:
        """Plot training history (cost).

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
        """
        if not self.cost_history:
            logger.warning("No training history available")
            return

        plt.figure(figsize=(10, 6))
        iterations = range(len(self.cost_history))
        plt.plot(iterations, self.cost_history, "b-", linewidth=2)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Cost (Cross-Entropy)", fontsize=12)
        plt.title(
            "Training History - Multinomial Logistic Regression",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Training history plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multinomial Logistic Regression for Multi-Class Classification"
    )
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
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations (default: from config)",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Don't scale features",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training history",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save training history plot",
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
        lr = (
            args.learning_rate
            if args.learning_rate is not None
            else model_config.get("learning_rate", 0.01)
        )
        max_iter = (
            args.max_iterations
            if args.max_iterations is not None
            else model_config.get("max_iterations", 1000)
        )
        scale_features = (
            not args.no_scale
            if args.no_scale
            else model_config.get("scale_features", True)
        )

        df = pd.read_csv(args.input)
        print(f"\n=== Multinomial Logistic Regression ===")
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
        print(f"Class distribution: {np.bincount([np.where(unique_classes == cls)[0][0] for cls in y])}")

        model = MultinomialLogisticRegression(
            learning_rate=lr,
            max_iterations=max_iter,
            scale_features=scale_features,
        )

        print(f"\nTraining model...")
        print(f"Learning rate: {lr}")
        print(f"Max iterations: {max_iter}")
        print(f"Feature scaling: {scale_features}")

        model.fit(X, y)

        print(f"\n=== Training Results ===")
        print(f"Final cost: {model.cost_history[-1]:.6f}")
        print(f"Iterations: {len(model.cost_history)}")
        print(f"Accuracy: {model.score(X, y):.6f}")

        print(f"\nIntercept (per class):")
        for i, cls in enumerate(model.classes):
            print(f"  Class {cls}: {model.intercept[i]:.6f}")

        print(f"\nWeights (per class):")
        for i, cls in enumerate(model.classes):
            print(f"  Class {cls}:")
            for j, feature in enumerate(feature_cols):
                print(f"    {feature}: {model.weights[j, i]:.6f}")

        if args.plot or args.save_plot:
            model.plot_training_history(
                save_path=args.save_plot, show=args.plot
            )

        if args.output:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            output_df = pd.DataFrame(
                {
                    "actual": y,
                    "predicted": predictions,
                }
            )
            for i, cls in enumerate(model.classes):
                output_df[f"prob_class_{cls}"] = probabilities[:, i]
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


if __name__ == "__main__":
    main()
