"""Multiple Linear Regression with Feature Scaling and Regularization.

This module provides functionality to implement multiple linear regression
from scratch with feature scaling and regularization (Ridge and Lasso).
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


class FeatureScaler:
    """Feature scaling utilities."""

    @staticmethod
    def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    @staticmethod
    def normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features to [0, 1] range.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (scaled_X, min, max).
        """
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        scaled_X = (X - min_val) / range_val
        return scaled_X, min_val, max_val

    @staticmethod
    def apply_standardization(
        X: np.ndarray, mean: np.ndarray, std: np.ndarray
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

    @staticmethod
    def apply_normalization(
        X: np.ndarray, min_val: np.ndarray, max_val: np.ndarray
    ) -> np.ndarray:
        """Apply normalization using pre-computed min and max.

        Args:
            X: Feature matrix.
            min_val: Minimum values.
            max_val: Maximum values.

        Returns:
            Scaled feature matrix.
        """
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        return (X - min_val) / range_val


class MultipleLinearRegression:
    """Multiple linear regression with feature scaling and regularization."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        regularization: Optional[str] = None,
        alpha: float = 0.1,
        scale_features: bool = True,
        scaling_method: str = "standardize",
        fit_intercept: bool = True,
    ) -> None:
        """Initialize MultipleLinearRegression.

        Args:
            learning_rate: Initial learning rate.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.
            regularization: Regularization type. Options: None, 'ridge', 'lasso'.
            alpha: Regularization strength.
            scale_features: Whether to scale features.
            scaling_method: Scaling method. Options: 'standardize', 'normalize'.
            fit_intercept: Whether to fit intercept term.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.alpha = alpha
        self.scale_features = scale_features
        self.scaling_method = scaling_method
        self.fit_intercept = fit_intercept

        self.weights: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.scale_params: Optional[Dict] = None
        self.cost_history: List[float] = []

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

    def _compute_cost(
        self, X: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> float:
        """Compute cost function with optional regularization.

        Args:
            X: Feature matrix (with intercept if applicable).
            y: Target values.
            theta: Model parameters.

        Returns:
            Cost value.
        """
        m = len(y)
        predictions = X @ theta
        mse = np.mean((predictions - y) ** 2)

        if self.regularization == "ridge":
            regularization_term = self.alpha * np.sum(theta[1:] ** 2)
            cost = mse + regularization_term
        elif self.regularization == "lasso":
            regularization_term = self.alpha * np.sum(np.abs(theta[1:]))
            cost = mse + regularization_term
        else:
            cost = mse

        return float(cost)

    def _compute_gradient(
        self, X: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute gradient with optional regularization.

        Args:
            X: Feature matrix (with intercept if applicable).
            y: Target values.
            theta: Model parameters.

        Returns:
            Gradient vector.
        """
        m = len(y)
        predictions = X @ theta
        error = predictions - y
        gradient = (1 / m) * (X.T @ error)

        if self.regularization == "ridge":
            regularization_gradient = np.zeros_like(theta)
            regularization_gradient[1:] = 2 * self.alpha * theta[1:]
            gradient += regularization_gradient
        elif self.regularization == "lasso":
            regularization_gradient = np.zeros_like(theta)
            regularization_gradient[1:] = self.alpha * np.sign(theta[1:])
            gradient += regularization_gradient

        return gradient

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> "MultipleLinearRegression":
        """Fit multiple linear regression model using gradient descent.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If inputs are invalid.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X has {len(X)} samples, "
                f"y has {len(y)} samples"
            )

        if len(X) == 0:
            raise ValueError("Input data cannot be empty")

        original_X = X.copy()

        if self.scale_features:
            if self.scaling_method == "standardize":
                X, mean, std = FeatureScaler.standardize(X)
                self.scale_params = {"method": "standardize", "mean": mean, "std": std}
            elif self.scaling_method == "normalize":
                X, min_val, max_val = FeatureScaler.normalize(X)
                self.scale_params = {
                    "method": "normalize",
                    "min": min_val,
                    "max": max_val,
                }
            else:
                raise ValueError(
                    f"Unknown scaling method: {self.scaling_method}"
                )
            logger.info(f"Features scaled using {self.scaling_method}")
        else:
            self.scale_params = None

        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]

        theta = np.zeros(n_features)
        self.cost_history = []

        reg_str = f" ({self.regularization}, alpha={self.alpha})" if self.regularization else ""
        logger.info(
            f"Starting gradient descent: {self.max_iterations} "
            f"iterations, LR={self.learning_rate}{reg_str}"
        )

        for iteration in range(self.max_iterations):
            gradient = self._compute_gradient(X_with_intercept, y, theta)
            theta = theta - self.learning_rate * gradient

            cost = self._compute_cost(X_with_intercept, y, theta)
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
                logger.debug(
                    f"Iteration {iteration + 1}: cost={cost:.6f}"
                )

        if self.fit_intercept:
            self.intercept = float(theta[0])
            self.weights = theta[1:]
        else:
            self.intercept = 0.0
            self.weights = theta

        logger.info(
            f"Training complete: final cost={self.cost_history[-1]:.6f}, "
            f"iterations={len(self.cost_history)}"
        )

        return self

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using stored parameters.

        Args:
            X: Feature matrix.

        Returns:
            Scaled feature matrix.
        """
        if not self.scale_features or self.scale_params is None:
            return X

        method = self.scale_params["method"]
        if method == "standardize":
            return FeatureScaler.apply_standardization(
                X, self.scale_params["mean"], self.scale_params["std"]
            )
        elif method == "normalize":
            return FeatureScaler.apply_normalization(
                X, self.scale_params["min"], self.scale_params["max"]
            )
        return X

    def predict(
        self, X: Union[List, np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Make predictions using fitted model.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.

        Raises:
            ValueError: If model not fitted.
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self._scale_features(X)
        predictions = X_scaled @ self.weights + self.intercept
        return predictions

    def score(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> float:
        """Calculate R-squared score.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            R-squared score.
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return float(r2)

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
        plt.ylabel("Cost", fontsize=12)
        title = "Training History"
        if self.regularization:
            title += f" ({self.regularization.capitalize()}, α={self.alpha})"
        plt.title(title, fontsize=14, fontweight="bold")
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
        description="Multiple Linear Regression with Regularization"
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
        "--regularization",
        type=str,
        choices=["ridge", "lasso", "none"],
        default=None,
        help="Regularization type (default: from config)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Regularization strength (default: from config)",
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
        "--scaling",
        type=str,
        choices=["standardize", "normalize", "none"],
        default=None,
        help="Feature scaling method (default: from config)",
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
        reg = (
            args.regularization
            if args.regularization is not None
            else model_config.get("regularization")
        )
        if reg == "none":
            reg = None
        alpha = (
            args.alpha
            if args.alpha is not None
            else model_config.get("alpha", 0.1)
        )
        scaling = (
            args.scaling
            if args.scaling is not None
            else model_config.get("scaling_method", "standardize")
        )
        scale_features = scaling != "none"
        scaling_method = scaling if scale_features else "standardize"

        df = pd.read_csv(args.input)
        print(f"\n=== Multiple Linear Regression ===")
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

        model = MultipleLinearRegression(
            learning_rate=lr,
            max_iterations=max_iter,
            regularization=reg,
            alpha=alpha,
            scale_features=scale_features,
            scaling_method=scaling_method,
        )

        print(f"\nTraining model...")
        print(f"Learning rate: {lr}")
        print(f"Max iterations: {max_iter}")
        if reg:
            print(f"Regularization: {reg} (alpha={alpha})")
        else:
            print("Regularization: None")
        print(f"Feature scaling: {scaling_method if scale_features else 'None'}")

        model.fit(X, y)

        print(f"\n=== Training Results ===")
        print(f"Final cost: {model.cost_history[-1]:.6f}")
        print(f"Iterations: {len(model.cost_history)}")
        print(f"R-squared: {model.score(X, y):.6f}")

        if model.fit_intercept:
            print(f"\nIntercept: {model.intercept:.6f}")
        print("Weights:")
        for i, weight in enumerate(model.weights):
            print(f"  {feature_cols[i]}: {weight:.6f}")

        if args.plot or args.save_plot:
            model.plot_training_history(
                save_path=args.save_plot, show=args.plot
            )

        if args.output:
            predictions = model.predict(X)
            output_df = pd.DataFrame(
                {
                    "actual": y,
                    "predicted": predictions,
                    "residual": y - predictions,
                }
            )
            output_df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


if __name__ == "__main__":
    main()
