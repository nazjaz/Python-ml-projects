"""Linear Regression with Gradient Descent.

This module provides functionality to implement linear regression
from scratch using gradient descent with learning rate scheduling.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    """Learning rate scheduling strategies."""

    @staticmethod
    def constant(initial_lr: float, **kwargs) -> Callable[[int], float]:
        """Constant learning rate.

        Args:
            initial_lr: Initial learning rate.

        Returns:
            Function that returns constant learning rate.
        """
        return lambda epoch: initial_lr

    @staticmethod
    def exponential_decay(
        initial_lr: float, decay_rate: float = 0.95, **kwargs
    ) -> Callable[[int], float]:
        """Exponential decay learning rate.

        Args:
            initial_lr: Initial learning rate.
            decay_rate: Decay rate per epoch.

        Returns:
            Function that returns exponentially decaying learning rate.
        """
        return lambda epoch: initial_lr * (decay_rate ** epoch)

    @staticmethod
    def step_decay(
        initial_lr: float,
        drop_rate: float = 0.5,
        epochs_drop: int = 10,
        **kwargs,
    ) -> Callable[[int], float]:
        """Step decay learning rate.

        Args:
            initial_lr: Initial learning rate.
            drop_rate: Factor to drop learning rate.
            epochs_drop: Number of epochs before dropping.

        Returns:
            Function that returns step-decaying learning rate.
        """
        return lambda epoch: initial_lr * (
            drop_rate ** (epoch // epochs_drop)
        )

    @staticmethod
    def polynomial_decay(
        initial_lr: float,
        end_lr: float = 0.001,
        power: float = 1.0,
        max_epochs: int = 100,
        **kwargs,
    ) -> Callable[[int], float]:
        """Polynomial decay learning rate.

        Args:
            initial_lr: Initial learning rate.
            end_lr: Final learning rate.
            power: Power of polynomial.
            max_epochs: Maximum number of epochs.

        Returns:
            Function that returns polynomially decaying learning rate.
        """
        return lambda epoch: (initial_lr - end_lr) * (
            1 - epoch / max_epochs
        ) ** power + end_lr


class LinearRegression:
    """Linear regression implemented from scratch using gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        scheduler: Optional[str] = None,
        scheduler_params: Optional[Dict] = None,
        fit_intercept: bool = True,
    ) -> None:
        """Initialize LinearRegression.

        Args:
            learning_rate: Initial learning rate.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.
            scheduler: Learning rate scheduler type. Options:
                'constant', 'exponential', 'step', 'polynomial'.
            scheduler_params: Parameters for scheduler.
            fit_intercept: Whether to fit intercept term.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.scheduler_type = scheduler or "constant"
        self.scheduler_params = scheduler_params or {}
        self.fit_intercept = fit_intercept

        self.weights: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.cost_history: List[float] = []
        self.lr_history: List[float] = []

        self._setup_scheduler()

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_map = {
            "constant": LearningRateScheduler.constant,
            "exponential": LearningRateScheduler.exponential_decay,
            "step": LearningRateScheduler.step_decay,
            "polynomial": LearningRateScheduler.polynomial_decay,
        }

        if self.scheduler_type not in scheduler_map:
            logger.warning(
                f"Unknown scheduler '{self.scheduler_type}', "
                "using constant"
            )
            self.scheduler_type = "constant"

        scheduler_func = scheduler_map[self.scheduler_type]
        params = {"initial_lr": self.learning_rate, **self.scheduler_params}
        self.scheduler = scheduler_func(**params)

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
        """Compute mean squared error cost.

        Args:
            X: Feature matrix (with intercept if applicable).
            y: Target values.
            theta: Model parameters.

        Returns:
            Mean squared error.
        """
        predictions = X @ theta
        mse = np.mean((predictions - y) ** 2)
        return float(mse)

    def _compute_gradient(
        self, X: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of cost function.

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
        return gradient

    def fit(
        self,
        X: Union[List, np.ndarray, pd.DataFrame],
        y: Union[List, np.ndarray, pd.Series],
    ) -> "LinearRegression":
        """Fit linear regression model using gradient descent.

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

        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]

        theta = np.zeros(n_features)
        self.cost_history = []
        self.lr_history = []

        logger.info(
            f"Starting gradient descent: {self.max_iterations} "
            f"iterations, initial LR={self.learning_rate}"
        )

        for iteration in range(self.max_iterations):
            current_lr = self.scheduler(iteration)
            self.lr_history.append(current_lr)

            gradient = self._compute_gradient(X_with_intercept, y, theta)
            theta = theta - current_lr * gradient

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
                    f"Iteration {iteration + 1}: cost={cost:.6f}, "
                    f"lr={current_lr:.6f}"
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

        predictions = X @ self.weights + self.intercept
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

    def get_lr_history(self) -> List[float]:
        """Get learning rate history from training.

        Returns:
            List of learning rate values per iteration.
        """
        return self.lr_history.copy()

    def plot_training_history(
        self, save_path: Optional[str] = None, show: bool = True
    ) -> None:
        """Plot training history (cost and learning rate).

        Args:
            save_path: Optional path to save figure.
            show: Whether to display plot.
        """
        if not self.cost_history:
            logger.warning("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        iterations = range(len(self.cost_history))

        ax1.plot(iterations, self.cost_history, "b-", linewidth=2)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Cost (MSE)", fontsize=12)
        ax1.set_title("Cost History", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        if self.lr_history:
            ax2.plot(iterations, self.lr_history, "r-", linewidth=2)
            ax2.set_xlabel("Iteration", fontsize=12)
            ax2.set_ylabel("Learning Rate", fontsize=12)
            ax2.set_title("Learning Rate History", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)

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
        description="Linear Regression with Gradient Descent"
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
        help="Initial learning rate (default: from config)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations (default: from config)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["constant", "exponential", "step", "polynomial"],
        default=None,
        help="Learning rate scheduler (default: from config)",
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
        scheduler = (
            args.scheduler
            if args.scheduler is not None
            else model_config.get("scheduler", "constant")
        )
        scheduler_params = model_config.get("scheduler_params", {})

        df = pd.read_csv(args.input)
        print(f"\n=== Linear Regression with Gradient Descent ===")
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

        model = LinearRegression(
            learning_rate=lr,
            max_iterations=max_iter,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
        )

        print(f"\nTraining model...")
        print(f"Learning rate: {lr}")
        print(f"Max iterations: {max_iter}")
        print(f"Scheduler: {scheduler}")

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
