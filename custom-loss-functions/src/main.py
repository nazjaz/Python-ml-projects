"""Custom Loss Functions for Regression and Classification.

This module provides implementations of custom loss functions for both
regression and classification tasks, including gradient computation
for use in gradient-based optimization algorithms.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class RegressionLoss:
    """Custom loss functions for regression tasks with gradient computation."""

    @staticmethod
    def mean_squared_error(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute Mean Squared Error (MSE) and its gradient.

        MSE = (1/n) * sum((y_true - y_pred)^2)
        Gradient = -2 * (y_true - y_pred) / n

        Args:
            y_true: Ground truth target values, shape (n_samples,)
            y_pred: Predicted target values, shape (n_samples,)

        Returns:
            Tuple containing:
                - loss: Scalar MSE loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )

        n_samples = len(y_true)
        error = y_true - y_pred
        loss = np.mean(error ** 2)
        gradient = -2 * error / n_samples

        return float(loss), gradient

    @staticmethod
    def mean_absolute_error(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute Mean Absolute Error (MAE) and its gradient.

        MAE = (1/n) * sum(|y_true - y_pred|)
        Gradient = -sign(y_true - y_pred) / n

        Args:
            y_true: Ground truth target values, shape (n_samples,)
            y_pred: Predicted target values, shape (n_samples,)

        Returns:
            Tuple containing:
                - loss: Scalar MAE loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )

        n_samples = len(y_true)
        error = y_true - y_pred
        loss = np.mean(np.abs(error))
        gradient = -np.sign(error) / n_samples

        return float(loss), gradient

    @staticmethod
    def huber_loss(
        y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0
    ) -> Tuple[float, np.ndarray]:
        """Compute Huber Loss and its gradient.

        Huber loss is less sensitive to outliers than MSE.
        For |error| <= delta: L = 0.5 * error^2
        For |error| > delta: L = delta * |error| - 0.5 * delta^2

        Args:
            y_true: Ground truth target values, shape (n_samples,)
            y_pred: Predicted target values, shape (n_samples,)
            delta: Threshold parameter (default: 1.0)

        Returns:
            Tuple containing:
                - loss: Scalar Huber loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes or delta <= 0
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

        n_samples = len(y_true)
        error = y_true - y_pred
        abs_error = np.abs(error)

        loss = np.where(
            abs_error <= delta,
            0.5 * error ** 2,
            delta * abs_error - 0.5 * delta ** 2,
        )
        loss = np.mean(loss)

        gradient = np.where(
            abs_error <= delta, -error, -delta * np.sign(error)
        )
        gradient = gradient / n_samples

        return float(loss), gradient

    @staticmethod
    def smooth_l1_loss(
        y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0
    ) -> Tuple[float, np.ndarray]:
        """Compute Smooth L1 Loss and its gradient.

        Similar to Huber loss, smooth L1 provides smooth gradient
        near zero while being robust to outliers.

        Args:
            y_true: Ground truth target values, shape (n_samples,)
            y_pred: Predicted target values, shape (n_samples,)
            beta: Smoothing parameter (default: 1.0)

        Returns:
            Tuple containing:
                - loss: Scalar smooth L1 loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes or beta <= 0
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        n_samples = len(y_true)
        error = y_true - y_pred
        abs_error = np.abs(error)

        loss = np.where(
            abs_error < beta,
            0.5 * error ** 2 / beta,
            abs_error - 0.5 * beta,
        )
        loss = np.mean(loss)

        gradient = np.where(
            abs_error < beta, -error / beta, -np.sign(error)
        )
        gradient = gradient / n_samples

        return float(loss), gradient

    @staticmethod
    def log_cosh_loss(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute Log-Cosh Loss and its gradient.

        Log-cosh loss is smooth everywhere and behaves like MSE
        for small errors and like MAE for large errors.

        Args:
            y_true: Ground truth target values, shape (n_samples,)
            y_pred: Predicted target values, shape (n_samples,)

        Returns:
            Tuple containing:
                - loss: Scalar log-cosh loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )

        n_samples = len(y_true)
        error = y_true - y_pred
        loss = np.mean(np.log(np.cosh(error)))
        gradient = -np.tanh(error) / n_samples

        return float(loss), gradient


class ClassificationLoss:
    """Custom loss functions for classification tasks with gradient computation."""

    @staticmethod
    def binary_cross_entropy(
        y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15
    ) -> Tuple[float, np.ndarray]:
        """Compute Binary Cross-Entropy Loss and its gradient.

        BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        Gradient = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n

        Args:
            y_true: Ground truth binary labels (0 or 1), shape (n_samples,)
            y_pred: Predicted probabilities, shape (n_samples,)
            epsilon: Small value to prevent numerical instability (default: 1e-15)

        Returns:
            Tuple containing:
                - loss: Scalar binary cross-entropy loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )
        if np.any((y_true != 0) & (y_true != 1)):
            raise ValueError("y_true must contain only 0 and 1")
        if np.any((y_pred < 0) | (y_pred > 1)):
            raise ValueError("y_pred must be in [0, 1]")

        n_samples = len(y_true)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        gradient = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n_samples

        return float(loss), gradient

    @staticmethod
    def categorical_cross_entropy(
        y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15
    ) -> Tuple[float, np.ndarray]:
        """Compute Categorical Cross-Entropy Loss and its gradient.

        CCE = -mean(sum(y_true * log(y_pred), axis=1))
        Gradient = -y_true / y_pred / n

        Args:
            y_true: Ground truth one-hot encoded labels, shape (n_samples, n_classes)
            y_pred: Predicted probabilities, shape (n_samples, n_classes)
            epsilon: Small value to prevent numerical instability (default: 1e-15)

        Returns:
            Tuple containing:
                - loss: Scalar categorical cross-entropy loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples, n_classes)

        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )
        if y_true.ndim != 2:
            raise ValueError("y_true and y_pred must be 2D arrays")
        if not np.allclose(np.sum(y_true, axis=1), 1.0):
            raise ValueError("y_true must be one-hot encoded (sum to 1 per row)")
        if np.any((y_pred < 0) | (y_pred > 1)):
            raise ValueError("y_pred must be in [0, 1]")

        n_samples = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        gradient = -y_true / y_pred / n_samples

        return float(loss), gradient

    @staticmethod
    def focal_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 1.0,
        gamma: float = 2.0,
        epsilon: float = 1e-15,
    ) -> Tuple[float, np.ndarray]:
        """Compute Focal Loss and its gradient for binary classification.

        Focal loss addresses class imbalance by down-weighting easy examples.
        FL = -alpha * (1 - p_t)^gamma * log(p_t)
        where p_t = y_pred if y_true=1, else 1-y_pred

        Args:
            y_true: Ground truth binary labels (0 or 1), shape (n_samples,)
            y_pred: Predicted probabilities, shape (n_samples,)
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            epsilon: Small value to prevent numerical instability (default: 1e-15)

        Returns:
            Tuple containing:
                - loss: Scalar focal loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )
        if np.any((y_true != 0) & (y_true != 1)):
            raise ValueError("y_true must contain only 0 and 1")
        if np.any((y_pred < 0) | (y_pred > 1)):
            raise ValueError("y_pred must be in [0, 1]")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        n_samples = len(y_true)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = (1 - p_t) ** gamma
        log_p_t = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

        loss = -alpha * np.mean(modulating_factor * log_p_t)

        gradient_p_t = y_true - y_pred
        gradient_modulating = -gamma * (1 - p_t) ** (gamma - 1) * gradient_p_t
        gradient_log = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
        gradient = (
            -alpha
            * (
                gradient_modulating * log_p_t
                + modulating_factor * gradient_log
            )
            / n_samples
        )

        return float(loss), gradient

    @staticmethod
    def hinge_loss(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute Hinge Loss and its gradient for binary classification.

        Hinge loss is used for maximum-margin classification.
        HL = mean(max(0, 1 - y_true * y_pred))
        where y_true is in {-1, 1}

        Args:
            y_true: Ground truth labels (-1 or 1), shape (n_samples,)
            y_pred: Raw predictions (not probabilities), shape (n_samples,)

        Returns:
            Tuple containing:
                - loss: Scalar hinge loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples,)

        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )
        if np.any((y_true != -1) & (y_true != 1)):
            raise ValueError("y_true must contain only -1 and 1")

        n_samples = len(y_true)
        margin = 1 - y_true * y_pred
        loss = np.mean(np.maximum(0, margin))

        gradient = np.where(margin > 0, -y_true, 0) / n_samples

        return float(loss), gradient

    @staticmethod
    def kullback_leibler_divergence(
        y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15
    ) -> Tuple[float, np.ndarray]:
        """Compute KL Divergence Loss and its gradient.

        KL divergence measures how one probability distribution diverges
        from another. Used for multi-class classification.

        Args:
            y_true: True probability distribution, shape (n_samples, n_classes)
            y_pred: Predicted probability distribution, shape (n_samples, n_classes)
            epsilon: Small value to prevent numerical instability (default: 1e-15)

        Returns:
            Tuple containing:
                - loss: Scalar KL divergence loss value
                - gradient: Gradient with respect to y_pred, shape (n_samples, n_classes)

        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )
        if y_true.ndim != 2:
            raise ValueError("y_true and y_pred must be 2D arrays")
        if np.any((y_pred < 0) | (y_pred > 1)):
            raise ValueError("y_pred must be in [0, 1]")

        n_samples = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = np.clip(y_true, epsilon, 1 - epsilon)

        loss = np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=1))
        gradient = -y_true / y_pred / n_samples

        return float(loss), gradient


class LossFunctionEvaluator:
    """Evaluator for testing and comparing loss functions."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the loss function evaluator.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/app.log")

        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

    def evaluate_regression_losses(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Evaluate all regression loss functions.

        Args:
            y_true: Ground truth target values
            y_pred: Predicted target values

        Returns:
            Dictionary mapping loss function names to their results
        """
        results = {}

        loss_functions = {
            "mse": RegressionLoss.mean_squared_error,
            "mae": RegressionLoss.mean_absolute_error,
            "huber": lambda yt, yp: RegressionLoss.huber_loss(yt, yp, delta=1.0),
            "smooth_l1": lambda yt, yp: RegressionLoss.smooth_l1_loss(
                yt, yp, beta=1.0
            ),
            "log_cosh": RegressionLoss.log_cosh_loss,
        }

        for name, func in loss_functions.items():
            try:
                loss, gradient = func(y_true, y_pred)
                results[name] = {
                    "loss": loss,
                    "gradient": gradient,
                    "gradient_norm": np.linalg.norm(gradient),
                }
                logger.info(f"{name.upper()} loss: {loss:.6f}")
            except Exception as e:
                logger.error(f"Error computing {name} loss: {e}")
                results[name] = {"error": str(e)}

        return results

    def evaluate_classification_losses(
        self, y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = "binary"
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Evaluate classification loss functions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities or raw predictions
            loss_type: Type of classification ("binary" or "multiclass")

        Returns:
            Dictionary mapping loss function names to their results
        """
        results = {}

        if loss_type == "binary":
            if y_pred.ndim == 1:
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            if np.any((y_true != 0) & (y_true != 1)):
                y_true_binary = (y_true > 0.5).astype(float)
            else:
                y_true_binary = y_true

            loss_functions = {
                "binary_cross_entropy": ClassificationLoss.binary_cross_entropy,
                "focal_loss": lambda yt, yp: ClassificationLoss.focal_loss(
                    yt, yp, alpha=1.0, gamma=2.0
                ),
            }

            for name, func in loss_functions.items():
                try:
                    loss, gradient = func(y_true_binary, y_pred)
                    results[name] = {
                        "loss": loss,
                        "gradient": gradient,
                        "gradient_norm": np.linalg.norm(gradient),
                    }
                    logger.info(f"{name} loss: {loss:.6f}")
                except Exception as e:
                    logger.error(f"Error computing {name} loss: {e}")
                    results[name] = {"error": str(e)}

        else:
            if y_true.ndim == 1:
                n_classes = len(np.unique(y_true))
                y_true_onehot = np.eye(n_classes)[y_true.astype(int)]
            else:
                y_true_onehot = y_true

            if y_pred.ndim == 1:
                y_pred = np.eye(len(y_true_onehot[0]))[y_pred.astype(int)]

            loss_functions = {
                "categorical_cross_entropy": ClassificationLoss.categorical_cross_entropy,
                "kl_divergence": lambda yt, yp: ClassificationLoss.kullback_leibler_divergence(
                    yt, yp
                ),
            }

            for name, func in loss_functions.items():
                try:
                    loss, gradient = func(y_true_onehot, y_pred)
                    results[name] = {
                        "loss": loss,
                        "gradient": gradient,
                        "gradient_norm": np.linalg.norm(gradient),
                    }
                    logger.info(f"{name} loss: {loss:.6f}")
                except Exception as e:
                    logger.error(f"Error computing {name} loss: {e}")
                    results[name] = {"error": str(e)}

        return results


def main():
    """Main entry point for the loss function evaluator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate custom loss functions for regression and classification"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input CSV file with true and predicted values",
    )
    parser.add_argument(
        "--true-col",
        type=str,
        default="y_true",
        help="Column name for true values (default: y_true)",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="y_pred",
        help="Column name for predicted values (default: y_pred)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Task type: regression or classification (default: regression)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSON file for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    evaluator = LossFunctionEvaluator(
        config_path=Path(args.config) if args.config else None
    )

    if args.input:
        df = pd.read_csv(args.input)
        y_true = df[args.true_col].values
        y_pred = df[args.pred_col].values
    else:
        logger.info("No input file provided, using synthetic data")
        np.random.seed(42)
        if args.task == "regression":
            y_true = np.random.randn(100)
            y_pred = y_true + 0.1 * np.random.randn(100)
        else:
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.rand(100)

    if args.task == "regression":
        results = evaluator.evaluate_regression_losses(y_true, y_pred)
    else:
        results = evaluator.evaluate_classification_losses(y_true, y_pred)

    if args.output:
        output_data = {}
        for name, result in results.items():
            if "error" not in result:
                output_data[name] = {
                    "loss": float(result["loss"]),
                    "gradient_norm": float(result["gradient_norm"]),
                }
            else:
                output_data[name] = {"error": result["error"]}

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print("\nLoss Function Evaluation Results:")
        print("=" * 50)
        for name, result in results.items():
            if "error" not in result:
                print(f"{name.upper()}:")
                print(f"  Loss: {result['loss']:.6f}")
                print(f"  Gradient Norm: {result['gradient_norm']:.6f}")
            else:
                print(f"{name.upper()}: Error - {result['error']}")


if __name__ == "__main__":
    main()
