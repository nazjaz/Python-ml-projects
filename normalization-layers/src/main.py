"""Batch normalization and layer normalization from scratch.

This module implements BatchNorm1D and LayerNorm1D using NumPy and integrates
them into a small two-layer feedforward network to demonstrate training
stability on a synthetic classification task.
"""

import argparse
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


class BatchNorm1D:
    """Batch normalization layer for 1D feature vectors.

    Normalizes each feature across the mini-batch using running statistics
    for evaluation and learnable scale and shift parameters.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.9,
        eps: float = 1e-5,
    ) -> None:
        """Initialize batch normalization parameters.

        Args:
            num_features: Number of features in the input.
            momentum: Momentum for running mean and variance.
            eps: Small constant for numerical stability.
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)

        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

        self._x_centered: Optional[np.ndarray] = None
        self._std_inv: Optional[np.ndarray] = None
        self._x_norm: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through batch normalization.

        Args:
            x: Input array of shape (batch_size, num_features).
            training: Whether the layer is in training mode.

        Returns:
            Normalized and scaled output with same shape as x.
        """
        if training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            self.running_mean = (
                self.momentum * self.running_mean
                + (1.0 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1.0 - self.momentum) * batch_var
            )

            self._x_centered = x - batch_mean
            self._std_inv = 1.0 / np.sqrt(batch_var + self.eps)
            self._x_norm = self._x_centered * self._std_inv
            out = self.gamma * self._x_norm + self.beta
        else:
            x_centered = x - self.running_mean
            x_norm = x_centered / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through batch normalization.

        Args:
            grad_output: Gradient from next layer, shape (batch_size, num_features).

        Returns:
            Gradient with respect to input x, same shape as grad_output.
        """
        if (
            self._x_centered is None
            or self._std_inv is None
            or self._x_norm is None
        ):
            raise RuntimeError("forward must be called in training mode first.")

        n, _ = grad_output.shape

        dgamma = np.sum(grad_output * self._x_norm, axis=0)
        dbeta = np.sum(grad_output, axis=0)

        dx_norm = grad_output * self.gamma
        dvar = np.sum(
            dx_norm * self._x_centered * -0.5 * self._std_inv**3, axis=0
        )
        dmean = (
            np.sum(dx_norm * -self._std_inv, axis=0)
            + dvar * np.mean(-2.0 * self._x_centered, axis=0)
        )

        dx = (
            dx_norm * self._std_inv
            + (2.0 / n) * dvar * self._x_centered
            + dmean / n
        )

        self.gamma -= 0.0 * dgamma
        self.beta -= 0.0 * dbeta

        return dx


class LayerNorm1D:
    """Layer normalization over the feature dimension for each sample."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        """Initialize layer normalization parameters."""
        self.num_features = num_features
        self.eps = eps

        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)

        self._x_centered: Optional[np.ndarray] = None
        self._std_inv: Optional[np.ndarray] = None
        self._x_norm: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through layer normalization."""
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        self._x_centered = x - mean
        self._std_inv = 1.0 / np.sqrt(var + self.eps)
        self._x_norm = self._x_centered * self._std_inv
        return self.gamma * self._x_norm + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through layer normalization."""
        if (
            self._x_centered is None
            or self._std_inv is None
            or self._x_norm is None
        ):
            raise RuntimeError("forward must be called before backward.")

        n, d = grad_output.shape

        dgamma = np.sum(grad_output * self._x_norm, axis=0)
        dbeta = np.sum(grad_output, axis=0)

        dx_norm = grad_output * self.gamma
        dvar = np.sum(
            dx_norm * self._x_centered * -0.5 * self._std_inv**3, axis=1,
            keepdims=True,
        )
        dmean = (
            np.sum(dx_norm * -self._std_inv, axis=1, keepdims=True)
            + dvar * np.mean(-2.0 * self._x_centered, axis=1, keepdims=True)
        )

        dx = (
            dx_norm * self._std_inv
            + (2.0 / d) * dvar * self._x_centered
            + dmean / d
        )

        self.gamma -= 0.0 * dgamma
        self.beta -= 0.0 * dbeta

        return dx


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax along last dimension."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def _cross_entropy(
    logits: np.ndarray, targets: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute cross-entropy loss and gradient."""
    probs = _softmax(logits)
    n = logits.shape[0]
    idx = np.arange(n)
    chosen = probs[idx, targets]
    epsilon = 1e-15
    loss = -np.mean(np.log(np.clip(chosen, epsilon, 1.0)))

    grad = probs
    grad[idx, targets] -= 1.0
    grad /= float(n)
    return float(loss), grad


class SimpleMLP:
    """Two-layer MLP with optional batch or layer normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        norm: str = "none",
    ) -> None:
        """Initialize MLP parameters."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm_type = norm

        limit = np.sqrt(1.0 / max(1, input_dim))
        self.w1 = np.random.uniform(
            -limit, limit, size=(input_dim, hidden_dim)
        ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w2 = np.random.uniform(
            -limit2, limit2, size=(hidden_dim, n_classes)
        ).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

        if norm == "batchnorm":
            self.norm_layer: Optional[object] = BatchNorm1D(hidden_dim)
        elif norm == "layernorm":
            self.norm_layer = LayerNorm1D(hidden_dim)
        else:
            self.norm_layer = None

        self._x_cache: Optional[np.ndarray] = None
        self._h_pre_cache: Optional[np.ndarray] = None
        self._h_cache: Optional[np.ndarray] = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0.0)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network."""
        self._x_cache = x
        h_pre = x @ self.w1 + self.b1
        if self.norm_layer is not None:
            if isinstance(self.norm_layer, BatchNorm1D):
                h_norm = self.norm_layer.forward(h_pre, training=training)
            else:
                h_norm = self.norm_layer.forward(h_pre)
            h_activated = self._relu(h_norm)
            self._h_pre_cache = h_norm
        else:
            h_activated = self._relu(h_pre)
            self._h_pre_cache = h_pre
        self._h_cache = h_activated
        logits = h_activated @ self.w2 + self.b2
        return logits

    def backward(
        self, grad_logits: np.ndarray, learning_rate: float
    ) -> None:
        """Backward pass updating all parameters."""
        if (
            self._x_cache is None
            or self._h_cache is None
            or self._h_pre_cache is None
        ):
            raise RuntimeError("forward must be called before backward.")

        x = self._x_cache
        h = self._h_cache
        h_pre = self._h_pre_cache

        n = x.shape[0]

        dw2 = h.T @ grad_logits / float(n)
        db2 = np.mean(grad_logits, axis=0)
        dh = grad_logits @ self.w2.T

        dh_pre = self._relu_backward(dh, h_pre)

        if self.norm_layer is not None:
            dh_pre = self.norm_layer.backward(dh_pre)

        dw1 = x.T @ dh_pre / float(n)
        db1 = np.mean(dh_pre, axis=0)

        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for input features."""
        logits = self.forward(x, training=False)
        return np.argmax(_softmax(logits), axis=1)


def generate_classification_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic multi-class classification dataset."""
    np.random.seed(random_seed)
    samples_per_class = n_samples // n_classes
    x_list = []
    y_list = []
    for cls in range(n_classes):
        center = np.random.randn(n_features) * 3.0
        points = center + np.random.randn(samples_per_class, n_features)
        x_list.append(points)
        y_list.append(np.full(samples_per_class, cls))
    x_all = np.vstack(x_list)
    y_all = np.concatenate(y_list)
    idx = np.random.permutation(len(x_all))
    x_all = x_all[idx]
    y_all = y_all[idx]
    split = int(0.8 * len(x_all))
    return x_all[:split], x_all[split:], y_all[:split], y_all[split:]


class NormalizationRunner:
    """Orchestrates training and evaluation for different normalization modes."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize runner with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load YAML configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found: %s, using defaults", config_path)
            return {}

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"))
        log_file = log_cfg.get("file", "logs/app.log")
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.setLevel(level)
        logger.addHandler(handler)

    def run(self, mode: str) -> Dict[str, float]:
        """Run training and evaluation for the specified normalization mode."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_train = data_cfg.get("n_train", 2000)
        n_test = data_cfg.get("n_test", 500)
        n_features = data_cfg.get("n_features", 20)
        n_classes = data_cfg.get("n_classes", 3)
        seed = data_cfg.get("random_seed", 42)

        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=n_train + n_test,
            n_features=n_features,
            n_classes=n_classes,
            random_seed=seed,
        )

        hidden_dim = model_cfg.get("hidden_dim", 32)
        model = SimpleMLP(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            norm=mode,
        )

        epochs = train_cfg.get("epochs", 30)
        learning_rate = train_cfg.get("learning_rate", 0.05)
        batch_size = train_cfg.get("batch_size", 32)

        history_loss: List[float] = []
        history_acc: List[float] = []

        n_samples = x_train.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            x_shuf = x_train[idx]
            y_shuf = y_train[idx]
            batch_losses: List[float] = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = x_shuf[start:end]
                yb = y_shuf[start:end]
                logits = model.forward(xb, training=True)
                loss, grad_logits = _cross_entropy(logits, yb)
                model.backward(grad_logits, learning_rate=learning_rate)
                batch_losses.append(loss)

            avg_loss = float(np.mean(batch_losses))
            preds_train = model.predict(x_train)
            train_acc = float(np.mean(preds_train == y_train))
            history_loss.append(avg_loss)
            history_acc.append(train_acc)

            if (epoch + 1) % max(1, epochs // 10) == 0:
                msg = (
                    f"[{mode}] Epoch {epoch + 1}/{epochs} - "
                    f"loss: {avg_loss:.6f} - acc: {train_acc:.4f}"
                )
                logger.info(msg)
                print(msg)

        preds_test = model.predict(x_test)
        test_acc = float(np.mean(preds_test == y_test))
        logits_test = model.forward(x_test, training=False)
        test_loss, _ = _cross_entropy(logits_test, y_test)

        results = {
            "train_loss": float(history_loss[-1]),
            "train_accuracy": float(history_acc[-1]),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        }

        logger.info(
            "[%s] Results - train_loss: %.6f, train_acc: %.4f, "
            "test_loss: %.6f, test_acc: %.4f",
            mode,
            results["train_loss"],
            results["train_accuracy"],
            results["test_loss"],
            results["test_accuracy"],
        )
        return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a simple MLP with optional batch or layer normalization"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["none", "batchnorm", "layernorm"],
        default="batchnorm",
        help="Normalization mode to use (default: batchnorm)",
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = NormalizationRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run(mode=args.mode)

    print("\nFinal Results:")
    print("=" * 40)
    for key, val in results.items():
        print(f"  {key}: {val:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()

