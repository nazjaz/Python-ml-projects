"""Dropout and regularization techniques from scratch.

This module implements dropout and L2 weight decay using NumPy and integrates
them into a small two-layer feedforward network to study overfitting and
regularization on a synthetic classification task.
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


class Dropout:
    """Inverted dropout regularization for fully connected layers."""

    def __init__(self, rate: float) -> None:
        """Initialize dropout layer.

        Args:
            rate: Drop probability in [0, 1).
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1).")
        self.rate = rate
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout to activations."""
        if not training or self.rate == 0.0:
            self._mask = None
            return x
        keep_prob = 1.0 - self.rate
        self._mask = (np.random.rand(*x.shape) < keep_prob).astype(x.dtype)
        return x * self._mask / keep_prob

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagate gradient through dropout mask."""
        if self._mask is None or self.rate == 0.0:
            return grad_output
        keep_prob = 1.0 - self.rate
        return grad_output * self._mask / keep_prob


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


class RegularizedMLP:
    """Two-layer MLP with optional dropout and L2 regularization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout_rate: float,
        l2_lambda: float,
        mode: str,
    ) -> None:
        """Initialize network parameters and regularization mode."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        if mode not in {"none", "dropout", "l2", "dropout_l2"}:
            raise ValueError(
                "mode must be one of 'none', 'dropout', 'l2', 'dropout_l2'"
            )
        self.mode = mode

        limit1 = np.sqrt(1.0 / max(1, input_dim))
        self.w1 = np.random.uniform(
            -limit1, limit1, size=(input_dim, hidden_dim)
        ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w2 = np.random.uniform(
            -limit2, limit2, size=(hidden_dim, n_classes)
        ).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

        self.dropout = Dropout(rate=dropout_rate)

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
        self._h_pre_cache = h_pre
        h = self._relu(h_pre)

        if self.mode in {"dropout", "dropout_l2"}:
            h = self.dropout.forward(h, training=training)
        self._h_cache = h

        logits = h @ self.w2 + self.b2
        return logits

    def compute_loss_and_gradients(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute loss (including L2 penalty when enabled) and gradient."""
        logits = self.forward(x, training=True)
        ce_loss, grad_logits = _cross_entropy(logits, y)

        l2_loss = 0.0
        if self.mode in {"l2", "dropout_l2"} and self.l2_lambda > 0.0:
            l2_loss = 0.5 * self.l2_lambda * (
                np.sum(self.w1**2) + np.sum(self.w2**2)
            )
        total_loss = ce_loss + l2_loss
        return float(total_loss), grad_logits

    def backward(
        self, grad_logits: np.ndarray, learning_rate: float
    ) -> None:
        """Backward pass updating parameters (including L2 gradients)."""
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

        if self.mode in {"l2", "dropout_l2"} and self.l2_lambda > 0.0:
            dw2 += self.l2_lambda * self.w2

        dh = grad_logits @ self.w2.T
        if self.mode in {"dropout", "dropout_l2"}:
            dh = self.dropout.backward(dh)

        dh_pre = self._relu_backward(dh, h_pre)

        dw1 = x.T @ dh_pre / float(n)
        db1 = np.mean(dh_pre, axis=0)

        if self.mode in {"l2", "dropout_l2"} and self.l2_lambda > 0.0:
            dw1 += self.l2_lambda * self.w1

        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for given input."""
        logits = self.forward(x, training=False)
        return np.argmax(_softmax(logits), axis=1)


def generate_classification_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic multi-class classification dataset."""
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


class RegularizationRunner:
    """Run experiments for different regularization modes."""

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
        """Train and evaluate a model with the specified regularization mode."""
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
        dropout_rate = model_cfg.get("dropout_rate", 0.5)
        l2_lambda = model_cfg.get("l2_lambda", 0.001)

        model = RegularizedMLP(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            l2_lambda=l2_lambda,
            mode=mode,
        )

        epochs = train_cfg.get("epochs", 30)
        learning_rate = train_cfg.get("learning_rate", 0.1)
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

                loss, grad_logits = model.compute_loss_and_gradients(xb, yb)
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
            "Train a simple MLP with dropout and L2 regularization options"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["none", "dropout", "l2", "dropout_l2"],
        default="dropout_l2",
        help="Regularization mode to use (default: dropout_l2)",
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = RegularizationRunner(
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

