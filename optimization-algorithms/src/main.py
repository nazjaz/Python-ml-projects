"""Adaptive optimization algorithms from scratch: Adam, RMSprop, AdaGrad.

This module implements several optimizers using NumPy and applies them to
train a small two-layer classifier on a synthetic classification task.
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


class MLPClassifier:
    """Two-layer MLP classifier used by all optimizers."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        """Initialize network parameters."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

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

        self._x_cache: Optional[np.ndarray] = None
        self._h_pre_cache: Optional[np.ndarray] = None
        self._h_cache: Optional[np.ndarray] = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning logits."""
        self._x_cache = x
        h_pre = x @ self.w1 + self.b1
        self._h_pre_cache = h_pre
        h = self._relu(h_pre)
        self._h_cache = h
        logits = h @ self.w2 + self.b2
        return logits

    def compute_loss_and_gradients(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute loss and gradients with respect to parameters."""
        logits = self.forward(x)
        loss, grad_logits = _cross_entropy(logits, y)

        if (
            self._x_cache is None
            or self._h_cache is None
            or self._h_pre_cache is None
        ):
            raise RuntimeError("forward must be called before gradients.")

        x_cache = self._x_cache
        h = self._h_cache
        h_pre = self._h_pre_cache
        n = x_cache.shape[0]

        grad_w2 = h.T @ grad_logits / float(n)
        grad_b2 = np.mean(grad_logits, axis=0)
        grad_h = grad_logits @ self.w2.T
        grad_h_pre = self._relu_backward(grad_h, h_pre)
        grad_w1 = x_cache.T @ grad_h_pre / float(n)
        grad_b1 = np.mean(grad_h_pre, axis=0)

        grads = {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }
        return loss, grads

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return a dict of parameter arrays."""
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        self.w1 = params["w1"]
        self.b1 = params["b1"]
        self.w2 = params["w2"]
        self.b2 = params["b2"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        logits = self.forward(x)
        return np.argmax(_softmax(logits), axis=1)


class BaseOptimizer:
    """Base class for parameter-space optimizers."""

    def __init__(self) -> None:
        self.state: Dict[str, Dict[str, np.ndarray]] = {}

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        raise NotImplementedError


class SGDOptimizer(BaseOptimizer):
    """Vanilla stochastic gradient descent."""

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        for name, p in params.items():
            p -= learning_rate * grads[name]


class AdaGradOptimizer(BaseOptimizer):
    """AdaGrad optimizer with per-parameter learning rate adaptation."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        for name, p in params.items():
            g = grads[name]
            if name not in self.state:
                self.state[name] = {"G": np.zeros_like(g)}
            state = self.state[name]
            state["G"] += g * g
            adjusted_lr = learning_rate / (
                np.sqrt(state["G"]) + self.eps
            )
            p -= adjusted_lr * g


class RMSpropOptimizer(BaseOptimizer):
    """RMSprop optimizer with exponential moving average of squared gradients."""

    def __init__(self, rho: float = 0.9, eps: float = 1e-8) -> None:
        super().__init__()
        self.rho = rho
        self.eps = eps

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        for name, p in params.items():
            g = grads[name]
            if name not in self.state:
                self.state[name] = {"E": np.zeros_like(g)}
            state = self.state[name]
            state["E"] = self.rho * state["E"] + (1.0 - self.rho) * (g * g)
            adjusted_lr = learning_rate / (
                np.sqrt(state["E"]) + self.eps
            )
            p -= adjusted_lr * g


class AdamOptimizer(BaseOptimizer):
    """Adam optimizer with bias correction."""

    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        self.t += 1
        for name, p in params.items():
            g = grads[name]
            if name not in self.state:
                self.state[name] = {
                    "m": np.zeros_like(g),
                    "v": np.zeros_like(g),
                }
            state = self.state[name]
            state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * g
            state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * (g * g)

            m_hat = state["m"] / (1.0 - self.beta1**self.t)
            v_hat = state["v"] / (1.0 - self.beta2**self.t)

            p -= learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


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


class OptimizationRunner:
    """Run training and evaluation for different optimizers."""

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

    def _make_optimizer(self, name: str) -> BaseOptimizer:
        """Construct optimizer given its name and config."""
        opt_cfg = self.config.get("optimizers", {})
        eps = float(opt_cfg.get("eps", 1e-8))
        rho = float(opt_cfg.get("rho", 0.9))
        beta1 = float(opt_cfg.get("beta1", 0.9))
        beta2 = float(opt_cfg.get("beta2", 0.999))

        if name == "sgd":
            return SGDOptimizer()
        if name == "adagrad":
            return AdaGradOptimizer(eps=eps)
        if name == "rmsprop":
            return RMSpropOptimizer(rho=rho, eps=eps)
        if name == "adam":
            return AdamOptimizer(beta1=beta1, beta2=beta2, eps=eps)
        raise ValueError("Unknown optimizer: {name}")

    def run(self, optimizer_name: str) -> Dict[str, float]:
        """Train and evaluate model with the selected optimizer."""
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
        model = MLPClassifier(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
        )

        optimizer = self._make_optimizer(optimizer_name)

        epochs = train_cfg.get("epochs", 30)
        learning_rate = train_cfg.get("learning_rate", 0.01)
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

                loss, grads = model.compute_loss_and_gradients(xb, yb)
                params = model.get_params()
                optimizer.step(params, grads, learning_rate=learning_rate)
                model.set_params(params)
                batch_losses.append(loss)

            avg_loss = float(np.mean(batch_losses))
            preds_train = model.predict(x_train)
            train_acc = float(np.mean(preds_train == y_train))
            history_loss.append(avg_loss)
            history_acc.append(train_acc)

            if (epoch + 1) % max(1, epochs // 10) == 0:
                msg = (
                    f"[{optimizer_name}] Epoch {epoch + 1}/{epochs} - "
                    f"loss: {avg_loss:.6f} - acc: {train_acc:.4f}"
                )
                logger.info(msg)
                print(msg)

        preds_test = model.predict(x_test)
        test_acc = float(np.mean(preds_test == y_test))
        logits_test = model.forward(x_test)
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
            optimizer_name,
            results["train_loss"],
            results["train_accuracy"],
            results["test_loss"],
            results["test_accuracy"],
        )
        return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train MLP with Adam, RMSprop, AdaGrad, or SGD optimizers"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adagrad", "rmsprop", "adam"],
        default="adam",
        help="Optimizer to use (default: adam)",
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = OptimizationRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run(optimizer_name=args.optimizer)

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

