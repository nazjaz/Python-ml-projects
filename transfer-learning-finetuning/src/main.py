"""Transfer learning and fine-tuning strategies from scratch.

Implements a simple two-layer MLP in NumPy and demonstrates:
- Training a base model on a source task
- Reusing learned weights on a target task via:
  - Feature extractor (freeze base, train head)
  - Head fine-tuning / full fine-tuning
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


class MLPBase:
    """Two-layer MLP used as base and target model."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        """Initialize MLP parameters."""
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
        """Compute loss and gradients for parameters."""
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

    def apply_gradients(
        self, grads: Dict[str, np.ndarray], learning_rate: float
    ) -> None:
        """Update parameters with gradient descent."""
        self.w1 -= learning_rate * grads["w1"]
        self.b1 -= learning_rate * grads["b1"]
        self.w2 -= learning_rate * grads["w2"]
        self.b2 -= learning_rate * grads["b2"]

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return parameters as a dict."""
        return {
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": self.b2.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        self.w1 = params["w1"].copy()
        self.b1 = params["b1"].copy()
        self.w2 = params["w2"].copy()
        self.b2 = params["b2"].copy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        logits = self.forward(x)
        return np.argmax(_softmax(logits), axis=1)


def generate_source_and_target_data(
    base_n_samples: int,
    target_n_samples: int,
    n_features: int,
    n_classes: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate related source and target datasets for transfer learning.

    Source and target share the same number of classes and features but have
    slightly shifted class centers to simulate domain shift.
    """
    rng = np.random.RandomState(random_seed)

    def make_dataset(n_samples: int, offset_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        samples_per_class = n_samples // n_classes
        x_list = []
        y_list = []
        for cls in range(n_classes):
            base_center = rng.randn(n_features) * 3.0
            offset = rng.randn(n_features) * offset_scale
            center = base_center + offset
            points = center + rng.randn(samples_per_class, n_features)
            x_list.append(points)
            y_list.append(np.full(samples_per_class, cls))
        x_all = np.vstack(x_list)
        y_all = np.concatenate(y_list)
        idx = rng.permutation(len(x_all))
        return x_all[idx], y_all[idx]

    x_base, y_base = make_dataset(base_n_samples, offset_scale=0.0)
    x_target, y_target = make_dataset(target_n_samples, offset_scale=1.0)
    return x_base, y_base, x_target, y_target


def train_model(
    model: MLPBase,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Dict[str, List[float]]:
    """Train a model using mini-batch gradient descent."""
    n_samples = x.shape[0]
    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        x_shuf = x[idx]
        y_shuf = y[idx]
        batch_losses: List[float] = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = x_shuf[start:end]
            yb = y_shuf[start:end]
            loss, grads = model.compute_loss_and_gradients(xb, yb)
            model.apply_gradients(grads, learning_rate=learning_rate)
            batch_losses.append(loss)

        avg_loss = float(np.mean(batch_losses))
        preds = model.predict(x)
        acc = float(np.mean(preds == y))
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(
                "Epoch %d/%d - loss: %.6f - acc: %.4f",
                epoch + 1,
                epochs,
                avg_loss,
                acc,
            )
    return history


class TransferLearningRunner:
    """Orchestrates base and target training with various strategies."""

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

    def _build_model(self) -> MLPBase:
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        n_features = data_cfg.get("n_features", 20)
        n_classes = data_cfg.get("n_classes", 3)
        hidden_dim = model_cfg.get("hidden_dim", 32)
        return MLPBase(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
        )

    def _train_base(self) -> Tuple[MLPBase, Dict[str, float]]:
        data_cfg = self.config.get("data", {})
        train_cfg = self.config.get("training", {})

        x_base, y_base, _, _ = generate_source_and_target_data(
            base_n_samples=data_cfg.get("base_n_samples", 2000),
            target_n_samples=data_cfg.get("target_n_samples", 800),
            n_features=data_cfg.get("n_features", 20),
            n_classes=data_cfg.get("n_classes", 3),
            random_seed=data_cfg.get("random_seed", 42),
        )
        model = self._build_model()
        history = train_model(
            model,
            x_base,
            y_base,
            epochs=train_cfg.get("base_epochs", 20),
            learning_rate=train_cfg.get("learning_rate_base", 0.05),
            batch_size=train_cfg.get("batch_size", 32),
        )
        preds = model.predict(x_base)
        acc = float(np.mean(preds == y_base))
        loss = float(history["loss"][-1])
        return model, {"loss": loss, "accuracy": acc}

    def run(self, strategy: str) -> Dict[str, float]:
        """Run transfer learning experiment for the given strategy."""
        if strategy not in {"scratch", "feature_extractor", "head_finetune", "full_finetune"}:
            raise ValueError(
                "strategy must be one of "
                "'scratch', 'feature_extractor', 'head_finetune', 'full_finetune'"
            )

        data_cfg = self.config.get("data", {})
        train_cfg = self.config.get("training", {})

        x_base, y_base, x_target, y_target = generate_source_and_target_data(
            base_n_samples=data_cfg.get("base_n_samples", 2000),
            target_n_samples=data_cfg.get("target_n_samples", 800),
            n_features=data_cfg.get("n_features", 20),
            n_classes=data_cfg.get("n_classes", 3),
            random_seed=data_cfg.get("random_seed", 42),
        )

        source_metrics = {"loss": float("nan"), "accuracy": float("nan")}

        if strategy == "scratch":
            model_target = self._build_model()
        else:
            base_model = self._build_model()
            history_base = train_model(
                base_model,
                x_base,
                y_base,
                epochs=train_cfg.get("base_epochs", 20),
                learning_rate=train_cfg.get("learning_rate_base", 0.05),
                batch_size=train_cfg.get("batch_size", 32),
            )
            preds_base = base_model.predict(x_base)
            source_metrics["loss"] = float(history_base["loss"][-1])
            source_metrics["accuracy"] = float(
                np.mean(preds_base == y_base)
            )

            if strategy == "feature_extractor":
                model_target = self._build_model()
                params = base_model.get_params()
                model_target.w1 = params["w1"].copy()
                model_target.b1 = params["b1"].copy()
                model_target.w2 = np.random.randn(
                    *model_target.w2.shape
                ).astype(np.float32) * 0.1
                model_target.b2 = np.zeros_like(model_target.b2)
            else:
                model_target = self._build_model()
                model_target.set_params(base_model.get_params())

        history_target = train_model(
            model_target,
            x_target,
            y_target,
            epochs=train_cfg.get("target_epochs", 20),
            learning_rate=train_cfg.get("learning_rate_target", 0.02),
            batch_size=train_cfg.get("batch_size", 32),
        )

        preds_target = model_target.predict(x_target)
        target_acc = float(np.mean(preds_target == y_target))
        target_loss = float(history_target["loss"][-1])

        results = {
            "source_loss": source_metrics["loss"],
            "source_accuracy": source_metrics["accuracy"],
            "target_loss": target_loss,
            "target_accuracy": target_acc,
        }
        logger.info(
            "[%s] Source - loss: %.6f, acc: %.4f | Target - loss: %.6f, acc: %.4f",
            strategy,
            results["source_loss"],
            results["source_accuracy"],
            results["target_loss"],
            results["target_accuracy"],
        )
        return results


def main() -> None:
    """Main entry point for transfer learning demo."""
    parser = argparse.ArgumentParser(
        description="Transfer learning and fine-tuning strategies demo"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["scratch", "feature_extractor", "head_finetune", "full_finetune"],
        default="feature_extractor",
        help="Transfer learning strategy to use (default: feature_extractor)",
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = TransferLearningRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run(strategy=args.strategy)

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

