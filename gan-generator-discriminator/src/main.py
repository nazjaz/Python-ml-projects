"""Generative Adversarial Network (GAN) with NumPy.

Implements a simple GAN on 2D synthetic data:
- Generator maps noise vectors to 2D samples
- Discriminator discriminates real vs generated samples
- Both are trained via adversarial binary cross-entropy losses
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _binary_cross_entropy(
    preds: np.ndarray, targets: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Binary cross-entropy loss and gradient."""
    epsilon = 1e-12
    preds_clipped = np.clip(preds, epsilon, 1.0 - epsilon)
    loss = -np.mean(
        targets * np.log(preds_clipped)
        + (1.0 - targets) * np.log(1.0 - preds_clipped)
    )
    grad = (preds_clipped - targets) / float(preds_clipped.shape[0])
    return float(loss), grad


class Generator:
    """Fully connected generator network."""

    def __init__(self, noise_dim: int, hidden_dim: int, data_dim: int) -> None:
        """Initialize generator parameters."""
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim

        limit1 = np.sqrt(1.0 / max(1, noise_dim))
        self.w1 = np.random.uniform(
            -limit1, limit1, size=(noise_dim, hidden_dim)
        ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w2 = np.random.uniform(
            -limit2, limit2, size=(hidden_dim, data_dim)
        ).astype(np.float32)
        self.b2 = np.zeros(data_dim, dtype=np.float32)

        self._z_cache: Optional[np.ndarray] = None
        self._h_cache: Optional[np.ndarray] = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0.0)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Generate data samples from noise."""
        self._z_cache = z
        h_pre = z @ self.w1 + self.b1
        h = self._relu(h_pre)
        self._h_cache = h
        x_fake = h @ self.w2 + self.b2
        return x_fake

    def parameters(self) -> Dict[str, np.ndarray]:
        """Return parameter dictionary."""
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    def update(self, params: Dict[str, np.ndarray]) -> None:
        """Update parameters from dictionary."""
        self.w1 = params["w1"]
        self.b1 = params["b1"]
        self.w2 = params["w2"]
        self.b2 = params["b2"]


class Discriminator:
    """Fully connected discriminator network."""

    def __init__(self, data_dim: int, hidden_dim: int) -> None:
        """Initialize discriminator parameters."""
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        limit1 = np.sqrt(1.0 / max(1, data_dim))
        self.w1 = np.random.uniform(
            -limit1, limit1, size=(data_dim, hidden_dim)
        ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w2 = np.random.uniform(
            -limit2, limit2, size=(hidden_dim, 1)
        ).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

        self._x_cache: Optional[np.ndarray] = None
        self._h_cache: Optional[np.ndarray] = None
        self._logits_cache: Optional[np.ndarray] = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute discriminator output probabilities."""
        self._x_cache = x
        h_pre = x @ self.w1 + self.b1
        h = self._relu(h_pre)
        self._h_cache = h
        logits = h @ self.w2 + self.b2
        self._logits_cache = logits
        probs = _sigmoid(logits)
        return probs

    def parameters(self) -> Dict[str, np.ndarray]:
        """Return parameter dictionary."""
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    def update(self, params: Dict[str, np.ndarray]) -> None:
        """Update parameters from dictionary."""
        self.w1 = params["w1"]
        self.b1 = params["b1"]
        self.w2 = params["w2"]
        self.b2 = params["b2"]

    def backward(
        self, grad_output: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute gradients w.r.t parameters from grad_output."""
        if (
            self._x_cache is None
            or self._h_cache is None
            or self._logits_cache is None
        ):
            raise RuntimeError("forward must be called before backward.")

        x = self._x_cache
        h = self._h_cache
        n = x.shape[0]

        grad_logits = grad_output

        grad_w2 = h.T @ grad_logits / float(n)
        grad_b2 = np.mean(grad_logits, axis=0)
        grad_h = grad_logits @ self.w2.T

        grad_h_pre = self._relu_backward(grad_h, h)

        grad_w1 = x.T @ grad_h_pre / float(n)
        grad_b1 = np.mean(grad_h_pre, axis=0)

        return {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }


def _gradient_step(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    learning_rate: float,
) -> Dict[str, np.ndarray]:
    """Apply a single gradient descent step."""
    updated = {}
    for name, p in params.items():
        updated[name] = p - learning_rate * grads[name]
    return updated


def generate_real_data(
    n_samples: int,
    data_dim: int,
    random_seed: int,
) -> np.ndarray:
    """Generate synthetic 2D data from a mixture of Gaussians."""
    rng = np.random.RandomState(random_seed)
    n_modes = 4
    samples_per_mode = n_samples // n_modes
    x_list = []
    for i in range(n_modes):
        angle = 2.0 * np.pi * i / n_modes
        center = np.array([np.cos(angle), np.sin(angle)]) * 4.0
        if data_dim > 2:
            pad = np.zeros(data_dim - 2)
            center = np.concatenate([center, pad])
        points = center + rng.randn(samples_per_mode, data_dim)
        x_list.append(points)
    x_all = np.vstack(x_list)
    rng.shuffle(x_all)
    return x_all.astype(np.float32)


class GANTrainer:
    """Coordinates adversarial training of generator and discriminator."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        noise_dim: int,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim

    def _sample_noise(self, n: int) -> np.ndarray:
        return np.random.randn(n, self.noise_dim).astype(np.float32)

    def train(
        self,
        x_real: np.ndarray,
        epochs: int,
        batch_size: int,
        lr_g: float,
        lr_d: float,
        d_steps: int = 1,
        g_steps: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the GAN on real data."""
        n_samples = x_real.shape[0]
        history: Dict[str, List[float]] = {"d_loss": [], "g_loss": []}

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            x_shuf = x_real[idx]
            d_losses: List[float] = []
            g_losses: List[float] = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuf[start:end]
                cur_batch_size = x_batch.shape[0]

                for _ in range(d_steps):
                    z = self._sample_noise(cur_batch_size)
                    x_fake = self.generator.forward(z)

                    d_real = self.discriminator.forward(x_batch)
                    d_fake = self.discriminator.forward(x_fake)

                    y_real = np.ones_like(d_real)
                    y_fake = np.zeros_like(d_fake)

                    loss_real, grad_real = _binary_cross_entropy(
                        d_real, y_real
                    )
                    loss_fake, grad_fake = _binary_cross_entropy(
                        d_fake, y_fake
                    )
                    d_loss = (loss_real + loss_fake) / 2.0

                    grad_d_total = (grad_real + grad_fake) / 2.0
                    grads_d = self.discriminator.backward(grad_d_total)
                    params_d = self.discriminator.parameters()
                    updated_d = _gradient_step(params_d, grads_d, lr_d)
                    self.discriminator.update(updated_d)
                    d_losses.append(d_loss)

                for _ in range(g_steps):
                    z = self._sample_noise(cur_batch_size)
                    x_fake = self.generator.forward(z)
                    d_fake = self.discriminator.forward(x_fake)

                    y_target = np.ones_like(d_fake)
                    loss_g, grad_d_out = _binary_cross_entropy(
                        d_fake, y_target
                    )

                    # Gradient through discriminator to fake inputs.
                    grad_logits = grad_d_out * d_fake * (1.0 - d_fake)
                    # dL/dh_d
                    h_d = self.discriminator._h_cache
                    grad_h_d = grad_logits @ self.discriminator.w2.T
                    # dL/d pre-activation of discriminator hidden
                    grad_h_d_pre = Discriminator._relu_backward(
                        grad_h_d, h_d
                    )
                    # dL/dx_fake
                    grad_x_fake = grad_h_d_pre @ self.discriminator.w1.T

                    if (
                        self.generator._z_cache is None
                        or self.generator._h_cache is None
                    ):
                        raise RuntimeError(
                            "Generator forward must be called before backward."
                        )
                    z_cache = self.generator._z_cache
                    h_cache = self.generator._h_cache
                    n_g = z_cache.shape[0]

                    # Generator: h_g -> x_fake = h_g @ w2_g + b2_g
                    # Gradient w.r.t generator parameters via chain rule.
                    grad_w2_gen = h_cache.T @ grad_x_fake / float(n_g)
                    grad_b2_gen = np.mean(grad_x_fake, axis=0)
                    grad_h_gen = grad_x_fake @ self.generator.w2.T

                    # Backprop through generator ReLU and first layer
                    grad_h_pre_gen = Generator._relu_backward(
                        grad_h_gen, h_cache
                    )
                    grad_w1_gen = z_cache.T @ grad_h_pre_gen / float(n_g)
                    grad_b1_gen = np.mean(grad_h_pre_gen, axis=0)

                    grads_g = {
                        "w1": grad_w1_gen,
                        "b1": grad_b1_gen,
                        "w2": grad_w2_gen,
                        "b2": grad_b2_gen,
                    }
                    params_g = self.generator.parameters()
                    updated_g = _gradient_step(params_g, grads_g, lr_g)
                    self.generator.update(updated_g)
                    g_losses.append(loss_g)

            avg_d_loss = float(np.mean(d_losses)) if d_losses else 0.0
            avg_g_loss = float(np.mean(g_losses)) if g_losses else 0.0
            history["d_loss"].append(avg_d_loss)
            history["g_loss"].append(avg_g_loss)

            if (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(
                    "Epoch %d/%d - D loss: %.6f - G loss: %.6f",
                    epoch + 1,
                    epochs,
                    avg_d_loss,
                    avg_g_loss,
                )

        return history


class GANRunner:
    """Orchestrates GAN training from configuration."""

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

    def run(self) -> Dict[str, float]:
        """Run GAN training and return final losses."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_samples = data_cfg.get("n_samples", 1000)
        data_dim = data_cfg.get("data_dim", 2)
        noise_dim = data_cfg.get("noise_dim", 8)
        seed = data_cfg.get("random_seed", 42)

        x_real = generate_real_data(
            n_samples=n_samples,
            data_dim=data_dim,
            random_seed=seed,
        )

        generator = Generator(
            noise_dim=noise_dim,
            hidden_dim=model_cfg.get("hidden_dim", 16),
            data_dim=data_dim,
        )
        discriminator = Discriminator(
            data_dim=data_dim,
            hidden_dim=model_cfg.get("hidden_dim", 16),
        )

        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            noise_dim=noise_dim,
        )

        history = trainer.train(
            x_real=x_real,
            epochs=train_cfg.get("epochs", 2000),
            batch_size=train_cfg.get("batch_size", 64),
            lr_g=train_cfg.get("learning_rate_generator", 0.001),
            lr_d=train_cfg.get("learning_rate_discriminator", 0.001),
            d_steps=train_cfg.get("d_steps", 1),
            g_steps=train_cfg.get("g_steps", 1),
        )

        final_d = history["d_loss"][-1] if history["d_loss"] else 0.0
        final_g = history["g_loss"][-1] if history["g_loss"] else 0.0
        logger.info(
            "Final losses - D: %.6f, G: %.6f",
            final_d,
            final_g,
        )
        return {
            "final_d_loss": float(final_d),
            "final_g_loss": float(final_g),
        }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a simple GAN (generator and discriminator) with NumPy"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = GANRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run()

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

