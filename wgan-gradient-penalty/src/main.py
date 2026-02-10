"""Wasserstein GAN with Gradient Penalty (WGAN-GP) using NumPy.

Implements a simple WGAN-GP on 2D synthetic data:
- Generator maps noise vectors to 2D samples
- Critic outputs scalar scores without sigmoid
- Training uses Wasserstein losses and a numerical gradient penalty term
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


class WGANGenerator:
    """Fully connected generator network for WGAN-GP."""

    def __init__(self, noise_dim: int, hidden_dim: int, data_dim: int) -> None:
        """Initialize generator parameters."""
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim

        limit1 = np.sqrt(1.0 / max(1, noise_dim))
        self.w1 = (np.random.randn(noise_dim, hidden_dim) * limit1).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w2 = (np.random.randn(hidden_dim, data_dim) * limit2).astype(
            np.float32
        )
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
        """Generate samples from noise."""
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


class WGANCritic:
    """Fully connected critic network for WGAN-GP."""

    def __init__(self, data_dim: int, hidden_dim: int) -> None:
        """Initialize critic parameters."""
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        limit1 = np.sqrt(1.0 / max(1, data_dim))
        self.w1 = (np.random.randn(data_dim, hidden_dim) * limit1).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w2 = (np.random.randn(hidden_dim, 1) * limit2).astype(
            np.float32
        )
        self.b2 = np.zeros(1, dtype=np.float32)

        self._x_cache: Optional[np.ndarray] = None
        self._h_cache: Optional[np.ndarray] = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute critic scores for inputs."""
        self._x_cache = x
        h_pre = x @ self.w1 + self.b1
        h = self._relu(h_pre)
        self._h_cache = h
        scores = h @ self.w2 + self.b2
        return scores  # shape: (batch_size, 1)

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

    def backward_params(self, grad_scores: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients of critic parameters given gradient on scores."""
        if self._x_cache is None or self._h_cache is None:
            raise RuntimeError("forward must be called before backward_params.")

        x = self._x_cache
        h = self._h_cache
        n = x.shape[0]

        grad_w2 = h.T @ grad_scores / float(n)
        grad_b2 = np.mean(grad_scores, axis=0)
        grad_h = grad_scores @ self.w2.T

        grad_h_pre = self._relu_backward(grad_h, h)
        grad_w1 = x.T @ grad_h_pre / float(n)
        grad_b1 = np.mean(grad_h_pre, axis=0)

        return {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }

    def backward_input(self, grad_scores: np.ndarray) -> np.ndarray:
        """Compute gradient of scores with respect to inputs (for GP)."""
        if self._x_cache is None or self._h_cache is None:
            raise RuntimeError("forward must be called before backward_input.")

        x = self._x_cache
        h = self._h_cache

        grad_h = grad_scores @ self.w2.T
        grad_h_pre = self._relu_backward(grad_h, h)
        grad_x = grad_h_pre @ self.w1.T
        assert grad_x.shape == x.shape
        return grad_x


def _gradient_step(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    learning_rate: float,
) -> Dict[str, np.ndarray]:
    """Apply a gradient descent step to parameters."""
    updated = {}
    for name, p in params.items():
        updated[name] = p - learning_rate * grads[name]
    return updated


def generate_real_data(
    n_samples: int, data_dim: int, random_seed: int
) :  # -> np.ndarray
    """Generate synthetic 2D Gaussian mixture data."""
    rng = np.random.RandomState(random_seed)
    samples_per_comp = n_samples // 4
    x_list = []
    for k in range(4):
        angle = 2.0 * np.pi * k / 4.0
        center = np.array([np.cos(angle), np.sin(angle)]) * 3.0
        if data_dim > 2:
            pad = np.zeros(data_dim - 2)
            center = np.concatenate([center, pad])
        points = center + rng.randn(samples_per_comp, data_dim)
        x_list.append(points)
    x = np.vstack(x_list)
    rng.shuffle(x)
    return x.astype(np.float32)


def compute_gradient_penalty(
    critic: WGANCritic,
    real: np.ndarray,
    fake: np.ndarray,
    lambda_gp: float,
) -> Tuple[float, float]:
    """Compute gradient penalty term and mean gradient norm.

    Note: This implementation estimates gradient norms using critic.backward_input
    and uses them only to compute the penalty term; parameter gradients from this
    term are not propagated (for simplicity in a NumPy-only setting).
    """
    batch_size = real.shape[0]
    alpha = np.random.rand(batch_size, 1).astype(np.float32)
    alpha = np.repeat(alpha, real.shape[1], axis=1)
    interpolates = alpha * real + (1.0 - alpha) * fake

    scores = critic.forward(interpolates)
    grad_scores = np.ones_like(scores, dtype=np.float32)
    grad_x = critic.backward_input(grad_scores)
    grad_norm = np.sqrt(np.sum(grad_x**2, axis=1) + 1e-12)
    penalty = lambda_gp * float(np.mean((grad_norm - 1.0) ** 2))
    return penalty, float(np.mean(grad_norm))


class WGANTrainer:
    """Training loop for WGAN-GP."""

    def __init__(
        self,
        generator: WGANGenerator,
        critic: WGANCritic,
        noise_dim: int,
        lambda_gp: float,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.noise_dim = noise_dim
        self.lambda_gp = lambda_gp

    def _sample_noise(self, batch_size: int) -> np.ndarray:
        return np.random.randn(batch_size, self.noise_dim).astype(np.float32)

    def train(
        self,
        x_real: np.ndarray,
        epochs: int,
        batch_size: int,
        lr_g: float,
        lr_c: float,
        critic_iters: int,
    ) -> Dict[str, List[float]]:
        """Train WGAN-GP on the provided real data."""
        n_samples = x_real.shape[0]
        history: Dict[str, List[float]] = {
            "critic_loss": [],
            "generator_loss": [],
        }

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            x_shuf = x_real[idx]
            critic_losses: List[float] = []
            gen_losses: List[float] = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                real_batch = x_shuf[start:end]
                bsz = real_batch.shape[0]

                # Train critic multiple times
                for _ in range(critic_iters):
                    z = self._sample_noise(bsz)
                    fake_batch = self.generator.forward(z)
                    real_scores = self.critic.forward(real_batch)
                    fake_scores = self.critic.forward(fake_batch)

                    wasserstein = float(
                        np.mean(fake_scores) - np.mean(real_scores)
                    )

                    gp, _ = compute_gradient_penalty(
                        self.critic,
                        real_batch,
                        fake_batch,
                        self.lambda_gp,
                    )
                    critic_loss = wasserstein + gp

                    grad_real = -np.ones_like(real_scores) / float(bsz)
                    grad_fake = np.ones_like(fake_scores) / float(bsz)
                    grad_scores = grad_real + grad_fake

                    grads_c = self.critic.backward_params(grad_scores)
                    params_c = self.critic.parameters()
                    updated_c = _gradient_step(params_c, grads_c, lr_c)
                    self.critic.update(updated_c)

                    critic_losses.append(critic_loss)

                # Train generator
                z = self._sample_noise(bsz)
                fake_batch = self.generator.forward(z)
                fake_scores = self.critic.forward(fake_batch)
                gen_loss = -float(np.mean(fake_scores))

                grad_scores_g = -np.ones_like(fake_scores) / float(bsz)
                grad_x_fake = self.critic.backward_input(grad_scores_g)

                h = self.generator._h_cache
                if h is None or self.generator._z_cache is None:
                    raise RuntimeError("Generator forward cache is empty.")
                z_cache = self.generator._z_cache

                grad_w2_g = h.T @ grad_x_fake / float(bsz)
                grad_b2_g = np.mean(grad_x_fake, axis=0)
                grad_h_g = grad_x_fake @ self.generator.w2.T
                grad_h_pre_g = WGANGenerator._relu_backward(grad_h_g, h)
                grad_w1_g = z_cache.T @ grad_h_pre_g / float(bsz)
                grad_b1_g = np.mean(grad_h_pre_g, axis=0)

                grads_g = {
                    "w1": grad_w1_g,
                    "b1": grad_b1_g,
                    "w2": grad_w2_g,
                    "b2": grad_b2_g,
                }
                params_g = self.generator.parameters()
                updated_g = _gradient_step(params_g, grads_g, lr_g)
                self.generator.update(updated_g)

                gen_losses.append(gen_loss)

            mean_c_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
            mean_g_loss = float(np.mean(gen_losses)) if gen_losses else 0.0
            history["critic_loss"].append(mean_c_loss)
            history["generator_loss"].append(mean_g_loss)

            if (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(
                    "Epoch %d/%d - critic_loss: %.6f - gen_loss: %.6f",
                    epoch + 1,
                    epochs,
                    mean_c_loss,
                    mean_g_loss,
                )

        return history


class WGANRunner:
    """Runner that loads configuration and executes WGAN-GP training."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found: %s, using defaults", config_path)
            return {}

    def _setup_logging(self) -> None:
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
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_samples = data_cfg.get("n_samples", 1000)
        data_dim = data_cfg.get("data_dim", 2)
        noise_dim = data_cfg.get("noise_dim", 8)
        seed = data_cfg.get("random_seed", 42)

        x_real = generate_real_data(n_samples, data_dim, seed)

        generator = WGANGenerator(
            noise_dim=noise_dim,
            hidden_dim=model_cfg.get("hidden_dim", 32),
            data_dim=data_dim,
        )
        critic = WGANCritic(
            data_dim=data_dim,
            hidden_dim=model_cfg.get("hidden_dim", 32),
        )

        trainer = WGANTrainer(
            generator=generator,
            critic=critic,
            noise_dim=noise_dim,
            lambda_gp=train_cfg.get("lambda_gp", 10.0),
        )

        history = trainer.train(
            x_real=x_real,
            epochs=train_cfg.get("epochs", 1000),
            batch_size=train_cfg.get("batch_size", 64),
            lr_g=train_cfg.get("learning_rate_generator", 0.0005),
            lr_c=train_cfg.get("learning_rate_critic", 0.0005),
            critic_iters=train_cfg.get("critic_iters", 5),
        )

        final_c = history["critic_loss"][-1]
        final_g = history["generator_loss"][-1]
        logger.info(
            "Final WGAN-GP - critic_loss: %.6f, generator_loss: %.6f",
            final_c,
            final_g,
        )
        return {
            "final_critic_loss": float(final_c),
            "final_generator_loss": float(final_g),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a WGAN-GP on synthetic 2D data"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = WGANRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run()

    print("\nFinal Results:")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()

