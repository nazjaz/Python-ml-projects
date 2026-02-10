"""Variational Autoencoder (VAE) with reparameterization and KL regularization.

Implements a simple fully connected VAE using NumPy:
- Encoder outputs mean and log-variance of latent Gaussian q(z|x)
- Reparameterization trick: z = mu + sigma * epsilon
- Decoder reconstructs inputs from latent samples
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


def _mse_loss(
    x_true: np.ndarray, x_pred: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute mean squared error loss and gradient."""
    diff = x_pred - x_true
    loss = float(np.mean(diff**2))
    grad = 2.0 * diff / float(x_true.shape[0])
    return loss, grad


def _kl_divergence_standard_normal(
    mu: np.ndarray, log_var: np.ndarray
) -> np.ndarray:
    """Compute KL divergence between N(mu, sigma^2) and N(0, 1) per sample."""
    return -0.5 * np.sum(1.0 + log_var - mu**2 - np.exp(log_var), axis=1)


class VariationalAutoencoder:
    """Fully connected variational autoencoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        """Initialize VAE parameters.

        Args:
            input_dim: Number of input features.
            hidden_dim: Size of hidden encoder/decoder layers.
            latent_dim: Size of latent representation.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        limit1 = np.sqrt(1.0 / max(1, input_dim))
        self.w1 = np.random.uniform(
            -limit1, limit1, size=(input_dim, hidden_dim)
        ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w_mu = np.random.uniform(
            -limit2, limit2, size=(hidden_dim, latent_dim)
        ).astype(np.float32)
        self.b_mu = np.zeros(latent_dim, dtype=np.float32)

        self.w_log_var = np.random.uniform(
            -limit2, limit2, size=(hidden_dim, latent_dim)
        ).astype(np.float32)
        self.b_log_var = np.zeros(latent_dim, dtype=np.float32)

        limit3 = np.sqrt(1.0 / max(1, latent_dim))
        self.w3 = np.random.uniform(
            -limit3, limit3, size=(latent_dim, hidden_dim)
        ).astype(np.float32)
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)

        limit4 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w4 = np.random.uniform(
            -limit4, limit4, size=(hidden_dim, input_dim)
        ).astype(np.float32)
        self.b4 = np.zeros(input_dim, dtype=np.float32)

        self._x_cache: Optional[np.ndarray] = None
        self._h1_cache: Optional[np.ndarray] = None
        self._mu_cache: Optional[np.ndarray] = None
        self._log_var_cache: Optional[np.ndarray] = None
        self._z_cache: Optional[np.ndarray] = None
        self._h3_cache: Optional[np.ndarray] = None

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        return grad * (x > 0.0)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input into latent distribution parameters."""
        h1 = self._relu(x @ self.w1 + self.b1)
        mu = h1 @ self.w_mu + self.b_mu
        log_var = h1 @ self.w_log_var + self.b_log_var
        return mu, log_var

    def reparameterize(
        self, mu: np.ndarray, log_var: np.ndarray
    ) -> np.ndarray:
        """Sample latent variable using reparameterization trick."""
        epsilon = np.random.randn(*mu.shape).astype(np.float32)
        std = np.exp(0.5 * log_var)
        return mu + std * epsilon

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent variable back to input space."""
        h3 = self._relu(z @ self.w3 + self.b3)
        x_recon = h3 @ self.w4 + self.b4
        return x_recon

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full VAE forward pass with caching for backpropagation."""
        self._x_cache = x
        h1_pre = x @ self.w1 + self.b1
        h1 = self._relu(h1_pre)
        self._h1_cache = h1

        mu = h1 @ self.w_mu + self.b_mu
        log_var = h1 @ self.w_log_var + self.b_log_var
        self._mu_cache = mu
        self._log_var_cache = log_var

        z = self.reparameterize(mu, log_var)
        self._z_cache = z

        h3_pre = z @ self.w3 + self.b3
        h3 = self._relu(h3_pre)
        self._h3_cache = h3

        x_recon = h3 @ self.w4 + self.b4
        return x_recon, mu, log_var

    def compute_loss_and_gradients(
        self, x: np.ndarray
    ) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
        """Compute total, reconstruction, and KL losses and gradients."""
        x_recon, mu, log_var = self.forward(x)
        recon_loss, grad_recon = _mse_loss(x, x_recon)

        kl_per_sample = _kl_divergence_standard_normal(mu, log_var)
        kl_loss = float(np.mean(kl_per_sample))

        total_loss = recon_loss + kl_loss

        if (
            self._x_cache is None
            or self._h1_cache is None
            or self._mu_cache is None
            or self._log_var_cache is None
            or self._z_cache is None
            or self._h3_cache is None
        ):
            raise RuntimeError("forward must be called before gradients.")

        x_cache = self._x_cache
        h1 = self._h1_cache
        mu = self._mu_cache
        log_var = self._log_var_cache
        z = self._z_cache
        h3 = self._h3_cache
        n = x_cache.shape[0]

        grad_w4 = h3.T @ grad_recon / float(n)
        grad_b4 = np.mean(grad_recon, axis=0)
        grad_h3 = grad_recon @ self.w4.T

        grad_h3_pre = self._relu_backward(grad_h3, h3)

        grad_w3 = z.T @ grad_h3_pre / float(n)
        grad_b3 = np.mean(grad_h3_pre, axis=0)
        grad_z = grad_h3_pre @ self.w3.T

        std = np.exp(0.5 * log_var)
        epsilon = (z - mu) / (std + 1e-8)
        grad_mu_from_z = grad_z
        grad_log_var_from_z = grad_z * std * 0.5 * epsilon

        grad_mu_kl = (mu / n).astype(np.float32)
        grad_log_var_kl = (
            0.5 * (np.exp(log_var) - 1.0) / n
        ).astype(np.float32)

        grad_mu_total = grad_mu_from_z + grad_mu_kl
        grad_log_var_total = grad_log_var_from_z + grad_log_var_kl

        grad_w_mu = h1.T @ grad_mu_total / float(1.0)
        grad_b_mu = np.mean(grad_mu_total, axis=0)

        grad_w_log_var = h1.T @ grad_log_var_total / float(1.0)
        grad_b_log_var = np.mean(grad_log_var_total, axis=0)

        grad_h1_from_mu = grad_mu_total @ self.w_mu.T
        grad_h1_from_log_var = grad_log_var_total @ self.w_log_var.T
        grad_h1_total = grad_h1_from_mu + grad_h1_from_log_var

        grad_h1_pre = self._relu_backward(grad_h1_total, h1)

        grad_w1 = x_cache.T @ grad_h1_pre / float(n)
        grad_b1 = np.mean(grad_h1_pre, axis=0)

        grads = {
            "w1": grad_w1,
            "b1": grad_b1,
            "w_mu": grad_w_mu,
            "b_mu": grad_b_mu,
            "w_log_var": grad_w_log_var,
            "b_log_var": grad_b_log_var,
            "w3": grad_w3,
            "b3": grad_b3,
            "w4": grad_w4,
            "b4": grad_b4,
        }
        return float(total_loss), float(recon_loss), float(kl_loss), grads

    def apply_gradients(
        self, grads: Dict[str, np.ndarray], learning_rate: float
    ) -> None:
        """Update parameters with gradient descent."""
        self.w1 -= learning_rate * grads["w1"]
        self.b1 -= learning_rate * grads["b1"]
        self.w_mu -= learning_rate * grads["w_mu"]
        self.b_mu -= learning_rate * grads["b_mu"]
        self.w_log_var -= learning_rate * grads["w_log_var"]
        self.b_log_var -= learning_rate * grads["b_log_var"]
        self.w3 -= learning_rate * grads["w3"]
        self.b3 -= learning_rate * grads["b3"]
        self.w4 -= learning_rate * grads["w4"]
        self.b4 -= learning_rate * grads["b4"]


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    random_seed: int,
) -> np.ndarray:
    """Generate synthetic Gaussian data for VAE training."""
    rng = np.random.RandomState(random_seed)
    center = rng.randn(n_features) * 2.0
    x = center + rng.randn(n_samples, n_features)
    return x.astype(np.float32)


def train_vae(
    model: VariationalAutoencoder,
    x: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Dict[str, List[float]]:
    """Train VAE using mini-batch gradient descent."""
    n_samples = x.shape[0]
    history: Dict[str, List[float]] = {
        "total_loss": [],
        "recon_loss": [],
        "kl_loss": [],
    }

    for epoch in range(epochs):
        idx = np.random.permutation(n_samples)
        x_shuf = x[idx]
        total_losses: List[float] = []
        recon_losses: List[float] = []
        kl_losses: List[float] = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = x_shuf[start:end]
            total_loss, recon_loss, kl_loss, grads = model.compute_loss_and_gradients(
                xb
            )
            model.apply_gradients(grads, learning_rate=learning_rate)
            total_losses.append(total_loss)
            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)

        avg_total = float(np.mean(total_losses))
        avg_recon = float(np.mean(recon_losses))
        avg_kl = float(np.mean(kl_losses))
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            logger.info(
                "Epoch %d/%d - total: %.6f - recon: %.6f - kl: %.6f",
                epoch + 1,
                epochs,
                avg_total,
                avg_recon,
                avg_kl,
            )
    return history


class VAERunner:
    """Orchestrates VAE training from configuration."""

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
        """Run VAE training and return final losses."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_samples = data_cfg.get("n_samples", 2000)
        n_features = data_cfg.get("n_features", 32)
        latent_dim = data_cfg.get("latent_dim", 4)
        seed = data_cfg.get("random_seed", 42)

        x = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            random_seed=seed,
        )

        model = VariationalAutoencoder(
            input_dim=n_features,
            hidden_dim=model_cfg.get("hidden_dim", 16),
            latent_dim=latent_dim,
        )

        history = train_vae(
            model,
            x,
            epochs=train_cfg.get("epochs", 50),
            learning_rate=train_cfg.get("learning_rate", 0.01),
            batch_size=train_cfg.get("batch_size", 64),
        )

        final_total = history["total_loss"][-1]
        final_recon = history["recon_loss"][-1]
        final_kl = history["kl_loss"][-1]
        logger.info(
            "Final losses - total: %.6f, recon: %.6f, kl: %.6f",
            final_total,
            final_recon,
            final_kl,
        )
        return {
            "final_total_loss": float(final_total),
            "final_recon_loss": float(final_recon),
            "final_kl_loss": float(final_kl),
        }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a variational autoencoder (VAE) with NumPy"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = VAERunner(
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

