"""Diffusion models for image generation with forward and reverse processes.

This module implements a simplified DDPM-style diffusion model using NumPy:
- Forward diffusion: gradually adds Gaussian noise to training images
- Reverse diffusion: learns to predict the noise and iteratively denoise

We use small 8x8 grayscale "images" (e.g. digits) flattened to vectors.
The model is a simple MLP that takes a noisy image and timestep embedding
and predicts the noise. Training objective is mean squared error between
true and predicted noise. Sampling starts from pure noise and runs the
reverse process.
"""

import argparse
import json
import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_EPS = 1e-8


def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Cosine schedule for betas (noise variance) as in DDPM++ style."""
    steps = T + 1
    x = np.linspace(0, T, steps)
    alphas_cumprod = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas.astype(np.float32), 1e-4, 0.999)


class MLP:
    """Two-layer MLP with ReLU for noise prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        limit1 = np.sqrt(1.0 / max(1, in_dim))
        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w1 = np.random.uniform(-limit1, limit1, (in_dim, hidden_dim)).astype(
            np.float32
        )
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w2 = np.random.uniform(-limit2, limit2, (hidden_dim, out_dim)).astype(
            np.float32
        )
        self.b2 = np.zeros((out_dim,), dtype=np.float32)
        self._x: Optional[np.ndarray] = None
        self._h: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        self._h = h
        return h @ self.w2 + self.b2

    def backward(self, grad_out: np.ndarray, lr: float) -> None:
        if self._x is None or self._h is None:
            raise RuntimeError("Forward must be called before backward.")
        x, h = self._x, self._h
        batch = x.shape[0]
        d_h = grad_out @ self.w2.T
        d_z = d_h * (h > 0.0).astype(np.float32)
        self.w2 -= lr * (self._h.T @ grad_out) / float(batch)
        self.b2 -= lr * np.mean(grad_out, axis=0)
        self.w1 -= lr * (x.T @ d_z) / float(batch)
        self.b1 -= lr * np.mean(d_z, axis=0)


def timestep_embedding(t: np.ndarray, dim: int) -> np.ndarray:
    """Simple sinusoidal timestep embedding."""
    half = dim // 2
    freqs = np.exp(
        -np.log(10000.0)
        * np.arange(0, half, dtype=np.float32)
        / max(half - 1, 1)
    )
    args = t[:, None].astype(np.float32) * freqs[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=1)
    if dim % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)))
    return emb


def load_data(
    max_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Load digits or synthetic 8x8 grayscale images; return flattened array in [-1, 1]."""
    try:
        from sklearn.datasets import load_digits

        data = load_digits()
        x = np.asarray(data.images, dtype=np.float32)
        x = (x / 16.0) * 2.0 - 1.0
        x = x.reshape(x.shape[0], -1)
        if max_samples is not None:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(len(x), min(max_samples, len(x)), replace=False)
            x = x[idx]
        return x
    except ImportError:
        rng = np.random.default_rng(random_seed or 42)
        n = max_samples or 500
        x = rng.standard_normal((n, 64)).astype(np.float32)
        x = np.tanh(x)
        return x


@dataclass
class DiffusionConfig:
    """Configuration for diffusion training and sampling."""

    timesteps: int = 100
    hidden_dim: int = 128
    emb_dim: int = 32
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    max_samples: Optional[int] = None
    random_seed: Optional[int] = 0


class DiffusionModel:
    """Forward and reverse diffusion with an MLP noise predictor."""

    def __init__(self, config: DiffusionConfig, data_dim: int) -> None:
        self.config = config
        self.data_dim = data_dim
        self.betas = cosine_beta_schedule(config.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.posterior_var = np.zeros_like(self.betas, dtype=np.float32)
        self.posterior_var[0] = self.betas[0]
        self.posterior_var[1:] = (
            self.betas[1:]
            * (1.0 - self.alphas_cumprod[:-1])
            / (1.0 - self.alphas_cumprod[1:] + _EPS)
        )
        in_dim = data_dim + config.emb_dim
        self.net = MLP(in_dim, config.hidden_dim, data_dim, random_seed=config.random_seed)

    def q_sample(self, x0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample x_t from q(x_t | x_0) with noise epsilon; returns (x_t, epsilon)."""
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(np.float32)
        alpha_bar = self.sqrt_alphas_cumprod[t][:, None]
        sigma = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        x_t = alpha_bar * x0 + sigma * noise
        return x_t, noise

    def predict_noise(self, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        emb = timestep_embedding(t, self.config.emb_dim)
        inp = np.concatenate([x_t, emb], axis=1)
        return self.net.forward(inp)

    def train(self, data: np.ndarray) -> List[float]:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        n = data.shape[0]
        losses: List[float] = []
        for epoch in range(cfg.epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, cfg.batch_size):
                end = min(start + cfg.batch_size, n)
                idx = perm[start:end]
                x0 = data[idx]
                bs = x0.shape[0]
                t = rng.integers(0, cfg.timesteps, size=bs, dtype=np.int64)
                x_t, eps = self.q_sample(x0, t)
                eps_pred = self.predict_noise(x_t, t)
                grad = (eps_pred - eps) * (2.0 / float(bs))
                self.net.backward(grad, cfg.learning_rate)
                mse = float(np.mean((eps_pred - eps) ** 2))
                epoch_loss += mse
                n_batches += 1
            loss = epoch_loss / max(n_batches, 1)
            losses.append(loss)
            logger.info("Diffusion epoch %d loss=%.4f", epoch, loss)
        return losses

    def p_sample(self, x_t: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        """Sample x_{t-1} from learned reverse process p_theta(x_{t-1} | x_t)."""
        bs = x_t.shape[0]
        t_arr = np.full(bs, t, dtype=np.int64)
        eps_theta = self.predict_noise(x_t, t_arr)
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        coef = (1.0 / np.sqrt(alpha_t))
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / np.sqrt(
            alpha_bar_t + _EPS
        )
        mean = coef * (
            x_t
            - self.betas[t]
            / (1.0 - alpha_bar_t + _EPS)
            * eps_theta
        )
        if t == 0:
            return mean
        var = self.posterior_var[t - 1]
        noise = rng.standard_normal(x_t.shape).astype(np.float32)
        return mean + np.sqrt(var) * noise

    def sample(self, num_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate samples by starting from noise and running reverse diffusion."""
        if rng is None:
            rng = np.random.default_rng(self.config.random_seed)
        x_t = rng.standard_normal((num_samples, self.data_dim)).astype(np.float32)
        for t in reversed(range(self.config.timesteps)):
            x_t = self.p_sample(x_t, t, rng)
        return np.clip(x_t, -1.0, 1.0)


def run_diffusion(config: DiffusionConfig) -> Dict:
    data = load_data(max_samples=config.max_samples, random_seed=config.random_seed)
    data_dim = data.shape[1]
    model = DiffusionModel(config, data_dim)
    losses = model.train(data)
    samples = model.sample(num_samples=4)
    return {
        "final_loss": losses[-1] if losses else 0.0,
        "num_samples": data.shape[0],
        "data_dim": data_dim,
        "samples_shape": samples.shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion model for image generation (forward and reverse processes)"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent.parent / "config.yaml"
    config_dict: Dict = {}
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config not found: %s", config_path)

    log_cfg = config_dict.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "logs/app.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.setLevel(level)
    logger.addHandler(handler)

    d = config_dict.get("diffusion", {})
    config = DiffusionConfig(
        timesteps=d.get("timesteps", 100),
        hidden_dim=d.get("hidden_dim", 128),
        emb_dim=d.get("emb_dim", 32),
        epochs=d.get("epochs", 10),
        batch_size=d.get("batch_size", 64),
        learning_rate=d.get("learning_rate", 0.001),
        max_samples=d.get("max_samples"),
        random_seed=d.get("random_seed", 0),
    )

    results = run_diffusion(config)
    print("\nDiffusion Results:")
    print("========================================")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
