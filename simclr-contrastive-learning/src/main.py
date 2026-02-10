"""Contrastive learning with SimCLR for self-supervised representation learning.

This module implements a SimCLR-style pipeline: two augmented views per
sample, shared encoder and projection head, and NT-Xent (InfoNCE) contrastive
loss. Uses a small input space (e.g. flattened digits) with simple
augmentations (noise, scaling). Implemented in NumPy.
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


def make_two_views(
    x: np.ndarray,
    noise_std: float = 0.1,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create two augmented views of each sample (SimCLR positive pairs)."""
    rng = np.random.default_rng(random_seed)
    x = np.asarray(x, dtype=np.float32)
    scale1 = rng.uniform(scale_range[0], scale_range[1], (x.shape[0], 1))
    scale2 = rng.uniform(scale_range[0], scale_range[1], (x.shape[0], 1))
    noise1 = rng.normal(0, noise_std, x.shape).astype(np.float32)
    noise2 = rng.normal(0, noise_std, x.shape).astype(np.float32)
    v1 = x * scale1 + noise1
    v2 = x * scale2 + noise2
    return v1, v2


class MLPBlock:
    """Two-layer MLP with ReLU."""

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

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        if self._x is None or self._h is None:
            raise RuntimeError("Forward before backward.")
        x, h = self._x, self._h
        batch = x.shape[0]
        d_h = grad_out @ self.w2.T
        d_z = d_h * (h > 0.0).astype(np.float32)
        self.w2 -= lr * (self._h.T @ grad_out) / float(batch)
        self.b2 -= lr * np.mean(grad_out, axis=0)
        self.w1 -= lr * (x.T @ d_z) / float(batch)
        self.b1 -= lr * np.mean(d_z, axis=0)
        return d_z @ self.w1.T


class SimCLRNet:
    """Encoder + projection head for SimCLR."""

    def __init__(
        self,
        in_dim: int,
        repr_dim: int,
        proj_dim: int,
        encoder_hidden: int = 256,
        proj_hidden: int = 128,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self.encoder = MLPBlock(in_dim, encoder_hidden, repr_dim, random_seed=random_seed)
        self.projection = MLPBlock(
            repr_dim, proj_hidden, proj_dim, random_seed=random_seed
        )
        self.repr_dim = repr_dim
        self.proj_dim = proj_dim
        self._repr: Optional[np.ndarray] = None

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Representation (before projection)."""
        self._repr = self.encoder.forward(x)
        return self._repr

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward: encode then project."""
        h = self.encode(x)
        return self.projection.forward(h)

    def backward_projection(self, grad_proj: np.ndarray, lr: float) -> np.ndarray:
        """Backward through projection head; returns grad w.r.t. representation."""
        return self.projection.backward(grad_proj, lr)

    def backward_encoder(self, grad_repr: np.ndarray, lr: float) -> None:
        """Backward through encoder."""
        self.encoder.backward(grad_repr, lr)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize along axis."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + _EPS)


def nt_xent_loss(
    z: np.ndarray,
    temperature: float,
    pair_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[float, np.ndarray]:
    """
    NT-Xent (InfoNCE) contrastive loss. z has shape (2*N, proj_dim); rows 2k and 2k+1
    are the two views of sample k. Returns (loss, grad_z) where grad_z is w.r.t. input z.
    """
    z_norm = l2_normalize(z, axis=1)
    N = z.shape[0] // 2
    tau = max(temperature, _EPS)
    sim = (z_norm @ z_norm.T) / tau
    np.fill_diagonal(sim, -1e9)
    exp_sim = np.exp(np.clip(sim, -50, 50))
    grad_z_norm = np.zeros_like(z_norm)
    loss_sum = 0.0
    for i in range(2 * N):
        pos_i = i + 1 if i % 2 == 0 else i - 1
        denom = np.sum(exp_sim[i, :]) + _EPS
        num = exp_sim[i, pos_i] + _EPS
        loss_sum -= np.log(num / denom)
        for j in range(2 * N):
            if i == j:
                continue
            p_ij = exp_sim[i, j] / denom
            if j == pos_i:
                grad_z_norm[i] += (p_ij - 1.0) * z_norm[j]
            else:
                grad_z_norm[i] += p_ij * z_norm[j]
        grad_z_norm[i] /= tau
    loss = loss_sum / (2.0 * N)
    grad_z_norm /= (2.0 * N)
    nrm = np.linalg.norm(z, axis=1, keepdims=True) + _EPS
    grad_z = (grad_z_norm - (grad_z_norm * z_norm).sum(axis=1, keepdims=True) * z_norm) / nrm
    return float(loss), grad_z.astype(np.float32)


def load_data(
    max_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Load digits or synthetic data; return feature matrix (no labels for SimCLR)."""
    try:
        from sklearn.datasets import load_digits
        data = load_digits()
        x = np.asarray(data.data, dtype=np.float32) / 16.0
        if max_samples is not None:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(len(x), min(max_samples, len(x)), replace=False)
            x = x[idx]
        return x
    except ImportError:
        rng = np.random.default_rng(random_seed or 42)
        n = max_samples or 500
        return np.asarray(rng.standard_normal((n, 64)), dtype=np.float32) * 0.1


def train_simclr(
    net: SimCLRNet,
    data: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    noise_std: float,
    scale_range: Tuple[float, float],
    random_seed: Optional[int] = None,
) -> List[float]:
    """Train SimCLR with NT-Xent; return per-epoch mean loss."""
    rng = np.random.default_rng(random_seed)
    losses: List[float] = []
    n = data.shape[0]

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            batch = data[idx]
            v1, v2 = make_two_views(
                batch, noise_std=noise_std, scale_range=scale_range,
                random_seed=rng.integers(0, 2**31),
            )
            views = np.concatenate([v1, v2], axis=0)
            z = net.forward(views)
            loss, grad_z = nt_xent_loss(z, temperature)
            grad_repr = net.backward_projection(grad_z, lr)
            net.backward_encoder(grad_repr, lr)
            epoch_loss += loss
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
        logger.info("SimCLR epoch %d loss=%.4f", epoch, losses[-1])
    return losses


def evaluate_representation(
    net: SimCLRNet,
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return representation vectors (no projection) for downstream use."""
    return net.encode(x)


@dataclass
class SimCLRConfig:
    """Configuration for SimCLR training."""

    repr_dim: int = 128
    proj_dim: int = 64
    encoder_hidden: int = 256
    proj_hidden: int = 128
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    temperature: float = 0.5
    noise_std: float = 0.1
    scale_low: float = 0.8
    scale_high: float = 1.2
    max_samples: Optional[int] = None
    random_seed: Optional[int] = 0


def run_simclr(config: SimCLRConfig) -> Dict:
    """Load data, train SimCLR, return final loss and config summary."""
    data = load_data(max_samples=config.max_samples, random_seed=config.random_seed)
    in_dim = data.shape[1]
    net = SimCLRNet(
        in_dim,
        config.repr_dim,
        config.proj_dim,
        encoder_hidden=config.encoder_hidden,
        proj_hidden=config.proj_hidden,
        random_seed=config.random_seed,
    )
    losses = train_simclr(
        net,
        data,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        temperature=config.temperature,
        noise_std=config.noise_std,
        scale_range=(config.scale_low, config.scale_high),
        random_seed=config.random_seed,
    )
    return {
        "final_loss": losses[-1] if losses else 0.0,
        "num_samples": data.shape[0],
        "in_dim": in_dim,
        "repr_dim": config.repr_dim,
        "proj_dim": config.proj_dim,
    }


def main() -> None:
    """Entry point: parse args, run SimCLR, print and optionally save results."""
    parser = argparse.ArgumentParser(
        description="SimCLR contrastive learning for self-supervised representations"
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

    d = config_dict.get("simclr", {})
    config = SimCLRConfig(
        repr_dim=d.get("repr_dim", 128),
        proj_dim=d.get("proj_dim", 64),
        encoder_hidden=d.get("encoder_hidden", 256),
        proj_hidden=d.get("proj_hidden", 128),
        epochs=d.get("epochs", 10),
        batch_size=d.get("batch_size", 64),
        learning_rate=d.get("learning_rate", 0.001),
        temperature=d.get("temperature", 0.5),
        noise_std=d.get("noise_std", 0.1),
        scale_low=d.get("scale_low", 0.8),
        scale_high=d.get("scale_high", 1.2),
        max_samples=d.get("max_samples"),
        random_seed=d.get("random_seed", 0),
    )

    results = run_simclr(config)
    print("\nSimCLR Results:")
    print("========================================")
    for k, v in results.items():
        if isinstance(v, float):
            print("  %s: %.4f" % (k, v))
        else:
            print("  %s: %s" % (k, v))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
