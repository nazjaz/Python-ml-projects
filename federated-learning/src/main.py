"""Federated learning framework with distributed training and aggregation.

This module implements a simulated federated learning setup: multiple
clients with local data perform local training; a central server aggregates
model updates (FedAvg) and broadcasts the global model. Privacy-preserving
techniques include gradient clipping and differential privacy (Gaussian
noise on updates). Implemented in NumPy with in-process client simulation.
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


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -500, 500))
    return e / (np.sum(e, axis=axis, keepdims=True) + _EPS)


class MLP:
    """Two-layer ReLU MLP with get/set_weights for federation."""

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

    def get_weights(self) -> List[np.ndarray]:
        """Return list of parameter arrays for aggregation."""
        return [self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set parameters from list of arrays (same order as get_weights)."""
        self.w1 = np.asarray(weights[0], dtype=np.float32).copy()
        self.b1 = np.asarray(weights[1], dtype=np.float32).copy()
        self.w2 = np.asarray(weights[2], dtype=np.float32).copy()
        self.b2 = np.asarray(weights[3], dtype=np.float32).copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        self._h = h
        return h @ self.w2 + self.b2

    def backward(self, grad_out: np.ndarray, lr: float) -> None:
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


def clip_gradient_norm(
    weights: List[np.ndarray],
    ref_weights: List[np.ndarray],
    max_norm: float,
) -> List[np.ndarray]:
    """Clip (weights - ref_weights) to have L2 norm at most max_norm; return ref + clipped_delta."""
    delta = [w - r for w, r in zip(weights, ref_weights)]
    total_norm = np.sqrt(sum(np.sum(d ** 2) for d in delta))
    if total_norm <= max_norm or total_norm < _EPS:
        return weights
    scale = max_norm / (total_norm + _EPS)
    return [r + scale * d for r, d in zip(ref_weights, delta)]


def add_gaussian_noise(weights: List[np.ndarray], sigma: float) -> List[np.ndarray]:
    """Add zero-mean Gaussian noise with std sigma to each parameter array."""
    rng = np.random.default_rng()
    return [
        w + np.asarray(rng.normal(0, sigma, w.shape), dtype=np.float32)
        for w in weights
    ]


def load_data(
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load digits or synthetic data; return train_x, train_y, val_x, val_y."""
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        data = load_digits()
        x = np.asarray(data.data, dtype=np.float32) / 16.0
        y = np.asarray(data.target, dtype=np.int64)
        if max_samples is not None:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(len(x), min(max_samples, len(x)), replace=False)
            x, y = x[idx], y[idx]
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=train_ratio, random_state=random_seed
        )
        return x_train, y_train, x_val, y_val
    except ImportError:
        rng = np.random.default_rng(random_seed or 42)
        n = max_samples or 500
        x = np.asarray(rng.standard_normal((n, 64)), dtype=np.float32) * 0.1
        y = np.asarray(rng.integers(0, 10, size=n), dtype=np.int64)
        split = int(n * train_ratio)
        return x[:split], y[:split], x[split:], y[split:]


def partition_data(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    iid: bool = True,
    random_seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition (x, y) into num_clients shards. If iid, random split; else by label."""
    rng = np.random.default_rng(random_seed)
    if iid:
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]
        sizes = [len(x) // num_clients] * num_clients
        for i in range(len(x) % num_clients):
            sizes[i] += 1
        start = 0
        shards = []
        for s in sizes:
            end = start + s
            shards.append((x[start:end], y[start:end]))
            start = end
        return shards
    order = np.argsort(y)
    x, y = x[order], y[order]
    sizes = [len(x) // num_clients] * num_clients
    for i in range(len(x) % num_clients):
        sizes[i] += 1
    start = 0
    shards = []
    for s in sizes:
        end = start + s
        shards.append((x[start:end], y[start:end]))
        start = end
    return shards


class FederatedClient:
    """Client with local data and model; performs local training and optional privacy."""

    def __init__(
        self,
        client_id: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        local_epochs: int = 1,
        lr: float = 0.01,
        batch_size: int = 32,
        max_grad_norm: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.client_id = client_id
        self.train_x = train_x
        self.train_y = train_y
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.noise_sigma = noise_sigma
        self.model = MLP(in_dim, hidden_dim, out_dim, random_seed=random_seed)
        self._rng = np.random.default_rng(random_seed)

    def train_local(
        self,
        global_weights: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], int]:
        """
        Set model to global weights, train on local data, return updated weights
        and local sample count (for weighted aggregation).
        """
        self.model.set_weights(global_weights)
        n = self.train_x.shape[0]
        out_dim = int(self.train_y.max()) + 1

        for _ in range(self.local_epochs):
            perm = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                x = self.train_x[idx]
                y = self.train_y[idx]
                logits = self.model.forward(x)
                probs = softmax(logits, axis=1)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(y)), y.astype(int)] = 1.0
                grad = (probs - one_hot) / float(x.shape[0])
                self.model.backward(grad, self.lr)

        new_weights = self.model.get_weights()
        if self.max_grad_norm is not None:
            new_weights = clip_gradient_norm(
                new_weights, global_weights, self.max_grad_norm
            )
        if self.noise_sigma is not None and self.noise_sigma > 0:
            delta = [nw - gw for nw, gw in zip(new_weights, global_weights)]
            noisy_delta = add_gaussian_noise(delta, self.noise_sigma)
            new_weights = [gw + nd for gw, nd in zip(global_weights, noisy_delta)]
        return new_weights, n


def aggregate_fedavg(
    client_weights: List[List[np.ndarray]],
    client_counts: List[int],
) -> List[np.ndarray]:
    """Weighted average of client weights by sample count (FedAvg)."""
    total = sum(client_counts)
    if total <= 0:
        return client_weights[0]
    out = []
    for i in range(len(client_weights[0])):
        w = np.zeros_like(client_weights[0][i], dtype=np.float64)
        for cw, n in zip(client_weights, client_counts):
            w += cw[i].astype(np.float64) * (n / total)
        out.append(w.astype(np.float32))
    return out


class FederatedServer:
    """Central server: holds global model, runs rounds of broadcast-aggregate-broadcast."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        clients: List[FederatedClient],
        random_seed: Optional[int] = None,
    ) -> None:
        self.global_model = MLP(in_dim, hidden_dim, out_dim, random_seed=random_seed)
        self.clients = clients

    def run_round(self) -> Dict[str, float]:
        """One round: broadcast global weights, collect client updates, aggregate, update global."""
        global_weights = self.global_model.get_weights()
        client_weights = []
        client_counts = []
        for client in self.clients:
            if client.train_x.shape[0] == 0:
                continue
            w, n = client.train_local(global_weights)
            client_weights.append(w)
            client_counts.append(n)
        if not client_weights:
            return {}
        new_global = aggregate_fedavg(client_weights, client_counts)
        self.global_model.set_weights(new_global)
        return {"clients_aggregated": len(client_weights)}

    def get_global_weights(self) -> List[np.ndarray]:
        return self.global_model.get_weights()


def evaluate_global(
    server: FederatedServer,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> Tuple[float, float]:
    """Compute accuracy and mean CE of global model on val data."""
    logits = server.global_model.forward(val_x)
    probs = softmax(logits, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred.astype(np.int64) == val_y.astype(np.int64)))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(val_y)), val_y.astype(int)] = 1.0
    ce = float(np.mean(-np.sum(one_hot * np.log(probs + _EPS), axis=1)))
    return acc, ce


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 1
    hidden_dim: int = 32
    learning_rate: float = 0.01
    batch_size: int = 32
    train_ratio: float = 0.8
    max_samples: Optional[int] = None
    iid: bool = True
    max_grad_norm: Optional[float] = 10.0
    noise_sigma: Optional[float] = None
    random_seed: Optional[int] = 0


def run_federated(config: FederatedConfig) -> Dict:
    """Partition data, create clients and server, run rounds, return metrics."""
    train_x, train_y, val_x, val_y = load_data(
        train_ratio=config.train_ratio,
        max_samples=config.max_samples,
        random_seed=config.random_seed,
    )
    in_dim = train_x.shape[1]
    out_dim = int(train_y.max()) + 1

    shards = partition_data(
        train_x, train_y,
        num_clients=config.num_clients,
        iid=config.iid,
        random_seed=config.random_seed,
    )
    clients = [
        FederatedClient(
            i,
            shards[i][0], shards[i][1],
            in_dim, config.hidden_dim, out_dim,
            local_epochs=config.local_epochs,
            lr=config.learning_rate,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            noise_sigma=config.noise_sigma,
            random_seed=config.random_seed + i if config.random_seed is not None else None,
        )
        for i in range(len(shards))
    ]
    server = FederatedServer(
        in_dim, config.hidden_dim, out_dim,
        clients,
        random_seed=config.random_seed,
    )

    history = []
    for r in range(config.num_rounds):
        server.run_round()
        acc, ce = evaluate_global(server, val_x, val_y)
        history.append({"round": r, "val_accuracy": acc, "val_ce": ce})
        logger.info("Round %d val_accuracy=%.4f", r, acc)

    final_acc, final_ce = evaluate_global(server, val_x, val_y)
    return {
        "final_val_accuracy": final_acc,
        "final_val_ce": final_ce,
        "num_rounds": config.num_rounds,
        "num_clients": config.num_clients,
        "history": history,
    }


def main() -> None:
    """Entry point: parse args, run federated learning, print and optionally save results."""
    parser = argparse.ArgumentParser(description="Federated learning with aggregation and privacy")
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

    d = config_dict.get("federated", {})
    config = FederatedConfig(
        num_clients=d.get("num_clients", 5),
        num_rounds=d.get("num_rounds", 10),
        local_epochs=d.get("local_epochs", 1),
        hidden_dim=d.get("hidden_dim", 32),
        learning_rate=d.get("learning_rate", 0.01),
        batch_size=d.get("batch_size", 32),
        train_ratio=d.get("train_ratio", 0.8),
        max_samples=d.get("max_samples"),
        iid=d.get("iid", True),
        max_grad_norm=d.get("max_grad_norm"),
        noise_sigma=d.get("noise_sigma"),
        random_seed=d.get("random_seed", 0),
    )

    results = run_federated(config)
    out = {k: v for k, v in results.items() if k != "history"}
    print("\nFederated Learning Results:")
    print("========================================")
    for k, v in out.items():
        if isinstance(v, float):
            print("  %s: %.4f" % (k, v))
        else:
            print("  %s: %s" % (k, v))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out["history_length"] = len(results["history"])
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
