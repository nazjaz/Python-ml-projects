"""Neural Architecture Search (NAS) using evolutionary algorithms and RL.

This module implements NAS with: (1) a discrete search space over MLP
architectures (number of layers, hidden size, activation); (2) evolutionary
search (population, fitness evaluation, selection, crossover, mutation);
(3) reinforcement learning (REINFORCE controller that samples architectures
and is updated with validation accuracy as reward). Uses a small classification
task (e.g. digits subset) for architecture evaluation.
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

# Search space: num_layers in [2,3,4], hidden_dim in [32,64,128], activation [relu, tanh]
NUM_LAYERS_OPTIONS = [2, 3, 4]
HIDDEN_DIM_OPTIONS = [32, 64, 128]
ACTIVATION_OPTIONS = ["relu", "tanh"]
NUM_CHOICES = [
    len(NUM_LAYERS_OPTIONS),
    len(HIDDEN_DIM_OPTIONS),
    len(ACTIVATION_OPTIONS),
]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -500, 500))
    return e / np.sum(e, axis=axis, keepdims=True)


def encode_architecture(
    num_layers: int, hidden_dim: int, activation: str
) -> List[int]:
    """Encode architecture as list of choice indices."""
    nl_idx = NUM_LAYERS_OPTIONS.index(num_layers)
    hd_idx = HIDDEN_DIM_OPTIONS.index(hidden_dim)
    act_idx = ACTIVATION_OPTIONS.index(activation)
    return [nl_idx, hd_idx, act_idx]


def decode_architecture(choices: List[int]) -> Tuple[int, int, str]:
    """Decode choice indices to (num_layers, hidden_dim, activation)."""
    return (
        NUM_LAYERS_OPTIONS[choices[0]],
        HIDDEN_DIM_OPTIONS[choices[1]],
        ACTIVATION_OPTIONS[choices[2]],
    )


def random_architecture(random_seed: Optional[int] = None) -> List[int]:
    """Sample a random architecture (list of choice indices)."""
    if random_seed is not None:
        np.random.seed(random_seed)
    return [
        int(np.random.randint(0, n)) for n in NUM_CHOICES
    ]


class TrainableMLP:
    """MLP with variable hidden layers and configurable activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int,
        hidden_dim: int,
        activation: str,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self.activation = activation
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._cache: List[Optional[np.ndarray]] = []
        self._input: Optional[np.ndarray] = None
        for i in range(len(dims) - 1):
            limit = np.sqrt(1.0 / max(1, dims[i]))
            self.weights.append(
                np.random.uniform(-limit, limit, (dims[i], dims[i + 1])).astype(
                    np.float32
                )
            )
            self.biases.append(np.zeros(dims[i + 1], dtype=np.float32))
            self._cache.append(None)

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        return np.tanh(x)

    def _act_grad(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return grad_out * (x > 0.0).astype(np.float32)
        return grad_out * (1.0 - np.tanh(x) ** 2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        self._cache = []
        h = x
        for i in range(len(self.weights) - 1):
            h = h @ self.weights[i] + self.biases[i]
            self._cache.append(h.copy())
            h = self._act(h)
        out = h @ self.weights[-1] + self.biases[-1]
        return out

    def backward(self, grad_out: np.ndarray, lr: float) -> None:
        batch = grad_out.shape[0]
        g = grad_out.copy()
        L = len(self.weights)
        h = self._act(self._cache[L - 2])
        self.weights[L - 1] -= lr * (h.T @ g) / float(batch)
        self.biases[L - 1] -= lr * np.mean(g, axis=0)
        g = self._act_grad(self._cache[L - 2], g @ self.weights[L - 1].T)
        for i in range(L - 3, -1, -1):
            h = self._act(self._cache[i])
            self.weights[i + 1] -= lr * (h.T @ g) / float(batch)
            self.biases[i + 1] -= lr * np.mean(g, axis=0)
            g = self._act_grad(self._cache[i], g @ self.weights[i + 1].T)
        if self._input is not None:
            self.weights[0] -= lr * (self._input.T @ g) / float(batch)
            self.biases[0] -= lr * np.mean(g, axis=0)


def train_and_evaluate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    choices: List[int],
    epochs: int = 5,
    lr: float = 0.01,
    batch_size: int = 32,
    random_seed: Optional[int] = None,
) -> float:
    """Train an MLP with the given architecture; return validation accuracy."""
    num_layers, hidden_dim, activation = decode_architecture(choices)
    in_dim = train_x.shape[1]
    out_dim = int(np.max(train_y)) + 1
    np.random.seed(random_seed)
    net = TrainableMLP(
        in_dim, out_dim, num_layers, hidden_dim, activation, random_seed=random_seed
    )
    n = train_x.shape[0]
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            x = train_x[idx]
            y = train_y[idx]
            logits = net.forward(x)
            probs = _softmax(logits, axis=1)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(y)), y.astype(int)] = 1.0
            grad = (probs - one_hot) / float(x.shape[0])
            net.backward(grad, lr)
    logits_val = net.forward(val_x)
    pred = np.argmax(logits_val, axis=1)
    acc = np.mean(pred.astype(np.int64) == val_y.astype(np.int64))
    return float(acc)


class EvolutionaryNAS:
    """NAS via evolutionary algorithm: population, fitness, selection, crossover, mutation."""

    def __init__(
        self,
        population_size: int,
        num_generations: int,
        mutation_prob: float,
        tournament_size: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        eval_epochs: int = 3,
        random_seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.eval_epochs = eval_epochs
        self.random_seed = random_seed
        self.population: List[List[int]] = []
        self.fitness_cache: Dict[Tuple[int, ...], float] = {}

    def _evaluate(self, choices: List[int]) -> float:
        key = tuple(choices)
        if key not in self.fitness_cache:
            seed = (self.random_seed + hash(key) % 10000) if self.random_seed else None
            self.fitness_cache[key] = train_and_evaluate(
                self.train_x,
                self.train_y,
                self.val_x,
                self.val_y,
                choices,
                epochs=self.eval_epochs,
                random_seed=seed,
            )
        return self.fitness_cache[key]

    def _select(self) -> List[int]:
        """Tournament selection."""
        idx = np.random.choice(len(self.population), self.tournament_size, replace=False)
        best = max(idx, key=lambda i: self._evaluate(self.population[i]))
        return self.population[best].copy()

    def _crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """Single-point style: each gene from p1 or p2 with equal probability."""
        return [p1[i] if np.random.rand() < 0.5 else p2[i] for i in range(3)]

    def _mutate(self, choices: List[int]) -> List[int]:
        out = choices.copy()
        for i in range(3):
            if np.random.rand() < self.mutation_prob:
                out[i] = int(np.random.randint(0, NUM_CHOICES[i]))
        return out

    def run(self) -> Tuple[List[int], float, List[Dict]]:
        """Run evolutionary search; return best architecture, its fitness, and history."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.population = [
            random_architecture(random_seed=self.random_seed + i)
            for i in range(self.population_size)
        ]
        history: List[Dict] = []
        best_arch = self.population[0]
        best_fitness = self._evaluate(best_arch)

        for gen in range(self.num_generations):
            fitnesses = [self._evaluate(ind) for ind in self.population]
            for i, f in enumerate(fitnesses):
                if f > best_fitness:
                    best_fitness = f
                    best_arch = self.population[i].copy()
            new_pop = [best_arch.copy()]
            while len(new_pop) < self.population_size:
                p1 = self._select()
                p2 = self._select()
                child = self._mutate(self._crossover(p1, p2))
                new_pop.append(child)
            self.population = new_pop
            history.append({"generation": gen, "best_fitness": best_fitness})
            logger.info("Evolution gen %d best_fitness=%.4f", gen, best_fitness)

        return best_arch, best_fitness, history


class Controller:
    """REINFORCE controller: outputs logits per step; sample architecture and compute log_prob."""

    def __init__(
        self,
        hidden_dim: int = 32,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self.num_steps = 3
        self.num_choices = NUM_CHOICES
        state_dim = 3
        total_actions = sum(NUM_CHOICES)
        limit = np.sqrt(1.0 / state_dim)
        self.w1 = np.random.uniform(-limit, limit, (state_dim, hidden_dim)).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        limit2 = np.sqrt(1.0 / hidden_dim)
        self.w2 = np.random.uniform(-limit2, limit2, (hidden_dim, total_actions)).astype(
            np.float32
        )
        self.b2 = np.zeros(total_actions, dtype=np.float32)
        self._state_onehot: Optional[np.ndarray] = None
        self._hidden: Optional[np.ndarray] = None
        self._logits_all: Optional[np.ndarray] = None
        self._samples: List[int] = []
        self._log_probs: List[float] = []

    def _forward_logits(self, state_onehot: np.ndarray) -> np.ndarray:
        h = np.maximum(state_onehot @ self.w1 + self.b1, 0.0)
        logits = h @ self.w2 + self.b2
        return logits

    def sample(self, random_seed: Optional[int] = None) -> List[int]:
        """Sample an architecture; store log_prob for REINFORCE."""
        if random_seed is not None:
            np.random.seed(random_seed)
        self._samples = []
        self._log_probs = []
        start = 0
        for step in range(self.num_steps):
            state = np.zeros(3, dtype=np.float32)
            state[step] = 1.0
            state = state.reshape(1, -1)
            logits_full = self._forward_logits(state)
            n = self.num_choices[step]
            logits = logits_full[0, start : start + n]
            probs = _softmax(logits.reshape(1, -1))[0]
            choice = int(np.random.choice(n, p=probs))
            log_prob = np.log(probs[choice] + 1e-8)
            self._samples.append(choice)
            self._log_probs.append(float(log_prob))
            start += n
        return self._samples

    def get_log_prob(self) -> float:
        """Sum of log probs of the last sampled trajectory."""
        return sum(self._log_probs)

    def backward_reinforce(self, reward: float, lr: float) -> None:
        """REINFORCE update: gradient = reward * grad_log_prob."""
        state = np.zeros((1, 3), dtype=np.float32)
        start = 0
        total_logits_dim = sum(self.num_choices)
        grad_logits = np.zeros((1, total_logits_dim), dtype=np.float32)
        for step in range(self.num_steps):
            state[0, :] = 0.0
            state[0, step] = 1.0
            logits_full = self._forward_logits(state)
            n = self.num_choices[step]
            logits = logits_full[0, start : start + n].copy()
            probs = _softmax(logits.reshape(1, -1))[0]
            choice = self._samples[step]
            probs[choice] -= 1.0
            grad_logits[0, start : start + n] = reward * probs
            start += n
        h = np.maximum(state @ self.w1 + self.b1, 0.0)
        grad_h = (grad_logits @ self.w2.T) * (h > 0.0).astype(np.float32)
        self.w2 -= lr * (h.T @ grad_logits)
        self.b2 -= lr * np.sum(grad_logits, axis=0)
        self.w1 -= lr * (state.T @ grad_h)
        self.b1 -= lr * np.sum(grad_h, axis=0)


class RLNAS:
    """NAS via REINFORCE: controller samples architectures; reward = validation accuracy."""

    def __init__(
        self,
        num_rollouts: int,
        controller_lr: float,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        eval_epochs: int = 3,
        random_seed: Optional[int] = None,
    ) -> None:
        self.num_rollouts = num_rollouts
        self.controller_lr = controller_lr
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.eval_epochs = eval_epochs
        self.random_seed = random_seed
        self.controller = Controller(random_seed=random_seed)

    def run(self) -> Tuple[List[int], float, List[Dict]]:
        """Run RL-based search; return best architecture, its reward, and history."""
        history: List[Dict] = []
        best_arch: Optional[List[int]] = None
        best_reward = -1.0

        for rollout in range(self.num_rollouts):
            seed = (self.random_seed + rollout * 1000) if self.random_seed else None
            choices = self.controller.sample(random_seed=seed)
            reward = train_and_evaluate(
                self.train_x,
                self.train_y,
                self.val_x,
                self.val_y,
                choices,
                epochs=self.eval_epochs,
                random_seed=seed,
            )
            if reward > best_reward:
                best_reward = reward
                best_arch = choices.copy()
            log_prob = self.controller.get_log_prob()
            self.controller.backward_reinforce(reward, self.controller_lr)
            history.append({"rollout": rollout, "reward": reward, "best": best_reward})
            logger.info("RL rollout %d reward=%.4f best=%.4f", rollout, reward, best_reward)

        return best_arch or random_architecture(), best_reward, history


def load_digits_data(
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load digits dataset (sklearn or fallback synthetic); return train_x, train_y, val_x, val_y."""
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        data = load_digits()
        x = np.asarray(data.data, dtype=np.float32) / 16.0
        y = np.asarray(data.target, dtype=np.int64)
        if max_samples is not None:
            if random_seed is not None:
                np.random.seed(random_seed)
            idx = np.random.choice(len(x), min(max_samples, len(x)), replace=False)
            x, y = x[idx], y[idx]
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=train_ratio, random_state=random_seed
        )
        return x_train, y_train, x_val, y_val
    except ImportError:
        np.random.seed(random_seed or 42)
        n = max_samples or 500
        x = np.random.randn(n, 64).astype(np.float32) * 0.1
        y = np.random.randint(0, 10, size=n, dtype=np.int64)
        split = int(n * train_ratio)
        return x[:split], y[:split], x[split:], y[split:]


@dataclass
class NASConfig:
    """Configuration for NAS runs."""

    method: str = "evolution"
    population_size: int = 8
    num_generations: int = 5
    mutation_prob: float = 0.2
    tournament_size: int = 3
    num_rollouts: int = 10
    controller_lr: float = 0.01
    eval_epochs: int = 3
    train_ratio: float = 0.8
    max_samples: Optional[int] = 200
    random_seed: Optional[int] = 0


def run_nas(config: NASConfig) -> Dict:
    """Run NAS with configured method (evolution or rl); return results dict."""
    train_x, train_y, val_x, val_y = load_digits_data(
        train_ratio=config.train_ratio,
        max_samples=config.max_samples,
        random_seed=config.random_seed,
    )
    if config.method == "evolution":
        nas = EvolutionaryNAS(
            population_size=config.population_size,
            num_generations=config.num_generations,
            mutation_prob=config.mutation_prob,
            tournament_size=config.tournament_size,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            eval_epochs=config.eval_epochs,
            random_seed=config.random_seed,
        )
        best_arch, best_fitness, history = nas.run()
    else:
        nas = RLNAS(
            num_rollouts=config.num_rollouts,
            controller_lr=config.controller_lr,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            eval_epochs=config.eval_epochs,
            random_seed=config.random_seed,
        )
        best_arch, best_fitness, history = nas.run()

    num_layers, hidden_dim, activation = decode_architecture(best_arch)
    return {
        "method": config.method,
        "best_architecture": {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "activation": activation,
        },
        "best_validation_accuracy": best_fitness,
        "history": history,
    }


def main() -> None:
    """Entry point: parse args, load config, run NAS, print and optionally save results."""
    parser = argparse.ArgumentParser(description="NAS with evolutionary and RL")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--output", type=Path, default=None, help="Path to write results JSON")
    parser.add_argument("--method", choices=["evolution", "rl"], default=None)
    args = parser.parse_args()

    config_dict: Dict = {}
    config_path = args.config or Path(__file__).parent.parent / "config.yaml"
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
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.setLevel(level)
    logger.addHandler(handler)

    nas_cfg = config_dict.get("nas", {})
    config = NASConfig(
        method=args.method or nas_cfg.get("method", "evolution"),
        population_size=nas_cfg.get("population_size", 8),
        num_generations=nas_cfg.get("num_generations", 5),
        mutation_prob=nas_cfg.get("mutation_prob", 0.2),
        tournament_size=nas_cfg.get("tournament_size", 3),
        num_rollouts=nas_cfg.get("num_rollouts", 10),
        controller_lr=nas_cfg.get("controller_lr", 0.01),
        eval_epochs=nas_cfg.get("eval_epochs", 3),
        train_ratio=nas_cfg.get("train_ratio", 0.8),
        max_samples=nas_cfg.get("max_samples", 200),
        random_seed=nas_cfg.get("random_seed", 0),
    )

    results = run_nas(config)
    print("\nNAS Results:")
    print("========================================")
    print("  method:", results["method"])
    print("  best_architecture:", results["best_architecture"])
    print("  best_validation_accuracy: %.4f" % results["best_validation_accuracy"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in results.items() if k != "history"}
        out["history_length"] = len(results["history"])
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
