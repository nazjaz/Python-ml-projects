"""Deep Q-Network (DQN) from scratch with experience replay and target network.

This module implements a minimal DQN agent using only NumPy. It includes:

- Fully connected Q-network and a separate target network
- Epsilon-greedy exploration strategy
- Experience replay buffer for decorrelated training samples
- Periodic target network updates for training stability
- A small tabular gridworld-like environment for demonstration
"""

import argparse
import json
import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for logging or policies if needed."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -500, 500))
    return e / np.sum(e, axis=axis, keepdims=True)


class SimpleGridworldEnv:
    """Small deterministic gridworld with discrete states and actions.

    The agent starts at (0, 0) on an N x N grid and aims to reach the goal
    at (N-1, N-1). Each step yields a reward of -1, and reaching the goal
    yields +10 and terminates the episode.
    """

    def __init__(self, size: int = 5, random_seed: Optional[int] = None) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.state: int = 0

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.size)

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def reset(self) -> int:
        """Reset environment to start state."""
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Apply action and return (next_state, reward, done)."""
        row, col = self._state_to_coord(self.state)
        if action == 0 and row > 0:  # up
            row -= 1
        elif action == 1 and row < self.size - 1:  # down
            row += 1
        elif action == 2 and col > 0:  # left
            col -= 1
        elif action == 3 and col < self.size - 1:  # right
            col += 1

        next_state = self._coord_to_state(row, col)
        done = next_state == self.n_states - 1
        reward = 10.0 if done else -1.0
        self.state = next_state
        return next_state, reward, done


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self._size = 0
        self._pos = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        idx = self._pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        """Return True if enough samples are stored."""
        return self._size >= batch_size

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch of transitions."""
        if batch_size > self._size:
            raise ValueError("Not enough elements in replay buffer to sample.")
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class QNetwork:
    """Simple fully connected Q-network with one hidden layer."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions

        limit1 = np.sqrt(1.0 / max(1, state_dim))
        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w1 = np.random.uniform(
            -limit1, limit1, (state_dim, hidden_dim)
        ).astype(np.float32)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w2 = np.random.uniform(
            -limit2, limit2, (hidden_dim, n_actions)
        ).astype(np.float32)
        self.b2 = np.zeros((n_actions,), dtype=np.float32)

        self._cache_input: Optional[np.ndarray] = None
        self._cache_hidden: Optional[np.ndarray] = None

    def forward(self, states: np.ndarray) -> np.ndarray:
        """Compute Q-values for a batch of states."""
        z1 = states @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        q_values = h1 @ self.w2 + self.b2
        self._cache_input = states
        self._cache_hidden = h1
        return q_values

    def backward(self, grad_q: np.ndarray, lr: float) -> None:
        """Backpropagate gradients and update parameters."""
        if self._cache_input is None or self._cache_hidden is None:
            raise RuntimeError("Forward must be called before backward.")
        x = self._cache_input
        h1 = self._cache_hidden
        batch_size = x.shape[0]

        d_w2 = h1.T @ grad_q / float(batch_size)
        d_b2 = np.mean(grad_q, axis=0)
        d_h1 = grad_q @ self.w2.T
        d_z1 = d_h1 * (h1 > 0.0).astype(np.float32)

        d_w1 = x.T @ d_z1 / float(batch_size)
        d_b1 = np.mean(d_z1, axis=0)

        self.w2 -= lr * d_w2.astype(np.float32)
        self.b2 -= lr * d_b2.astype(np.float32)
        self.w1 -= lr * d_w1.astype(np.float32)
        self.b1 -= lr * d_b1.astype(np.float32)

    def copy_from(self, other: "QNetwork") -> None:
        """Copy parameters from another QNetwork."""
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()


@dataclass
class DQNConfig:
    """Configuration for DQN training."""

    gamma: float = 0.99
    learning_rate: float = 0.001
    batch_size: int = 32
    buffer_capacity: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1000
    target_update_interval: int = 100
    max_episodes: int = 500
    max_steps_per_episode: int = 100


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        config: DQNConfig,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.config = config

        self.q_net = QNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        )
        self.target_net = QNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
        )
        self.target_net.copy_from(self.q_net)

        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_capacity, state_dim=state_dim
        )

        self._step_count = 0

    def _epsilon(self) -> float:
        """Linearly decaying epsilon for epsilon-greedy policy."""
        frac = min(1.0, self._step_count / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        eps = self._epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        q_values = self.q_net.forward(state[None, :])[0]
        return int(np.argmax(q_values))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self._step_count += 1

    def _update_target_network(self) -> None:
        """Synchronize target network with online network."""
        self.target_net.copy_from(self.q_net)

    def train_step(self) -> Optional[float]:
        """Perform one gradient step using a replay batch."""
        if not self.replay_buffer.can_sample(self.config.batch_size):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.config.batch_size)

        q_values = self.q_net.forward(states)
        q_selected = q_values[np.arange(self.config.batch_size), actions]

        with np.errstate(over="ignore"):
            next_q_values = self.target_net.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.config.gamma * max_next_q * (1.0 - dones)

        td_errors = q_selected - targets
        loss = float(np.mean(td_errors**2))

        grad_q = np.zeros_like(q_values)
        grad_q[np.arange(self.config.batch_size), actions] = (
            2.0 * td_errors / float(self.config.batch_size)
        )

        self.q_net.backward(grad_q, lr=self.config.learning_rate)

        if self._step_count % self.config.target_update_interval == 0:
            self._update_target_network()

        return loss


class DQNRunner:
    """Run DQN training loop from configuration."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_dict = self._load_config(config_path)
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
        log_cfg = self.config_dict.get("logging", {})
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

    def _build_agent_and_env(self) -> Tuple[DQNAgent, SimpleGridworldEnv]:
        env_cfg = self.config_dict.get("env", {})
        agent_cfg = self.config_dict.get("agent", {})

        grid_size = env_cfg.get("grid_size", 5)
        random_seed = env_cfg.get("random_seed", 0)

        env = SimpleGridworldEnv(size=grid_size, random_seed=random_seed)

        state_dim = env.n_states
        n_actions = env.n_actions
        hidden_dim = agent_cfg.get("hidden_dim", 64)

        dqn_cfg = DQNConfig(
            gamma=agent_cfg.get("gamma", 0.99),
            learning_rate=agent_cfg.get("learning_rate", 0.001),
            batch_size=agent_cfg.get("batch_size", 32),
            buffer_capacity=agent_cfg.get("buffer_capacity", 10000),
            epsilon_start=agent_cfg.get("epsilon_start", 1.0),
            epsilon_end=agent_cfg.get("epsilon_end", 0.1),
            epsilon_decay_steps=agent_cfg.get("epsilon_decay_steps", 1000),
            target_update_interval=agent_cfg.get(
                "target_update_interval", 100
            ),
            max_episodes=agent_cfg.get("max_episodes", 500),
            max_steps_per_episode=agent_cfg.get(
                "max_steps_per_episode", 100
            ),
        )

        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            config=dqn_cfg,
        )
        return agent, env

    def run(self) -> Dict[str, float]:
        """Train DQN agent and report episode rewards and average returns."""
        agent, env = self._build_agent_and_env()
        cfg = agent.config

        episode_rewards: List[float] = []
        moving_returns: List[float] = []

        for episode in range(cfg.max_episodes):
            state_idx = env.reset()
            state_vec = np.eye(env.n_states, dtype=np.float32)[state_idx]
            total_reward = 0.0

            for _ in range(cfg.max_steps_per_episode):
                action = agent.select_action(state_vec)
                next_state_idx, reward, done = env.step(action)
                next_state_vec = np.eye(env.n_states, dtype=np.float32)[
                    next_state_idx
                ]

                agent.store_transition(
                    state=state_vec,
                    action=action,
                    reward=reward,
                    next_state=next_state_vec,
                    done=done,
                )

                _ = agent.train_step()

                state_vec = next_state_vec
                total_reward += reward
                if done:
                    break

            episode_rewards.append(total_reward)
            window = min(50, len(episode_rewards))
            avg_return = float(np.mean(episode_rewards[-window:]))
            moving_returns.append(avg_return)

            if (episode + 1) % max(1, cfg.max_episodes // 10) == 0:
                logger.info(
                    "Episode %d/%d - reward: %.2f, avg_return(last_%d): %.2f",
                    episode + 1,
                    cfg.max_episodes,
                    total_reward,
                    window,
                    avg_return,
                )

        results = {
            "final_episode_reward": float(episode_rewards[-1]),
            "average_return_last_50": float(moving_returns[-1]),
        }
        logger.info(
            "Training complete - final_reward: %.2f, avg_return_last_50: %.2f",
            results["final_episode_reward"],
            results["average_return_last_50"],
        )
        return results


def main() -> None:
    """Entry point for running DQN training."""
    parser = argparse.ArgumentParser(
        description="Train DQN from scratch with replay and target network"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = DQNRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run()

    print("\nFinal Results:")
    print("=" * 40)
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()

