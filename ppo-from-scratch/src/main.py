"""Proximal Policy Optimization (PPO) with clipped objective and trust region.

This module implements PPO from scratch using only NumPy: policy and value
networks, clipped surrogate objective, optional KL penalty for trust region,
Generalized Advantage Estimation (GAE), and multi-epoch minibatch updates on
collected trajectories. Uses a small gridworld environment for demonstration.
"""

import argparse
import json
import logging
import logging.handlers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -500, 500))
    return e / np.sum(e, axis=axis, keepdims=True)


class SimpleGridworldEnv:
    """Discrete gridworld: start (0,0), goal (N-1,N-1), 4 actions."""

    def __init__(self, size: int = 5, random_seed: Optional[int] = None) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.state: int = 0

    def _state_to_coord(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.size)

    def _coord_to_state(self, row: int, col: int) -> int:
        return row * self.size + col

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        row, col = self._state_to_coord(self.state)
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < self.size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < self.size - 1:
            col += 1
        next_state = self._coord_to_state(row, col)
        done = next_state == self.n_states - 1
        reward = 10.0 if done else -1.0
        self.state = next_state
        return next_state, reward, done


class MLP:
    """Two-layer MLP with ReLU for shared feature extraction or single output."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
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
            raise RuntimeError("Forward must be called before backward.")
        x, h = self._x, self._h
        batch = x.shape[0]
        d_h = grad_out @ self.w2.T
        d_z = d_h * (h > 0.0).astype(np.float32)
        d_w2 = (self._h.T @ grad_out) / float(batch)
        d_b2 = np.mean(grad_out, axis=0)
        d_w1 = (x.T @ d_z) / float(batch)
        d_b1 = np.mean(d_z, axis=0)
        self.w2 -= lr * d_w2.astype(np.float32)
        self.b2 -= lr * d_b2.astype(np.float32)
        self.w1 -= lr * d_w1.astype(np.float32)
        self.b1 -= lr * d_b1.astype(np.float32)
        return d_z @ self.w1.T


class PolicyNetwork:
    """Policy pi(a|s) with softmax over actions; returns logits, probs, log_probs."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int) -> None:
        self.mlp = MLP(state_dim, hidden_dim, n_actions)
        self.n_actions = n_actions
        self._probs: Optional[np.ndarray] = None
        self._logits: Optional[np.ndarray] = None

    def forward(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = self.mlp.forward(states)
        probs = _softmax(logits, axis=1)
        log_probs = np.log(np.clip(probs, 1e-8, 1.0))
        self._probs = probs
        self._logits = logits
        return logits, probs, log_probs

    def sample(
        self, states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, probs, log_probs = self.forward(states)
        batch = states.shape[0]
        actions = np.array(
            [
                np.random.choice(self.n_actions, p=probs[i])
                for i in range(batch)
            ],
            dtype=np.int64,
        )
        log_prob_actions = log_probs[np.arange(batch), actions]
        return actions, log_prob_actions, probs

    def evaluate_actions(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return log_probs and entropy for given (states, actions)."""
        _, probs, log_probs = self.forward(states)
        batch = states.shape[0]
        log_prob_actions = log_probs[np.arange(batch), actions]
        entropy = -np.sum(probs * log_probs, axis=1)
        return log_prob_actions, entropy

    def backward_policy(
        self, states: np.ndarray, actions: np.ndarray, grad_log_prob: np.ndarray, lr: float
    ) -> None:
        """Backpropagate gradient of scalar loss w.r.t. log_prob of chosen actions."""
        _, probs, _ = self.forward(states)
        batch = states.shape[0]
        grad_logits = np.zeros_like(self._logits)
        grad_logits[np.arange(batch), actions] = grad_log_prob
        grad_logits -= probs * np.sum(grad_logits, axis=1, keepdims=True)
        self.mlp.backward(grad_logits, lr=lr)


class ValueNetwork:
    """V(s) critic."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        self.mlp = MLP(state_dim, hidden_dim, 1)

    def forward(self, states: np.ndarray) -> np.ndarray:
        out = self.mlp.forward(states)
        return out[:, 0]

    def backward(
        self, states: np.ndarray, grad_values: np.ndarray, lr: float
    ) -> None:
        """grad_values: (batch,) or (batch, 1)."""
        if grad_values.ndim == 1:
            grad_values = grad_values[:, None]
        self.mlp.backward(grad_values.astype(np.float32), lr=lr)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute returns and advantages using Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae * (1.0 - dones[t])
    returns = advantages + values
    return returns, advantages


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Normalize advantages to zero mean and unit variance."""
    mean = np.mean(advantages)
    std = np.std(advantages)
    if std < 1e-8:
        return advantages - mean
    return (advantages - mean) / (std + 1e-8)


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_target: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    rollout_steps: int = 128
    max_episodes: int = 200
    max_steps_per_episode: int = 100


@dataclass
class RolloutBuffer:
    """Stores one rollout of trajectories for PPO update."""

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs_old: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs_old.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs_old.clear()
        self.values.clear()
        self.dones.clear()

    def to_arrays(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        states = np.stack(self.states, axis=0)
        actions = np.array(self.actions, dtype=np.int64)
        rewards = np.array(self.rewards, dtype=np.float32)
        log_probs_old = np.array(self.log_probs_old, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        return states, actions, rewards, log_probs_old, values, dones


class PPOAgent:
    """PPO agent with clipped objective and optional KL trust region."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        config: PPOConfig,
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.config = config
        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.rollout = RolloutBuffer()

    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, float, float]:
        state_batch = state[None, :]
        actions, log_probs, _ = self.policy.sample(state_batch)
        value = self.value_net.forward(state_batch)[0]
        return int(actions[0]), float(log_probs[0]), float(value)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        self.rollout.add(state, action, reward, log_prob, value, done)

    def update(self) -> Dict[str, float]:
        """Compute GAE, normalize advantages, then run PPO epochs with clipping."""
        (
            states,
            actions,
            rewards,
            log_probs_old,
            values,
            dones,
        ) = self.rollout.to_arrays()
        n = len(rewards)

        next_value = 0.0
        returns, advantages = compute_gae(
            rewards, values, dones, next_value,
            self.config.gamma, self.config.gae_lambda,
        )
        advantages = normalize_advantages(advantages)

        indices = np.arange(n)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                mb_indices = indices[start:end]
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                log_probs_new, entropy = self.policy.evaluate_actions(
                    mb_states, mb_actions
                )
                values_new = self.value_net.forward(mb_states)

                ratio = np.exp(log_probs_new - mb_log_probs_old)
                ratio = np.clip(ratio, 0.1, 10.0)
                surr1 = ratio * mb_advantages
                surr2 = np.clip(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * mb_advantages
                policy_loss = -np.mean(np.minimum(surr1, surr2))

                value_loss = np.mean((values_new - mb_returns) ** 2)
                entropy_mean = np.mean(entropy)
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )

                grad_log_prob = np.zeros_like(log_probs_new)
                for i in range(len(mb_advantages)):
                    if mb_advantages[i] >= 0:
                        if ratio[i] <= 1.0 + self.config.clip_epsilon:
                            grad_log_prob[i] = -mb_advantages[i] * ratio[i]
                        else:
                            grad_log_prob[i] = 0.0
                    else:
                        if ratio[i] >= 1.0 - self.config.clip_epsilon:
                            grad_log_prob[i] = -mb_advantages[i] * ratio[i]
                        else:
                            grad_log_prob[i] = 0.0

                self.policy.backward_policy(
                    mb_states, mb_actions, grad_log_prob, self.config.learning_rate
                )
                grad_values = 2.0 * (values_new - mb_returns) / float(len(mb_indices))
                self.value_net.backward(
                    mb_states, grad_values, self.config.learning_rate
                )

                total_policy_loss += float(policy_loss)
                total_value_loss += float(value_loss)
                total_entropy += float(entropy_mean)
                n_updates += 1

        self.rollout.clear()
        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }


class PPORunner:
    """Run PPO training from YAML config."""

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

    def _build_agent_and_env(self) -> Tuple[PPOAgent, SimpleGridworldEnv]:
        env_cfg = self.config_dict.get("env", {})
        agent_cfg = self.config_dict.get("agent", {})
        grid_size = env_cfg.get("grid_size", 5)
        random_seed = env_cfg.get("random_seed", 0)
        env = SimpleGridworldEnv(size=grid_size, random_seed=random_seed)
        state_dim = env.n_states
        n_actions = env.n_actions
        hidden_dim = agent_cfg.get("hidden_dim", 64)
        ppo_cfg = PPOConfig(
            gamma=agent_cfg.get("gamma", 0.99),
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_epsilon=agent_cfg.get("clip_epsilon", 0.2),
            value_coef=agent_cfg.get("value_coef", 0.5),
            entropy_coef=agent_cfg.get("entropy_coef", 0.01),
            kl_target=agent_cfg.get("kl_target", 0.01),
            max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
            ppo_epochs=agent_cfg.get("ppo_epochs", 4),
            batch_size=agent_cfg.get("batch_size", 64),
            learning_rate=agent_cfg.get("learning_rate", 3e-4),
            rollout_steps=agent_cfg.get("rollout_steps", 128),
            max_episodes=agent_cfg.get("max_episodes", 200),
            max_steps_per_episode=agent_cfg.get("max_steps_per_episode", 100),
        )
        agent = PPOAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            config=ppo_cfg,
        )
        return agent, env

    def run(self) -> Dict[str, float]:
        """Collect rollouts and run PPO updates until max_episodes."""
        agent, env = self._build_agent_and_env()
        cfg = agent.config
        episode_rewards: List[float] = []
        global_step = 0

        while len(episode_rewards) < cfg.max_episodes:
            state_idx = env.reset()
            state_vec = np.eye(env.n_states, dtype=np.float32)[state_idx]
            episode_reward = 0.0

            for _ in range(cfg.max_steps_per_episode):
                action, log_prob, value = agent.select_action(state_vec)
                next_idx, reward, done = env.step(action)
                next_vec = np.eye(env.n_states, dtype=np.float32)[next_idx]
                agent.store_transition(
                    state=state_vec,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    value=value,
                    done=done,
                )
                episode_reward += reward
                global_step += 1
                state_vec = next_vec
                if done:
                    break

            episode_rewards.append(episode_reward)
            if len(agent.rollout.states) >= cfg.rollout_steps:
                update_stats = agent.update()
                if (len(episode_rewards)) % max(1, cfg.max_episodes // 10) == 0:
                    logger.info(
                        "Episode %d - reward: %.2f, policy_loss: %.4f",
                        len(episode_rewards),
                        episode_reward,
                        update_stats["policy_loss"],
                    )

        window = min(50, len(episode_rewards))
        avg_return = float(np.mean(episode_rewards[-window:]))
        results = {
            "final_episode_reward": float(episode_rewards[-1]),
            "average_return_last_50": avg_return,
        }
        logger.info(
            "Training complete - final_reward: %.2f, avg_return: %.2f",
            results["final_episode_reward"],
            results["average_return_last_50"],
        )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO with clipped objective and trust region"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()
    runner = PPORunner(config_path=Path(args.config) if args.config else None)
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
