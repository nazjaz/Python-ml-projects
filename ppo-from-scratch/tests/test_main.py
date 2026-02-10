"""Tests for PPO implementation with clipped objective and trust region."""

from pathlib import Path

import numpy as np
import pytest

from src.main import (
    PPOAgent,
    PPOConfig,
    PPORunner,
    PolicyNetwork,
    RolloutBuffer,
    ValueNetwork,
    compute_gae,
    normalize_advantages,
    SimpleGridworldEnv,
)


class TestEnv:
    """Tests for gridworld environment."""

    def test_reset_and_step(self) -> None:
        env = SimpleGridworldEnv(size=4, random_seed=42)
        s = env.reset()
        assert s == 0
        next_s, reward, done = env.step(1)
        assert next_s == 4
        assert not done
        next_s, reward, done = env.step(3)
        assert next_s == 5


class TestRolloutBuffer:
    """Tests for rollout buffer."""

    def test_add_and_to_arrays_shapes(self) -> None:
        buf = RolloutBuffer()
        state_dim, n = 9, 5
        for i in range(n):
            buf.add(
                state=np.eye(state_dim, dtype=np.float32)[i],
                action=i % 4,
                reward=1.0,
                log_prob=-1.0,
                value=0.0,
                done=(i == n - 1),
            )
        s, a, r, lp, v, d = buf.to_arrays()
        assert s.shape == (n, state_dim)
        assert a.shape == (n,)
        assert r.shape == (n,)
        assert lp.shape == (n,)
        assert v.shape == (n,)
        assert d.shape == (n,)
        buf.clear()
        assert len(buf.states) == 0


class TestGAE:
    """Tests for GAE computation."""

    def test_gae_shapes(self) -> None:
        n = 10
        rewards = np.random.randn(n).astype(np.float32)
        values = np.random.randn(n).astype(np.float32)
        dones = np.zeros(n, dtype=np.float32)
        returns, advantages = compute_gae(
            rewards, values, dones, next_value=0.0, gamma=0.99, lam=0.95
        )
        assert returns.shape == (n,)
        assert advantages.shape == (n,)

    def test_normalize_advantages(self) -> None:
        adv = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = normalize_advantages(adv)
        assert np.abs(np.mean(out)) < 1e-5
        assert np.abs(np.std(out) - 1.0) < 1e-5


class TestPolicyNetwork:
    """Tests for policy network."""

    def test_forward_shapes(self) -> None:
        np.random.seed(0)
        policy = PolicyNetwork(state_dim=9, n_actions=4, hidden_dim=8)
        states = np.eye(9, dtype=np.float32)[:4]
        logits, probs, log_probs = policy.forward(states)
        assert logits.shape == (4, 4)
        assert probs.shape == (4, 4)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

    def test_sample_returns_valid_action(self) -> None:
        np.random.seed(1)
        policy = PolicyNetwork(state_dim=9, n_actions=4, hidden_dim=8)
        states = np.eye(9, dtype=np.float32)[:2]
        actions, log_probs, _ = policy.sample(states)
        assert actions.shape == (2,)
        assert np.all(actions >= 0)
        assert np.all(actions < 4)
        assert log_probs.shape == (2,)


class TestValueNetwork:
    """Tests for value network."""

    def test_forward_shape(self) -> None:
        np.random.seed(2)
        value_net = ValueNetwork(state_dim=9, hidden_dim=8)
        states = np.eye(9, dtype=np.float32)[:5]
        values = value_net.forward(states)
        assert values.shape == (5,)


class TestPPOAgent:
    """Tests for PPO agent."""

    def test_select_action_returns_valid(self) -> None:
        np.random.seed(3)
        cfg = PPOConfig(max_episodes=1)
        agent = PPOAgent(
            state_dim=25, n_actions=4, hidden_dim=16, config=cfg
        )
        state = np.eye(25, dtype=np.float32)[0]
        action, log_prob, value = agent.select_action(state)
        assert 0 <= action < 4
        assert np.isfinite(log_prob)
        assert np.isfinite(value)

    def test_update_after_rollout(self) -> None:
        np.random.seed(4)
        cfg = PPOConfig(
            rollout_steps=32,
            batch_size=16,
            ppo_epochs=2,
            max_episodes=1,
        )
        agent = PPOAgent(
            state_dim=25, n_actions=4, hidden_dim=16, config=cfg
        )
        for i in range(40):
            s = np.eye(25, dtype=np.float32)[i % 25]
            a = i % 4
            agent.store_transition(
                state=s, action=a, reward=1.0, log_prob=-1.0,
                value=0.5, done=(i % 10 == 9),
            )
        stats = agent.update()
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats
        assert len(agent.rollout.states) == 0


class TestRunner:
    """Smoke test for PPORunner."""

    def test_runner_returns_metrics(self, tmp_path: "Path") -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join([
                "logging:",
                "  level: \"INFO\"",
                "  file: \"logs/app.log\"",
                "env:",
                "  grid_size: 4",
                "  random_seed: 1",
                "agent:",
                "  hidden_dim: 16",
                "  rollout_steps: 32",
                "  batch_size: 16",
                "  ppo_epochs: 2",
                "  max_episodes: 15",
                "  max_steps_per_episode: 50",
            ])
        )
        runner = PPORunner(config_path=cfg_path)
        results = runner.run()
        assert "final_episode_reward" in results
        assert "average_return_last_50" in results
        assert np.isfinite(results["final_episode_reward"])
