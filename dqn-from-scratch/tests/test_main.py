"""Tests for DQN implementation with replay and target network."""

import numpy as np
import pytest

from src.main import (
    DQNAgent,
    DQNConfig,
    DQNRunner,
    QNetwork,
    ReplayBuffer,
    SimpleGridworldEnv,
)


class TestReplayBuffer:
    """Tests for the replay buffer."""

    def test_add_and_sample_shapes(self) -> None:
        state_dim = 4
        buffer = ReplayBuffer(capacity=100, state_dim=state_dim)
        state = np.zeros((state_dim,), dtype=np.float32)
        next_state = np.ones((state_dim,), dtype=np.float32)
        for i in range(50):
            buffer.add(
                state=state,
                action=i % 2,
                reward=float(i),
                next_state=next_state,
                done=bool(i % 3 == 0),
            )
        assert buffer.can_sample(16)
        batch = buffer.sample(16)
        states, actions, rewards, next_states, dones = batch
        assert states.shape == (16, state_dim)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, state_dim)
        assert dones.shape == (16,)


class TestQNetwork:
    """Tests for Q-network forward and backward."""

    def test_forward_output_shape(self) -> None:
        np.random.seed(0)
        net = QNetwork(state_dim=4, n_actions=3, hidden_dim=8)
        states = np.random.randn(5, 4).astype(np.float32)
        q_values = net.forward(states)
        assert q_values.shape == (5, 3)

    def test_backward_updates_parameters(self) -> None:
        np.random.seed(1)
        net = QNetwork(state_dim=4, n_actions=3, hidden_dim=8)
        states = np.random.randn(10, 4).astype(np.float32)
        q_values = net.forward(states)
        grad_q = np.random.randn(*q_values.shape).astype(np.float32) * 0.01
        w1_before = net.w1.copy()
        net.backward(grad_q, lr=0.1)
        assert not np.allclose(net.w1, w1_before)


class TestDQNAgent:
    """Tests for DQNAgent behavior."""

    def test_select_action_returns_valid_action(self) -> None:
        np.random.seed(2)
        state_dim = 9
        n_actions = 4
        config = DQNConfig(max_episodes=1)
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=8,
            config=config,
        )
        state = np.eye(state_dim, dtype=np.float32)[0]
        action = agent.select_action(state)
        assert 0 <= action < n_actions

    def test_train_step_runs_after_buffer_fill(self) -> None:
        np.random.seed(3)
        state_dim = 9
        n_actions = 4
        config = DQNConfig(batch_size=8, buffer_capacity=100, max_episodes=1)
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=8,
            config=config,
        )
        state = np.eye(state_dim, dtype=np.float32)[0]
        next_state = np.eye(state_dim, dtype=np.float32)[1]
        for _ in range(20):
            agent.store_transition(
                state=state,
                action=0,
                reward=1.0,
                next_state=next_state,
                done=False,
            )
        loss = agent.train_step()
        assert loss is not None
        assert np.isfinite(loss)


class TestRunner:
    """Smoke tests for DQNRunner."""

    def test_runner_runs_and_returns_metrics(self, tmp_path: "Path") -> None:
        from src.main import DQNRunner

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "logging:",
                    "  level: \"INFO\"",
                    "  file: \"logs/app.log\"",
                    "env:",
                    "  grid_size: 4",
                    "  random_seed: 1",
                    "agent:",
                    "  hidden_dim: 16",
                    "  gamma: 0.99",
                    "  learning_rate: 0.001",
                    "  batch_size: 8",
                    "  buffer_capacity: 200",
                    "  epsilon_start: 1.0",
                    "  epsilon_end: 0.1",
                    "  epsilon_decay_steps: 100",
                    "  target_update_interval: 20",
                    "  max_episodes: 20",
                    "  max_steps_per_episode: 50",
                ]
            )
        )

        runner = DQNRunner(config_path=cfg_path)
        results = runner.run()
        assert "final_episode_reward" in results
        assert "average_return_last_50" in results
        assert np.isfinite(results["final_episode_reward"])
        assert np.isfinite(results["average_return_last_50"])

