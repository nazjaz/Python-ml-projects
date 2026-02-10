"""Tests for WGAN-GP module."""

import numpy as np

from src.main import (
    WGANGenerator,
    WGANCritic,
    WGANRunner,
    WGANTrainer,
    compute_gradient_penalty,
    generate_real_data,
)


class TestDataGeneration:
    """Tests for synthetic real data generation."""

    def test_generate_real_data_shapes(self) -> None:
        """Generated data should have expected shape and dtype."""
        x = generate_real_data(n_samples=200, data_dim=3, random_seed=0)
        assert x.shape == (200, 3)
        assert x.dtype == np.float32

    def test_generate_real_data_reproducible(self) -> None:
        """Data generation should be reproducible with fixed seed."""
        x1 = generate_real_data(100, 2, random_seed=1)
        x2 = generate_real_data(100, 2, random_seed=1)
        np.testing.assert_allclose(x1, x2)


class TestNetworks:
    """Tests for generator and critic networks."""

    def test_generator_forward_shape(self) -> None:
        """Generator output should have correct shape."""
        np.random.seed(0)
        gen = WGANGenerator(noise_dim=4, hidden_dim=8, data_dim=2)
        z = np.random.randn(5, 4).astype(np.float32)
        x_fake = gen.forward(z)
        assert x_fake.shape == (5, 2)

    def test_critic_forward_shape(self) -> None:
        """Critic scores should be scalar per sample."""
        np.random.seed(0)
        crit = WGANCritic(data_dim=2, hidden_dim=8)
        x = np.random.randn(7, 2).astype(np.float32)
        scores = crit.forward(x)
        assert scores.shape == (7, 1)

    def test_critic_backward_input_shape(self) -> None:
        """Gradient of scores w.r.t inputs should match input shape."""
        np.random.seed(0)
        crit = WGANCritic(data_dim=2, hidden_dim=8)
        x = np.random.randn(6, 2).astype(np.float32)
        scores = crit.forward(x)
        grad_scores = np.ones_like(scores, dtype=np.float32)
        grad_x = crit.backward_input(grad_scores)
        assert grad_x.shape == x.shape


class TestGradientPenalty:
    """Tests for gradient penalty computation."""

    def test_compute_gradient_penalty_outputs_finite(self) -> None:
        """Gradient penalty and mean norm should be finite scalars."""
        np.random.seed(1)
        crit = WGANCritic(data_dim=2, hidden_dim=8)
        real = np.random.randn(10, 2).astype(np.float32)
        fake = np.random.randn(10, 2).astype(np.float32)
        penalty, mean_norm = compute_gradient_penalty(
            critic=crit,
            real=real,
            fake=fake,
            lambda_gp=10.0,
        )
        assert np.isfinite(penalty)
        assert np.isfinite(mean_norm)


class TestTraining:
    """Smoke tests for WGAN-GP training."""

    def test_trainer_runs(self) -> None:
        """WGANTrainer should train for a few epochs without NaNs."""
        np.random.seed(2)
        x_real = generate_real_data(256, 2, random_seed=2)
        gen = WGANGenerator(noise_dim=4, hidden_dim=8, data_dim=2)
        crit = WGANCritic(data_dim=2, hidden_dim=8)
        trainer = WGANTrainer(
            generator=gen,
            critic=crit,
            noise_dim=4,
            lambda_gp=10.0,
        )
        history = trainer.train(
            x_real=x_real,
            epochs=5,
            batch_size=32,
            lr_g=0.0005,
            lr_c=0.0005,
            critic_iters=3,
        )
        assert len(history["critic_loss"]) == 5
        assert len(history["generator_loss"]) == 5
        assert np.isfinite(history["critic_loss"][-1])
        assert np.isfinite(history["generator_loss"][-1])


class TestRunner:
    """Smoke test for WGANRunner."""

    def test_runner_executes(self) -> None:
        """WGANRunner.run should return final loss metrics."""
        runner = WGANRunner(config_path=None)
        results = runner.run()
        assert "final_critic_loss" in results
        assert "final_generator_loss" in results
        assert np.isfinite(results["final_critic_loss"])
        assert np.isfinite(results["final_generator_loss"])

