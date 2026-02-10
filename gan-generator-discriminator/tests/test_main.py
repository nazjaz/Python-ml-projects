"""Tests for GAN generator-discriminator module."""

import numpy as np

from src.main import (
    GANRunner,
    GANTrainer,
    Discriminator,
    Generator,
    generate_real_data,
)


class TestDataGeneration:
    """Tests for synthetic real data generation."""

    def test_generate_real_data_shapes(self) -> None:
        """Generated data should have correct shape and dtype."""
        x = generate_real_data(
            n_samples=120,
            data_dim=2,
            random_seed=42,
        )
        assert x.shape == (120, 2)
        assert x.dtype == np.float32

    def test_generate_real_data_reproducible(self) -> None:
        """Data generation should be reproducible for same seed."""
        x1 = generate_real_data(50, 2, random_seed=7)
        x2 = generate_real_data(50, 2, random_seed=7)
        np.testing.assert_allclose(x1, x2)


class TestNetworks:
    """Tests for generator and discriminator network shapes."""

    def test_generator_forward_shape(self) -> None:
        """Generator should map noise to correct data shape."""
        np.random.seed(0)
        gen = Generator(noise_dim=8, hidden_dim=16, data_dim=2)
        z = np.random.randn(10, 8).astype(np.float32)
        x_fake = gen.forward(z)
        assert x_fake.shape == (10, 2)

    def test_discriminator_forward_shape(self) -> None:
        """Discriminator should output scalar probabilities per sample."""
        np.random.seed(1)
        disc = Discriminator(data_dim=2, hidden_dim=16)
        x = np.random.randn(10, 2).astype(np.float32)
        probs = disc.forward(x)
        assert probs.shape == (10, 1)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


class TestGANTraining:
    """Smoke test to ensure GAN training runs and losses are finite."""

    def test_gan_trainer_runs(self) -> None:
        """GANTrainer.train should produce finite losses."""
        np.random.seed(2)
        x_real = generate_real_data(
            n_samples=200,
            data_dim=2,
            random_seed=2,
        )
        gen = Generator(noise_dim=4, hidden_dim=8, data_dim=2)
        disc = Discriminator(data_dim=2, hidden_dim=8)
        trainer = GANTrainer(generator=gen, discriminator=disc, noise_dim=4)

        history = trainer.train(
            x_real=x_real,
            epochs=10,
            batch_size=32,
            lr_g=0.001,
            lr_d=0.001,
            d_steps=1,
            g_steps=1,
        )
        assert len(history["d_loss"]) == 10
        assert len(history["g_loss"]) == 10
        assert np.isfinite(history["d_loss"][-1])
        assert np.isfinite(history["g_loss"][-1])


class TestGANRunner:
    """Smoke test for GANRunner."""

    def test_runner_executes(self) -> None:
        """GANRunner.run should return final loss metrics."""
        runner = GANRunner(config_path=None)
        results = runner.run()
        assert "final_d_loss" in results
        assert "final_g_loss" in results
        assert np.isfinite(results["final_d_loss"])
        assert np.isfinite(results["final_g_loss"])

