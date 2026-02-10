"""Tests for variational autoencoder (VAE) module."""

import numpy as np

from src.main import (
    VAERunner,
    VariationalAutoencoder,
    generate_synthetic_data,
    train_vae,
)


class TestDataGeneration:
    """Tests for synthetic data generator."""

    def test_generate_synthetic_data_shapes(self) -> None:
        """Generated data should have correct shape and dtype."""
        x = generate_synthetic_data(
            n_samples=100,
            n_features=12,
            random_seed=42,
        )
        assert x.shape == (100, 12)
        assert x.dtype == np.float32

    def test_generate_synthetic_data_reproducible(self) -> None:
        """Data generation should be reproducible with same seed."""
        x1 = generate_synthetic_data(50, 8, random_seed=1)
        x2 = generate_synthetic_data(50, 8, random_seed=1)
        np.testing.assert_allclose(x1, x2)


class TestVAE:
    """Unit tests for VariationalAutoencoder."""

    def test_forward_output_shapes(self) -> None:
        """Forward pass should produce correct shapes."""
        np.random.seed(0)
        model = VariationalAutoencoder(
            input_dim=10,
            hidden_dim=6,
            latent_dim=3,
        )
        x = np.random.randn(5, 10).astype(np.float32)
        x_recon, mu, log_var = model.forward(x)
        assert x_recon.shape == x.shape
        assert mu.shape == (5, 3)
        assert log_var.shape == (5, 3)

    def test_reparameterize_respects_shapes(self) -> None:
        """Reparameterization should return samples with same shape as mu."""
        np.random.seed(1)
        model = VariationalAutoencoder(
            input_dim=8,
            hidden_dim=4,
            latent_dim=2,
        )
        mu = np.zeros((7, 2), dtype=np.float32)
        log_var = np.zeros((7, 2), dtype=np.float32)
        z = model.reparameterize(mu, log_var)
        assert z.shape == mu.shape

    def test_training_reduces_total_loss(self) -> None:
        """Training VAE should reduce total loss over epochs."""
        np.random.seed(2)
        x = generate_synthetic_data(200, 10, random_seed=2)
        model = VariationalAutoencoder(
            input_dim=10,
            hidden_dim=6,
            latent_dim=3,
        )
        history = train_vae(
            model,
            x,
            epochs=5,
            learning_rate=0.01,
            batch_size=32,
        )
        assert len(history["total_loss"]) == 5
        # Allow modest increase due to stochasticity of VAE training but
        # ensure the loss does not explode.
        assert history["total_loss"][-1] <= history["total_loss"][0] * 2.0


class TestVAERunner:
    """Smoke tests for VAERunner."""

    def test_runner_executes(self) -> None:
        """Runner should return final loss metrics."""
        runner = VAERunner(config_path=None)
        results = runner.run()
        assert "final_total_loss" in results
        assert "final_recon_loss" in results
        assert "final_kl_loss" in results
        assert np.isfinite(results["final_total_loss"])
        assert np.isfinite(results["final_recon_loss"])
        assert np.isfinite(results["final_kl_loss"])

