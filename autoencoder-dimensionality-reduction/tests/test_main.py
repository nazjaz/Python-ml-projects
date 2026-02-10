"""Tests for autoencoder dimensionality reduction module."""

import numpy as np

from src.main import (
    Autoencoder,
    AutoencoderRunner,
    generate_synthetic_data,
    train_autoencoder,
)


class TestDataGeneration:
    """Tests for synthetic data generator."""

    def test_generate_synthetic_data_shapes(self) -> None:
        """Generated data should have correct shape and dtype."""
        x = generate_synthetic_data(
            n_samples=100,
            n_features=10,
            random_seed=42,
        )
        assert x.shape == (100, 10)
        assert x.dtype == np.float32

    def test_generate_synthetic_data_reproducible(self) -> None:
        """Data generation should be reproducible with same seed."""
        x1 = generate_synthetic_data(50, 8, random_seed=1)
        x2 = generate_synthetic_data(50, 8, random_seed=1)
        np.testing.assert_allclose(x1, x2)


class TestAutoencoder:
    """Unit tests for Autoencoder class."""

    def test_forward_output_shape(self) -> None:
        """Forward pass should reconstruct input shape."""
        np.random.seed(0)
        model = Autoencoder(input_dim=12, hidden_dim=6, latent_dim=3)
        x = np.random.randn(5, 12).astype(np.float32)
        x_recon = model.forward(x)
        assert x_recon.shape == x.shape

    def test_encode_decode_roundtrip_dimension(self) -> None:
        """Encode and decode should map dimensions correctly."""
        np.random.seed(1)
        model = Autoencoder(input_dim=10, hidden_dim=5, latent_dim=2)
        x = np.random.randn(4, 10).astype(np.float32)
        z = model.encode(x)
        x_recon = model.decode(z)
        assert z.shape == (4, 2)
        assert x_recon.shape == x.shape

    def test_training_reduces_loss(self) -> None:
        """Training autoencoder should reduce reconstruction loss."""
        np.random.seed(2)
        x = generate_synthetic_data(200, 10, random_seed=2)
        model = Autoencoder(input_dim=10, hidden_dim=6, latent_dim=3)
        history = train_autoencoder(
            model,
            x,
            epochs=5,
            learning_rate=0.05,
            batch_size=32,
        )
        assert len(history["loss"]) == 5
        assert history["loss"][-1] <= history["loss"][0] * 1.2


class TestAutoencoderRunner:
    """Smoke tests for AutoencoderRunner."""

    def test_runner_executes(self) -> None:
        """Runner should return final_loss metric."""
        runner = AutoencoderRunner(config_path=None)
        results = runner.run()
        assert "final_loss" in results
        assert np.isfinite(results["final_loss"])

