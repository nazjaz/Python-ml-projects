"""Tests for diffusion image generation model."""

import numpy as np

from src.main import (
    DiffusionConfig,
    DiffusionModel,
    MLP,
    cosine_beta_schedule,
    load_data,
    run_diffusion,
    timestep_embedding,
)


class TestCosineSchedule:
    def test_length_and_range(self) -> None:
        betas = cosine_beta_schedule(10)
        assert betas.shape == (10,)
        assert np.all(betas > 0) and np.all(betas < 1)


class TestTimestepEmbedding:
    def test_embedding_shape(self) -> None:
        t = np.array([0, 1, 5], dtype=np.int64)
        emb = timestep_embedding(t, dim=8)
        assert emb.shape == (3, 8)


class TestMLP:
    def test_forward_backward(self) -> None:
        mlp = MLP(4, 8, 4, random_seed=0)
        x = np.random.randn(2, 4).astype(np.float32)
        y = mlp.forward(x)
        assert y.shape == (2, 4)
        w_before = mlp.w1.copy()
        mlp.backward(np.ones_like(y) / 2.0, lr=0.01)
        assert not np.allclose(w_before, mlp.w1)


class TestDiffusionModel:
    def test_q_sample_shapes(self) -> None:
        cfg = DiffusionConfig(timesteps=5)
        model = DiffusionModel(cfg, data_dim=4)
        x0 = np.random.randn(3, 4).astype(np.float32)
        t = np.array([0, 1, 4], dtype=np.int64)
        x_t, eps = model.q_sample(x0, t)
        assert x_t.shape == x0.shape
        assert eps.shape == x0.shape

    def test_train_and_sample(self) -> None:
        x = load_data(max_samples=32, random_seed=0)
        cfg = DiffusionConfig(timesteps=10, epochs=1, batch_size=16, max_samples=32, random_seed=0)
        model = DiffusionModel(cfg, data_dim=x.shape[1])
        losses = model.train(x)
        assert len(losses) == 1
        samples = model.sample(num_samples=4)
        assert samples.shape == (4, x.shape[1])


class TestRunDiffusion:
    def test_returns_metrics(self) -> None:
        cfg = DiffusionConfig(timesteps=10, epochs=1, batch_size=16, max_samples=32, random_seed=0)
        results = run_diffusion(cfg)
        assert "final_loss" in results
        assert "samples_shape" in results
