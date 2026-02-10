"""Tests for SimCLR contrastive learning."""

import numpy as np
import pytest

from src.main import (
    SimCLRConfig,
    SimCLRNet,
    evaluate_representation,
    load_data,
    l2_normalize,
    make_two_views,
    nt_xent_loss,
    run_simclr,
    train_simclr,
)


class TestMakeTwoViews:
    def test_shapes(self) -> None:
        x = np.random.randn(10, 64).astype(np.float32)
        v1, v2 = make_two_views(x, random_seed=0)
        assert v1.shape == v2.shape == (10, 64)

    def test_different_views(self) -> None:
        x = np.zeros((5, 64), dtype=np.float32)
        v1, v2 = make_two_views(x, noise_std=0.5, random_seed=0)
        assert not np.allclose(v1, v2)


class TestL2Normalize:
    def test_unit_norm(self) -> None:
        x = np.random.randn(4, 8).astype(np.float32)
        y = l2_normalize(x, axis=1)
        n = np.linalg.norm(y, axis=1)
        assert np.allclose(n, np.ones(4))


class TestSimCLRNet:
    def test_forward_shape(self) -> None:
        net = SimCLRNet(64, 32, 16, encoder_hidden=32, proj_hidden=16, random_seed=0)
        x = np.random.randn(6, 64).astype(np.float32) * 0.1
        z = net.forward(x)
        assert z.shape == (6, 16)

    def test_encode_shape(self) -> None:
        net = SimCLRNet(64, 32, 16, random_seed=0)
        x = np.random.randn(4, 64).astype(np.float32) * 0.1
        h = net.encode(x)
        assert h.shape == (4, 32)


class TestNTXentLoss:
    def test_loss_positive_and_grad_shape(self) -> None:
        z = np.random.randn(8, 16).astype(np.float32) * 0.1
        loss, grad = nt_xent_loss(z, temperature=0.5)
        assert loss > 0
        assert grad.shape == z.shape

    def test_lower_loss_for_aligned_pairs(self) -> None:
        z_random = np.random.randn(4, 8).astype(np.float32) * 0.5
        z_aligned = np.tile(np.random.randn(2, 8).astype(np.float32), (2, 1))
        z_aligned = l2_normalize(z_aligned, axis=1)
        loss_rand, _ = nt_xent_loss(z_random, 0.5)
        loss_align, _ = nt_xent_loss(z_aligned, 0.5)
        assert loss_align <= loss_rand + 1.0


class TestLoadData:
    def test_returns_2d(self) -> None:
        x = load_data(max_samples=20, random_seed=0)
        assert x.ndim == 2
        assert x.shape[0] <= 20


class TestTrainSimclr:
    def test_returns_losses(self) -> None:
        data = load_data(max_samples=40, random_seed=0)
        net = SimCLRNet(
            data.shape[1], 16, 8,
            encoder_hidden=32, proj_hidden=16,
            random_seed=0,
        )
        losses = train_simclr(
            net, data, epochs=2, batch_size=16,
            lr=0.01, temperature=0.5, noise_std=0.1,
            scale_range=(0.9, 1.1), random_seed=0,
        )
        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)


class TestEvaluateRepresentation:
    def test_shape(self) -> None:
        net = SimCLRNet(64, 32, 16, random_seed=0)
        x = np.random.randn(5, 64).astype(np.float32) * 0.1
        h = evaluate_representation(net, x)
        assert h.shape == (5, 32)


class TestRunSimclr:
    def test_returns_metrics(self) -> None:
        cfg = SimCLRConfig(
            epochs=2,
            batch_size=16,
            max_samples=50,
            random_seed=0,
        )
        results = run_simclr(cfg)
        assert "final_loss" in results
        assert "num_samples" in results
        assert "repr_dim" in results
