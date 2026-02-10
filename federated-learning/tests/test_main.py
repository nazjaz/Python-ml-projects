"""Tests for federated learning (distributed training, aggregation, privacy)."""

import numpy as np
import pytest

from src.main import (
    MLP,
    FederatedClient,
    FederatedConfig,
    FederatedServer,
    add_gaussian_noise,
    aggregate_fedavg,
    clip_gradient_norm,
    evaluate_global,
    load_data,
    partition_data,
    run_federated,
)


class TestMLP:
    def test_get_set_weights_roundtrip(self) -> None:
        np.random.seed(0)
        m = MLP(8, 4, 2)
        w = m.get_weights()
        assert len(w) == 4
        m.set_weights([x + 0.1 for x in w])
        w2 = m.get_weights()
        assert all(np.allclose(a, b + 0.1) for a, b in zip(w2, w))

    def test_forward_shape(self) -> None:
        m = MLP(64, 16, 10)
        x = np.random.randn(5, 64).astype(np.float32) * 0.1
        y = m.forward(x)
        assert y.shape == (5, 10)


class TestClipGradientNorm:
    def test_clips_large_delta(self) -> None:
        ref = [np.zeros((2, 2)), np.zeros(2)]
        w = [np.ones((2, 2)) * 10, np.ones(2) * 10]
        out = clip_gradient_norm(w, ref, max_norm=1.0)
        total = np.sqrt(sum(np.sum((o - r) ** 2) for o, r in zip(out, ref)))
        assert total <= 1.01

    def test_leaves_small_delta_unchanged(self) -> None:
        ref = [np.ones((2, 2)), np.ones(2)]
        w = [np.ones((2, 2)) * 1.01, np.ones(2) * 1.01]
        out = clip_gradient_norm(w, ref, max_norm=10.0)
        assert np.allclose(out[0], w[0])


class TestAddGaussianNoise:
    def test_shape_unchanged(self) -> None:
        w = [np.ones((2, 3)), np.ones(4)]
        out = add_gaussian_noise(w, sigma=0.01)
        assert out[0].shape == w[0].shape and out[1].shape == w[1].shape

    def test_different_with_positive_sigma(self) -> None:
        w = [np.zeros((2, 2))]
        out = add_gaussian_noise(w, sigma=1.0)
        assert not np.allclose(out[0], w[0])


class TestAggregateFedAvg:
    def test_weighted_average(self) -> None:
        w1 = [np.array([[1.0, 1.0], [1.0, 1.0]])]
        w2 = [np.array([[3.0, 3.0], [3.0, 3.0]])]
        agg = aggregate_fedavg([w1, w2], [1, 1])
        assert len(agg) == 1
        assert np.allclose(agg[0], 2.0)

    def test_weights_by_count(self) -> None:
        w1 = [np.array([1.0, 1.0])]
        w2 = [np.array([5.0, 5.0])]
        agg = aggregate_fedavg([w1, w2], [3, 1])
        assert np.allclose(agg[0], 2.0)


class TestPartitionData:
    def test_sizes_sum_to_total(self) -> None:
        x = np.random.randn(100, 8).astype(np.float32)
        y = np.random.randint(0, 5, size=100)
        shards = partition_data(x, y, num_clients=5, iid=True, random_seed=0)
        assert sum(s[0].shape[0] for s in shards) == 100
        assert len(shards) == 5


class TestFederatedClient:
    def test_train_local_returns_weights_and_count(self) -> None:
        tx = np.random.randn(20, 8).astype(np.float32) * 0.1
        ty = np.random.randint(0, 3, size=20)
        client = FederatedClient(
            0, tx, ty, 8, 4, 3,
            local_epochs=1, lr=0.01, batch_size=8, random_seed=0,
        )
        global_w = client.model.get_weights()
        new_w, n = client.train_local(global_w)
        assert len(new_w) == len(global_w)
        assert n == 20


class TestFederatedServer:
    def test_run_round_aggregates(self) -> None:
        tx, ty, vx, vy = load_data(max_samples=40, random_seed=0)
        shards = partition_data(tx, ty, 2, iid=True, random_seed=0)
        in_dim, out_dim = tx.shape[1], int(ty.max()) + 1
        clients = [
            FederatedClient(0, shards[0][0], shards[0][1], in_dim, 8, out_dim, random_seed=0),
            FederatedClient(1, shards[1][0], shards[1][1], in_dim, 8, out_dim, random_seed=1),
        ]
        server = FederatedServer(in_dim, 8, out_dim, clients, random_seed=0)
        w_before = server.get_global_weights()[0].copy()
        server.run_round()
        w_after = server.get_global_weights()[0]
        assert not np.allclose(w_before, w_after)


class TestEvaluateGlobal:
    def test_returns_accuracy_and_ce(self) -> None:
        tx, ty, vx, vy = load_data(max_samples=30, random_seed=0)
        shards = partition_data(tx, ty, 2, random_seed=0)
        in_dim, out_dim = tx.shape[1], int(ty.max()) + 1
        clients = [
            FederatedClient(0, shards[0][0], shards[0][1], in_dim, 8, out_dim, random_seed=0),
            FederatedClient(1, shards[1][0], shards[1][1], in_dim, 8, out_dim, random_seed=1),
        ]
        server = FederatedServer(in_dim, 8, out_dim, clients, random_seed=0)
        acc, ce = evaluate_global(server, vx, vy)
        assert 0 <= acc <= 1.0
        assert ce >= 0.0


class TestRunFederated:
    def test_returns_metrics(self) -> None:
        cfg = FederatedConfig(
            num_clients=3,
            num_rounds=2,
            max_samples=60,
            random_seed=0,
        )
        results = run_federated(cfg)
        assert "final_val_accuracy" in results
        assert "num_rounds" in results
        assert "num_clients" in results
        assert len(results["history"]) == 2

    def test_with_privacy_options(self) -> None:
        cfg = FederatedConfig(
            num_clients=2,
            num_rounds=1,
            max_samples=40,
            max_grad_norm=5.0,
            noise_sigma=0.01,
            random_seed=0,
        )
        results = run_federated(cfg)
        assert results["final_val_accuracy"] >= 0.0
