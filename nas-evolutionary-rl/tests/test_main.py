"""Tests for NAS (evolutionary and RL)."""

from pathlib import Path

import numpy as np
import pytest

from src.main import (
    ACTIVATION_OPTIONS,
    HIDDEN_DIM_OPTIONS,
    NUM_LAYERS_OPTIONS,
    Controller,
    EvolutionaryNAS,
    RLNAS,
    NASConfig,
    TrainableMLP,
    decode_architecture,
    encode_architecture,
    load_digits_data,
    random_architecture,
    run_nas,
    train_and_evaluate,
)


class TestSearchSpace:
    """Encode/decode and random architecture."""

    def test_encode_decode_roundtrip(self) -> None:
        for nl in NUM_LAYERS_OPTIONS:
            for hd in HIDDEN_DIM_OPTIONS:
                for act in ACTIVATION_OPTIONS:
                    c = encode_architecture(nl, hd, act)
                    assert len(c) == 3
                    nl2, hd2, act2 = decode_architecture(c)
                    assert (nl2, hd2, act2) == (nl, hd, act)

    def test_random_architecture_valid(self) -> None:
        np.random.seed(0)
        for _ in range(20):
            c = random_architecture()
            nl, hd, act = decode_architecture(c)
            assert nl in NUM_LAYERS_OPTIONS
            assert hd in HIDDEN_DIM_OPTIONS
            assert act in ACTIVATION_OPTIONS


class TestTrainableMLP:
    """TrainableMLP forward and backward."""

    def test_forward_shape(self) -> None:
        np.random.seed(0)
        mlp = TrainableMLP(64, 10, 2, 32, "relu")
        x = np.random.randn(5, 64).astype(np.float32) * 0.1
        y = mlp.forward(x)
        assert y.shape == (5, 10)

    def test_backward_updates_params(self) -> None:
        np.random.seed(1)
        mlp = TrainableMLP(8, 4, 2, 8, "tanh")
        w0 = mlp.weights[0].copy()
        x = np.random.randn(2, 8).astype(np.float32) * 0.1
        _ = mlp.forward(x)
        mlp.backward(np.ones((2, 4), dtype=np.float32) / 2, 0.01)
        assert not np.allclose(mlp.weights[0], w0)


class TestTrainAndEvaluate:
    """train_and_evaluate returns accuracy in [0, 1]."""

    def test_returns_accuracy(self) -> None:
        tx, ty, vx, vy = load_digits_data(max_samples=80, random_seed=0)
        acc = train_and_evaluate(
            tx, ty, vx, vy, [0, 0, 0], epochs=2, random_seed=0
        )
        assert isinstance(acc, float)
        assert 0 <= acc <= 1.0


class TestLoadDigitsData:
    """Data loading."""

    def test_shapes_and_splits(self) -> None:
        tx, ty, vx, vy = load_digits_data(
            train_ratio=0.8, max_samples=100, random_seed=0
        )
        assert tx.shape[0] + vx.shape[0] <= 100
        assert tx.shape[1] == vx.shape[1]
        assert len(ty) == tx.shape[0] and len(vy) == vx.shape[0]


class TestEvolutionaryNAS:
    """Evolutionary NAS run."""

    def test_run_returns_best_and_history(self) -> None:
        tx, ty, vx, vy = load_digits_data(max_samples=60, random_seed=0)
        nas = EvolutionaryNAS(
            3, 1, 0.3, 2, tx, ty, vx, vy,
            eval_epochs=1, random_seed=0,
        )
        best, fitness, history = nas.run()
        assert len(best) == 3
        assert 0 <= fitness <= 1.0
        assert len(history) == 1
        assert history[0]["best_fitness"] == fitness


class TestRLNAS:
    """RL-based NAS run."""

    def test_run_returns_best_and_history(self) -> None:
        tx, ty, vx, vy = load_digits_data(max_samples=60, random_seed=0)
        nas = RLNAS(
            3, 0.01, tx, ty, vx, vy,
            eval_epochs=1, random_seed=0,
        )
        best, reward, history = nas.run()
        assert len(best) == 3
        assert 0 <= reward <= 1.0
        assert len(history) == 3


class TestController:
    """REINFORCE controller."""

    def test_sample_valid_architecture(self) -> None:
        np.random.seed(0)
        ctrl = Controller(random_seed=0)
        choices = ctrl.sample(random_seed=1)
        assert len(choices) == 3
        assert choices[0] in range(3)
        assert choices[1] in range(3)
        assert choices[2] in range(2)
        assert isinstance(ctrl.get_log_prob(), float)


class TestRunNAS:
    """run_nas with config."""

    def test_evolution_method(self) -> None:
        config = NASConfig(
            method="evolution",
            population_size=3,
            num_generations=1,
            eval_epochs=1,
            max_samples=50,
            random_seed=0,
        )
        results = run_nas(config)
        assert results["method"] == "evolution"
        assert "best_architecture" in results
        assert "best_validation_accuracy" in results
        assert 0 <= results["best_validation_accuracy"] <= 1.0

    def test_rl_method(self) -> None:
        config = NASConfig(
            method="rl",
            num_rollouts=2,
            eval_epochs=1,
            max_samples=50,
            random_seed=0,
        )
        results = run_nas(config)
        assert results["method"] == "rl"
        assert "best_architecture" in results
        assert "best_validation_accuracy" in results
