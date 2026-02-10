"""Tests for knowledge distillation (teacher-student, temperature scaling)."""

import numpy as np
import pytest

from src.main import (
    MLP,
    DistillationConfig,
    evaluate,
    kl_divergence,
    load_data,
    run_distillation,
    softmax,
    train_teacher,
    train_student_with_distillation,
)


class TestSoftmax:
    def test_temperature_scaling(self) -> None:
        z = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        p1 = softmax(z, temperature=1.0)
        p2 = softmax(z, temperature=4.0)
        assert np.allclose(p1.sum(), 1.0) and np.allclose(p2.sum(), 1.0)
        assert p2[0, 0] > p1[0, 0]

    def test_sum_one(self) -> None:
        z = np.random.randn(3, 5).astype(np.float32)
        p = softmax(z, temperature=2.0)
        assert np.allclose(np.sum(p, axis=1), np.ones(3))


class TestKL:
    def test_kl_non_negative(self) -> None:
        p = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
        q = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float32)
        k = kl_divergence(p, q, axis=1)
        assert np.all(k >= -1e-6)

    def test_kl_shape(self) -> None:
        p = np.ones((4, 10)) / 10.0
        q = np.ones((4, 10)) / 10.0
        assert kl_divergence(p, q, axis=1).shape == (4,)


class TestMLP:
    def test_forward_shape(self) -> None:
        np.random.seed(0)
        mlp = MLP(64, 32, 10)
        x = np.random.randn(5, 64).astype(np.float32) * 0.1
        y = mlp.forward(x)
        assert y.shape == (5, 10)

    def test_backward_updates(self) -> None:
        np.random.seed(1)
        mlp = MLP(8, 8, 4)
        w0 = mlp.w1.copy()
        x = np.random.randn(2, 8).astype(np.float32) * 0.1
        mlp.forward(x)
        mlp.backward(np.ones((2, 4), dtype=np.float32) / 2, 0.01)
        assert not np.allclose(mlp.w1, w0)


class TestTrainTeacher:
    def test_returns_losses(self) -> None:
        tx, ty, vx, vy = load_data(max_samples=50, random_seed=0)
        np.random.seed(0)
        teacher = MLP(tx.shape[1], 16, int(ty.max()) + 1)
        losses = train_teacher(teacher, tx, ty, epochs=2, lr=0.01, batch_size=16)
        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)


class TestTrainStudentDistillation:
    def test_runs_and_student_updates(self) -> None:
        tx, ty, vx, vy = load_data(max_samples=50, random_seed=0)
        np.random.seed(0)
        teacher = MLP(tx.shape[1], 16, int(ty.max()) + 1)
        train_teacher(teacher, tx, ty, epochs=1, lr=0.01, batch_size=16)
        student = MLP(tx.shape[1], 8, int(ty.max()) + 1)
        w_before = student.w1.copy()
        train_student_with_distillation(
            teacher, student, tx, ty,
            temperature=2.0, alpha=0.7, epochs=1, lr=0.01, batch_size=16,
        )
        assert not np.allclose(student.w1, w_before)


class TestEvaluate:
    def test_accuracy_and_ce(self) -> None:
        tx, ty, vx, vy = load_data(max_samples=30, random_seed=0)
        np.random.seed(0)
        model = MLP(tx.shape[1], 8, int(ty.max()) + 1)
        acc, ce = evaluate(model, vx, vy)
        assert 0 <= acc <= 1.0
        assert ce >= 0.0


class TestLoadData:
    def test_shapes(self) -> None:
        tx, ty, vx, vy = load_data(train_ratio=0.8, max_samples=100)
        assert tx.shape[0] + vx.shape[0] <= 100
        assert tx.shape[1] == vx.shape[1]


class TestRunDistillation:
    def test_returns_metrics(self) -> None:
        cfg = DistillationConfig(
            teacher_hidden=16,
            student_hidden=8,
            teacher_epochs=1,
            student_epochs=1,
            max_samples=60,
            random_seed=0,
        )
        results = run_distillation(cfg)
        assert "teacher_val_accuracy" in results
        assert "student_val_accuracy" in results
        assert "compression_ratio" in results
        assert results["compression_ratio"] >= 1.0
        assert results["teacher_params"] > results["student_params"]
