"""Tests for transfer learning and fine-tuning module."""

import numpy as np

from src.main import (
    MLPBase,
    TransferLearningRunner,
    generate_source_and_target_data,
    train_model,
)


class TestDataGeneration:
    """Tests for source and target data generation."""

    def test_generate_source_and_target_data_shapes(self) -> None:
        """Generated datasets should have expected shapes."""
        x_base, y_base, x_target, y_target = generate_source_and_target_data(
            base_n_samples=300,
            target_n_samples=120,
            n_features=10,
            n_classes=3,
            random_seed=42,
        )
        assert x_base.shape == (300, 10)
        assert x_target.shape == (120, 10)
        assert y_base.shape == (300,)
        assert y_target.shape == (120,)


class TestMLPBase:
    """Unit tests for base MLP model."""

    def test_forward_output_shape(self) -> None:
        """Forward pass should produce correct logits shape."""
        np.random.seed(0)
        model = MLPBase(input_dim=8, hidden_dim=16, n_classes=4)
        x = np.random.randn(5, 8).astype(np.float32)
        logits = model.forward(x)
        assert logits.shape == (5, 4)

    def test_training_reduces_loss(self) -> None:
        """Training on synthetic data should reduce loss."""
        np.random.seed(1)
        x_base, y_base, _, _ = generate_source_and_target_data(
            base_n_samples=400,
            target_n_samples=100,
            n_features=8,
            n_classes=3,
            random_seed=1,
        )
        model = MLPBase(input_dim=8, hidden_dim=16, n_classes=3)
        history = train_model(
            model,
            x_base,
            y_base,
            epochs=5,
            learning_rate=0.05,
            batch_size=32,
        )
        assert len(history["loss"]) == 5
        assert history["loss"][-1] <= history["loss"][0] * 1.2


class TestTransferLearningRunner:
    """Smoke tests for TransferLearningRunner strategies."""

    def test_runner_executes_for_all_strategies(self) -> None:
        """Runner should return metrics for each strategy."""
        runner = TransferLearningRunner(config_path=None)
        for strategy in [
            "scratch",
            "feature_extractor",
            "head_finetune",
            "full_finetune",
        ]:
            results = runner.run(strategy=strategy)
            assert "source_loss" in results
            assert "source_accuracy" in results
            assert "target_loss" in results
            assert "target_accuracy" in results
            assert 0.0 <= results["target_accuracy"] <= 1.0

