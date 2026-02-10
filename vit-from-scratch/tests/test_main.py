"""Tests for Vision Transformer implementation."""

import numpy as np
import pytest

from src.main import (
    PatchEmbedding,
    ViTClassifier,
    ViTRunner,
    generate_synthetic_images,
)


class TestPatchEmbedding:
    """Tests for patch embedding."""

    def test_patches_shape(self) -> None:
        """Patch embedding should produce correct sequence length and dim."""
        image_size = 32
        patch_size = 4
        in_channels = 3
        dim_model = 16
        batch_size = 2
        images = np.random.rand(
            batch_size, image_size, image_size, in_channels
        ).astype(np.float32)
        patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim_model=dim_model,
        )
        tokens = patch_embed.forward(images)
        num_patches = (image_size // patch_size) ** 2
        assert tokens.shape == (batch_size, num_patches, dim_model)


class TestSyntheticImages:
    """Tests for synthetic image generator."""

    def test_generate_synthetic_images_shapes(self) -> None:
        images, labels = generate_synthetic_images(
            n_samples=10,
            image_size=16,
            in_channels=3,
            num_classes=5,
            random_seed=123,
        )
        assert images.shape == (10, 16, 16, 3)
        assert labels.shape == (10,)
        assert np.all(labels >= 0)
        assert np.all(labels < 5)


class TestViTClassifier:
    """Tests for ViT classifier."""

    def test_forward_output_shape(self) -> None:
        np.random.seed(0)
        model = ViTClassifier(
            image_size=16,
            patch_size=4,
            in_channels=3,
            dim_model=32,
            num_heads=4,
            dim_ff=64,
            num_layers=1,
            num_classes=7,
        )
        images = np.random.rand(4, 16, 16, 3).astype(np.float32)
        logits = model.forward(images)
        assert logits.shape == (4, 7)

    def test_cross_entropy_loss_and_grad(self) -> None:
        np.random.seed(1)
        logits = np.random.randn(5, 3).astype(np.float32) * 0.1
        labels = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        loss, grad = ViTClassifier.cross_entropy_loss(logits, labels)
        assert np.isfinite(loss)
        assert grad.shape == logits.shape
        assert np.all(np.isfinite(grad))

    def test_backward_updates_parameters(self) -> None:
        np.random.seed(2)
        model = ViTClassifier(
            image_size=16,
            patch_size=4,
            in_channels=3,
            dim_model=32,
            num_heads=4,
            dim_ff=64,
            num_layers=1,
            num_classes=4,
        )
        images = np.random.rand(4, 16, 16, 3).astype(np.float32)
        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        w_before = model.w_cls.copy()
        logits = model.forward(images)
        loss, grad_logits = ViTClassifier.cross_entropy_loss(logits, labels)
        assert np.isfinite(loss)
        model.backward(grad_logits, lr=0.01)
        assert not np.allclose(model.w_cls, w_before)


class TestRunner:
    """Smoke tests for ViTRunner using a tiny config."""

    def test_runner_runs_and_returns_metrics(self, tmp_path: "Path") -> None:
        # Local import to avoid circular imports in type checking
        from src.main import ViTRunner

        # Minimal config for quick run
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "logging:",
                    "  level: \"INFO\"",
                    "  file: \"logs/app.log\"",
                    "data:",
                    "  image_size: 16",
                    "  in_channels: 3",
                    "  num_classes: 4",
                    "  n_train: 32",
                    "  n_test: 16",
                    "  random_seed: 7",
                    "model:",
                    "  dim_model: 32",
                    "  patch_size: 4",
                    "  num_heads: 4",
                    "  dim_ff: 64",
                    "  num_layers: 1",
                    "training:",
                    "  epochs: 2",
                    "  batch_size: 8",
                    "  learning_rate: 0.001",
                ]
            )
        )

        runner = ViTRunner(config_path=cfg_path)
        results = runner.run()
        assert "train_loss" in results
        assert "test_accuracy" in results
        assert np.isfinite(results["train_loss"])
        assert 0.0 <= results["test_accuracy"] <= 1.0

