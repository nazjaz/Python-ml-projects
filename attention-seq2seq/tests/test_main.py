"""Tests for scaled dot-product attention and seq2seq attention model."""

import numpy as np

from src.main import (
    AttentionRunner,
    Seq2SeqAttentionModel,
    generate_copy_task_data,
    scaled_dot_product_attention,
)


class TestScaledDotProductAttention:
    """Unit tests for scaled dot-product attention."""

    def test_attention_output_shapes(self) -> None:
        """Output and weights should have correct shapes."""
        np.random.seed(0)
        batch_size, tgt_len, src_len, d_k, d_v = 2, 3, 4, 5, 6
        q = np.random.randn(batch_size, tgt_len, d_k).astype(np.float32)
        k = np.random.randn(batch_size, src_len, d_k).astype(np.float32)
        v = np.random.randn(batch_size, src_len, d_v).astype(np.float32)

        out, weights = scaled_dot_product_attention(q, k, v)

        assert out.shape == (batch_size, tgt_len, d_v)
        assert weights.shape == (batch_size, tgt_len, src_len)

    def test_attention_weights_sum_to_one(self) -> None:
        """Attention weights should sum to 1 along source dimension."""
        np.random.seed(1)
        q = np.random.randn(2, 3, 4).astype(np.float32)
        k = np.random.randn(2, 5, 4).astype(np.float32)
        v = np.random.randn(2, 5, 6).astype(np.float32)

        _, weights = scaled_dot_product_attention(q, k, v)

        sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5)

    def test_attention_respects_mask(self) -> None:
        """Masked positions should receive zero weight."""
        np.random.seed(2)
        batch_size, tgt_len, src_len, d_k, d_v = 1, 2, 4, 3, 3
        q = np.random.randn(batch_size, tgt_len, d_k).astype(np.float32)
        k = np.random.randn(batch_size, src_len, d_k).astype(np.float32)
        v = np.random.randn(batch_size, src_len, d_v).astype(np.float32)

        mask = np.ones((batch_size, tgt_len, src_len), dtype=np.float32)
        mask[:, :, -1] = 0.0

        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        assert np.allclose(weights[:, :, -1], 0.0, atol=1e-6)


class TestDataGeneration:
    """Tests for synthetic copy task data generation."""

    def test_generate_copy_task_shapes(self) -> None:
        """Generated source and target should have correct shapes."""
        x_src, x_tgt = generate_copy_task_data(
            n_samples=10,
            src_length=5,
            tgt_length=5,
            vocab_size=7,
            random_seed=42,
        )
        assert x_src.shape == (10, 5)
        assert x_tgt.shape == (10, 5)

    def test_generate_copy_task_reproducible(self) -> None:
        """Data generation should be reproducible with fixed seed."""
        x_src1, x_tgt1 = generate_copy_task_data(
            n_samples=8,
            src_length=4,
            tgt_length=4,
            vocab_size=6,
            random_seed=7,
        )
        x_src2, x_tgt2 = generate_copy_task_data(
            n_samples=8,
            src_length=4,
            tgt_length=4,
            vocab_size=6,
            random_seed=7,
        )
        np.testing.assert_array_equal(x_src1, x_src2)
        np.testing.assert_array_equal(x_tgt1, x_tgt2)


class TestSeq2SeqAttentionModel:
    """End-to-end tests for attention-based seq2seq model."""

    def test_train_reduces_loss(self) -> None:
        """Training should reduce loss on the copy task."""
        np.random.seed(0)
        x_src, x_tgt = generate_copy_task_data(
            n_samples=120,
            src_length=5,
            tgt_length=5,
            vocab_size=10,
            random_seed=0,
        )
        x_src_train = x_src[:100]
        x_tgt_train = x_tgt[:100]

        model = Seq2SeqAttentionModel(
            vocab_size=10,
            src_length=5,
            tgt_length=5,
            d_model=8,
            d_k=8,
            d_v=8,
        )
        history = model.train(
            x_src_train,
            x_tgt_train,
            epochs=5,
            learning_rate=0.05,
            batch_size=16,
            verbose=False,
        )
        assert len(history["loss"]) == 5
        assert history["loss"][-1] <= history["loss"][0] * 1.2

    def test_evaluate_returns_metrics(self) -> None:
        """Evaluate should return finite loss and accuracy."""
        np.random.seed(1)
        x_src, x_tgt = generate_copy_task_data(
            n_samples=60,
            src_length=4,
            tgt_length=4,
            vocab_size=9,
            random_seed=1,
        )
        x_src_train = x_src[:40]
        x_tgt_train = x_tgt[:40]
        x_src_test = x_src[40:]
        x_tgt_test = x_tgt[40:]

        model = Seq2SeqAttentionModel(
            vocab_size=9,
            src_length=4,
            tgt_length=4,
            d_model=8,
            d_k=8,
            d_v=8,
        )
        model.train(
            x_src_train,
            x_tgt_train,
            epochs=3,
            learning_rate=0.05,
            batch_size=8,
            verbose=False,
        )
        results = model.evaluate(x_src_test, x_tgt_test)
        assert "loss" in results and "accuracy" in results
        assert np.isfinite(results["loss"])
        assert 0.0 <= results["accuracy"] <= 1.0


class TestAttentionRunner:
    """Smoke test for AttentionRunner."""

    def test_runner_executes(self) -> None:
        """AttentionRunner.run should produce a results dictionary."""
        runner = AttentionRunner(config_path=None)
        results = runner.run()
        assert "train_loss" in results
        assert "test_loss" in results
        assert "train_accuracy" in results
        assert "test_accuracy" in results

