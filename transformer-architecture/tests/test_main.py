"""Tests for Transformer encoder implementation."""

import numpy as np
import pytest

from src.main import (
    MultiHeadSelfAttention,
    PositionalEncoding,
    TransformerEncoderClassifier,
    TransformerEncoderLayer,
    generate_synthetic_classification_data,
)


class TestPositionalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_positional_encoding_shape(self) -> None:
        """PositionalEncoding should preserve input shape."""
        dim_model = 16
        seq_len = 10
        batch_size = 4
        pe = PositionalEncoding(dim_model=dim_model, max_len=seq_len)
        x = np.zeros((batch_size, seq_len, dim_model), dtype=np.float32)
        out = pe.forward(x)
        assert out.shape == x.shape


class TestMultiHeadSelfAttention:
    """Tests for multi-head self-attention."""

    def test_attention_output_shape(self) -> None:
        """Attention output should match input shape."""
        np.random.seed(0)
        batch_size = 2
        seq_len = 5
        dim_model = 8
        num_heads = 2
        attn = MultiHeadSelfAttention(dim_model=dim_model, num_heads=num_heads)
        x = np.random.randn(batch_size, seq_len, dim_model).astype(np.float32)

        out = attn.forward(x)
        assert out.shape == (batch_size, seq_len, dim_model)

    def test_attention_backward_updates_weights(self) -> None:
        """Backward pass should update attention weights."""
        np.random.seed(1)
        batch_size = 2
        seq_len = 4
        dim_model = 8
        num_heads = 2
        attn = MultiHeadSelfAttention(dim_model=dim_model, num_heads=num_heads)
        x = np.random.randn(batch_size, seq_len, dim_model).astype(np.float32)

        _ = attn.forward(x)
        grad_out = np.random.randn(batch_size, seq_len, dim_model).astype(
            np.float32
        )
        w_q_before = attn.w_q.copy()

        _ = attn.backward(grad_out, lr=0.01)
        assert not np.allclose(attn.w_q, w_q_before)


class TestTransformerEncoderLayer:
    """Tests for a single Transformer encoder layer."""

    def test_encoder_layer_shape(self) -> None:
        """Encoder layer should preserve sequence and feature dimensions."""
        np.random.seed(2)
        batch_size = 3
        seq_len = 6
        dim_model = 12
        num_heads = 3
        dim_ff = 24
        layer = TransformerEncoderLayer(
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
        )
        x = np.random.randn(batch_size, seq_len, dim_model).astype(np.float32)

        out = layer.forward(x)
        assert out.shape == (batch_size, seq_len, dim_model)


class TestSyntheticData:
    """Tests for synthetic classification data generator."""

    def test_generate_synthetic_classification_data_shapes(self) -> None:
        """Generated token sequences and labels should have correct shapes."""
        tokens, labels = generate_synthetic_classification_data(
            n_samples=50,
            seq_len=10,
            vocab_size=16,
            random_seed=42,
        )
        assert tokens.shape == (50, 10)
        assert labels.shape == (50,)

    def test_generate_synthetic_classification_data_reproducible(
        self,
    ) -> None:
        """Data generation should be reproducible with fixed seed."""
        tokens1, labels1 = generate_synthetic_classification_data(
            n_samples=20,
            seq_len=8,
            vocab_size=10,
            random_seed=7,
        )
        tokens2, labels2 = generate_synthetic_classification_data(
            n_samples=20,
            seq_len=8,
            vocab_size=10,
            random_seed=7,
        )
        np.testing.assert_allclose(tokens1, tokens2)
        np.testing.assert_allclose(labels1, labels2)


class TestTransformerEncoderClassifier:
    """End-to-end tests for TransformerEncoderClassifier."""

    def test_train_reduces_loss_like_behavior(self) -> None:
        """Training should improve classification accuracy on synthetic data."""
        np.random.seed(3)
        tokens, labels = generate_synthetic_classification_data(
            n_samples=200,
            seq_len=10,
            vocab_size=20,
            random_seed=3,
        )
        x_train, y_train = tokens[:160], labels[:160]
        x_val, y_val = tokens[160:], labels[160:]

        model = TransformerEncoderClassifier(
            vocab_size=20,
            dim_model=16,
            num_heads=4,
            dim_ff=32,
            num_layers=1,
            num_classes=2,
            max_seq_len=10,
        )

        # Short training loop inside the test to check learning signal.
        epochs = 5
        batch_size = 16
        lr = 0.02

        for _ in range(epochs):
            indices = np.random.permutation(x_train.shape[0])
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            for start in range(0, x_train.shape[0], batch_size):
                end = min(start + batch_size, x_train.shape[0])
                batch_tokens = x_shuffled[start:end]
                batch_labels = y_shuffled[start:end]
                logits = model.forward(batch_tokens)
                _, grad_logits = model._cross_entropy_loss(
                    logits, batch_labels
                )
                model.backward(grad_logits, lr=lr)

        logits_val = model.forward(x_val)
        preds = np.argmax(logits_val, axis=1)
        accuracy = np.mean(preds == y_val)
        assert accuracy > 0.5

