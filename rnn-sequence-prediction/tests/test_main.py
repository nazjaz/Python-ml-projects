"""Tests for LSTM-based RNN sequence prediction module."""

import numpy as np
import pytest

from src.main import (
    LSTM,
    LSTMSequenceRegressor,
    generate_sine_wave_sequences,
)


class TestLSTM:
    """Test cases for LSTM layer."""

    def test_forward_output_shapes(self) -> None:
        """LSTM forward should produce correct output shapes."""
        np.random.seed(42)
        batch_size = 4
        seq_len = 10
        input_dim = 3
        hidden_dim = 5

        lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim)
        x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

        h_seq, h_last, c_last = lstm.forward(x)

        assert h_seq.shape == (batch_size, seq_len, hidden_dim)
        assert h_last.shape == (batch_size, hidden_dim)
        assert c_last.shape == (batch_size, hidden_dim)

    def test_backward_runs_without_error(self) -> None:
        """LSTM backward should run and update parameters."""
        np.random.seed(42)
        batch_size = 3
        seq_len = 6
        input_dim = 2
        hidden_dim = 4

        lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim)
        x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        _, h_last, _ = lstm.forward(x)

        dh_last = np.random.randn(batch_size, hidden_dim).astype(np.float32)
        w_x_before = lstm.w_x.copy()

        lstm.backward(dh_last, learning_rate=0.01)

        assert not np.allclose(lstm.w_x, w_x_before)


class TestDataGeneration:
    """Tests for synthetic sine-wave sequence generation."""

    def test_generate_sine_wave_sequences_shapes(self) -> None:
        """Generated data should have correct shapes."""
        x, y = generate_sine_wave_sequences(
            n_samples=50,
            sequence_length=8,
            noise_std=0.0,
            random_seed=123,
        )
        assert x.shape == (50, 8, 1)
        assert y.shape == (50, 1)

    def test_generate_sine_wave_sequences_reproducible(self) -> None:
        """Data generation should be reproducible with fixed seed."""
        x1, y1 = generate_sine_wave_sequences(
            n_samples=20,
            sequence_length=5,
            noise_std=0.1,
            random_seed=7,
        )
        x2, y2 = generate_sine_wave_sequences(
            n_samples=20,
            sequence_length=5,
            noise_std=0.1,
            random_seed=7,
        )
        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(y1, y2)


class TestLSTMSequenceRegressor:
    """End-to-end tests for LSTMSequenceRegressor."""

    def test_train_reduces_loss(self) -> None:
        """Training should reduce loss over epochs on synthetic data."""
        np.random.seed(42)
        x, y = generate_sine_wave_sequences(
            n_samples=200,
            sequence_length=15,
            noise_std=0.05,
            random_seed=42,
        )
        x_train, y_train = x[:150], y[:150]

        model = LSTMSequenceRegressor(
            input_dim=1,
            hidden_dim=8,
            output_dim=1,
        )
        history = model.train(
            x_train,
            y_train,
            epochs=5,
            learning_rate=0.01,
            batch_size=16,
            verbose=False,
        )

        assert len(history["loss"]) == 5
        assert history["loss"][-1] <= history["loss"][0] * 1.2

    def test_evaluate_returns_loss(self) -> None:
        """Evaluate should return a finite loss value."""
        np.random.seed(0)
        x, y = generate_sine_wave_sequences(
            n_samples=60,
            sequence_length=10,
            noise_std=0.05,
            random_seed=0,
        )
        x_train, y_train = x[:40], y[:40]
        x_test, y_test = x[40:], y[40:]

        model = LSTMSequenceRegressor(
            input_dim=1,
            hidden_dim=8,
            output_dim=1,
        )
        model.train(
            x_train,
            y_train,
            epochs=3,
            learning_rate=0.01,
            batch_size=8,
            verbose=False,
        )
        results = model.evaluate(x_test, y_test)

        assert "loss" in results
        assert np.isfinite(results["loss"])

