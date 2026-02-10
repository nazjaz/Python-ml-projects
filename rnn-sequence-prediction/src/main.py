"""LSTM-based RNN from scratch for sequence prediction.

This module implements a single-layer LSTM network and a small regression
head for next-step sequence prediction using only NumPy. It includes:

- LSTM layer with input, forget, output, and candidate gates
- Backpropagation through time with gradient updates
- Synthetic sine-wave sequence generator
- Training runner driven by a YAML configuration file
"""

import argparse
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable sigmoid."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class LSTM:
    """Single-layer LSTM network for many-to-one sequence prediction.

    The implementation follows the standard LSTM equations with input,
    forget, output, and candidate gates. It maintains internal caches
    of per-timestep activations to support backpropagation through time.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize LSTM parameters.

        Args:
            input_dim: Number of input features per time step.
            hidden_dim: Number of hidden units in the LSTM cell.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weight matrices for input and recurrent connections
        limit = np.sqrt(1.0 / max(1, input_dim))
        self.w_x = np.random.uniform(
            -limit, limit, size=(input_dim, 4 * hidden_dim)
        )
        self.w_h = np.random.uniform(
            -limit, limit, size=(hidden_dim, 4 * hidden_dim)
        )
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)

        # Caches used during backpropagation through time
        self._x_seq_cache: Optional[np.ndarray] = None
        self._h_seq_cache: Optional[np.ndarray] = None
        self._c_seq_cache: Optional[np.ndarray] = None
        self._gates_cache: Optional[List[Tuple[np.ndarray, ...]]] = None

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run forward pass over a batch of sequences.

        Args:
            x: Input sequences of shape
                (batch_size, sequence_length, input_dim).

        Returns:
            Tuple of:
                - h_seq: Hidden states for all time steps,
                  shape (batch_size, sequence_length, hidden_dim).
                - h_last: Final hidden state,
                  shape (batch_size, hidden_dim).
                - c_last: Final cell state,
                  shape (batch_size, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        h_seq = np.zeros(
            (batch_size, seq_len, self.hidden_dim), dtype=np.float32
        )

        gates_cache: List[Tuple[np.ndarray, ...]] = []
        h_list = []
        c_list = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            z = x_t @ self.w_x + h @ self.w_h + self.b

            i = _sigmoid(z[:, 0 : self.hidden_dim])
            f = _sigmoid(z[:, self.hidden_dim : 2 * self.hidden_dim])
            o = _sigmoid(z[:, 2 * self.hidden_dim : 3 * self.hidden_dim])
            g = np.tanh(z[:, 3 * self.hidden_dim : 4 * self.hidden_dim])

            c = f * c + i * g
            h = o * np.tanh(c)

            h_seq[:, t, :] = h
            h_list.append(h.copy())
            c_list.append(c.copy())
            gates_cache.append((i, f, o, g))

        self._x_seq_cache = x
        self._h_seq_cache = np.stack(h_list, axis=1)
        self._c_seq_cache = np.stack(c_list, axis=1)
        self._gates_cache = gates_cache

        return h_seq, h, c

    def backward(self, dh_last: np.ndarray, learning_rate: float) -> None:
        """Backpropagate gradient from final hidden state through time.

        Args:
            dh_last: Gradient of loss with respect to final hidden state,
                shape (batch_size, hidden_dim).
            learning_rate: Step size for gradient descent.
        """
        if (
            self._x_seq_cache is None
            or self._h_seq_cache is None
            or self._c_seq_cache is None
            or self._gates_cache is None
        ):
            raise RuntimeError("Forward pass must be called before backward.")

        x = self._x_seq_cache
        h_seq = self._h_seq_cache
        c_seq = self._c_seq_cache
        gates = self._gates_cache

        batch_size, seq_len, _ = x.shape

        d_w_x = np.zeros_like(self.w_x)
        d_w_h = np.zeros_like(self.w_h)
        d_b = np.zeros_like(self.b)

        dh_next = dh_last.copy()
        dc_next = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        for t in reversed(range(seq_len)):
            x_t = x[:, t, :]
            h_t = h_seq[:, t, :]
            c_t = c_seq[:, t, :]
            c_prev = (
                c_seq[:, t - 1, :] if t > 0 else np.zeros_like(c_t)
            )
            h_prev = (
                h_seq[:, t - 1, :] if t > 0 else np.zeros_like(h_t)
            )

            i, f, o, g = gates[t]

            tanh_c = np.tanh(c_t)
            do = dh_next * tanh_c
            dc = dh_next * o * (1.0 - tanh_c**2) + dc_next

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            di_input = di * i * (1.0 - i)
            df_input = df * f * (1.0 - f)
            do_input = do * o * (1.0 - o)
            dg_input = dg * (1.0 - g**2)

            dz = np.concatenate(
                [di_input, df_input, do_input, dg_input],
                axis=1,
            )

            d_w_x += x_t.T @ dz
            d_w_h += h_prev.T @ dz
            d_b += np.sum(dz, axis=0)

            dh_next = dz @ self.w_h.T
            dc_next = dc_prev

        scale = 1.0 / float(batch_size * max(1, seq_len))
        self.w_x -= learning_rate * d_w_x * scale
        self.w_h -= learning_rate * d_w_h * scale
        self.b -= learning_rate * d_b * scale


class Dense:
    """Fully connected output layer with linear activation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize dense layer parameters.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        limit = np.sqrt(1.0 / max(1, input_dim))
        self.weights = np.random.uniform(
            -limit, limit, size=(input_dim, output_dim)
        )
        self.biases = np.zeros(output_dim, dtype=np.float32)

        self._input_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass."""
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim {self.input_dim}, got {x.shape[1]}"
            )
        self._input_cache = x
        return x @ self.weights + self.biases

    def backward(
        self, dout: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        """Backpropagate gradient and update parameters."""
        if self._input_cache is None:
            raise RuntimeError(
                "Forward pass must be called before backward."
            )
        x = self._input_cache
        batch_size = x.shape[0]

        dw = x.T @ dout / float(batch_size)
        db = np.mean(dout, axis=0)
        dx = dout @ self.weights.T

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db

        return dx


class LSTMSequenceRegressor:
    """LSTM-based sequence regressor using MSE loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize LSTM regressor."""
        self.lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim)
        self.dense = Dense(input_dim=hidden_dim, output_dim=output_dim)
        self._history: Dict[str, List[float]] = {"loss": []}

    @staticmethod
    def _mse_loss(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute mean squared error loss and gradient."""
        error = y_pred - y_true
        loss = float(np.mean(error**2))
        grad = 2.0 * error / float(y_true.shape[0])
        return loss, grad

    def _forward_with_cache(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through LSTM and dense output."""
        _, h_last, _ = self.lstm.forward(x)
        y_pred = self.dense.forward(h_last)
        return y_pred, h_last

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        learning_rate: float = 0.005,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the LSTM regressor using mini-batch gradient descent."""
        n_samples = x_train.shape[0]
        history_losses: List[float] = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            batch_losses: List[float] = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred, _ = self._forward_with_cache(x_batch)
                loss, grad_y = self._mse_loss(y_batch, y_pred)

                dh_last = self.dense.backward(
                    grad_y, learning_rate=learning_rate
                )
                self.lstm.backward(dh_last, learning_rate=learning_rate)

                batch_losses.append(loss)

            epoch_loss = float(np.mean(batch_losses))
            history_losses.append(epoch_loss)
            self._history["loss"] = history_losses

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = (
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"loss: {epoch_loss:.6f}"
                )
                logger.info(msg)
                print(msg)

        return self._history

    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test sequences."""
        y_pred, _ = self._forward_with_cache(x_test)
        loss, _ = self._mse_loss(y_test, y_pred)
        return {"loss": loss}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for input sequences."""
        y_pred, _ = self._forward_with_cache(x)
        return y_pred


def generate_sine_wave_sequences(
    n_samples: int,
    sequence_length: int,
    noise_std: float = 0.05,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sine-wave sequences with Gaussian noise.

    Each sample consists of a contiguous subsequence of a sine wave
    and a target equal to the next value after the sequence.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    total_length = n_samples + sequence_length + 1
    t = np.linspace(0, 8 * np.pi, total_length)
    signal = np.sin(t)
    noise = np.random.normal(scale=noise_std, size=signal.shape)
    series = signal + noise

    x = np.zeros((n_samples, sequence_length, 1), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)

    for i in range(n_samples):
        x[i, :, 0] = series[i : i + sequence_length]
        y[i, 0] = series[i + sequence_length]

    return x, y


class RNNRunner:
    """Orchestrates LSTM training and evaluation from configuration."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize runner with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load YAML configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found: %s, using defaults", config_path)
            return {}

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"))
        log_file = log_cfg.get("file", "logs/app.log")
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.setLevel(level)
        logger.addHandler(handler)

    def run(self) -> Dict[str, float]:
        """Run sequence prediction experiment."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_train = data_cfg.get("n_train", 2000)
        n_test = data_cfg.get("n_test", 500)
        sequence_length = data_cfg.get("sequence_length", 20)
        random_seed = data_cfg.get("random_seed", 42)
        noise_std = data_cfg.get("noise_std", 0.05)

        x_all, y_all = generate_sine_wave_sequences(
            n_samples=n_train + n_test,
            sequence_length=sequence_length,
            noise_std=noise_std,
            random_seed=random_seed,
        )
        x_train, x_test = x_all[:n_train], x_all[n_train:]
        y_train, y_test = y_all[:n_train], y_all[n_train:]

        input_dim = model_cfg.get("input_dim", 1)
        hidden_dim = model_cfg.get("hidden_dim", 32)
        output_dim = model_cfg.get("output_dim", 1)

        model = LSTMSequenceRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        history = model.train(
            x_train,
            y_train,
            epochs=train_cfg.get("epochs", 20),
            learning_rate=train_cfg.get("learning_rate", 0.005),
            batch_size=train_cfg.get("batch_size", 32),
            verbose=True,
        )

        test_results = model.evaluate(x_test, y_test)
        train_loss = history["loss"][-1] if history["loss"] else float("nan")

        results = {
            "train_loss": float(train_loss),
            "test_loss": float(test_results["loss"]),
        }

        logger.info(
            "Final results - train_loss: %.6f, test_loss: %.6f",
            results["train_loss"],
            results["test_loss"],
        )
        return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train LSTM from scratch for sequence prediction"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = RNNRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run()

    print("\nFinal Results:")
    print("=" * 40)
    for key, val in results.items():
        print(f"  {key}: {val:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()

