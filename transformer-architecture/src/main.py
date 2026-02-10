"""Transformer encoder from scratch with multi-head attention.

This module implements a minimal but fully functional Transformer encoder
stack using only NumPy. It includes:

- Token embedding and sinusoidal positional encoding
- Scaled dot-product attention and multi-head self-attention
- Position-wise feed-forward network
- Stacked encoder layers with residual connections and layer normalization
- Simple training loop for sequence classification on synthetic data
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


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along a given axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


class LayerNorm:
    """Layer normalization module."""

    def __init__(self, features: int, eps: float = 1e-5) -> None:
        """Initialize layer normalization parameters.

        Args:
            features: Number of feature dimensions.
            eps: Small constant for numerical stability.
        """
        self.features = features
        self.eps = eps
        self.gamma = np.ones((features,), dtype=np.float32)
        self.beta = np.zeros((features,), dtype=np.float32)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        self._cache = (x_norm, std)
        return self.gamma * x_norm + self.beta

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Backpropagate gradients through layer norm."""
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x_norm, std = self._cache
        n = x_norm.shape[-1]

        dgamma = np.sum(grad_out * x_norm, axis=tuple(range(grad_out.ndim - 1)))
        dbeta = np.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))

        self.gamma -= lr * dgamma.astype(np.float32)
        self.beta -= lr * dbeta.astype(np.float32)

        dx_norm = grad_out * self.gamma
        dx = (dx_norm - np.mean(dx_norm, axis=-1, keepdims=True) -
              x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True)) / std
        return dx.astype(np.float32)


class PositionalEncoding:
    """Sinusoidal positional encoding as in the original Transformer."""

    def __init__(self, dim_model: int, max_len: int = 5000) -> None:
        """Precompute positional encodings.

        Args:
            dim_model: Embedding dimension.
            max_len: Maximum supported sequence length.
        """
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, dim_model, 2) * (-np.log(10000.0) / dim_model)
        )
        pe = np.zeros((max_len, dim_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Add positional encodings to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim_model).
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len][None, :, :]


class MultiHeadSelfAttention:
    """Multi-head self-attention layer."""

    def __init__(self, dim_model: int, num_heads: int) -> None:
        """Initialize attention parameters.

        Args:
            dim_model: Embedding dimension.
            num_heads: Number of attention heads.
        """
        if dim_model % num_heads != 0:
            raise ValueError("dim_model must be divisible by num_heads")

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads

        limit = np.sqrt(1.0 / max(1, dim_model))
        self.w_q = np.random.uniform(
            -limit, limit, size=(dim_model, dim_model)
        ).astype(np.float32)
        self.w_k = np.random.uniform(
            -limit, limit, size=(dim_model, dim_model)
        ).astype(np.float32)
        self.w_v = np.random.uniform(
            -limit, limit, size=(dim_model, dim_model)
        ).astype(np.float32)
        self.w_o = np.random.uniform(
            -limit, limit, size=(dim_model, dim_model)
        ).astype(np.float32)

        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
            None
        )

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute multi-head self-attention.

        Args:
            x: Input embeddings, shape (batch_size, seq_len, dim_model).
        """
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        q_h = self._split_heads(q)
        k_h = self._split_heads(k)
        v_h = self._split_heads(v)

        dk = float(self.head_dim)
        scores = (q_h @ np.transpose(k_h, (0, 1, 3, 2))) / np.sqrt(dk)
        weights = _softmax(scores, axis=-1)
        attn = weights @ v_h

        context = self._combine_heads(attn)
        out = context @ self.w_o

        self._cache = (x, weights, v_h, context)
        return out

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Backpropagate gradients and update attention parameters."""
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x, weights, v_h, context = self._cache

        batch_size, seq_len, _ = x.shape

        d_w_o = (context.reshape(-1, self.dim_model).T @ grad_out.reshape(
            -1, self.dim_model
        ))
        d_context = grad_out @ self.w_o.T

        d_context_h = self._split_heads(d_context)
        d_weights = d_context_h @ np.transpose(v_h, (0, 1, 3, 2))
        d_v_h = np.transpose(weights, (0, 1, 3, 2)) @ d_context_h

        d_scores = d_weights * weights - weights * np.sum(
            d_weights * weights, axis=-1, keepdims=True
        )

        dk = float(self.head_dim)
        d_scores_scaled = d_scores / np.sqrt(dk)

        q_h = self._split_heads(x @ self.w_q)
        k_h = self._split_heads(x @ self.w_k)

        d_q_h = d_scores_scaled @ k_h
        d_k_h = np.transpose(d_scores_scaled, (0, 1, 3, 2)) @ q_h

        d_q = self._combine_heads(d_q_h)
        d_k = self._combine_heads(d_k_h)
        d_v = self._combine_heads(d_v_h)

        d_w_q = (x.reshape(-1, self.dim_model).T @ d_q.reshape(
            -1, self.dim_model
        ))
        d_w_k = (x.reshape(-1, self.dim_model).T @ d_k.reshape(
            -1, self.dim_model
        ))
        d_w_v = (x.reshape(-1, self.dim_model).T @ d_v.reshape(
            -1, self.dim_model
        ))

        dx_q = d_q @ self.w_q.T
        dx_k = d_k @ self.w_k.T
        dx_v = d_v @ self.w_v.T
        dx = dx_q + dx_k + dx_v

        scale = 1.0 / float(batch_size * max(1, seq_len))
        self.w_o -= lr * d_w_o.astype(np.float32) * scale
        self.w_q -= lr * d_w_q.astype(np.float32) * scale
        self.w_k -= lr * d_w_k.astype(np.float32) * scale
        self.w_v -= lr * d_w_v.astype(np.float32) * scale

        return dx.astype(np.float32)


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, dim_model: int, dim_hidden: int) -> None:
        """Initialize feed-forward parameters."""
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden

        limit1 = np.sqrt(1.0 / max(1, dim_model))
        limit2 = np.sqrt(1.0 / max(1, dim_hidden))
        self.w1 = np.random.uniform(
            -limit1, limit1, size=(dim_model, dim_hidden)
        ).astype(np.float32)
        self.b1 = np.zeros((dim_hidden,), dtype=np.float32)
        self.w2 = np.random.uniform(
            -limit2, limit2, size=(dim_hidden, dim_model)
        ).astype(np.float32)
        self.b2 = np.zeros((dim_model,), dtype=np.float32)
        self._cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through feed-forward network."""
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        self._cache = (x, a1)
        return z2

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Backpropagate gradients and update parameters."""
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x, a1 = self._cache

        da1 = grad_out @ self.w2.T
        dz1 = da1 * (a1 > 0.0).astype(np.float32)

        batch_size, seq_len, _ = x.shape

        d_w2 = (a1.reshape(-1, self.dim_hidden).T @ grad_out.reshape(
            -1, self.dim_model
        ))
        d_b2 = np.sum(grad_out, axis=(0, 1))
        d_w1 = (x.reshape(-1, self.dim_model).T @ dz1.reshape(
            -1, self.dim_hidden
        ))
        d_b1 = np.sum(dz1, axis=(0, 1))

        dx = dz1 @ self.w1.T

        scale = 1.0 / float(batch_size * max(1, seq_len))
        self.w1 -= lr * d_w1.astype(np.float32) * scale
        self.b1 -= lr * d_b1.astype(np.float32) * scale
        self.w2 -= lr * d_w2.astype(np.float32) * scale
        self.b2 -= lr * d_b2.astype(np.float32) * scale

        return dx.astype(np.float32)


class TransformerEncoderLayer:
    """Single Transformer encoder layer."""

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
    ) -> None:
        """Initialize encoder layer."""
        self.attn = MultiHeadSelfAttention(dim_model=dim_model, num_heads=num_heads)
        self.norm1 = LayerNorm(features=dim_model)
        self.ff = FeedForward(dim_model=dim_model, dim_hidden=dim_ff)
        self.norm2 = LayerNorm(features=dim_model)

    def forward(self, x: np.ndarray, lr: Optional[float] = None) -> np.ndarray:
        """Forward pass through encoder layer.

        If lr is provided, backpropagation-ready state is kept for training.
        """
        attn_out = self.attn.forward(x)
        x = self.norm1.forward(x + attn_out)
        ff_out = self.ff.forward(x)
        x = self.norm2.forward(x + ff_out)
        return x

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Backpropagate gradients through encoder layer."""
        grad_norm2 = self.norm2.backward(grad_out, lr=lr)
        grad_ff = self.ff.backward(grad_norm2, lr=lr)
        grad_res2 = grad_ff + grad_norm2

        grad_norm1 = self.norm1.backward(grad_res2, lr=lr)
        grad_attn = self.attn.backward(grad_norm1, lr=lr)
        grad_res1 = grad_attn + grad_norm1
        return grad_res1


class TransformerEncoderClassifier:
    """Classifier built on top of a Transformer encoder stack."""

    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        num_layers: int,
        num_classes: int,
        max_seq_len: int,
    ) -> None:
        """Initialize Transformer encoder classifier."""
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_classes = num_classes

        limit = np.sqrt(1.0 / max(1, vocab_size))
        self.token_embed = np.random.uniform(
            -limit, limit, size=(vocab_size, dim_model)
        ).astype(np.float32)
        self.pos_encoding = PositionalEncoding(dim_model=dim_model, max_len=max_seq_len)

        self.layers: List[TransformerEncoderLayer] = [
            TransformerEncoderLayer(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_ff=dim_ff,
            )
            for _ in range(num_layers)
        ]

        limit_cls = np.sqrt(1.0 / max(1, dim_model))
        self.w_cls = np.random.uniform(
            -limit_cls, limit_cls, size=(dim_model, num_classes)
        ).astype(np.float32)
        self.b_cls = np.zeros((num_classes,), dtype=np.float32)
        self._cache_last_hidden: Optional[np.ndarray] = None

    def _embed(self, tokens: np.ndarray) -> np.ndarray:
        """Embed token ids and add positional encodings."""
        x = self.token_embed[tokens]
        return self.pos_encoding.forward(x)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Forward pass from token ids to class logits."""
        x = self._embed(tokens)
        for layer in self.layers:
            x = layer.forward(x)
        cls_repr = x[:, 0, :]
        logits = cls_repr @ self.w_cls + self.b_cls
        self._cache_last_hidden = cls_repr
        return logits

    @staticmethod
    def _cross_entropy_loss(
        logits: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradient."""
        probs = _softmax(logits, axis=1)
        batch_size = logits.shape[0]
        one_hot = np.eye(probs.shape[1], dtype=np.float32)[labels]
        loss = -float(np.sum(one_hot * np.log(probs + 1e-12)) / batch_size)
        grad_logits = (probs - one_hot) / float(batch_size)
        return loss, grad_logits

    def backward(self, grad_logits: np.ndarray, lr: float) -> None:
        """Backpropagate gradients through classifier and encoder."""
        if self._cache_last_hidden is None:
            raise RuntimeError("Forward must be called before backward.")
        cls_repr = self._cache_last_hidden

        d_w_cls = cls_repr.T @ grad_logits
        d_b_cls = np.sum(grad_logits, axis=0)
        d_cls_repr = grad_logits @ self.w_cls.T

        batch_size = d_cls_repr.shape[0]
        grad_enc = np.zeros((batch_size, 1, self.dim_model), dtype=np.float32)
        grad_enc[:, 0, :] = d_cls_repr

        for layer in reversed(self.layers):
            grad_enc = layer.backward(grad_enc, lr=lr)

        scale = 1.0 / float(batch_size)
        self.w_cls -= lr * d_w_cls.astype(np.float32) * scale
        self.b_cls -= lr * d_b_cls.astype(np.float32) * scale


def generate_synthetic_classification_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic token sequences for classification.

    The label is the parity (even or odd) of the sum of token ids.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    tokens = np.random.randint(
        low=0, high=vocab_size, size=(n_samples, seq_len), dtype=np.int32
    )
    sums = np.sum(tokens, axis=1)
    labels = (sums % 2).astype(np.int64)
    return tokens, labels


class TransformerRunner:
    """Orchestrate Transformer training and evaluation from configuration."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize runner with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load YAML configuration from file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found: %s, using defaults", config_path)
            return {}

    def _setup_logging(self) -> None:
        """Configure rotating file logging."""
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
        """Run Transformer training and evaluation."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_train = data_cfg.get("n_train", 512)
        n_test = data_cfg.get("n_test", 128)
        seq_len = data_cfg.get("sequence_length", 12)
        vocab_size = data_cfg.get("vocab_size", 32)
        random_seed = data_cfg.get("random_seed", 7)

        tokens_all, labels_all = generate_synthetic_classification_data(
            n_samples=n_train + n_test,
            seq_len=seq_len,
            vocab_size=vocab_size,
            random_seed=random_seed,
        )
        x_train, x_test = tokens_all[:n_train], tokens_all[n_train:]
        y_train, y_test = labels_all[:n_train], labels_all[n_train:]

        dim_model = model_cfg.get("dim_model", 32)
        num_heads = model_cfg.get("num_heads", 4)
        dim_ff = model_cfg.get("dim_ff", 64)
        num_layers = model_cfg.get("num_layers", 2)
        num_classes = model_cfg.get("num_classes", 2)

        model = TransformerEncoderClassifier(
            vocab_size=vocab_size,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=num_layers,
            num_classes=num_classes,
            max_seq_len=seq_len,
        )

        epochs = train_cfg.get("epochs", 10)
        batch_size = train_cfg.get("batch_size", 32)
        learning_rate = train_cfg.get("learning_rate", 0.01)

        history_losses: List[float] = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_train)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses: List[float] = []

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                batch_tokens = x_shuffled[start:end]
                batch_labels = y_shuffled[start:end]

                logits = model.forward(batch_tokens)
                loss, grad_logits = model._cross_entropy_loss(
                    logits, batch_labels
                )
                model.backward(grad_logits, lr=learning_rate)
                epoch_losses.append(loss)

            epoch_loss = float(np.mean(epoch_losses))
            history_losses.append(epoch_loss)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(
                    "Epoch %d/%d - loss: %.6f",
                    epoch + 1,
                    epochs,
                    epoch_loss,
                )

        logits_test = model.forward(x_test)
        preds_test = np.argmax(logits_test, axis=1)
        accuracy = float(np.mean(preds_test == y_test))

        results = {"train_loss": float(history_losses[-1]), "test_accuracy": accuracy}
        logger.info(
            "Final results - train_loss: %.6f, test_accuracy: %.4f",
            results["train_loss"],
            results["test_accuracy"],
        )
        return results


def main() -> None:
    """Command-line entry point for Transformer training."""
    parser = argparse.ArgumentParser(
        description="Train Transformer encoder from scratch on synthetic data"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = TransformerRunner(
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

