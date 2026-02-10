"""Scaled dot-product attention and seq2seq model from scratch.

This module implements:
- Scaled dot-product attention with optional masking
- A minimal attention-based encoder-decoder seq2seq model
- Synthetic copy-task data generation
- A configuration-driven training and evaluation runner
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


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch_size, target_len, d_k).
        key: Key tensor of shape (batch_size, source_len, d_k).
        value: Value tensor of shape (batch_size, source_len, d_v).
        mask: Optional mask of shape (batch_size, target_len, source_len)
            where masked positions are indicated by 0 and unmasked by 1.

    Returns:
        Tuple of:
            - output: Attention output of shape (batch_size, target_len, d_v)
            - weights: Attention weights of shape
              (batch_size, target_len, source_len)
    """
    batch_size, tgt_len, d_k = query.shape
    _, src_len, _ = key.shape

    scores = np.matmul(
        query, np.transpose(key, (0, 2, 1))
    ) / np.sqrt(float(d_k))

    if mask is not None:
        scores = np.where(mask > 0, scores, -1e9)

    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores_shifted)
    weights /= np.sum(weights, axis=-1, keepdims=True)

    output = np.matmul(weights, value)
    return output, weights


class Embedding:
    """Simple token embedding layer."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initialize embedding matrix."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        limit = np.sqrt(1.0 / max(1, vocab_size))
        self.weights = np.random.uniform(
            -limit, limit, size=(vocab_size, d_model)
        ).astype(np.float32)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Lookup embeddings for integer token array."""
        return self.weights[tokens]


class SimpleRNN:
    """Minimal tanh RNN used for encoder and decoder."""

    def __init__(self, d_model: int) -> None:
        """Initialize RNN parameters."""
        self.d_model = d_model
        limit = np.sqrt(1.0 / max(1, d_model))
        self.w_in = np.random.uniform(
            -limit, limit, size=(d_model, d_model)
        ).astype(np.float32)
        self.w_h = np.random.uniform(
            -limit, limit, size=(d_model, d_model)
        ).astype(np.float32)
        self.b = np.zeros(d_model, dtype=np.float32)

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run RNN over a sequence of embeddings."""
        batch_size, seq_len, _ = x.shape
        h = np.zeros((batch_size, self.d_model), dtype=np.float32)
        h_seq = np.zeros(
            (batch_size, seq_len, self.d_model), dtype=np.float32
        )
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = np.tanh(x_t @ self.w_in + h @ self.w_h + self.b)
            h_seq[:, t, :] = h
        return h_seq, h


class Seq2SeqAttentionModel:
    """Attention-based encoder-decoder model for copy task."""

    def __init__(
        self,
        vocab_size: int,
        src_length: int,
        tgt_length: int,
        d_model: int,
        d_k: int,
        d_v: int,
    ) -> None:
        """Initialize seq2seq model components."""
        self.vocab_size = vocab_size
        self.src_length = src_length
        self.tgt_length = tgt_length

        self.embedding = Embedding(vocab_size, d_model)
        self.encoder = SimpleRNN(d_model)
        self.decoder = SimpleRNN(d_model)

        limit = np.sqrt(1.0 / max(1, d_model))
        self.w_q = np.random.uniform(
            -limit, limit, size=(d_model, d_k)
        ).astype(np.float32)
        self.w_k = np.random.uniform(
            -limit, limit, size=(d_model, d_k)
        ).astype(np.float32)
        self.w_v = np.random.uniform(
            -limit, limit, size=(d_model, d_v)
        ).astype(np.float32)

        self.w_out = np.random.uniform(
            -limit, limit, size=(d_model + d_v, vocab_size)
        ).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def _cross_entropy(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradient."""
        probs = self._softmax(logits)
        batch, seq_len, vocab = probs.shape
        flat_targets = targets.reshape(-1)
        flat_probs = probs.reshape(-1, vocab)

        indices = np.arange(flat_targets.shape[0])
        chosen = flat_probs[indices, flat_targets]
        epsilon = 1e-15
        loss = -np.mean(np.log(np.clip(chosen, epsilon, 1.0)))

        grad = flat_probs
        grad[indices, flat_targets] -= 1.0
        grad /= float(batch * seq_len)
        return float(loss), grad.reshape(batch, seq_len, vocab)

    def _encode(self, src_tokens: np.ndarray) -> np.ndarray:
        """Encode source tokens to hidden state sequence."""
        src_emb = self.embedding.forward(src_tokens)
        h_enc_seq, _ = self.encoder.forward(src_emb)
        return h_enc_seq

    def _decode_teacher_forcing(
        self,
        h_enc_seq: np.ndarray,
        tgt_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Decode with teacher forcing using attention over encoder states."""
        batch_size = tgt_tokens.shape[0]
        tgt_emb = self.embedding.forward(tgt_tokens)
        h_dec = np.zeros(
            (batch_size, self.encoder.d_model), dtype=np.float32
        )
        logits = np.zeros(
            (batch_size, self.tgt_length, self.vocab_size),
            dtype=np.float32,
        )

        h_enc_proj_k = h_enc_seq @ self.w_k
        h_enc_proj_v = h_enc_seq @ self.w_v

        cache: Dict[str, np.ndarray] = {
            "tgt_emb": tgt_emb,
            "h_enc_seq": h_enc_seq,
        }
        h_dec_seq = []
        attn_weights_seq = []
        contexts = []

        for t in range(self.tgt_length):
            x_t = tgt_emb[:, t, :]
            h_dec = np.tanh(
                x_t @ self.decoder.w_in + h_dec @ self.decoder.w_h
                + self.decoder.b
            )
            h_dec_seq.append(h_dec.copy())

            q_t = h_dec @ self.w_q
            q_t_expanded = q_t[:, np.newaxis, :]
            ctx_t, w_t = scaled_dot_product_attention(
                q_t_expanded,
                h_enc_proj_k,
                h_enc_proj_v,
                mask=None,
            )
            ctx_t = ctx_t[:, 0, :]
            attn_weights_seq.append(w_t[:, 0, :])
            contexts.append(ctx_t)

            joint = np.concatenate([h_dec, ctx_t], axis=1)
            logits[:, t, :] = joint @ self.w_out + self.b_out

        cache["h_dec_seq"] = np.stack(h_dec_seq, axis=1)
        cache["contexts"] = np.stack(contexts, axis=1)
        cache["attn_weights"] = np.stack(attn_weights_seq, axis=1)
        cache["h_enc_proj_k"] = h_enc_proj_k
        cache["h_enc_proj_v"] = h_enc_proj_v
        return logits, cache

    def _backward(
        self,
        grad_logits: np.ndarray,
        cache: Dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        """Backpropagate gradients and update parameters."""
        tgt_emb = cache["tgt_emb"]
        h_enc_seq = cache["h_enc_seq"]
        h_dec_seq = cache["h_dec_seq"]
        contexts = cache["contexts"]
        h_enc_proj_k = cache["h_enc_proj_k"]
        h_enc_proj_v = cache["h_enc_proj_v"]

        batch_size = tgt_emb.shape[0]

        d_w_out = np.zeros_like(self.w_out)
        d_b_out = np.zeros_like(self.b_out)
        d_w_q = np.zeros_like(self.w_q)
        d_w_k = np.zeros_like(self.w_k)
        d_w_v = np.zeros_like(self.w_v)
        d_w_in_dec = np.zeros_like(self.decoder.w_in)
        d_w_h_dec = np.zeros_like(self.decoder.w_h)
        d_b_dec = np.zeros_like(self.decoder.b)

        d_h_enc_seq = np.zeros_like(h_enc_seq)
        d_h_dec_next = np.zeros_like(h_dec_seq[:, 0, :])

        for t in reversed(range(self.tgt_length)):
            h_dec_t = h_dec_seq[:, t, :]
            ctx_t = contexts[:, t, :]

            dhdc = np.concatenate([h_dec_t, ctx_t], axis=1)
            d_w_out += dhdc.T @ grad_logits[:, t, :]
            d_b_out += np.sum(grad_logits[:, t, :], axis=0)

            d_dhdc = grad_logits[:, t, :] @ self.w_out.T
            d_h_dec_out = d_dhdc[:, : self.encoder.d_model]
            d_ctx_t = d_dhdc[:, self.encoder.d_model :]

            q_t = h_dec_t @ self.w_q
            q_t_expanded = q_t[:, np.newaxis, :]
            _, attn_w = scaled_dot_product_attention(
                q_t_expanded,
                h_enc_proj_k,
                h_enc_proj_v,
            )
            attn_w = attn_w[:, 0, :]

            d_attn_weights = np.matmul(
                d_ctx_t[:, np.newaxis, :], h_enc_proj_v.transpose(0, 2, 1)
            )[:, 0, :]
            d_h_enc_proj_v = (
                attn_w[:, :, np.newaxis] * d_ctx_t[:, np.newaxis, :]
            )

            d_scores = d_attn_weights
            d_scores -= np.sum(d_scores * attn_w, axis=1, keepdims=True)
            d_scores *= attn_w

            d_q = np.matmul(d_scores[:, np.newaxis, :], h_enc_proj_k)[
                :, 0, :
            ]
            d_k = np.matmul(
                d_scores[:, :, np.newaxis],
                q_t_expanded,
            )

            scale = 1.0 / np.sqrt(float(self.w_q.shape[1]))
            d_q *= scale
            d_k *= scale

            d_w_q += h_dec_t.T @ d_q
            d_h_dec_out += d_q @ self.w_q.T

            d_w_k += np.tensordot(
                h_enc_seq, d_k, axes=([0, 1], [0, 1])
            )
            d_w_v += np.tensordot(
                h_enc_seq, d_h_enc_proj_v, axes=([0, 1], [0, 1])
            )

            d_h_enc_seq += (
                d_h_enc_proj_v @ self.w_v.T + d_k @ self.w_k.T
            ).reshape(h_enc_seq.shape)

            d_h_dec_total = d_h_dec_out + d_h_dec_next
            dtanh = (1.0 - h_dec_t**2) * d_h_dec_total
            d_w_in_dec += tgt_emb[:, t, :].T @ dtanh
            if t > 0:
                d_w_h_dec += h_dec_seq[:, t - 1, :].T @ dtanh
            d_b_dec += np.sum(dtanh, axis=0)
            if t > 0:
                d_h_dec_next = dtanh @ self.decoder.w_h.T

        scale = 1.0 / float(max(1, batch_size * self.tgt_length))
        self.w_out -= learning_rate * d_w_out * scale
        self.b_out -= learning_rate * d_b_out * scale
        self.w_q -= learning_rate * d_w_q * scale
        self.w_k -= learning_rate * d_w_k * scale
        self.w_v -= learning_rate * d_w_v * scale
        self.decoder.w_in -= learning_rate * d_w_in_dec * scale
        self.decoder.w_h -= learning_rate * d_w_h_dec * scale
        self.decoder.b -= learning_rate * d_b_dec * scale

    def train(
        self,
        x_src: np.ndarray,
        x_tgt: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model using teacher forcing on the copy task."""
        n_samples = x_src.shape[0]
        history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_src_shuf = x_src[indices]
            x_tgt_shuf = x_tgt[indices]

            epoch_losses: List[float] = []
            epoch_correct = 0
            epoch_tokens = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                src_batch = x_src_shuf[start:end]
                tgt_batch = x_tgt_shuf[start:end]

                h_enc_seq = self._encode(src_batch)
                logits, cache = self._decode_teacher_forcing(
                    h_enc_seq, tgt_batch
                )
                loss, grad_logits = self._cross_entropy(logits, tgt_batch)
                self._backward(grad_logits, cache, learning_rate)

                epoch_losses.append(loss)
                preds = np.argmax(logits, axis=-1)
                epoch_correct += int(np.sum(preds == tgt_batch))
                epoch_tokens += int(np.prod(tgt_batch.shape))

            avg_loss = float(np.mean(epoch_losses))
            accuracy = float(epoch_correct / max(1, epoch_tokens))
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = (
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"loss: {avg_loss:.6f} - accuracy: {accuracy:.4f}"
                )
                logger.info(msg)
                print(msg)

        return history

    def evaluate(
        self, x_src: np.ndarray, x_tgt: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        h_enc_seq = self._encode(x_src)
        logits, _ = self._decode_teacher_forcing(h_enc_seq, x_tgt)
        loss, _ = self._cross_entropy(logits, x_tgt)
        preds = np.argmax(logits, axis=-1)
        correct = int(np.sum(preds == x_tgt))
        tokens = int(np.prod(x_tgt.shape))
        accuracy = float(correct / max(1, tokens))
        return {"loss": float(loss), "accuracy": accuracy}

    def predict(self, x_src: np.ndarray) -> np.ndarray:
        """Predict target sequences for given sources using greedy decoding."""
        h_enc_seq = self._encode(x_src)
        batch_size = x_src.shape[0]

        preds = np.zeros(
            (batch_size, self.tgt_length), dtype=np.int64
        )
        current_tokens = np.zeros_like(preds)

        for t in range(self.tgt_length):
            h_enc_proj_k = h_enc_seq @ self.w_k
            h_enc_proj_v = h_enc_seq @ self.w_v
            tgt_emb = self.embedding.forward(current_tokens)
            h_dec_seq, _ = self.decoder.forward(tgt_emb)
            h_dec_t = h_dec_seq[:, t, :]
            q_t = h_dec_t @ self.w_q
            q_t_expanded = q_t[:, np.newaxis, :]
            ctx_t, _ = scaled_dot_product_attention(
                q_t_expanded,
                h_enc_proj_k,
                h_enc_proj_v,
            )
            ctx_t = ctx_t[:, 0, :]
            joint = np.concatenate([h_dec_t, ctx_t], axis=1)
            logits_t = joint @ self.w_out + self.b_out
            preds[:, t] = np.argmax(self._softmax(logits_t), axis=-1)
            current_tokens[:, t] = preds[:, t]

        return preds


def generate_copy_task_data(
    n_samples: int,
    src_length: int,
    tgt_length: int,
    vocab_size: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate integer copy-task sequences."""
    np.random.seed(random_seed)
    src = np.random.randint(
        1, vocab_size, size=(n_samples, src_length), dtype=np.int64
    )
    tgt = src.copy()
    if tgt_length > src_length:
        pad = np.zeros(
            (n_samples, tgt_length - src_length), dtype=np.int64
        )
        tgt = np.concatenate([tgt, pad], axis=1)
    elif tgt_length < src_length:
        tgt = tgt[:, :tgt_length]
        src = src[:, :tgt_length]
    return src, tgt


class AttentionRunner:
    """Orchestrates attention-based seq2seq training and evaluation."""

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
        """Run training and evaluation."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_train = data_cfg.get("n_train", 2000)
        n_test = data_cfg.get("n_test", 500)
        src_length = data_cfg.get("src_length", 6)
        tgt_length = data_cfg.get("tgt_length", 6)
        vocab_size = data_cfg.get("vocab_size", 12)
        seed = data_cfg.get("random_seed", 42)

        x_src_all, x_tgt_all = generate_copy_task_data(
            n_samples=n_train + n_test,
            src_length=src_length,
            tgt_length=tgt_length,
            vocab_size=vocab_size,
            random_seed=seed,
        )
        x_src_train = x_src_all[:n_train]
        x_tgt_train = x_tgt_all[:n_train]
        x_src_test = x_src_all[n_train:]
        x_tgt_test = x_tgt_all[n_train:]

        d_model = model_cfg.get("d_model", 16)
        d_k = model_cfg.get("d_k", 16)
        d_v = model_cfg.get("d_v", 16)

        model = Seq2SeqAttentionModel(
            vocab_size=vocab_size,
            src_length=src_length,
            tgt_length=tgt_length,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
        )

        history = model.train(
            x_src_train,
            x_tgt_train,
            epochs=train_cfg.get("epochs", 15),
            learning_rate=train_cfg.get("learning_rate", 0.05),
            batch_size=train_cfg.get("batch_size", 32),
            verbose=True,
        )
        train_loss = history["loss"][-1]
        train_acc = history["accuracy"][-1]

        test_results = model.evaluate(x_src_test, x_tgt_test)

        results = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "test_loss": float(test_results["loss"]),
            "test_accuracy": float(test_results["accuracy"]),
        }
        logger.info(
            "Results - train_loss: %.6f, train_acc: %.4f, "
            "test_loss: %.6f, test_acc: %.4f",
            results["train_loss"],
            results["train_accuracy"],
            results["test_loss"],
            results["test_accuracy"],
        )
        return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Train attention-based seq2seq model with scaled dot-product "
            "attention"
        )
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = AttentionRunner(
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

