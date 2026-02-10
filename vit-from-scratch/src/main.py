"""Vision Transformer (ViT) from scratch for image classification.

This module implements a minimal Vision Transformer using only NumPy:

- Patch embedding of images into a sequence of tokens
- Learnable class token and positional embeddings
- Multi-head self-attention and feed-forward transformer encoder blocks
- Classification head over the [CLS] representation

The model is trained on synthetic image data for demonstration purposes.
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
    e = np.exp(np.clip(x - x_max, -500, 500))
    return e / np.sum(e, axis=axis, keepdims=True)


class LayerNorm:
    """Layer normalization over the last dimension."""

    def __init__(self, features: int, eps: float = 1e-5) -> None:
        """Initialize layer normalization parameters."""
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
        """Backpropagate gradients and update scale and shift."""
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x_norm, std = self._cache

        dgamma = np.sum(
            grad_out * x_norm, axis=tuple(range(grad_out.ndim - 1))
        )
        dbeta = np.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))

        self.gamma -= lr * dgamma.astype(np.float32)
        self.beta -= lr * dbeta.astype(np.float32)

        dx_norm = grad_out * self.gamma
        dx = (
            dx_norm
            - np.mean(dx_norm, axis=-1, keepdims=True)
            - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True)
        ) / std
        return dx.astype(np.float32)


class MultiHeadSelfAttention:
    """Multi-head scaled dot-product self-attention."""

    def __init__(self, dim_model: int, num_heads: int) -> None:
        if dim_model % num_heads != 0:
            raise ValueError("dim_model must be divisible by num_heads")
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads

        limit = np.sqrt(1.0 / max(1, dim_model))
        self.w_q = np.random.uniform(
            -limit, limit, (dim_model, dim_model)
        ).astype(np.float32)
        self.w_k = np.random.uniform(
            -limit, limit, (dim_model, dim_model)
        ).astype(np.float32)
        self.w_v = np.random.uniform(
            -limit, limit, (dim_model, dim_model)
        ).astype(np.float32)
        self.w_o = np.random.uniform(
            -limit, limit, (dim_model, dim_model)
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
        """Compute multi-head self-attention."""
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
        """Backpropagate gradients and update parameters."""
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        x, weights, v_h, context = self._cache
        batch_size, seq_len, _ = x.shape

        d_w_o = context.reshape(-1, self.dim_model).T @ grad_out.reshape(
            -1, self.dim_model
        )
        d_context = grad_out @ self.w_o.T

        d_context_h = self._split_heads(d_context)
        d_weights = d_context_h @ np.transpose(v_h, (0, 1, 3, 2))
        d_v_h = np.transpose(weights, (0, 1, 3, 2)) @ d_context_h

        d_scores = d_weights * weights - weights * np.sum(
            d_weights * weights, axis=-1, keepdims=True
        )
        d_scores_scaled = d_scores / np.sqrt(float(self.head_dim))

        q_h = self._split_heads(x @ self.w_q)
        k_h = self._split_heads(x @ self.w_k)

        d_q_h = d_scores_scaled @ k_h
        d_k_h = np.transpose(d_scores_scaled, (0, 1, 3, 2)) @ q_h

        d_q = self._combine_heads(d_q_h)
        d_k = self._combine_heads(d_k_h)
        d_v = self._combine_heads(d_v_h)

        d_w_q = x.reshape(-1, self.dim_model).T @ d_q.reshape(
            -1, self.dim_model
        )
        d_w_k = x.reshape(-1, self.dim_model).T @ d_k.reshape(
            -1, self.dim_model
        )
        d_w_v = x.reshape(-1, self.dim_model).T @ d_v.reshape(
            -1, self.dim_model
        )

        dx = d_q @ self.w_q.T + d_k @ self.w_k.T + d_v @ self.w_v.T

        scale = 1.0 / float(batch_size * max(1, seq_len))
        self.w_o -= lr * d_w_o.astype(np.float32) * scale
        self.w_q -= lr * d_w_q.astype(np.float32) * scale
        self.w_k -= lr * d_w_k.astype(np.float32) * scale
        self.w_v -= lr * d_w_v.astype(np.float32) * scale

        return dx.astype(np.float32)


class FeedForward:
    """Position-wise feed-forward network with ReLU."""

    def __init__(self, dim_model: int, dim_hidden: int) -> None:
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden

        limit1 = np.sqrt(1.0 / max(1, dim_model))
        limit2 = np.sqrt(1.0 / max(1, dim_hidden))
        self.w1 = np.random.uniform(
            -limit1, limit1, (dim_model, dim_hidden)
        ).astype(np.float32)
        self.b1 = np.zeros((dim_hidden,), dtype=np.float32)
        self.w2 = np.random.uniform(
            -limit2, limit2, (dim_hidden, dim_model)
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

        d_w2 = a1.reshape(-1, self.dim_hidden).T @ grad_out.reshape(
            -1, self.dim_model
        )
        d_b2 = np.sum(grad_out, axis=(0, 1))
        d_w1 = x.reshape(-1, self.dim_model).T @ dz1.reshape(
            -1, self.dim_hidden
        )
        d_b1 = np.sum(dz1, axis=(0, 1))

        dx = dz1 @ self.w1.T

        scale = 1.0 / float(batch_size * max(1, seq_len))
        self.w1 -= lr * d_w1.astype(np.float32) * scale
        self.b1 -= lr * d_b1.astype(np.float32) * scale
        self.w2 -= lr * d_w2.astype(np.float32) * scale
        self.b2 -= lr * d_b2.astype(np.float32) * scale

        return dx.astype(np.float32)


class ViTEncoderLayer:
    """Single Vision Transformer encoder layer."""

    def __init__(self, dim_model: int, num_heads: int, dim_ff: int) -> None:
        self.attn = MultiHeadSelfAttention(dim_model=dim_model, num_heads=num_heads)
        self.norm1 = LayerNorm(features=dim_model)
        self.ff = FeedForward(dim_model=dim_model, dim_hidden=dim_ff)
        self.norm2 = LayerNorm(features=dim_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through encoder layer."""
        x = x + self.attn.forward(self.norm1.forward(x))
        x = x + self.ff.forward(self.norm2.forward(x))
        return x

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Backpropagate gradients through encoder layer."""
        # Second residual block
        grad_ff = grad_out
        grad_norm2 = self.ff.backward(grad_ff, lr=lr)
        grad_x1 = grad_out + self.norm2.backward(grad_norm2, lr=lr)

        # First residual block
        grad_attn = self.attn.backward(grad_x1, lr=lr)
        grad_x0 = grad_x1 + self.norm1.backward(grad_attn, lr=lr)
        return grad_x0


class PatchEmbedding:
    """Image to patch embedding."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        dim_model: int,
    ) -> None:
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches
        patch_dim = patch_size * patch_size * in_channels
        self.patch_dim = patch_dim

        limit = np.sqrt(1.0 / max(1, patch_dim))
        self.w = np.random.uniform(
            -limit, limit, (patch_dim, dim_model)
        ).astype(np.float32)
        self.b = np.zeros((dim_model,), dtype=np.float32)

        self._cache: Optional[np.ndarray] = None

    def _images_to_patches(self, images: np.ndarray) -> np.ndarray:
        """Convert images to flattened patch vectors."""
        batch_size, h, w, c = images.shape
        ph = pw = self.patch_size
        images = images.reshape(
            batch_size,
            h // ph,
            ph,
            w // pw,
            pw,
            c,
        )
        patches = np.transpose(images, (0, 1, 3, 2, 4, 5))
        patches = patches.reshape(batch_size, -1, ph * pw * c)
        return patches

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Project image patches to embeddings."""
        patches = self._images_to_patches(images)
        embeddings = patches @ self.w + self.b
        self._cache = patches
        return embeddings

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Backpropagate gradients to patch projection."""
        if self._cache is None:
            raise RuntimeError("Forward must be called before backward.")
        patches = self._cache
        batch_size, num_patches, _ = patches.shape

        d_w = patches.reshape(-1, self.patch_dim).T @ grad_out.reshape(
            -1, grad_out.shape[-1]
        )
        d_b = np.sum(grad_out, axis=(0, 1))
        d_patches = grad_out @ self.w.T

        scale = 1.0 / float(batch_size * max(1, num_patches))
        self.w -= lr * d_w.astype(np.float32) * scale
        self.b -= lr * d_b.astype(np.float32) * scale

        return d_patches


class ViTClassifier:
    """Vision Transformer classifier."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim_model = dim_model
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim_model=dim_model,
        )
        num_patches = self.patch_embedding.num_patches
        self.num_patches = num_patches

        limit = np.sqrt(1.0 / max(1, dim_model))
        self.cls_token = np.random.uniform(
            -limit, limit, (1, 1, dim_model)
        ).astype(np.float32)
        self.pos_embed = np.random.uniform(
            -limit, limit, (1, num_patches + 1, dim_model)
        ).astype(np.float32)

        self.encoder_layers: List[ViTEncoderLayer] = [
            ViTEncoderLayer(dim_model=dim_model, num_heads=num_heads, dim_ff=dim_ff)
            for _ in range(num_layers)
        ]

        self.w_cls = np.random.uniform(
            -limit, limit, (dim_model, num_classes)
        ).astype(np.float32)
        self.b_cls = np.zeros((num_classes,), dtype=np.float32)
        self._cache_tokens: Optional[np.ndarray] = None

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Forward pass from images to class logits."""
        batch_size = images.shape[0]
        patch_tokens = self.patch_embedding.forward(images)
        cls_tokens = np.repeat(self.cls_token, repeats=batch_size, axis=0)
        tokens = np.concatenate([cls_tokens, patch_tokens], axis=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        for layer in self.encoder_layers:
            tokens = layer.forward(tokens)

        self._cache_tokens = tokens
        cls_repr = tokens[:, 0, :]
        logits = cls_repr @ self.w_cls + self.b_cls
        return logits

    def backward(self, grad_logits: np.ndarray, lr: float) -> None:
        """Backpropagate gradients from logits to all parameters."""
        if self._cache_tokens is None:
            raise RuntimeError("Forward must be called before backward.")
        tokens = self._cache_tokens
        batch_size = tokens.shape[0]

        d_w_cls = tokens[:, 0, :].T @ grad_logits
        d_b_cls = np.sum(grad_logits, axis=0)
        d_tokens = np.zeros_like(tokens)
        d_tokens[:, 0, :] = grad_logits @ self.w_cls.T

        scale = 1.0 / float(batch_size)
        self.w_cls -= lr * d_w_cls.astype(np.float32) * scale
        self.b_cls -= lr * d_b_cls.astype(np.float32) * scale

        for layer in reversed(self.encoder_layers):
            d_tokens = layer.backward(d_tokens, lr=lr)

        d_tokens = d_tokens
        d_tokens_no_cls = d_tokens[:, 1:, :]
        self.pos_embed[:, : tokens.shape[1], :] -= lr * np.mean(
            d_tokens, axis=0, keepdims=True
        ).astype(np.float32)
        self.cls_token -= lr * np.mean(
            d_tokens[:, :1, :], axis=0, keepdims=True
        ).astype(np.float32)

        self.patch_embedding.backward(d_tokens_no_cls, lr=lr)

    @staticmethod
    def cross_entropy_loss(
        logits: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradient."""
        probs = _softmax(logits, axis=1)
        batch_size, num_classes = probs.shape
        labels_clipped = np.clip(labels, 0, num_classes - 1).astype(np.int64)
        one_hot = np.eye(num_classes, dtype=np.float32)[labels_clipped]
        loss = -float(
            np.sum(one_hot * np.log(probs + 1e-12)) / float(batch_size)
        )
        grad_logits = (probs - one_hot) / float(batch_size)
        return loss, grad_logits


def generate_synthetic_images(
    n_samples: int,
    image_size: int,
    in_channels: int,
    num_classes: int,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and integer labels."""
    if random_seed is not None:
        np.random.seed(random_seed)
    images = np.random.rand(n_samples, image_size, image_size, in_channels).astype(
        np.float32
    )
    labels = np.random.randint(0, num_classes, size=(n_samples,), dtype=np.int64)
    return images, labels


class ViTRunner:
    """Train and evaluate Vision Transformer from configuration."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found: %s, using defaults", config_path)
            return {}

    def _setup_logging(self) -> None:
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
        """Run ViT training and evaluation on synthetic data."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        image_size = data_cfg.get("image_size", 32)
        in_channels = data_cfg.get("in_channels", 3)
        num_classes = data_cfg.get("num_classes", 10)
        n_train = data_cfg.get("n_train", 512)
        n_test = data_cfg.get("n_test", 128)
        random_seed = data_cfg.get("random_seed", 42)

        dim_model = model_cfg.get("dim_model", 64)
        patch_size = model_cfg.get("patch_size", 4)
        num_heads = model_cfg.get("num_heads", 4)
        dim_ff = model_cfg.get("dim_ff", 128)
        num_layers = model_cfg.get("num_layers", 2)

        epochs = train_cfg.get("epochs", 5)
        batch_size = train_cfg.get("batch_size", 32)
        learning_rate = train_cfg.get("learning_rate", 0.001)

        images, labels = generate_synthetic_images(
            n_samples=n_train + n_test,
            image_size=image_size,
            in_channels=in_channels,
            num_classes=num_classes,
            random_seed=random_seed,
        )
        x_train, x_test = images[:n_train], images[n_train:]
        y_train, y_test = labels[:n_train], labels[n_train:]

        model = ViTClassifier(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        n_train_samples = x_train.shape[0]
        history_loss: List[float] = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_train_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses: List[float] = []

            for start in range(0, n_train_samples, batch_size):
                end = min(start + batch_size, n_train_samples)
                batch_images = x_shuffled[start:end]
                batch_labels = y_shuffled[start:end]

                logits = model.forward(batch_images)
                loss, grad_logits = model.cross_entropy_loss(
                    logits, batch_labels
                )
                model.backward(grad_logits, lr=learning_rate)
                epoch_losses.append(loss)

            epoch_loss = float(np.mean(epoch_losses))
            history_loss.append(epoch_loss)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                logger.info(
                    "Epoch %d/%d - loss: %.4f",
                    epoch + 1,
                    epochs,
                    epoch_loss,
                )

        logits_test = model.forward(x_test)
        preds_test = np.argmax(logits_test, axis=1)
        accuracy = float(np.mean(preds_test == y_test))

        results = {
            "train_loss": float(history_loss[-1]),
            "test_accuracy": accuracy,
        }
        logger.info(
            "Final results - train_loss: %.4f, test_accuracy: %.4f",
            results["train_loss"],
            results["test_accuracy"],
        )
        return results


def main() -> None:
    """Command-line entry point for ViT training."""
    parser = argparse.ArgumentParser(
        description="Train Vision Transformer from scratch on synthetic data"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = ViTRunner(
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

