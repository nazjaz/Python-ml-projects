"""Neural style transfer with content and style loss optimization.

This module implements neural style transfer using a small feature extractor
(Conv2D + ReLU) in NumPy. Content loss is L2 distance between feature maps
of the generated and content images; style loss is L2 distance between
Gram matrices of the generated and style images. The generated image is
optimized via gradient descent to minimize the weighted sum of both losses.
"""

import argparse
import json
import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_EPS = 1e-8


def conv2d_forward(
    x: np.ndarray,
    w: np.ndarray,
    pad: int = 1,
) -> np.ndarray:
    """Conv2D: x (H, W, C_in), w (kH, kW, C_in, C_out). Same padding."""
    H, W, C_in = x.shape
    kH, kW, _, C_out = w.shape
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)
    out_h = H + 2 * pad - kH + 1
    out_w = W + 2 * pad - kW + 1
    out = np.zeros((out_h, out_w, C_out), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = x_pad[i : i + kH, j : j + kW, :]
            out[i, j, :] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2]))
    return out


def conv2d_backward_input(
    grad_out: np.ndarray,
    w: np.ndarray,
    x_shape: Tuple[int, ...],
    pad: int = 1,
) -> np.ndarray:
    """Gradient of conv output w.r.t. input (for fixed weights)."""
    kH, kW, C_in, C_out = w.shape
    H, W, _ = x_shape
    x_pad_shape = (H + 2 * pad, W + 2 * pad, C_in)
    grad_pad = np.zeros(x_pad_shape, dtype=np.float32)
    out_h, out_w, _ = grad_out.shape
    for i in range(out_h):
        for j in range(out_w):
            for c in range(C_out):
                grad_pad[i : i + kH, j : j + kW, :] += (
                    grad_out[i, j, c] * w[:, :, :, c]
                )
    return grad_pad[pad : pad + H, pad : pad + W, :]


def gram_matrix(features: np.ndarray) -> np.ndarray:
    """Compute Gram matrix from feature map (H, W, C). Returns (C, C)."""
    H, W, C = features.shape
    F = features.reshape(-1, C)
    n = F.shape[0] * C
    return (F.T @ F) / max(n, 1)


def gram_matrix_backward(
    grad_gram: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """Gradient of Gram (w.r.t. features) for backprop."""
    H, W, C = features.shape
    F = features.reshape(-1, C)
    n = max(F.shape[0] * C, 1)
    grad_F = (2.0 / n) * (F @ grad_gram)
    return grad_F.reshape(H, W, C)


def load_content_style_images(
    size: int = 8,
    content_index: int = 0,
    style_index: int = 1,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load content and style images as (H, W, 1) in [-1, 1]. Uses digits or synthetic."""
    try:
        from sklearn.datasets import load_digits

        data = load_digits()
        images = np.asarray(data.images, dtype=np.float32)
        images = (images / 16.0) * 2.0 - 1.0
        n = images.shape[0]
        content = images[content_index % n]
        style = images[style_index % n]
        content = content.reshape(size, size, 1)
        style = style.reshape(size, size, 1)
        return content, style
    except ImportError:
        rng = np.random.default_rng(random_seed or 42)
        content = np.tanh(rng.standard_normal((size, size, 1)).astype(np.float32))
        style = np.tanh(rng.standard_normal((size, size, 1)).astype(np.float32))
        return content, style


class FeatureExtractor:
    """Small CNN: one Conv2D + ReLU. Fixed weights; used for content and style loss."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 16,
        kernel_size: int = 3,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self.pad = kernel_size // 2
        scale = 1.0 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.w = np.random.randn(
            kernel_size, kernel_size, in_channels, out_channels
        ).astype(np.float32) * scale
        self._input: Optional[np.ndarray] = None
        self._activated: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (H, W, C_in) -> (H', W', C_out)."""
        self._input = x
        h = conv2d_forward(x, self.w, pad=self.pad)
        self._activated = np.maximum(h, 0.0)
        return self._activated

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backprop gradient to input (grad_out same shape as forward output)."""
        if self._activated is None:
            raise RuntimeError("Forward before backward.")
        grad_h = grad_out * (self._activated > 0.0).astype(np.float32)
        return conv2d_backward_input(
            grad_h, self.w, self._input.shape, pad=self.pad
        )


def content_loss(feat_gen: np.ndarray, feat_content: np.ndarray) -> float:
    """MSE between feature maps."""
    return float(np.mean((feat_gen - feat_content) ** 2))


def content_loss_grad(feat_gen: np.ndarray, feat_content: np.ndarray) -> np.ndarray:
    """Gradient of content loss w.r.t. feat_gen."""
    n = feat_gen.size
    return 2.0 * (feat_gen - feat_content) / max(n, 1)


def style_loss(gram_gen: np.ndarray, gram_style: np.ndarray) -> float:
    """MSE between Gram matrices."""
    return float(np.mean((gram_gen - gram_style) ** 2))


def style_loss_grad(gram_gen: np.ndarray, gram_style: np.ndarray) -> np.ndarray:
    """Gradient of style loss w.r.t. gram_gen."""
    n = gram_gen.size
    return 2.0 * (gram_gen - gram_style) / max(n, 1)


def run_style_transfer(
    content: np.ndarray,
    style: np.ndarray,
    extractor: FeatureExtractor,
    num_steps: int,
    content_weight: float,
    style_weight: float,
    lr: float,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Optimize generated image to minimize content_loss + style_loss.
    Returns (generated_image, history).
    """
    rng = np.random.default_rng(random_seed)
    generated = content.copy()
    feat_content = extractor.forward(content)
    gram_style = gram_matrix(extractor.forward(style))
    history: List[Dict] = []

    for step in range(num_steps):
        feat_gen = extractor.forward(generated)
        gram_gen = gram_matrix(feat_gen)
        c_loss = content_loss(feat_gen, feat_content)
        s_loss = style_loss(gram_gen, gram_style)
        total = content_weight * c_loss + style_weight * s_loss
        history.append({"step": step, "content_loss": c_loss, "style_loss": s_loss, "total": total})

        grad_feat = content_weight * content_loss_grad(feat_gen, feat_content)
        grad_gram = style_weight * style_loss_grad(gram_gen, gram_style)
        grad_feat += gram_matrix_backward(grad_gram, feat_gen)
        grad_image = extractor.backward(grad_feat)
        generated = generated - lr * grad_image
        generated = np.clip(generated, -1.0, 1.0).astype(np.float32)

        if (step + 1) % max(1, num_steps // 5) == 0:
            logger.info("Step %d total_loss=%.4f content=%.4f style=%.4f", step + 1, total, c_loss, s_loss)

    return generated, history


@dataclass
class StyleTransferConfig:
    """Configuration for neural style transfer."""

    num_steps: int = 200
    content_weight: float = 1.0
    style_weight: float = 1e3
    learning_rate: float = 1.0
    extractor_channels: int = 16
    image_size: int = 8
    content_index: int = 0
    style_index: int = 1
    random_seed: Optional[int] = 0


def run_pipeline(config: StyleTransferConfig) -> Dict:
    """Load content/style, build extractor, run optimization, return metrics."""
    content, style = load_content_style_images(
        size=config.image_size,
        content_index=config.content_index,
        style_index=config.style_index,
        random_seed=config.random_seed,
    )
    extractor = FeatureExtractor(
        in_channels=1,
        out_channels=config.extractor_channels,
        kernel_size=3,
        random_seed=config.random_seed,
    )
    generated, history = run_style_transfer(
        content,
        style,
        extractor,
        num_steps=config.num_steps,
        content_weight=config.content_weight,
        style_weight=config.style_weight,
        lr=config.learning_rate,
        random_seed=config.random_seed,
    )
    final = history[-1] if history else {}
    return {
        "final_content_loss": final.get("content_loss", 0.0),
        "final_style_loss": final.get("style_loss", 0.0),
        "final_total_loss": final.get("total", 0.0),
        "generated_shape": generated.shape,
        "num_steps": config.num_steps,
    }


def main() -> None:
    """Entry point: parse args, run style transfer, print and optionally save results."""
    parser = argparse.ArgumentParser(
        description="Neural style transfer with content and style loss"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent.parent / "config.yaml"
    config_dict: Dict = {}
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config not found: %s", config_path)

    log_cfg = config_dict.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "logs/app.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.setLevel(level)
    logger.addHandler(handler)

    d = config_dict.get("style_transfer", {})
    config = StyleTransferConfig(
        num_steps=d.get("num_steps", 200),
        content_weight=d.get("content_weight", 1.0),
        style_weight=d.get("style_weight", 1e3),
        learning_rate=d.get("learning_rate", 1.0),
        extractor_channels=d.get("extractor_channels", 16),
        image_size=d.get("image_size", 8),
        content_index=d.get("content_index", 0),
        style_index=d.get("style_index", 1),
        random_seed=d.get("random_seed", 0),
    )

    results = run_pipeline(config)
    print("\nStyle Transfer Results:")
    print("========================================")
    for k, v in results.items():
        if isinstance(v, float):
            print("  %s: %.4f" % (k, v))
        else:
            print("  %s: %s" % (k, v))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in results.items() if k != "generated_shape"}
        out["generated_shape"] = list(results["generated_shape"])
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
