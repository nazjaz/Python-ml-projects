"""Explainable AI: Integrated Gradients, LIME, and attention visualization.

This module provides model interpretability via integrated gradients (IG)
for input attribution, LIME for local linear explanations, and attention
visualization for models that expose attention weights.
"""

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_logging(level: str, log_file: Optional[str]) -> None:
    """Configure logging to console and optionally to file."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)


class SimpleClassifier(nn.Module):
    """MLP classifier for use with integrated gradients and LIME."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelWithAttention(nn.Module):
    """Classifier with a single self-attention layer over features for visualization.

    Input is treated as a sequence of feature dim; we learn an embedding per
    position and apply self-attention, then pool and classify. Stores the
    last attention weights in last_attention.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(embed_dim * input_dim, num_classes)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x_seq = x.unsqueeze(-1)
        emb = self.embed(x_seq)
        attn_out, attn_weights = self.attn(emb, emb, emb)
        self.last_attention = attn_weights.detach()
        flat = attn_out.reshape(batch, -1)
        return self.proj(flat)


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50,
) -> torch.Tensor:
    """Compute integrated gradients attribution for the target class.

    Integrates gradients along the path from baseline to input. Attribution
    approximates (x - baseline) * mean(gradients along path).

    Args:
        model: Differentiable model; forward takes input and returns logits.
        x: Input tensor of shape (1, input_dim) or (N, input_dim); uses first row.
        target_class: Class index for which to compute attribution.
        baseline: Baseline input; if None, uses zeros. Same shape as x.
        steps: Number of interpolation steps.

    Returns:
        Attribution tensor of same shape as x (single sample).
    """
    model.eval()
    x = x[:1].detach().requires_grad_(True)
    if baseline is None:
        baseline = torch.zeros_like(x, device=x.device)
    else:
        baseline = baseline[:1].detach()

    grads: List[torch.Tensor] = []
    for i in range(steps + 1):
        alpha = (i / steps) if steps > 0 else 1.0
        s = (baseline + alpha * (x - baseline)).clone().detach().requires_grad_(True)
        logits = model(s)
        score = logits[0, target_class]
        model.zero_grad()
        score.backward()
        if s.grad is not None:
            grads.append(s.grad.detach())
    grad_avg = torch.stack(grads, dim=0).mean(dim=0)
    attribution = (x.detach() - baseline) * grad_avg
    return attribution[0]


def lime_explain(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int,
    num_features: int,
    kernel_width: float = 0.25,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """LIME: local linear approximation of model predictions.

    Samples perturbed inputs around x, gets model predictions, fits a
    weighted linear model. Returns feature coefficients as importance.

    Args:
        model: Callable model; forward returns logits.
        x: Input tensor (1, D) or (D,). Single instance.
        num_samples: Number of perturbed samples to draw.
        num_features: Number of features (D).
        kernel_width: Width for exponential kernel (distance weighting).
        device: Device for model and tensors.

    Returns:
        One-dimensional array of length num_features: LIME feature importance.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x_flat = x.flatten()[:num_features]
    x_np = x_flat.detach().cpu().numpy()
    x_np = np.reshape(x_np, (1, -1))

    std = np.maximum(np.std(x_np, axis=0), 1e-6)
    samples = np.random.normal(
        loc=x_np,
        scale=std,
        size=(num_samples, num_features),
    )
    samples[0] = x_np[0]

    with torch.no_grad():
        samples_t = torch.from_numpy(samples).float().to(device)
        logits = model(samples_t)
        preds = logits.softmax(dim=1).cpu().numpy()

    pred_class = np.argmax(preds[0])
    y = preds[:, pred_class]

    distances = np.sqrt(np.sum((samples - x_np) ** 2, axis=1))
    weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))
    weights = np.maximum(weights, 1e-6)

    X = np.hstack([samples, np.ones((num_samples, 1))])
    W = np.diag(weights)
    XtWX = X.T @ W @ X + 1e-6 * np.eye(X.shape[1])
    XtWy = X.T @ W @ y
    coef = np.linalg.solve(XtWX, XtWy)
    importance = coef[:-1]
    return importance.astype(np.float64)


def get_attention_weights(model: nn.Module) -> Optional[torch.Tensor]:
    """Return the last stored attention weights if the model exposes them.

    Models like ModelWithAttention set last_attention after forward.
    Returns tensor of shape (batch, num_heads, seq_len, seq_len) or similar.
    """
    if hasattr(model, "last_attention") and model.last_attention is not None:
        return model.last_attention.detach().cpu()
    return None


def format_attention_for_visualization(
    attention: torch.Tensor,
    num_decimals: int = 3,
) -> str:
    """Format attention tensor as a string for logging or text visualization.

    Args:
        attention: Attention tensor (e.g. num_heads, seq, seq) or (seq, seq).
        num_decimals: Decimal places for values.

    Returns:
        Multi-line string describing the attention matrix or matrices.
    """
    a = attention.numpy()
    if a.ndim == 3:
        parts = []
        for h in range(a.shape[0]):
            parts.append(f"Head {h}:\n{np.round(a[h], num_decimals)}")
        return "\n".join(parts)
    return str(np.round(a, num_decimals))


def generate_synthetic_data(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification data."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(num_samples, input_dim, device=device)
    w = torch.randn(input_dim, num_classes, device=device)
    logits = x @ w
    y = logits.argmax(dim=1)
    return x, y


def run_demo(config: Dict[str, Any]) -> None:
    """Train a small model and run IG, LIME, and attention visualization."""
    model_cfg = config.get("model", {})
    interp_cfg = config.get("interpretability", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    input_dim = model_cfg.get("input_dim", 8)
    hidden_dim = model_cfg.get("hidden_dim", 32)
    num_classes = model_cfg.get("num_classes", 2)
    num_train = data_cfg.get("num_train", 400)
    seed = data_cfg.get("random_seed", 42)
    epochs = train_cfg.get("epochs", 15)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)

    x_train, y_train = generate_synthetic_data(
        num_train, input_dim, num_classes, device, seed
    )
    model = SimpleClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_train, device=device)
        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            idx = perm[start:end]
            optimizer.zero_grad()
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            loss.backward()
            optimizer.step()
    logger.info("Training complete")

    ig_cfg = interp_cfg.get("integrated_gradients", {})
    steps = ig_cfg.get("steps", 50)
    baseline = torch.zeros(1, input_dim, device=device)
    sample = x_train[:1].detach()
    target = y_train[0].item()
    ig_attr = integrated_gradients(model, sample, target, baseline, steps)
    logger.info(
        "Integrated gradients attribution (sample, target=%d): norm=%.4f",
        target,
        ig_attr.norm().item(),
    )

    lime_cfg = interp_cfg.get("lime", {})
    num_samples = lime_cfg.get("num_samples", 500)
    num_features = lime_cfg.get("num_features", input_dim)
    kernel_width = lime_cfg.get("kernel_width", 0.25)
    lime_imp = lime_explain(
        model, sample, num_samples, num_features, kernel_width, device
    )
    logger.info(
        "LIME feature importance: min=%.4f max=%.4f",
        float(np.min(lime_imp)),
        float(np.max(lime_imp)),
    )

    attn_cfg = interp_cfg.get("attention", {})
    embed_dim = attn_cfg.get("embed_dim", 16)
    num_heads = attn_cfg.get("num_heads", 2)
    attn_model = ModelWithAttention(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_classes=num_classes,
    ).to(device)
    _ = attn_model(sample)
    attn_weights = get_attention_weights(attn_model)
    if attn_weights is not None:
        viz = format_attention_for_visualization(attn_weights[0])
        logger.info("Attention visualization (first head):\n%s", viz[:500])


def main() -> None:
    """Entry point: load config, set up logging, run interpretability demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Explainable AI: Integrated Gradients, LIME, "
            "and attention visualization."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    log_cfg = config.get("logging", {})
    _setup_logging(
        log_cfg.get("level", "INFO"),
        log_cfg.get("file"),
    )

    logger.info("Starting explainable AI interpretability demo")
    run_demo(config)
    logger.info("Done")


if __name__ == "__main__":
    main()
