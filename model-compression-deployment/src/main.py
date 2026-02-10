"""Model compression: quantization, pruning, and distillation for efficient deployment.

This module implements post-training quantization (dynamic int8), magnitude-based
pruning (L1 unstructured), and knowledge distillation from a teacher to a smaller
student model. A demo trains a teacher, optionally compresses it, and trains a
student via distillation.
"""

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.nn.utils import prune
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


class MLP(nn.Module):
    """MLP classifier with configurable hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
    ) -> None:
        """Initialize MLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer sizes.
            num_classes: Number of output classes.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def apply_quantization(
    model: nn.Module,
    dtype: str = "qint8",
) -> nn.Module:
    """Apply dynamic quantization to linear layers for efficient deployment.

    Uses torch.quantization.quantize_dynamic to convert Linear layers to
    quantized (int8) versions, reducing memory and compute.

    Args:
        model: Model containing Linear layers (e.g. MLP).
        dtype: Quantization dtype; "qint8" is typical.

    Returns:
        Quantized model (in-place modification may also apply).
    """
    qdtype = getattr(torch, dtype, torch.qint8)
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=qdtype,
    )
    return quantized


def apply_pruning(
    model: nn.Module,
    amount: float,
    make_permanent: bool = True,
) -> None:
    """Apply L1 unstructured pruning to Linear layers.

    Prunes the smallest magnitude weights by fraction `amount`. Optionally
    removes the pruning reparametrization so masks are applied permanently.

    Args:
        model: Model containing Linear layers.
        amount: Fraction of parameters to prune in (0, 1).
        make_permanent: If True, remove reparametrization and keep masks.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            if make_permanent:
                prune.remove(module, "weight")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Compute distillation loss: soft target KL + hard label cross-entropy.

    Loss = alpha * KL(soft_teacher || soft_student) + (1 - alpha) * CE(student, labels).
    Soft logits are computed with temperature scaling.

    Args:
        student_logits: Student model logits (N, C).
        teacher_logits: Teacher model logits (N, C).
        labels: Hard labels (N,) long.
        temperature: Temperature for softmax (T > 1 softens).
        alpha: Weight for distillation term; (1 - alpha) for CE.

    Returns:
        Scalar loss tensor.
    """
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kl = F.kl_div(
        soft_student,
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1.0 - alpha) * ce


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def evaluate_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            logits = model(x[start:end])
            pred = logits.argmax(dim=1)
            correct += (pred == y[start:end]).sum().item()
            total += end - start
    return correct / max(total, 1)


def run_demo(config: Dict[str, Any]) -> None:
    """Train teacher, apply compression, train student via distillation."""
    model_cfg = config.get("model", {})
    comp_cfg = config.get("compression", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    input_dim = model_cfg.get("input_dim", 16)
    teacher_hidden = model_cfg.get("teacher_hidden", [64, 64])
    student_hidden = model_cfg.get("student_hidden", [32])
    num_classes = model_cfg.get("num_classes", 3)
    num_train = data_cfg.get("num_train", 600)
    seed = data_cfg.get("random_seed", 42)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    teacher_epochs = train_cfg.get("teacher_epochs", 20)
    student_epochs = train_cfg.get("student_epochs", 25)
    temp = comp_cfg.get("distillation", {}).get("temperature", 4.0)
    alpha = comp_cfg.get("distillation", {}).get("alpha", 0.7)
    prune_amount = comp_cfg.get("pruning", {}).get("amount", 0.3)

    x_train, y_train = generate_synthetic_data(
        num_train, input_dim, num_classes, device, seed
    )
    x_val, y_val = generate_synthetic_data(
        150, input_dim, num_classes, device, seed + 1
    )

    teacher = MLP(
        input_dim=input_dim,
        hidden_dims=teacher_hidden,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    for epoch in range(teacher_epochs):
        teacher.train()
        perm = torch.randperm(num_train, device=device)
        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            idx = perm[start:end]
            optimizer.zero_grad()
            logits = teacher(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            loss.backward()
            optimizer.step()
    acc_teacher = evaluate_accuracy(teacher, x_val, y_val, batch_size)
    logger.info("Teacher accuracy: %.4f, params: %d", acc_teacher, count_parameters(teacher))

    teacher_for_quant = MLP(
        input_dim=input_dim,
        hidden_dims=teacher_hidden,
        num_classes=num_classes,
    ).to(device)
    teacher_for_quant.load_state_dict(teacher.state_dict())
    if comp_cfg.get("quantization", {}).get("enabled", True):
        try:
            teacher_quant = apply_quantization(teacher_for_quant)
            acc_quant = evaluate_accuracy(teacher_quant, x_val, y_val, batch_size)
            logger.info("Quantized teacher accuracy: %.4f", acc_quant)
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "linear_prepack" in str(e):
                logger.warning(
                    "Quantization skipped: engine not available on this build"
                )
            else:
                raise

    teacher_for_prune = MLP(
        input_dim=input_dim,
        hidden_dims=teacher_hidden,
        num_classes=num_classes,
    ).to(device)
    teacher_for_prune.load_state_dict(teacher.state_dict())
    if comp_cfg.get("pruning", {}).get("enabled", True):
        apply_pruning(teacher_for_prune, amount=prune_amount, make_permanent=True)
        acc_prune = evaluate_accuracy(teacher_for_prune, x_val, y_val, batch_size)
        logger.info("Pruned teacher accuracy: %.4f, params: %d", acc_prune, count_parameters(teacher_for_prune))

    student = MLP(
        input_dim=input_dim,
        hidden_dims=student_hidden,
        num_classes=num_classes,
    ).to(device)
    optimizer_s = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for epoch in range(student_epochs):
        student.train()
        perm = torch.randperm(num_train, device=device)
        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            idx = perm[start:end]
            optimizer_s.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(x_train[idx])
            student_logits = student(x_train[idx])
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                y_train[idx],
                temperature=temp,
                alpha=alpha,
            )
            loss.backward()
            optimizer_s.step()
    acc_student = evaluate_accuracy(student, x_val, y_val, batch_size)
    logger.info("Student (distilled) accuracy: %.4f, params: %d", acc_student, count_parameters(student))


def main() -> None:
    """Entry point: load config, set up logging, run compression demo."""
    parser = argparse.ArgumentParser(
        description=(
            "Model compression: quantization, pruning, and distillation "
            "for efficient deployment."
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

    logger.info("Starting model compression demo")
    run_demo(config)
    logger.info("Done")


if __name__ == "__main__":
    main()
