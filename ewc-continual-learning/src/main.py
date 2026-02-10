"""Continual learning with Elastic Weight Consolidation (EWC).

This module implements EWC to mitigate catastrophic forgetting when training
a model sequentially on multiple tasks. After training on a task, the Fisher
information is estimated and used to constrain future updates via a quadratic
penalty on important parameters.
"""

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """Configure logging to console and optionally to file.

    Args:
        level: Log level (e.g. INFO, DEBUG).
        log_file: Optional path to log file.
    """
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


class SimpleMLP(nn.Module):
    """Simple MLP for classification across tasks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        """Initialize MLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.net(x)


def generate_task_dataset(
    task_id: int,
    num_samples: int,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification dataset for a specific task.

    Each task has its own class prototypes (Gaussian means). Catastrophic
    forgetting arises because tasks have different prototypes.

    Args:
        task_id: Identifier used to change prototypes between tasks.
        num_samples: Number of (x, y) pairs to generate.
        input_dim: Feature dimension.
        num_classes: Number of classes.
        device: Torch device.
        seed: Optional random seed.

    Returns:
        features: Tensor of shape (num_samples, input_dim).
        labels: Tensor of shape (num_samples,) with values in [0, num_classes-1].
    """
    if seed is not None:
        torch.manual_seed(seed + task_id * 1000)
    prototypes = torch.randn(num_classes, input_dim, device=device) + task_id
    labels = torch.randint(0, num_classes, (num_samples,), device=device)
    features = prototypes[labels] + 0.3 * torch.randn(
        num_samples, input_dim, device=device
    )
    return features, labels


def _iter_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Yield mini-batches from tensors."""
    num = x.size(0)
    for start in range(0, num, batch_size):
        end = min(start + batch_size, num)
        yield x[start:end], y[start:end]


def compute_accuracy(
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
        for xb, yb in _iter_batches(x, y, batch_size):
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return float(correct) / max(total, 1)


def estimate_fisher_diagonal(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> List[torch.Tensor]:
    """Estimate diagonal Fisher information matrix for a trained task.

    Uses squared gradients of log-likelihood (here approximated via cross-entropy
    loss) averaged over data.

    Args:
        model: Trained model for current task.
        x: Task inputs.
        y: Task labels.
        batch_size: Mini-batch size.

    Returns:
        List of tensors matching model parameters, containing Fisher diagonals.
    """
    model.eval()
    fisher_diags: List[torch.Tensor] = [
        torch.zeros_like(p) for p in model.parameters()
    ]
    total = 0

    for xb, yb in _iter_batches(x, y, batch_size):
        model.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        total += xb.size(0)
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                fisher_diags[i] += (p.grad.detach() ** 2) * xb.size(0)

    if total > 0:
        fisher_diags = [fd / float(total) for fd in fisher_diags]
    return fisher_diags


def snapshot_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Create a detached copy of model parameters."""
    return [p.detach().clone() for p in model.parameters()]


def ewc_penalty(
    model: nn.Module,
    fisher_diags: List[torch.Tensor],
    prev_params: List[torch.Tensor],
    lambda_ewc: float,
) -> torch.Tensor:
    """Compute EWC quadratic penalty for current parameters.

    Args:
        model: Current model.
        fisher_diags: Estimated Fisher diagonals from previous task.
        prev_params: Parameter snapshot from previous task.
        lambda_ewc: Regularization strength.

    Returns:
        Scalar penalty tensor.
    """
    if not fisher_diags or not prev_params:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for p, f, p_old in zip(model.parameters(), fisher_diags, prev_params):
        penalty = penalty + torch.sum(f * (p - p_old) ** 2)
    return (lambda_ewc / 2.0) * penalty


def train_task(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    fisher_diags: Optional[List[torch.Tensor]] = None,
    prev_params: Optional[List[torch.Tensor]] = None,
    lambda_ewc: float = 0.0,
) -> None:
    """Train model on a single task, optionally with EWC regularization."""
    device = next(model.parameters()).device
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in _iter_batches(x, y, batch_size):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            ce_loss = F.cross_entropy(logits, yb)
            reg = torch.tensor(0.0, device=device)
            if fisher_diags is not None and prev_params is not None:
                reg = ewc_penalty(model, fisher_diags, prev_params, lambda_ewc)
            loss = ce_loss + reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(1, (x.size(0) // batch_size))
        logger.info("Task train epoch %d/%d loss: %.4f", epoch + 1, epochs, avg)


def run_training(config: Dict[str, Any]) -> None:
    """Run continual learning experiment with EWC on synthetic tasks."""
    log_cfg = config.get("logging", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    input_dim = model_cfg.get("input_dim", 16)
    hidden_dim = model_cfg.get("hidden_dim", 64)
    num_classes = model_cfg.get("num_classes", 3)
    tasks = int(train_cfg.get("tasks", 2))
    epochs_per_task = int(train_cfg.get("epochs_per_task", 20))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("learning_rate", 0.001))
    lambda_ewc = float(train_cfg.get("lambda_ewc", 50.0))
    samples_per_task = int(data_cfg.get("samples_per_task", 1000))
    seed = int(data_cfg.get("random_seed", 42))

    model = SimpleMLP(input_dim, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    prev_fisher: Optional[List[torch.Tensor]] = None
    prev_params: Optional[List[torch.Tensor]] = None

    # Train sequential tasks with EWC after the first.
    for task_id in range(tasks):
        logger.info("===== Training task %d/%d =====", task_id + 1, tasks)
        x, y = generate_task_dataset(
            task_id=task_id,
            num_samples=samples_per_task,
            input_dim=input_dim,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )
        acc_before = 0.0
        if task_id > 0:
            acc_before = compute_accuracy(
                model, x, y, batch_size=batch_size
            )
            logger.info("Accuracy on new task before training: %.3f", acc_before)

        train_task(
            model=model,
            x=x,
            y=y,
            optimizer=optimizer,
            epochs=epochs_per_task,
            batch_size=batch_size,
            fisher_diags=prev_fisher,
            prev_params=prev_params,
            lambda_ewc=lambda_ewc,
        )

        acc_after = compute_accuracy(model, x, y, batch_size=batch_size)
        logger.info("Accuracy on task %d after training: %.3f", task_id + 1, acc_after)

        # After finishing a task, compute Fisher and snapshot parameters.
        prev_fisher = estimate_fisher_diagonal(
            model, x, y, batch_size=batch_size
        )
        prev_params = snapshot_parameters(model)


def main() -> None:
    """Entry point: load config, set up logging, run EWC training."""
    parser = argparse.ArgumentParser(
        description=(
            "Continual learning with Elastic Weight Consolidation to mitigate "
            "catastrophic forgetting."
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

    logger.info("Starting EWC continual learning experiment")
    run_training(config)
    logger.info("Done")


if __name__ == "__main__":
    main()

