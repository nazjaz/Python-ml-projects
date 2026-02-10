"""Multi-task learning with shared representations and task-specific heads.

This module implements a multi-task model: a shared encoder produces a common
representation from the input; task-specific heads map that representation to
each task's outputs. Training minimizes a weighted sum of per-task losses to
learn both the shared representation and the heads jointly.
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


class SharedEncoder(nn.Module):
    """MLP that maps input features to a shared representation.

    Layers: input_dim -> hidden[0] -> ... -> shared_dim with ReLU between.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        shared_dim: int,
    ) -> None:
        """Initialize shared encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer sizes.
            shared_dim: Output representation dimension.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, shared_dim))
        self.net = nn.Sequential(*layers)
        self.shared_dim = shared_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns representation of shape (N, shared_dim)."""
        return self.net(x)


class TaskHead(nn.Module):
    """Task-specific head: shared_dim -> num_classes (classification)."""

    def __init__(self, shared_dim: int, num_classes: int) -> None:
        """Initialize task head.

        Args:
            shared_dim: Size of shared representation.
            num_classes: Number of output classes for this task.
        """
        super().__init__()
        self.linear = nn.Linear(shared_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (N, num_classes)."""
        return self.linear(shared)


class MultiTaskModel(nn.Module):
    """Multi-task model: shared encoder plus one task-specific head per task."""

    def __init__(
        self,
        input_dim: int,
        shared_hidden: List[int],
        shared_dim: int,
        task_configs: List[Dict[str, Any]],
    ) -> None:
        """Initialize multi-task model.

        Args:
            input_dim: Input feature dimension.
            shared_hidden: Hidden layer sizes for shared encoder.
            shared_dim: Shared representation dimension.
            task_configs: List of dicts with 'type' and 'num_classes' (for classification).
        """
        super().__init__()
        self.encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=shared_hidden,
            shared_dim=shared_dim,
        )
        self.heads = nn.ModuleList()
        for cfg in task_configs:
            if cfg.get("type") == "classification":
                head = TaskHead(shared_dim, cfg["num_classes"])
            else:
                raise ValueError(
                    f"Unknown task type: {cfg.get('type')}. Use 'classification'."
                )
            self.heads.append(head)
        self.task_configs = task_configs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through shared encoder and all task heads.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            List of logits tensors, one per task; each of shape (N, num_classes).
        """
        shared = self.encoder(x)
        return [head(shared) for head in self.heads]

    def num_tasks(self) -> int:
        """Return number of tasks."""
        return len(self.heads)


def multi_task_loss(
    logits_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Compute weighted sum of cross-entropy losses for each task.

    Args:
        logits_list: List of logits (N, num_classes_i) per task.
        labels_list: List of label tensors (N,) per task.
        weights: Optional per-task loss weights; default 1.0 per task.

    Returns:
        Scalar combined loss tensor.
    """
    if weights is None:
        weights = [1.0] * len(logits_list)
    total = torch.tensor(0.0, device=logits_list[0].device)
    for logits, labels, w in zip(logits_list, labels_list, weights):
        total = total + w * F.cross_entropy(logits, labels.long())
    return total


def generate_synthetic_multi_task_data(
    num_samples: int,
    input_dim: int,
    task_num_classes: List[int],
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Generate synthetic data: one input set, one label set per task.

    Inputs are random. Each task's labels are generated from a task-specific
    linear function of the inputs plus noise (then discretized to classes).

    Args:
        num_samples: Number of samples.
        input_dim: Input feature dimension.
        task_num_classes: Number of classes for each task.
        device: Torch device.
        seed: Optional random seed.

    Returns:
        features: (num_samples, input_dim).
        labels_list: List of (num_samples,) integer label tensors.
    """
    if seed is not None:
        torch.manual_seed(seed)
    features = torch.randn(num_samples, input_dim, device=device)
    labels_list: List[torch.Tensor] = []
    for num_classes in task_num_classes:
        w = torch.randn(input_dim, 1, device=device)
        logits = features @ w
        logits = logits.squeeze(-1) + 0.5 * torch.randn(num_samples, device=device)
        probs = torch.sigmoid(logits)
        bins = torch.linspace(0, 1, num_classes + 1, device=device)
        labels = torch.bucketize(probs.clamp(0.01, 0.99), bins) - 1
        labels = labels.clamp(0, num_classes - 1)
        labels_list.append(labels)
    return features, labels_list


def run_training(config: Dict[str, Any]) -> None:
    """Train multi-task model on synthetic data.

    Args:
        config: Full config with model, training, data, logging.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    input_dim = model_cfg.get("input_dim", 16)
    shared_hidden = model_cfg.get("shared_hidden", [64, 64])
    shared_dim = model_cfg.get("shared_dim", 32)
    task_configs = model_cfg.get("tasks", [])

    if not task_configs:
        raise ValueError("model.tasks must be a non-empty list of task configs")

    model = MultiTaskModel(
        input_dim=input_dim,
        shared_hidden=shared_hidden,
        shared_dim=shared_dim,
        task_configs=task_configs,
    ).to(device)

    epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    loss_weights = train_cfg.get("loss_weights")
    if loss_weights is None:
        loss_weights = [1.0] * len(task_configs)
    num_train = data_cfg.get("num_train", 800)
    seed = data_cfg.get("random_seed", 42)

    task_num_classes = [
        t["num_classes"] for t in task_configs if t.get("type") == "classification"
    ]
    features, labels_list = generate_synthetic_multi_task_data(
        num_samples=num_train,
        input_dim=input_dim,
        task_num_classes=task_num_classes,
        device=device,
        seed=seed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, num_train // batch_size)

    logger.info(
        "Training multi-task model: %d tasks, %d epochs, %d steps/epoch",
        model.num_tasks(),
        epochs,
        steps_per_epoch,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(num_train, device=device)
        features_shuf = features[perm]
        labels_shuf = [lab[perm] for lab in labels_list]

        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            batch_x = features_shuf[start:end]
            batch_labels = [lab[start:end] for lab in labels_shuf]

            optimizer.zero_grad()
            logits_list = model(batch_x)
            loss = multi_task_loss(logits_list, batch_labels, loss_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / steps_per_epoch
        logger.info("Epoch %d/%d combined loss: %.4f", epoch + 1, epochs, avg)


def main() -> None:
    """Entry point: load config, set up logging, run training."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-task learning with shared representations and "
            "task-specific heads."
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

    logger.info("Starting multi-task learning training")
    run_training(config)
    logger.info("Done")


if __name__ == "__main__":
    main()
