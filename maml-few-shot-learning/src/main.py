"""Model-Agnostic Meta-Learning (MAML) for few-shot learning.

This module implements MAML: an inner loop adapts model parameters on a task
support set with a few gradient steps; the outer loop minimizes query loss
after adaptation across tasks. Includes N-way K-shot task sampling and a
functional MLP for differentiable inner updates.
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


class FewShotMLP(nn.Module):
    """MLP that supports forward with an explicit parameter list for MAML inner loop.

    Layers are linear -> relu -> ... -> linear (logits). Parameter list order
    matches: [W0, b0, W1, b1, ..., W_out, b_out].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
    ) -> None:
        """Initialize MLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Output class count.
            num_layers: Number of hidden layers (>= 1).
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        layers: List[nn.Module] = []
        prev = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden_dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward using module parameters."""
        return self.net(x)

    def forward_with_params(
        self,
        x: torch.Tensor,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward using a flat list of parameters (for MAML inner loop).

        Args:
            x: Input tensor (N, input_dim).
            params: List of (weight, bias) in order for each linear layer.

        Returns:
            Logits tensor (N, num_classes).
        """
        h = x
        idx = 0
        for _ in range(self.num_layers - 1):
            w, b = params[idx], params[idx + 1]
            h = F.linear(h, w, b)
            h = F.relu(h)
            idx += 2
        w, b = params[idx], params[idx + 1]
        return F.linear(h, w, b)

    def get_param_list(self) -> List[torch.Tensor]:
        """Return list of parameters in same order as forward_with_params expects."""
        out: List[torch.Tensor] = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                out.append(m.weight)
                out.append(m.bias)
        return out


def sample_few_shot_task(
    n_way: int,
    k_shot: int,
    query_size: int,
    input_dim: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one N-way K-shot classification task (synthetic).

    Each class is a random Gaussian prototype; support and query points
    are prototype + noise. Labels are class indices in [0, n_way - 1].

    Args:
        n_way: Number of classes in the task.
        k_shot: Support samples per class.
        query_size: Total query samples (split across classes).
        input_dim: Feature dimension.
        device: Torch device.
        seed: Optional random seed.

    Returns:
        support_x: (n_way * k_shot, input_dim).
        support_y: (n_way * k_shot,) long.
        query_x: (query_size, input_dim).
        query_y: (query_size,) long.
    """
    if seed is not None:
        torch.manual_seed(seed)
    prototypes = torch.randn(n_way, input_dim, device=device)
    support_list_x: List[torch.Tensor] = []
    support_list_y: List[torch.Tensor] = []
    for c in range(n_way):
        pts = prototypes[c : c + 1] + 0.1 * torch.randn(
            k_shot, input_dim, device=device
        )
        support_list_x.append(pts)
        support_list_y.append(torch.full((k_shot,), c, dtype=torch.long, device=device))
    support_x = torch.cat(support_list_x, dim=0)
    support_y = torch.cat(support_list_y, dim=0)

    q_per_class = max(1, query_size // n_way)
    query_list_x: List[torch.Tensor] = []
    query_list_y: List[torch.Tensor] = []
    for c in range(n_way):
        pts = prototypes[c : c + 1] + 0.1 * torch.randn(
            q_per_class, input_dim, device=device
        )
        query_list_x.append(pts)
        query_list_y.append(
            torch.full((q_per_class,), c, dtype=torch.long, device=device)
        )
    query_x = torch.cat(query_list_x, dim=0)
    query_y = torch.cat(query_list_y, dim=0)
    if query_x.size(0) > query_size:
        query_x = query_x[:query_size]
        query_y = query_y[:query_size]
    return support_x, support_y, query_x, query_y


def maml_inner_outer_step(
    model: FewShotMLP,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    first_order: bool,
) -> torch.Tensor:
    """Perform one MAML inner adaptation and return query loss for meta-update.

    Inner loop: clone params and take inner_steps gradient steps on support loss.
    Query loss is computed with adapted parameters; backward on it gives
    meta-gradient (unless first_order, which stops gradient through inner loop).

    Args:
        model: FewShotMLP with get_param_list and forward_with_params.
        support_x: Support inputs (S, F).
        support_y: Support labels (S,) long.
        query_x: Query inputs (Q, F).
        query_y: Query labels (Q,) long.
        inner_lr: Learning rate for inner steps.
        inner_steps: Number of inner gradient steps.
        first_order: If True, do not backprop through inner updates (FOMAML).

    Returns:
        Scalar query loss tensor (for backward and logging).
    """
    params = [p.clone().detach().requires_grad_(True) for p in model.get_param_list()]
    for step in range(inner_steps):
        support_logits = model.forward_with_params(support_x, params)
        inner_loss = F.cross_entropy(support_logits, support_y)
        grads = torch.autograd.grad(
            inner_loss,
            params,
            create_graph=not first_order,
            allow_unused=True,
        )
        params = [
            p - inner_lr * (g if g is not None else torch.zeros_like(p))
            for p, g in zip(params, grads)
        ]

    query_logits = model.forward_with_params(query_x, params)
    query_loss = F.cross_entropy(query_logits, query_y)
    return query_loss


def run_training(config: Dict[str, Any]) -> None:
    """Run MAML meta-training on synthetic few-shot tasks.

    Args:
        config: Full config with model, maml, training, data, logging.
    """
    model_cfg = config.get("model", {})
    maml_cfg = config.get("maml", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    log_cfg = config.get("logging", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    input_dim = model_cfg.get("input_dim", data_cfg.get("input_dim", 8))
    hidden_dim = model_cfg.get("hidden_dim", 64)
    num_classes = model_cfg.get("num_classes", data_cfg.get("n_way", 5))
    num_layers = model_cfg.get("num_layers", 2)

    model = FewShotMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
    ).to(device)

    outer_lr = maml_cfg.get("outer_lr", 0.001)
    inner_lr = maml_cfg.get("inner_lr", 0.01)
    inner_steps = maml_cfg.get("inner_steps", 5)
    first_order = maml_cfg.get("first_order", False)

    optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    meta_epochs = train_cfg.get("meta_epochs", 30)
    tasks_per_epoch = train_cfg.get("tasks_per_epoch", 100)
    batch_tasks = train_cfg.get("batch_tasks", 4)
    n_way = data_cfg.get("n_way", 5)
    k_shot = data_cfg.get("k_shot", 3)
    query_size = data_cfg.get("query_size", 15)
    seed = data_cfg.get("random_seed", 42)

    steps_per_epoch = max(1, tasks_per_epoch // batch_tasks)
    logger.info(
        "MAML: meta_epochs=%d, tasks_per_epoch=%d, batch_tasks=%d",
        meta_epochs,
        tasks_per_epoch,
        batch_tasks,
    )
    logger.info(
        "Inner lr=%.4f, steps=%d, first_order=%s",
        inner_lr,
        inner_steps,
        first_order,
    )

    for epoch in range(meta_epochs):
        model.train()
        total_loss = 0.0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            batch_loss = 0.0
            for _ in range(batch_tasks):
                support_x, support_y, query_x, query_y = sample_few_shot_task(
                    n_way=n_way,
                    k_shot=k_shot,
                    query_size=query_size,
                    input_dim=input_dim,
                    device=device,
                    seed=seed + epoch * 10000 + step * batch_tasks + _,
                )
                loss = maml_inner_outer_step(
                    model,
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    inner_lr=inner_lr,
                    inner_steps=inner_steps,
                    first_order=first_order,
                )
                batch_loss = batch_loss + loss
            (batch_loss / batch_tasks).backward()
            total_loss += batch_loss.item()
        optimizer.step()
        avg = total_loss / steps_per_epoch
        logger.info("Epoch %d/%d meta-loss: %.4f", epoch + 1, meta_epochs, avg)


def main() -> None:
    """Entry point: load config, set up logging, run MAML training."""
    parser = argparse.ArgumentParser(
        description="MAML for few-shot learning (meta-learning)."
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

    logger.info("Starting MAML few-shot learning")
    run_training(config)
    logger.info("Done")


if __name__ == "__main__":
    main()
