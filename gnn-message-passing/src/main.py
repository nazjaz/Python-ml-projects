"""Graph neural network (GNN) with message passing and graph convolution.

This module implements a basic graph convolutional network (GCN) using
message passing. A normalized adjacency matrix aggregates neighbor messages,
and stacked graph convolution layers perform node classification on graphs.
"""

import argparse
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        level: Log level as string (e.g. INFO, DEBUG).
        log_file: Optional path to log file.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="-%Y-%m-%d %H:%M:%S",
    )
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Compute symmetric normalized adjacency with self-loops.

    Uses A_hat = D^{-1/2} (A + I) D^{-1/2} where D is the degree of A + I.

    Args:
        adj: Adjacency matrix of shape (N, N) with non-negative entries.

    Returns:
        Normalized adjacency matrix of shape (N, N).
    """
    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("adjacency must be square matrix of shape (N, N)")
    device = adj.device
    n = adj.size(0)
    # Symmetrize adjacency to ensure undirected message passing.
    adj_sym = 0.5 * (adj + adj.t())
    eye = torch.eye(n, device=device, dtype=adj.dtype)
    a_hat = adj_sym + eye
    deg = a_hat.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    d_hat = torch.diag(deg_inv_sqrt)
    return d_hat @ a_hat @ d_hat


class GraphConvolution(nn.Module):
    """Graph convolution layer using normalized adjacency for message passing.

    Computes H' = A_hat H W + b where A_hat is normalized adjacency and
    H are node features.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize graph convolution layer.

        Args:
            in_features: Size of input node features.
            out_features: Size of output node features.
            bias: Whether to include learnable bias term.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and bias."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply graph convolution.

        Args:
            x: Node feature matrix of shape (N, F_in).
            adj: Adjacency matrix of shape (N, N).

        Returns:
            Updated node features of shape (N, F_out).
        """
        a_norm = _normalize_adjacency(adj)
        support = x @ self.weight
        out = a_norm @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class GNN(nn.Module):
    """Stacked graph convolution network for node classification."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize GNN model.

        Args:
            num_features: Input feature dimension F_in.
            hidden_dim: Hidden feature dimension.
            num_classes: Number of node classes.
            num_layers: Total number of GCN layers (>= 2).
            dropout: Dropout rate applied after hidden layers.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.num_layers = num_layers
        self.dropout = dropout
        layers = []
        layers.append(GraphConvolution(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(GraphConvolution(hidden_dim, hidden_dim))
        layers.append(GraphConvolution(hidden_dim, num_classes))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass over all GCN layers.

        Args:
            x: Node features (N, F_in).
            adj: Adjacency matrix (N, N).

        Returns:
            Node logits (N, num_classes).
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < self.num_layers - 1:
                h = F.relu(h)
                if self.dropout > 0.0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def node_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for node classification.

    Args:
        logits: Predicted node logits of shape (N, C).
        labels: Integer node labels of shape (N,) with values in [0, C-1].

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(logits, labels.long())


def generate_synthetic_graph(
    num_nodes: int,
    num_features: int,
    num_classes: int,
    edge_prob: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random undirected graph with node features and labels.

    Args:
        num_nodes: Number of nodes in the graph.
        num_features: Feature dimension F.
        num_classes: Number of node classes.
        edge_prob: Probability of edge between any node pair.
        device: Torch device.
        seed: Optional random seed for reproducibility.

    Returns:
        adj: Adjacency matrix (N, N) with 0/1 entries.
        features: Node features (N, F).
        labels: Node labels (N,) in [0, num_classes-1].
    """
    if seed is not None:
        torch.manual_seed(seed)
    adj_upper = torch.bernoulli(
        torch.full((num_nodes, num_nodes), edge_prob, device=device)
    )
    adj_upper = torch.triu(adj_upper, diagonal=1)
    adj = adj_upper + adj_upper.t()
    features = torch.randn(num_nodes, num_features, device=device)
    labels = torch.randint(0, num_classes, (num_nodes,), device=device)
    return adj, features, labels


def run_training(config: Dict[str, Any]) -> None:
    """Train GNN on synthetic graphs according to configuration.

    Args:
        config: Configuration dictionary with model, training, and data keys.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    num_features = model_cfg.get("num_features", 16)
    hidden_dim = model_cfg.get("hidden_dim", 32)
    num_classes = model_cfg.get("num_classes", 3)
    num_layers = model_cfg.get("num_layers", 3)

    net = GNN(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    epochs = train_cfg.get("epochs", 20)
    num_graphs = data_cfg.get("num_train_graphs", 200)
    batch_size = max(1, int(train_cfg.get("batch_size", 1)))
    seed = data_cfg.get("random_seed", 42)
    num_nodes = data_cfg.get("num_nodes", 50)
    edge_prob = float(data_cfg.get("edge_prob", 0.1))

    steps_per_epoch = max(1, num_graphs // batch_size)
    logger.info(
        "Training for %d epochs, %d steps/epoch, batch_size=%d",
        epochs,
        steps_per_epoch,
        batch_size,
    )

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        for step in range(steps_per_epoch):
            adj, features, labels = generate_synthetic_graph(
                num_nodes=num_nodes,
                num_features=num_features,
                num_classes=num_classes,
                edge_prob=edge_prob,
                device=device,
                seed=seed + epoch * 1000 + step,
            )
            optimizer.zero_grad()
            logits = net(features, adj)
            loss = node_classification_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / steps_per_epoch
        logger.info("Epoch %d/%d loss: %.4f", epoch + 1, epochs, avg)


def main() -> None:
    """Entry point: load config, set up logging, run training."""
    parser = argparse.ArgumentParser(
        description=(
            "Graph neural network with message passing and graph "
            "convolution layers."
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

    logger.info("Starting GNN message passing training")
    run_training(config)
    logger.info("Done")


if __name__ == "__main__":
    main()

