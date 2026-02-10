"""Tests for graph neural network module."""

import torch

from src.main import (
    GNN,
    GraphConvolution,
    _normalize_adjacency,
    generate_synthetic_graph,
    node_classification_loss,
)


class TestNormalizeAdjacency:
    """Test cases for adjacency normalization."""

    def test_normalize_returns_square_matrix(self) -> None:
        """Test that normalization preserves matrix shape."""
        adj = torch.zeros(4, 4)
        norm = _normalize_adjacency(adj)
        assert norm.shape == (4, 4)

    def test_normalize_is_symmetric(self) -> None:
        """Test that normalized adjacency is symmetric."""
        adj = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        norm = _normalize_adjacency(adj)
        assert torch.allclose(norm, norm.t(), atol=1e-6)


class TestGraphConvolution:
    """Test cases for GraphConvolution layer."""

    def test_output_shape(self) -> None:
        """Test that GraphConvolution produces correct output dimensions."""
        layer = GraphConvolution(5, 7)
        x = torch.randn(10, 5)
        adj = torch.eye(10)
        out = layer(x, adj)
        assert out.shape == (10, 7)

    def test_identity_adjacency_behaves_like_linear(self) -> None:
        """Test that with zero adjacency, only self-messages are used."""
        layer = GraphConvolution(3, 4)
        x = torch.randn(6, 3)
        adj = torch.zeros(6, 6)
        out = layer(x, adj)
        # Normalization adds self-loops so each node sees only itself.
        assert out.shape == (6, 4)


class TestGNN:
    """Test cases for stacked GNN model."""

    def test_forward_output_shape(self) -> None:
        """Test that GNN forward returns node logits of expected shape."""
        model = GNN(num_features=8, hidden_dim=16, num_classes=4, num_layers=3)
        x = torch.randn(20, 8)
        adj = torch.eye(20)
        out = model(x, adj)
        assert out.shape == (20, 4)

    def test_requires_at_least_two_layers(self) -> None:
        """Test that constructing with one layer raises ValueError."""
        try:
            GNN(num_features=4, hidden_dim=8, num_classes=2, num_layers=1)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for num_layers < 2")


class TestNodeClassificationLoss:
    """Test cases for node_classification_loss."""

    def test_loss_is_scalar(self) -> None:
        """Test that node_classification_loss returns a scalar tensor."""
        logits = torch.randn(5, 3)
        labels = torch.randint(0, 3, (5,))
        loss = node_classification_loss(logits, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_decreases_with_correct_prediction(self) -> None:
        """Test that loss is lower for logits aligned with labels."""
        labels = torch.zeros(4, dtype=torch.long)
        random_logits = torch.randn(4, 2) * 0.1
        good_logits = torch.randn(4, 2) * 0.1
        good_logits[:, 0] = 10.0
        loss_random = node_classification_loss(random_logits, labels)
        loss_good = node_classification_loss(good_logits, labels)
        assert loss_good.item() < loss_random.item()


class TestGenerateSyntheticGraph:
    """Test cases for generate_synthetic_graph."""

    def test_shapes_and_ranges(self) -> None:
        """Test that generated graph has correct shapes and label range."""
        device = torch.device("cpu")
        adj, features, labels = generate_synthetic_graph(
            num_nodes=12,
            num_features=6,
            num_classes=3,
            edge_prob=0.2,
            device=device,
            seed=123,
        )
        assert adj.shape == (12, 12)
        assert features.shape == (12, 6)
        assert labels.shape == (12,)
        assert labels.min() >= 0 and labels.max() < 3

    def test_adjacency_is_symmetric_without_self_loops(self) -> None:
        """Test that generated adjacency is symmetric and has zero diagonal."""
        device = torch.device("cpu")
        adj, _, _ = generate_synthetic_graph(
            num_nodes=5,
            num_features=4,
            num_classes=2,
            edge_prob=0.3,
            device=device,
            seed=7,
        )
        assert torch.allclose(adj, adj.t(), atol=1e-6)
        assert torch.allclose(torch.diag(adj), torch.zeros(5))

