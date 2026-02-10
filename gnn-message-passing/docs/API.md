# GNN Message Passing API Documentation

## Module: src.main

### Functions

#### _load_config(config_path)

Load YAML configuration file. Returns configuration dict. Raises FileNotFoundError
if the path does not exist.

#### _setup_logging(level, log_file)

Configure logging to console and optionally to a rotating log file.

#### _normalize_adjacency(adj)

Compute symmetric normalized adjacency with self-loops using
A_hat = D^{-1/2} (A + I) D^{-1/2}. Expects a square (N, N) tensor and returns an
(N, N) tensor.

#### node_classification_loss(logits, labels)

Cross-entropy loss for node classification. `logits` has shape (N, C) and
`labels` shape (N,) with integer class indices.

#### generate_synthetic_graph(num_nodes, num_features, num_classes, edge_prob, device, seed=None)

Generate a random undirected graph with Bernoulli edges, node features, and
integer labels. Returns `(adj, features, labels)` where:

- `adj`: (N, N) adjacency (0/1), symmetric with zero diagonal
- `features`: (N, F) feature matrix
- `labels`: (N,) integer labels in [0, num_classes-1]

#### run_training(config)

Train the GNN on synthetic graphs. Uses configuration sections:

- `model`: num_features, hidden_dim, num_classes, num_layers
- `training`: epochs, learning_rate, weight_decay, batch_size
- `data`: random_seed, num_train_graphs, num_nodes, edge_prob

#### main()

CLI entry point. Parses `--config` and runs training.

### Classes

#### GraphConvolution

Graph convolution layer using normalized adjacency for message passing.

**Constructor**: `GraphConvolution(in_features, out_features, bias=True)`

- `in_features`: Input feature dimension
- `out_features`: Output feature dimension
- `bias`: Include bias term

**Methods**:

- `forward(x, adj)`: Apply graph convolution given node features `(N, F_in)` and
  adjacency `(N, N)`, returning `(N, F_out)`.

#### GNN

Stacked graph convolutional network for node classification.

**Constructor**:

`GNN(num_features, hidden_dim, num_classes, num_layers=2, dropout=0.0)`

- `num_features`: Input feature dimension
- `hidden_dim`: Hidden feature dimension
- `num_classes`: Number of classes
- `num_layers`: Total GCN layers (>= 2)
- `dropout`: Dropout rate applied after hidden layers

**Methods**:

- `forward(x, adj)`: Compute node logits `(N, num_classes)` from features and
  adjacency.

