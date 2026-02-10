# GNN Message Passing and Graph Convolutions

A Python implementation of graph neural networks (GNNs) with message passing
and graph convolution layers. The model operates on graphs represented by
adjacency matrices and node feature matrices, and performs node-level
classification using stacked graph convolutional layers.

## Project Title and Description

This project provides a self-contained graph neural network based on the
graph convolutional network (GCN) formulation. Message passing is implemented
through a normalized adjacency matrix that aggregates neighbor information at
each layer. The network supports configurable feature dimensions, hidden size,
number of layers, and number of classes. Synthetic graphs are generated for
demonstration and testing.

**Target Audience**: Developers and researchers exploring GNNs, students
learning message passing and graph convolutions, and engineers integrating
graph-based models into pipelines.

## Features

- GraphConvolution layer implementing H' = A_hat H W + b
- Symmetric normalized adjacency with self-loops for message passing
- Stacked GNN model for node classification
- Cross-entropy loss for node-level supervision
- Synthetic graph generator (adjacency, features, labels)
- Config-driven architecture and training (YAML)
- Logging to console and rotating file handler

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for faster training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/gnn-message-passing
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows: `venv\Scripts\activate`

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --config config.yaml
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  num_features: 16
  hidden_dim: 32
  num_classes: 3
  num_layers: 3

training:
  epochs: 20
  learning_rate: 0.001
  weight_decay: 0.0
  batch_size: 1

data:
  random_seed: 42
  num_train_graphs: 200
  num_nodes: 50
  edge_prob: 0.1
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set:

- `RANDOM_SEED`: Integer seed for reproducibility
- `DATA_ROOT`: Path to dataset root (optional; script runs with synthetic graphs)

## Usage

### Command-Line

Train with default config:

```bash
python src/main.py
```

Train with custom config path:

```bash
python src/main.py --config config.yaml
```

### Programmatic Usage

```python
from src.main import GNN, generate_synthetic_graph, node_classification_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adj, features, labels = generate_synthetic_graph(
    num_nodes=50,
    num_features=16,
    num_classes=3,
    edge_prob=0.1,
    device=device,
)
model = GNN(num_features=16, hidden_dim=32, num_classes=3, num_layers=3).to(device)
logits = model(features, adj)
loss = node_classification_loss(logits, labels)
```

## Project Structure

```text
gnn-message-passing/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/
│   └── main.py          # GCN layers, model, loss, synthetic data, training
├── tests/
│   └── test_main.py
├── docs/
│   └── API.md
└── logs/
    └── .gitkeep
```

## Testing

Run tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Troubleshooting

- **CUDA out of memory**: Reduce number of nodes or hidden_dim, or run on CPU.
- **Configuration file not found**: Run from project root or pass `--config` with full path.
- **Loss NaN**: Lower learning rate or check that labels are in [0, num_classes-1].

## Contributing

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Follow PEP 8 and the project docstring and type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request with a clear description.

## License

See repository license.

