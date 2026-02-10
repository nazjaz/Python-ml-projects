# Federated Learning Framework

This project implements a federated learning setup with distributed training, central aggregation (FedAvg), and privacy-preserving options. Multiple clients train on local data; a server aggregates their model updates and broadcasts the global model. Privacy techniques include gradient clipping and optional Gaussian noise (differential privacy). Implemented in NumPy with in-process simulation of clients.

### Description

Federated learning trains a global model without centralizing raw data: clients perform local SGD and send only model updates to a server, which aggregates (e.g. weighted average) and redistributes the global model. This code simulates that flow and adds gradient clipping and optional noise for privacy.

**Target audience**: Developers and students learning federated learning and privacy-preserving ML.

### Features

- **Distributed training**: Data partitioned across multiple clients; each client trains a local copy of the model on its shard.
- **FedAvg aggregation**: Server computes a weighted average of client weights by local sample count and updates the global model.
- **Privacy-preserving techniques**:
  - **Gradient clipping**: Clip the norm of the update (local weights minus global weights) before aggregation to bound influence of any client.
  - **Differential privacy**: Optional Gaussian noise added to each client’s update before aggregation (noise_sigma in config).
- **Data partitioning**: IID (random) or non-IID (by label) splits across clients.
- **Config and CLI**: YAML config for number of clients, rounds, local epochs, learning rate, clipping, noise; optional JSON output.

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd Python-ml-projects/federated-learning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Default: `config.yaml`

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

federated:
  num_clients: 5
  num_rounds: 10
  local_epochs: 1
  hidden_dim: 32
  learning_rate: 0.01
  batch_size: 32
  train_ratio: 0.8
  max_samples: null
  iid: true
  max_grad_norm: 10.0
  noise_sigma: null
  random_seed: 0
```

- **max_grad_norm**: If set, clip each client’s update (relative to global) to this L2 norm.
- **noise_sigma**: If set, add N(0, sigma^2) noise to each client’s update (DP-style).
- **iid**: true = random partition; false = partition by label (non-IID).

### Usage

```bash
python src/main.py
python src/main.py --config path/to/config.yaml --output results.json
```

Output includes final validation accuracy and number of rounds/clients.

### Project structure

```
federated-learning/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/main.py
├── tests/test_main.py
├── docs/API.md
└── logs/.gitkeep
```

### Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

### Troubleshooting

- **Low accuracy**: Increase num_rounds or local_epochs; try iid=true; reduce noise_sigma or increase max_grad_norm.
- **Instability**: Lower learning_rate or set max_grad_norm.

### License

Part of Python ML Projects; see repository license.
