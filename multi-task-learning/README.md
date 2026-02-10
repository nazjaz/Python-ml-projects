# Multi-Task Learning with Shared Representations and Task-Specific Heads

A Python implementation of multi-task learning: a shared encoder produces a common representation from the input, and task-specific heads map that representation to each task's outputs. Training minimizes a weighted sum of per-task losses so the shared representation and all heads are learned jointly.

## Project Title and Description

This project provides a minimal multi-task learning setup for classification tasks. The shared encoder (MLP) learns features useful across tasks; each task has its own head (linear layer) that predicts task-specific labels. The combined loss encourages the representation to support all tasks while allowing task-specific specialization in the heads.

**Target Audience**: Practitioners and students working on multi-task learning, transfer learning, and shared representations.

## Features

- SharedEncoder: MLP mapping input to a shared representation
- TaskHead: Linear layer from shared representation to task output (e.g. class logits)
- MultiTaskModel: Shared encoder plus one head per task
- Weighted combined loss (cross-entropy per task, summed with configurable weights)
- Synthetic multi-task data generator (one input set, multiple label sets)
- Config-driven model and training via YAML

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for faster training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/multi-task-learning
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
  input_dim: 16
  shared_hidden: [64, 64]
  shared_dim: 32
  tasks:
    - name: task_a
      type: classification
      num_classes: 3
    - name: task_b
      type: classification
      num_classes: 2

training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.001
  loss_weights: [1.0, 1.0]

data:
  random_seed: 42
  num_train: 800
  num_val: 200
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set `RANDOM_SEED`.

## Usage

### Command-Line

```bash
python src/main.py
python src/main.py --config config.yaml
```

### Programmatic Usage

```python
from src.main import (
    MultiTaskModel,
    multi_task_loss,
    generate_synthetic_multi_task_data,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_configs = [
    {"type": "classification", "num_classes": 3},
    {"type": "classification", "num_classes": 2},
]
model = MultiTaskModel(
    input_dim=16,
    shared_hidden=[64, 64],
    shared_dim=32,
    task_configs=task_configs,
).to(device)
features, labels_list = generate_synthetic_multi_task_data(
    100, 16, [3, 2], device, seed=42
)
logits_list = model(features)
loss = multi_task_loss(logits_list, labels_list, weights=[1.0, 1.0])
loss.backward()
```

## Project Structure

```
multi-task-learning/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/
│   └── main.py
├── tests/
│   └── test_main.py
├── docs/
│   └── API.md
└── logs/
    └── .gitkeep
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Troubleshooting

- **Loss not decreasing**: Increase shared capacity (shared_hidden, shared_dim) or epochs.
- **One task dominates**: Adjust loss_weights to balance tasks.
- **Configuration file not found**: Run from project root or pass full path to --config.

## Contributing

1. Create a virtual environment and install dependencies.
2. Follow PEP 8 and project docstring and type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request.

## License

See repository license.
