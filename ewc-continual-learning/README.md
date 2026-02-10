# EWC Continual Learning

A Python implementation of continual learning with Elastic Weight Consolidation
(EWC) to mitigate catastrophic forgetting. The model is trained sequentially on
multiple synthetic classification tasks. After each task, a diagonal Fisher
information matrix and parameter snapshot are computed; subsequent tasks are
trained with an EWC penalty that discourages large changes to important weights.

## Project Title and Description

This project demonstrates EWC for simple continual learning scenarios. EWC
addresses catastrophic forgetting by adding a quadratic regularization term
based on the Fisher information of previous tasks. Parameters that are more
important (higher Fisher values) are penalized more strongly when they drift
from their previous optimal values.

**Target Audience**: Practitioners and researchers exploring continual
learning, and engineers who want a reference implementation of EWC.

## Features

- Simple MLP classifier shared across tasks
- Synthetic task generator with task-specific Gaussian class prototypes
- Estimation of diagonal Fisher information after each task
- EWC quadratic penalty to protect important parameters
- Accuracy reporting before and after training each task
- Config-driven experiments via YAML

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for faster training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/ewc-continual-learning
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
  hidden_dim: 64
  num_classes: 3

training:
  tasks: 2
  epochs_per_task: 20
  batch_size: 32
  learning_rate: 0.001
  lambda_ewc: 50.0

data:
  random_seed: 42
  samples_per_task: 1000
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set:

- `RANDOM_SEED`: Integer seed for reproducibility

## Usage

### Command-Line

```bash
python src/main.py
python src/main.py --config config.yaml
```

### Programmatic Usage

```python
from src.main import (
    SimpleMLP,
    generate_task_dataset,
    estimate_fisher_diagonal,
    snapshot_parameters,
    ewc_penalty,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(input_dim=16, hidden_dim=64, num_classes=3).to(device)
x, y = generate_task_dataset(
    task_id=0,
    num_samples=1000,
    input_dim=16,
    num_classes=3,
    device=device,
    seed=42,
)
fisher = estimate_fisher_diagonal(model, x, y, batch_size=32)
prev_params = snapshot_parameters(model)
penalty = ewc_penalty(model, fisher, prev_params, lambda_ewc=50.0)
```

## Project Structure

```text
ewc-continual-learning/
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

- **Loss remains high**: Increase `epochs_per_task` or decrease `lambda_ewc`.
- **Previous task accuracy drops heavily**: Increase `lambda_ewc` or reduce
  the learning rate.
- **Slow training**: Reduce `samples_per_task`, `epochs_per_task`, or model
  size.

## Contributing

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Follow PEP 8 and the project docstring and type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request with a clear description.

## License

See repository license.

