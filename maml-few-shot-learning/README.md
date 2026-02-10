# MAML Few-Shot Learning

A Python implementation of Model-Agnostic Meta-Learning (MAML) for few-shot learning. The inner loop adapts model parameters on each task's support set with a few gradient steps; the outer loop minimizes query loss after adaptation across tasks. Includes N-way K-shot synthetic task sampling and an optional first-order (FOMAML) mode.

## Project Title and Description

This project provides a self-contained MAML implementation for few-shot classification. It addresses the problem of learning a model that can quickly adapt to new tasks with few examples. The meta-learner optimizes initial parameters so that one or several gradient steps on a task's support set yield good performance on that task's query set. Synthetic N-way K-shot tasks (random class prototypes plus noise) are used for training and demonstration.

**Target Audience**: Researchers and developers working on meta-learning and few-shot learning, and engineers integrating fast adaptation into pipelines.

## Features

- FewShotMLP: MLP with `forward_with_params` for differentiable inner-loop updates
- N-way K-shot task sampler (synthetic class prototypes and support/query points)
- MAML inner loop: clone parameters, take K gradient steps on support loss
- MAML outer loop: query loss after adaptation, meta-gradient, Adam update
- Optional first-order MAML (FOMAML): no second-order derivatives through inner loop
- Config-driven: inner/outer learning rates, inner steps, N-way, K-shot, etc.
- Logging to console and rotating file

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for faster meta-training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/maml-few-shot-learning
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
  input_dim: 8
  hidden_dim: 64
  num_classes: 5
  num_layers: 2

maml:
  inner_lr: 0.01
  outer_lr: 0.001
  inner_steps: 5
  first_order: false

training:
  meta_epochs: 30
  tasks_per_epoch: 100
  batch_tasks: 4

data:
  random_seed: 42
  n_way: 5
  k_shot: 3
  query_size: 15
  input_dim: 8
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
    FewShotMLP,
    sample_few_shot_task,
    maml_inner_outer_step,
)

model = FewShotMLP(input_dim=8, hidden_dim=64, num_classes=5, num_layers=2)
support_x, support_y, query_x, query_y = sample_few_shot_task(
    5, 3, 15, 8, torch.device("cpu"), seed=42
)
loss = maml_inner_outer_step(
    model, support_x, support_y, query_x, query_y,
    inner_lr=0.01, inner_steps=5, first_order=False,
)
loss.backward()
```

## Project Structure

```
maml-few-shot-learning/
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

- **CUDA out of memory**: Reduce `batch_tasks`, `query_size`, or `hidden_dim`.
- **Configuration file not found**: Run from project root or pass full `--config` path.
- **Loss NaN**: Lower `inner_lr` or `outer_lr`; try `first_order: true`.

## Contributing

1. Create a virtual environment and install dependencies.
2. Follow PEP 8 and project docstring/type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request.

## License

See repository license.
