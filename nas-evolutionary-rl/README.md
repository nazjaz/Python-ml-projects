# Neural Architecture Search (NAS) with Evolutionary and RL

This project implements Neural Architecture Search using two strategies: (1) an **evolutionary algorithm** (population, fitness evaluation, tournament selection, crossover, mutation) and (2) **reinforcement learning** (REINFORCE controller that samples architectures and is updated with validation accuracy as reward). The search space is a small set of MLP architectures (number of layers, hidden dimension, activation). Architectures are evaluated by training on a classification task (digits dataset or synthetic fallback).

### Project Title and Description

The goal is to provide a self-contained reference implementation of NAS that combines evolutionary search and policy-gradient RL for architecture search, without large AutoML frameworks. Users can run either method from config or CLI and compare results.

**Target audience**: Practitioners and students interested in automated architecture design and NAS algorithms.

### Features

- **Discrete search space**: Number of layers (2, 3, 4), hidden dimension (32, 64, 128), activation (relu, tanh). Encoded as three choice indices.
- **Trainable MLP**: Variable-depth MLP with configurable activation; NumPy-only forward and backward for fast evaluation.
- **Evolutionary NAS**: Population of architectures; fitness = validation accuracy; tournament selection; crossover and mutation; generational replacement.
- **RL-based NAS**: Controller network outputs logits per choice; sample architecture, evaluate, use validation accuracy as reward; REINFORCE update.
- **Evaluation**: Train each candidate architecture for a few epochs on a small train/val split (digits or synthetic); return validation accuracy.
- **Configuration**: YAML config for method (evolution vs rl), population size, generations, rollouts, learning rates, and data limits.
- **CLI and logging**: Optional config path, output JSON path, and method override; rotating log file.

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd /path/to/Python-ml-projects/nas-evolutionary-rl
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Default config is `config.yaml` in the project root.

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

nas:
  method: "evolution"
  population_size: 8
  num_generations: 5
  mutation_prob: 0.2
  tournament_size: 3
  num_rollouts: 10
  controller_lr: 0.01
  eval_epochs: 3
  train_ratio: 0.8
  max_samples: 200
  random_seed: 0
```

- **method**: `"evolution"` or `"rl"`.
- **evolution**: population_size, num_generations, mutation_prob, tournament_size.
- **rl**: num_rollouts, controller_lr.
- **eval_epochs**: Training epochs per architecture evaluation.
- **max_samples**: Cap on dataset size for speed (digits subset or synthetic size).

Copy `.env.example` to `.env` for optional overrides (e.g. RANDOM_SEED).

### Usage

Run with default config (evolution):

```bash
python src/main.py
```

Run with RL:

```bash
python src/main.py --method rl
```

Custom config and output:

```bash
python src/main.py --config path/to/config.yaml --output results.json
```

The script prints the chosen method, best architecture (num_layers, hidden_dim, activation), and best validation accuracy; optionally writes a JSON summary.

### Project structure

```
nas-evolutionary-rl/
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

- `src/main.py`: Search space, TrainableMLP, train_and_evaluate, EvolutionaryNAS, Controller, RLNAS, load_digits_data, run_nas, main.
- `config.yaml`: NAS and logging settings.
- `tests/test_main.py`: Tests for encode/decode, MLP, train_and_evaluate, evolutionary/RL runs, run_nas.
- `docs/API.md`: API reference.

### Testing

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Troubleshooting

- **Slow runs**: Reduce population_size, num_generations, num_rollouts, or max_samples; reduce eval_epochs.
- **ImportError for sklearn**: The code falls back to synthetic data if scikit-learn is not installed; install with `pip install scikit-learn` for the digits dataset.
- **Poor best accuracy**: Increase eval_epochs or max_samples; try both methods and compare.

### Contributing

Follow PEP 8 and project standards; add tests for new behavior; ensure tests pass before submitting changes.

### License

Part of the Python ML Projects collection; see repository license.
