# Dropout and Regularization Techniques from Scratch

This project implements dropout and other regularization techniques from scratch using NumPy and demonstrates their impact on overfitting in a simple neural network.

## Project Title and Description

The project provides a clear reference implementation of:

- Dropout regularization applied to hidden activations
- L2 weight decay applied to model parameters

These techniques are integrated into a small feedforward neural network trained on a synthetic classification task to illustrate how they can reduce overfitting.

**Target Audience**: Machine learning engineers and students who want to understand and experiment with regularization techniques without deep learning frameworks.

## Features

- Dropout regularization:
  - Bernoulli masks applied to hidden activations during training
  - Inverted dropout scaling to keep activation magnitude stable at test time
- L2 weight decay:
  - Applied to weights of all linear layers
  - Included in loss and gradients
- Simple two-layer feedforward classifier:
  - Modes: no regularization, dropout only, L2 only, dropout + L2
- Synthetic multi-class classification data generator
- Cross-entropy loss with softmax
- YAML-based configuration for data, model, and training
- Logging with rotating file handler
- Command-line interface for experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/regularization-techniques
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py
```

## Configuration

### Configuration File Structure

The project is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

data:
  n_train: 2000
  n_test: 500
  n_features: 20
  n_classes: 3
  random_seed: 42

model:
  hidden_dim: 32
  dropout_rate: 0.5
  l2_lambda: 0.001

training:
  epochs: 30
  learning_rate: 0.1
  batch_size: 32
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Run training with dropout and L2 regularization:

```bash
python src/main.py --mode dropout_l2
```

Run without regularization:

```bash
python src/main.py --mode none
```

Run with dropout only:

```bash
python src/main.py --mode dropout
```

Run with L2 only:

```bash
python src/main.py --mode l2
```

### With Custom Config and Output

```bash
python src/main.py --mode dropout_l2 --config config.yaml --output results.json
```

The script prints final training and test accuracy and loss and can write them to a JSON file.

## Project Structure

```text
regularization-techniques/
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

- `src/main.py`: Implementations of dropout, L2 regularization, a small MLP, and a training runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for dropout behavior, L2 loss contribution, and training
- `docs/API.md`: API-level documentation for the main classes and functions

## Testing

Run tests with `pytest`:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Troubleshooting

### Overfitting Persists

- Increase `dropout_rate` or `l2_lambda`.
- Increase `n_train` or reduce `hidden_dim`.

### Training Diverges

- Reduce `learning_rate`.
- Lower `dropout_rate` to keep gradients stable.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

