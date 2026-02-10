# Adam, RMSprop, and AdaGrad Optimizers from Scratch

This project implements popular adaptive optimization algorithms - Adam, RMSprop, and AdaGrad - from scratch using NumPy and compares their behavior on a simple neural network.

## Project Title and Description

The project provides reference implementations of:

- AdaGrad: per-parameter learning rate scaled by accumulated squared gradients
- RMSprop: exponential moving average of squared gradients for adaptive steps
- Adam: adaptive moment estimation combining momentum and RMSprop-style scaling

These optimizers are plugged into a small feedforward classifier trained on synthetic data, allowing you to compare convergence speed and stability across optimizers.

**Target Audience**: Machine learning practitioners and students who want to understand and experiment with optimization algorithms without deep learning frameworks.

## Features

- Optimizers implemented from first principles:
  - `SGD` (baseline)
  - `AdaGrad`
  - `RMSprop`
  - `Adam`
- Each optimizer:
  - Tracks its own state (e.g., accumulated gradients or moments)
  - Adapts the effective learning rate per parameter
  - Exposes a common `step` interface for updating parameter arrays
- Simple two-layer feedforward classifier:
  - Shared across optimizers for fair comparison
- Synthetic multi-class classification data generator
- Cross-entropy loss with softmax
- YAML-based configuration for data, model, training, and optimizer hyperparameters
- Logging with rotating file handler
- Command-line interface to select optimizer and run experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/optimization-algorithms
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

training:
  epochs: 30
  learning_rate: 0.01
  batch_size: 32

optimizers:
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
  rho: 0.9
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Run training with Adam:

```bash
python src/main.py --optimizer adam
```

Run with RMSprop:

```bash
python src/main.py --optimizer rmsprop
```

Run with AdaGrad:

```bash
python src/main.py --optimizer adagrad
```

Run baseline SGD:

```bash
python src/main.py --optimizer sgd
```

### With Custom Config and Output

```bash
python src/main.py --optimizer adam --config config.yaml --output results.json
```

The script prints final training and test accuracy and loss and can write them to a JSON file.

## Project Structure

```text
optimization-algorithms/
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

- `src/main.py`: Implementations of optimizers, an MLP classifier, and a training runner
- `config.yaml`: Data, model, training, and optimizer hyperparameters
- `tests/test_main.py`: Unit tests for optimizers and training behavior
- `docs/API.md`: API-level documentation for core classes and functions

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

### Loss Does Not Decrease

- Try reducing `learning_rate`.
- For Adam and RMSprop, verify `beta1`, `beta2`, `rho` are in a reasonable range.
- Increase `epochs` if convergence is slow.

### Training Diverges or Becomes Unstable

- Lower `learning_rate` or increase batch size.
- Ensure data features are not extremely large in magnitude.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

