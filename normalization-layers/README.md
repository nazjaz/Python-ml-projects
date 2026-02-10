# Batch Normalization and Layer Normalization from Scratch

This project implements batch normalization and layer normalization from scratch using NumPy and demonstrates their effect on training stability in a simple neural network.

## Project Title and Description

The goal of this project is to provide clear, from-scratch implementations of:

- Batch normalization: normalizing activations per feature across a mini-batch
- Layer normalization: normalizing activations per sample across features

These normalization techniques are integrated into a small feedforward network trained on a synthetic classification task, showing how they can stabilize and accelerate training.

**Target Audience**: Machine learning engineers and students who want a readable reference implementation of normalization layers without relying on deep learning frameworks.

## Features

- Batch normalization layer (`BatchNorm1D`) with:
  - Learnable scale (`gamma`) and shift (`beta`)
  - Running mean and variance for evaluation
  - Forward and backward passes for training
- Layer normalization layer (`LayerNorm1D`) with:
  - Learnable `gamma` and `beta`
  - Per-sample normalization across feature dimension
  - Forward and backward passes
- Simple two-layer feedforward classifier that can use:
  - No normalization
  - Batch normalization
  - Layer normalization
- Synthetic classification data generator
- Mean cross-entropy loss with gradient
- Configuration-driven training with logging

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/normalization-layers
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
  learning_rate: 0.05
  batch_size: 32
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Run training with batch normalization:

```bash
python src/main.py --mode batchnorm
```

Run training with layer normalization:

```bash
python src/main.py --mode layernorm
```

Run baseline without normalization:

```bash
python src/main.py --mode none
```

### With Custom Config and Output

```bash
python src/main.py --mode batchnorm --config config.yaml --output results.json
```

The script prints final training and test accuracy and loss and can write them to a JSON file.

## Project Structure

```text
normalization-layers/
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

- `src/main.py`: Implementations of batch normalization, layer normalization, a small MLP, and a training runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for normalization layers and training behavior
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

### Training Loss Does Not Decrease

- Ensure `learning_rate` is not excessively high.
- Try increasing `epochs` or `hidden_dim`.
- Verify that `n_classes` and labels from the data generator match.

### NaN or Inf Values

- Reduce `learning_rate`.
- Confirm that inputs are finite and that batch size is not extremely small.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

