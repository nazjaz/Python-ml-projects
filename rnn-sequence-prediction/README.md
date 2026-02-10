# RNN with LSTM Cells from Scratch for Sequence Prediction

This project provides a complete implementation of a recurrent neural network (RNN) with long short-term memory (LSTM) cells built from scratch using only NumPy. It supports training on synthetic time-series data for next-step sequence prediction and exposes a simple command-line interface.

## Project Title and Description

The goal of this project is to demonstrate how sequence models based on LSTM cells can be implemented without relying on high-level deep learning frameworks. The implementation includes forward propagation, backpropagation through time (BPTT), and training logic for many-to-one sequence prediction tasks such as forecasting the next value in a time series.

The core components are:

- A single-layer LSTM network implemented directly with NumPy
- A dense output layer for regression-style predictions
- Synthetic sine-wave sequence generator for supervised learning
- Configuration-driven training via YAML

**Target Audience**: Machine learning students, practitioners studying sequence models from first principles, and developers who need a transparent, framework-free reference implementation of LSTM-based sequence prediction.

## Features

- LSTM layer with:
  - Input, forget, output, and candidate gates
  - Cell and hidden state propagation through time
  - Backpropagation through time with parameter updates
- Dense output layer for real-valued prediction
- Synthetic sine-wave sequence generator with configurable noise
- Mean squared error (MSE) loss for regression-style sequence prediction
- Mini-batch training with configurable batch size, learning rate, and epochs
- Configuration via `config.yaml`
- Logging with rotating file handler
- Simple command-line interface for running experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/rnn-sequence-prediction
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
  sequence_length: 20
  random_seed: 42
  noise_std: 0.05

model:
  input_dim: 1
  hidden_dim: 32
  output_dim: 1

training:
  epochs: 20
  learning_rate: 0.005
  batch_size: 32
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

```bash
python src/main.py
```

### With Custom Config

```bash
python src/main.py --config path/to/config.yaml
```

### Save Results to File

```bash
python src/main.py --output results.json
```

The script prints final training and test loss to the console and optionally writes them to a JSON file.

## Project Structure

```text
rnn-sequence-prediction/
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

- `src/main.py`: LSTM implementation, synthetic data generation, and training runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for LSTM, model, and data utilities
- `docs/API.md`: API-level documentation for main classes and functions

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

### Training Is Slow

LSTM implemented in NumPy operates purely on CPU without GPU acceleration. To speed up experiments, reduce `n_train`, `epochs`, or increase `batch_size` in `config.yaml`.

### Loss Does Not Decrease

- Ensure `learning_rate` is not too high or too low
- Verify that `sequence_length` is appropriate for the pattern being learned
- Try increasing `hidden_dim` to provide more model capacity

### Numerical Instability

If you see extremely large losses or `nan` values:

- Reduce `learning_rate`
- Decrease `sequence_length`
- Check that inputs are within a reasonable numeric range

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

