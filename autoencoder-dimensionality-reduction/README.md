# Autoencoder for Dimensionality Reduction and Feature Learning

This project implements a simple autoencoder using NumPy to perform dimensionality reduction and feature learning with an encoder-decoder architecture.

## Project Title and Description

The project provides a minimal yet complete implementation of:

- An encoder network that maps high-dimensional inputs to a low-dimensional latent space
- A decoder network that reconstructs the original inputs from latent representations
- Mean squared error reconstruction loss

The focus is on clarity and educational value, showing how autoencoders can learn compact representations of data without relying on deep learning frameworks.

**Target Audience**: Machine learning practitioners and students who want a transparent reference implementation of autoencoders using only NumPy.

## Features

- Fully connected autoencoder:
  - Input layer → hidden encoder layer → latent layer → hidden decoder layer → reconstruction
- Mean squared error (MSE) reconstruction loss
- Mini-batch training with gradient descent
- Synthetic Gaussian data generator for demonstration
- Encoded feature extraction for downstream use
- YAML-based configuration for data, model, and training
- Logging with rotating file handler
- Command-line interface for running experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/autoencoder-dimensionality-reduction
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
  n_samples: 2000
  n_features: 32
  latent_dim: 4
  random_seed: 42

model:
  hidden_dim: 16

training:
  epochs: 50
  learning_rate: 0.05
  batch_size: 64
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Train the autoencoder on synthetic data and print final reconstruction loss:

```bash
python src/main.py
```

### With Custom Config and Output

```bash
python src/main.py --config config.yaml --output results.json
```

The script prints final training loss and can write a results JSON file.

## Project Structure

```text
autoencoder-dimensionality-reduction/
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

- `src/main.py`: Autoencoder implementation, data generator, and training runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for autoencoder and data utilities
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

### Reconstruction Quality Is Poor

- Increase `hidden_dim` or `latent_dim`.
- Train for more `epochs`.
- Reduce `learning_rate` if loss oscillates.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

