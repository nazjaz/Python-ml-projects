# Variational Autoencoder (VAE) with Reparameterization Trick

This project implements a simple variational autoencoder (VAE) using NumPy for dimensionality reduction and feature learning, including the reparameterization trick and KL divergence regularization.

## Project Title and Description

The project demonstrates how to build a VAE from first principles:

- Encoder network produces mean and log-variance of a latent Gaussian distribution
- Reparameterization trick: sampling latent variables via \( z = \mu + \sigma \odot \epsilon \)
- Decoder network reconstructs inputs from latent samples
- Loss combines reconstruction error and KL divergence between posterior and standard normal prior

**Target Audience**: Practitioners and students who want to understand VAEs at the implementation level without relying on deep learning frameworks.

## Features

- Fully connected VAE:
  - Encoder: input → hidden → latent mean/log-variance
  - Decoder: latent sample → hidden → reconstruction
- Reparameterization trick for differentiable sampling
- KL divergence regularization for latent distribution
- Reconstruction loss: mean squared error
- Mini-batch training with gradient descent
- Synthetic Gaussian data generator
- YAML-based configuration for data, model, and training
- Logging with rotating file handler
- Command-line interface for running experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/variational-autoencoder
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
  learning_rate: 0.01
  batch_size: 64
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Train the VAE on synthetic data and print final total, reconstruction, and KL losses:

```bash
python src/main.py
```

### With Custom Config and Output

```bash
python src/main.py --config config.yaml --output results.json
```

The script prints final losses and can write a results JSON file.

## Project Structure

```text
variational-autoencoder/
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

- `src/main.py`: VAE implementation, data generator, and training runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for VAE and data utilities
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

### KL Divergence Too Small or Too Large

- Adjust learning rate or number of `epochs`.
- Increase or decrease `hidden_dim` or `latent_dim` to change model capacity.

### Reconstruction Quality Is Poor

- Train for more `epochs`.
- Increase `hidden_dim`.
- Reduce `learning_rate` if loss oscillates strongly.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

