# Generative Adversarial Network (GAN) from Scratch

This project implements a simple generative adversarial network (GAN) using NumPy, with separate generator and discriminator networks and alternating training dynamics.

## Project Title and Description

The goal of this project is to illustrate GAN training mechanics without relying on deep learning frameworks:

- A generator network maps random noise to synthetic samples
- A discriminator network tries to distinguish real data from generated samples
- Both networks are trained adversarially via gradient-based updates

The example uses a low-dimensional synthetic dataset so that behavior is easy to inspect and extend.

**Target Audience**: Practitioners and students who want a transparent, framework-free implementation of basic GAN concepts.

## Features

- Fully connected generator and discriminator:
  - Generator: noise vector → hidden layer → 2D output
  - Discriminator: 2D input → hidden layer → scalar probability
- Adversarial training loop:
  - Binary cross-entropy losses for generator and discriminator
  - Alternating updates for discriminator and generator
- Synthetic 2D data (Gaussian mixture) as "real" distribution
- Mini-batch training
- YAML-based configuration for data, model, and training
- Logging with rotating file handler
- Command-line interface for experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/gan-generator-discriminator
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
  n_samples: 1000
  noise_dim: 8
  data_dim: 2
  random_seed: 42

model:
  hidden_dim: 16

training:
  epochs: 2000
  learning_rate_generator: 0.001
  learning_rate_discriminator: 0.001
  batch_size: 64
  d_steps: 1
  g_steps: 1
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Train the GAN on synthetic 2D data:

```bash
python src/main.py
```

### With Custom Config and Output

```bash
python src/main.py --config config.yaml --output results.json
```

The script prints final discriminator and generator losses and can write a results JSON file.

## Project Structure

```text
gan-generator-discriminator/
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

- `src/main.py`: Generator, discriminator, GAN training loop, and runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for networks and training loop
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

### Mode Collapse or Unstable Training

- Reduce learning rates (`learning_rate_generator`, `learning_rate_discriminator`).
- Increase `hidden_dim` or training `epochs`.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

