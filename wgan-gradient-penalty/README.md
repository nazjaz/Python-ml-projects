# Wasserstein GAN with Gradient Penalty (WGAN-GP) from Scratch

This project implements a simple Wasserstein GAN with gradient penalty (WGAN-GP) using NumPy for stable training and improved sample quality on a low-dimensional synthetic dataset.

## Project Title and Description

The project demonstrates the core ideas behind WGAN-GP:

- A generator network maps noise vectors to synthetic samples
- A critic (discriminator without sigmoid) assigns scalar scores to inputs
- The Wasserstein distance is approximated via the critic
- A gradient penalty term encourages the critic to satisfy a 1-Lipschitz constraint

The implementation operates on a 2D Gaussian mixture distribution, making it easy to inspect and extend.

**Target Audience**: Practitioners and students who want a clear, framework-free reference implementation of WGAN-GP training mechanics.

## Features

- Fully connected generator and critic:
  - Generator: noise vector → hidden layer → 2D output
  - Critic: 2D input → hidden layer → scalar score
- Wasserstein losses:
  - Critic loss: \( \mathbb{E}[D(\hat{x}_{\text{fake}})] - \mathbb{E}[D(x_{\text{real}})] + \lambda \cdot \mathbb{E}[(\lVert \nabla_{\hat{x}} D(\hat{x}) \rVert_2 - 1)^2] \)
  - Generator loss: \( -\mathbb{E}[D(\hat{x}_{\text{fake}})] \)
- Gradient penalty:
  - Interpolated samples between real and fake data
  - Numerical estimate of gradient norm of critic with respect to inputs
- Mini-batch training with multiple critic updates per generator update
- YAML-based configuration for data, model, and training
- Logging with rotating file handler
- Command-line interface for experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/wgan-gradient-penalty
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
  hidden_dim: 32

training:
  epochs: 1000
  learning_rate_generator: 0.0005
  learning_rate_critic: 0.0005
  batch_size: 64
  critic_iters: 5
  lambda_gp: 10.0
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Train the WGAN-GP on synthetic 2D data:

```bash
python src/main.py
```

### With Custom Config and Output

```bash
python src/main.py --config config.yaml --output results.json
```

The script prints final critic and generator losses and can write a results JSON file.

## Project Structure

```text
wgan-gradient-penalty/
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

- `src/main.py`: Generator, critic, WGAN-GP training loop, and runner
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

### Training Instability

- Reduce learning rates for the critic and generator.
- Increase `critic_iters` to train the critic more relative to the generator.

### Poor Sample Quality

- Increase the model `hidden_dim` or the number of `epochs`.
- Tune `lambda_gp` to improve Lipschitz enforcement.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

