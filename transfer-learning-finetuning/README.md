# Transfer Learning and Fine-Tuning Strategies from Scratch

This project implements simple transfer learning and fine-tuning strategies using NumPy-based models on synthetic data. It demonstrates how to adapt a pre-trained model to a new but related classification task.

## Project Title and Description

The project simulates transfer learning without deep learning frameworks by:

- Training a base model on a "source" task
- Reusing the learned representation on a "target" task via different strategies:
  - Feature extractor: freeze base layers, train a new head only
  - Head fine-tuning: start from pre-trained base, retrain head and lightly fine-tune base
  - Full fine-tuning: start from pre-trained weights and train all layers

**Target Audience**: Practitioners and students who want a clear, framework-free reference for transfer learning and fine-tuning workflows.

## Features

- Simple two-layer feedforward base model (MLP) built with NumPy
- Synthetic source and target classification tasks with related distributions
- Transfer learning strategies:
  - `feature_extractor`: freeze base, train new head
  - `head_finetune`: initialize from base, train head and base with same rate
  - `full_finetune`: train all layers from pre-trained weights
- Baseline training from scratch on target task for comparison
- Training runner that reports source and target metrics
- YAML-based configuration for data, model, and training
- Logging with rotating file handler
- Command-line interface to select strategy

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/transfer-learning-finetuning
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
  base_n_samples: 2000
  target_n_samples: 800
  n_features: 20
  n_classes: 3
  random_seed: 42

model:
  hidden_dim: 32

training:
  base_epochs: 20
  target_epochs: 20
  learning_rate_base: 0.05
  learning_rate_target: 0.02
  batch_size: 32
```

### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides:

- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Basic Usage

Run transfer learning using the model as a frozen feature extractor:

```bash
python src/main.py --strategy feature_extractor
```

Run fine-tuning of the head and base together:

```bash
python src/main.py --strategy head_finetune
```

Run full fine-tuning of all layers:

```bash
python src/main.py --strategy full_finetune
```

Run baseline training from scratch on the target task:

```bash
python src/main.py --strategy scratch
```

### With Custom Config and Output

```bash
python src/main.py --strategy feature_extractor --config config.yaml --output results.json
```

The script prints source and target losses and accuracies and can write them to a JSON file.

## Project Structure

```text
transfer-learning-finetuning/
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

- `src/main.py`: Implementation of the base and target models, transfer strategies, and runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for model behavior and transfer strategies
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

### Target Accuracy Low

- Increase `target_epochs` or `hidden_dim`.
- Try `head_finetune` or `full_finetune` instead of `feature_extractor`.

### No Benefit from Transfer

- Verify that base and target tasks are related (e.g., both multi-class with similar structure).
- Increase `base_n_samples` or `base_epochs` to get a better pre-trained model.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

