## Transformer Encoder from Scratch with Multi-Head Attention

This project provides a pure-NumPy implementation of a Transformer encoder
architecture for sequence classification. It demonstrates how to build
multi-head self-attention, sinusoidal positional encoding, layer
normalization, and a stacked encoder from first principles without using
high-level deep learning frameworks.

### Project Title and Description

The goal of this project is to show how the core components of the
Transformer encoder can be implemented and trained end to end using only
NumPy. The model is trained on a synthetic token classification task where
the label depends on the parity of the token sequence, which exercises the
attention mechanism and positional encoding.

**Target Audience**: Machine learning practitioners and students who want a
transparent, framework-free reference implementation of the Transformer
encoder, including multi-head attention and positional encoding.

### Features

- **Transformer encoder stack implemented from scratch**:
  - Multi-head scaled dot-product self-attention
  - Sinusoidal positional encodings
  - Position-wise feed-forward networks
  - Residual connections and layer normalization
- **Token-level sequence classification model** using the representation of
  the first token as a classification token
- **Synthetic data generator** for a parity-based classification task
- **Configuration-driven experimentation** via `config.yaml`
- **Logging** with rotating file handler
- **Command-line interface** for training and evaluation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation

#### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/transformer-architecture
```

#### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
python src/main.py
```

### Configuration

#### Configuration File Structure

The project is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

data:
  n_train: 512
  n_test: 128
  sequence_length: 12
  vocab_size: 32
  random_seed: 7

model:
  dim_model: 32
  num_heads: 4
  dim_ff: 64
  num_layers: 2
  num_classes: 2

training:
  epochs: 10
  learning_rate: 0.01
  batch_size: 32
```

#### Environment Variables

Copy `.env.example` to `.env` and configure optional overrides as needed.
The current implementation does not require mandatory secrets or external
services. Environment variables are reserved for future extensions.

### Usage

#### Basic Usage

```bash
python src/main.py
```

#### With Custom Config

```bash
python src/main.py --config path/to/config.yaml
```

#### Save Results to File

```bash
python src/main.py --output results.json
```

The script prints final training loss and test accuracy to the console and
optionally writes them to a JSON file.

### Project Structure

```text
transformer-architecture/
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

- `src/main.py`: Transformer implementation, synthetic data generation, and
  training runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for attention, encoder, and data utilities
- `docs/API.md`: API-level documentation for the main classes and functions

### Testing

Run tests with `pytest`:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Troubleshooting

#### Training Is Slow

The Transformer is implemented in NumPy and runs on CPU only. To speed up
experiments, reduce `n_train`, `epochs`, or increase `batch_size` in
`config.yaml`.

#### Accuracy Is Low

- Verify that the model capacity (`dim_model`, `dim_ff`, `num_layers`) is
  sufficient for the task.
- Try adjusting `learning_rate` and `batch_size`.
- Ensure that `sequence_length` and `vocab_size` in `config.yaml` match your
  expectations.

#### Numerical Instability

If you observe `nan` losses or probabilities:

- Reduce `learning_rate`.
- Decrease `sequence_length`.
- Confirm that configuration values are within a reasonable range.

### Contributing

1. Create a feature branch from the main repository.
2. Follow PEP 8 and the shared Python automation standards.
3. Add or update tests for any new functionality.
4. Ensure all tests pass before opening a pull request.

### License

This project is part of the Python ML Projects collection. Refer to the main
repository license for details.

