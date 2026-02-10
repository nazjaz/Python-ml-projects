# Scaled Dot-Product Attention from Scratch for Sequence-to-Sequence Models

This project implements scaled dot-product attention and an attention-based sequence-to-sequence (seq2seq) model from scratch using only NumPy. It demonstrates how attention can be used to align encoder and decoder hidden states for sequence transformation tasks.

## Project Title and Description

The project provides a minimal yet complete implementation of:

- Scaled dot-product attention with optional masking
- An encoder-decoder seq2seq model using attention over encoder states
- A synthetic copy task for training and evaluation

The focus is on clarity and correctness rather than performance, making it suitable as a reference implementation for learning or experimentation.

**Target Audience**: Machine learning practitioners and students who want to understand attention mechanisms and seq2seq models at the implementation level without relying on high-level deep learning frameworks.

## Features

- Scaled dot-product attention:
  - Query-key dot products with \(1 / \sqrt{d_k}\) scaling
  - Softmax normalization over encoder positions
  - Support for attention masks
- Simple attention-based seq2seq model:
  - Learned token embeddings
  - Encoder and decoder recurrent layers
  - Attention over encoder states at each decoder step
  - Cross-entropy loss over target token sequences
- Synthetic integer copy task data generator
- YAML-based configuration for data, model, and training
- Logging with rotating file handler
- Command-line interface for experiments

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/attention-seq2seq
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
  src_length: 6
  tgt_length: 6
  vocab_size: 12
  random_seed: 42

model:
  d_model: 16
  d_k: 16
  d_v: 16

training:
  epochs: 15
  learning_rate: 0.05
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

The script prints final train and test accuracy and loss to the console and optionally writes them to a JSON file.

## Project Structure

```text
attention-seq2seq/
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

- `src/main.py`: Implementation of scaled dot-product attention, encoder-decoder seq2seq model, data generation, and training runner
- `config.yaml`: Data, model, and training configuration
- `tests/test_main.py`: Unit tests for attention mechanism, data generation, and training loop
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

### Training Is Slow or Unstable

- Reduce `epochs` or increase `batch_size` in `config.yaml`.
- Lower the `learning_rate` if the loss oscillates or diverges.

### Accuracy Remains Low

- Increase `d_model`, `d_k`, and `d_v` for more model capacity.
- Train for more epochs.
- Ensure `vocab_size` is appropriate for the synthetic task.

## Contributing

1. Create a feature branch from the main repository
2. Follow PEP 8 and the project coding standards
3. Add or update tests for new functionality
4. Ensure all tests pass before opening a pull request

## License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.

