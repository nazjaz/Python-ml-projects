# BERT-like Model from Scratch with MLM and NSP

This project implements a BERT-style encoder from scratch using only NumPy, with masked language modeling (MLM) and next sentence prediction (NSP) pretraining objectives. It provides a transparent, framework-free reference for how BERT combines token, position, and segment embeddings with a transformer encoder and dual task heads.

## Project Title and Description

The goal is to demonstrate a minimal but complete BERT-like pretraining setup: token and segment embeddings, learned positional embeddings, stacked transformer encoder layers with multi-head self-attention and feed-forward blocks, a pooler over the [CLS] token, an MLM head over masked positions, and an NSP binary classifier. Training is performed jointly on synthetic data that mimics MLM (random token masking) and NSP (next vs random segment pairs).

**Target Audience**: Machine learning practitioners and students who want a from-scratch, NumPy-only implementation of BERT-style pretraining with MLM and NSP, without relying on PyTorch or TensorFlow.

## Features

- **BERT-style embeddings**: Token, learned position, and segment (sentence) embeddings summed per token.
- **Transformer encoder**: Multi-head self-attention and position-wise feed-forward layers with residual connections and layer normalization.
- **Masked language modeling (MLM)**: Random masking of input tokens; model predicts original token ids at masked positions via cross-entropy.
- **Next sentence prediction (NSP)**: Binary classification from [CLS] representation (pooler) to predict whether segment B follows segment A or is random.
- **Joint training**: Combined MLM and NSP loss with configurable weights.
- **Synthetic data**: Generators for MLM batches (single-segment sequences with masking) and NSP batches (pairs with correlated or random second segment).
- **Configuration**: YAML-based config for data, model, and training; logging to a rotating file.
- **CLI**: Run training and evaluation with optional config path and results JSON output.

## Prerequisites

- Python 3.8 or higher
- pip

## Installation

### Step 1: Navigate to project directory

```bash
cd /path/to/Python-ml-projects/bert-from-scratch
```

### Step 2: Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify

```bash
python src/main.py
```

## Configuration

Configuration is read from `config.yaml` (or a path given by `--config`).

### Configuration file structure

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

data:
  batch_size: 8
  sequence_length: 32
  vocab_size: 64
  mask_probability: 0.15
  num_train_batches: 100
  num_test_batches: 20
  random_seed: 42

model:
  dim_model: 32
  num_heads: 4
  dim_ff: 64
  num_layers: 2

training:
  epochs: 5
  learning_rate: 0.001
  mlm_loss_weight: 1.0
  nsp_loss_weight: 1.0
```

### Environment variables

Copy `.env.example` to `.env` for optional overrides (e.g. random seed). No secrets or external APIs are required.

## Usage

### Basic run

```bash
python src/main.py
```

### Custom config

```bash
python src/main.py --config path/to/config.yaml
```

### Save results

```bash
python src/main.py --output results.json
```

The script prints final train/test MLM loss, test NSP loss, and test NSP accuracy, and can write them to a JSON file.

## Project structure

```text
bert-from-scratch/
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

- `src/main.py`: BERT embeddings, encoder, pooler, MLM/NSP heads, data generators, training loop, and CLI.
- `config.yaml`: Data, model, and training hyperparameters.
- `tests/test_main.py`: Unit tests for batch creation, model forward/backward, and losses.
- `docs/API.md`: API documentation for main classes and functions.

## Testing

Run tests:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Troubleshooting

### Training is slow

The implementation uses NumPy only and runs on CPU. Reduce `num_train_batches`, `epochs`, or increase `batch_size` to shorten runs.

### Loss is NaN

Lower `learning_rate`; ensure `vocab_size` and `sequence_length` are consistent with data generators.

### NSP accuracy near 0.5

Increase `epochs` or model size (`dim_model`, `num_layers`); check that NSP data generator produces distinguishable "next" vs "random" segments (see `create_nsp_pair_batch`).

## Contributing

1. Create a feature branch.
2. Follow PEP 8 and project standards.
3. Add or update tests for new behavior.
4. Ensure tests pass before submitting a pull request.

## License

This project is part of the Python ML Projects collection. See the main repository license for details.
