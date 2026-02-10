## Vision Transformer (ViT) from Scratch for Image Classification

This project implements a Vision Transformer (ViT) model using only NumPy. The
implementation covers patch embedding, class token and positional embeddings,
stacked transformer encoder blocks with multi-head self-attention and
feed-forward networks, and a classification head over the [CLS] token.

### Project Title and Description

The project demonstrates how a ViT-style architecture can be built from first
principles without deep learning frameworks. Images are split into fixed-size
patches, linearly projected to token embeddings, combined with a learned class
token and positional embeddings, passed through transformer encoder layers,
and classified into discrete categories.

**Target Audience**: Practitioners and students who want a transparent,
framework-free reference implementation of Vision Transformers for image
classification.

### Features

- **Patch embedding**: Images are decomposed into non-overlapping patches and
  projected into an embedding space.
- **Class token and positional embeddings**: A learned [CLS] token and learned
  positional embeddings are added to patch tokens.
- **Transformer encoder stack**: Multi-head self-attention, position-wise
  feed-forward blocks, residual connections, and layer normalization.
- **Classification head**: Linear projection from [CLS] representation to
  class logits.
- **Synthetic data generator**: Random RGB images and labels for quick
  experimentation.
- **Configuration-driven experiments**: `config.yaml` controls data, model, and
  training hyperparameters.
- **Logging and CLI**: Rotating log file and a command-line interface to run
  training and evaluation.

### Prerequisites

- Python 3.8 or higher
- `pip`

### Installation

#### Step 1: Navigate to project directory

```bash
cd /path/to/Python-ml-projects/vit-from-scratch
```

#### Step 2: Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Verify installation

```bash
python src/main.py
```

### Configuration

The project is configured via `config.yaml`.

#### Configuration file structure

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

data:
  image_size: 32
  in_channels: 3
  num_classes: 10
  n_train: 512
  n_test: 128
  random_seed: 42

model:
  dim_model: 64
  patch_size: 4
  num_heads: 4
  dim_ff: 128
  num_layers: 2

training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
```

### Environment variables

Copy `.env.example` to `.env` and set optional overrides such as random seed.
No external services or secrets are required.

### Usage

#### Basic usage

```bash
python src/main.py
```

#### With custom config

```bash
python src/main.py --config path/to/config.yaml
```

#### Save results to file

```bash
python src/main.py --output results.json
```

The script logs training loss and prints final train loss and test accuracy,
optionally writing results to a JSON file.

### Project structure

```text
vit-from-scratch/
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

- `src/main.py`: ViT implementation, synthetic data generation, training
  runner, and CLI entry point.
- `config.yaml`: Data, model, and training configuration.
- `tests/test_main.py`: Unit tests for patch embedding, model forward/backward,
  losses, and the runner.
- `docs/API.md`: API documentation for the main classes and functions.

### Testing

Run tests:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Troubleshooting

#### Training is slow

Reduce `n_train`, `epochs`, or increase `batch_size` in `config.yaml`. The
implementation uses NumPy only and runs on CPU.

#### Accuracy is low

- Increase `dim_model`, `dim_ff`, or `num_layers`.
- Train for more epochs.
- Verify that `image_size` and `patch_size` are compatible.

#### Numerical issues

- Lower `learning_rate` if you observe unstable loss values.

### Contributing

1. Create a feature branch.
2. Follow PEP 8 and the shared project standards.
3. Add or update tests for new behavior.
4. Ensure all tests pass before submitting a pull request.

### License

This project is part of the Python ML Projects collection. Refer to the main
repository license for details.

