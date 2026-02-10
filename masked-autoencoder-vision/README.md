# Masked Autoencoder for Vision (MAE)

A Python implementation of self-supervised learning with masked autoencoders
(MAE) for vision tasks. Images are split into patches, a large subset of
patches is masked, and the model learns to reconstruct the original pixels of
masked patches, enabling visual representation learning without labels.

## Project Title and Description

This project provides a minimal MAE-style model for vision. It follows the
core idea of MAE: encode only visible patches, use a lightweight decoder with
mask tokens to reconstruct all patches, and optimize reconstruction loss on
masked regions. Synthetic images are used for demonstration.

**Target Audience**: Practitioners and students exploring self-supervised
vision methods and masked autoencoders.

## Features

- Patchify/unpatchify of images into flat patch sequences
- Transformer-based encoder for visible patch tokens
- Decoder with learned mask token and positional embeddings
- Random patch masking with configurable mask ratio
- Reconstruction loss (MSE) computed only on masked patches
- Configurable model and training via YAML

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for faster training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/masked-autoencoder-vision
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows: `venv\Scripts\activate`

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --config config.yaml
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  image_size: 32
  patch_size: 4
  in_channels: 3
  embed_dim: 64
  encoder_layers: 4
  decoder_layers: 2
  mask_ratio: 0.75

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001

data:
  random_seed: 42
  num_train: 1000
  num_val: 200
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set:

- `RANDOM_SEED`: Integer seed for reproducibility

## Usage

### Command-Line

```bash
python src/main.py
python src/main.py --config config.yaml
```

### Programmatic Usage

```python
from src.main import MaskedAutoencoder, generate_synthetic_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskedAutoencoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    encoder_layers=4,
    decoder_layers=2,
    mask_ratio=0.75,
).to(device)
images = generate_synthetic_images(8, 3, 32, 32, device=device, seed=42)
loss, recon = model(images)
loss.backward()
```

## Project Structure

```text
masked-autoencoder-vision/
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

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Troubleshooting

- **Loss not decreasing**: Increase model size or number of epochs, or reduce
  mask ratio.
- **High memory usage**: Reduce image size, batch size, or number of encoder
  layers.

## Contributing

1. Create a virtual environment and install dependencies.
2. Follow PEP 8 and project docstring and type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request.

## License

See repository license.

