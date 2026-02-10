# U-Net Semantic Segmentation

A Python implementation of semantic segmentation using the U-Net architecture with an encoder-decoder structure and skip connections. The encoder downsamples the input through conv blocks and max pooling; the decoder upsamples with skip connections from the encoder to restore spatial resolution; the final layer outputs per-pixel class logits.

## Project Title and Description

This project provides a self-contained U-Net for semantic segmentation. It addresses the problem of assigning a class label to each pixel in an image. The encoder captures multi-scale features; skip connections pass high-resolution encoder features to the decoder to preserve fine spatial detail; the decoder produces a dense prediction map. The implementation is configurable via YAML and can be trained on custom or synthetic data.

**Target Audience**: Developers and researchers working on image segmentation, students learning U-Net, and engineers integrating segmentation into pipelines.

## Features

- DoubleConv blocks: two 3x3 convolutions with BatchNorm and ReLU
- Encoder: repeated EncoderBlock (DoubleConv then MaxPool) with skip outputs
- Bottleneck: DoubleConv at the lowest resolution
- Decoder: ConvTranspose2d upsample, concatenation with encoder skip, then DoubleConv
- Skip connections: encoder feature maps concatenated with decoder at each level
- Final 1x1 convolution to num_classes for per-pixel logits
- Cross-entropy segmentation loss with optional ignore index
- Synthetic data generation for training without external datasets
- Config-driven architecture and training (YAML)
- Logging to file and console

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for faster training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/unet-semantic-segmentation
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
  in_channels: 3
  num_classes: 2
  base_channels: 64
  depth: 4

training:
  epochs: 20
  learning_rate: 0.001
  batch_size: 8
  loss: "cross_entropy"

data:
  random_seed: 42
  num_train: 400
  num_val: 100
  image_height: 128
  image_width: 128
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set:

- `RANDOM_SEED`: Integer seed for reproducibility
- `DATA_ROOT`: Path to dataset root (optional; script runs with synthetic data if not provided)

## Usage

### Command-Line

Train with default config:

```bash
python src/main.py
```

Train with custom config path:

```bash
python src/main.py --config config.yaml
```

### Programmatic Usage

```python
from src.main import UNet, segmentation_loss, generate_synthetic_batch

model = UNet(
    in_channels=3,
    num_classes=2,
    base_channels=64,
    depth=4,
)
images = ...  # (N, 3, H, W)
logits = model(images)  # (N, num_classes, H, W)
predictions = logits.argmax(dim=1)  # (N, H, W)
```

### Common Use Cases

- Train on synthetic data to verify the pipeline
- Replace synthetic data with a custom DataLoader and label maps
- Use output logits or argmax for downstream tasks or evaluation metrics

## Project Structure

```
unet-semantic-segmentation/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/
│   └── main.py          # U-Net, loss, synthetic data, training
├── tests/
│   └── test_main.py
├── docs/
│   └── API.md
└── logs/
    └── .gitkeep
```

- `src/main.py`: DoubleConv, EncoderBlock, DecoderBlock, UNet, segmentation_loss, synthetic batch, training loop
- `config.yaml`: Model and training hyperparameters, logging, data settings
- `tests/test_main.py`: Unit tests for model and helpers
- `docs/API.md`: API reference for public classes and functions

## Testing

Run tests:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Troubleshooting

- **CUDA out of memory**: Reduce `batch_size` or `image_height`/`image_width` in config.
- **Configuration file not found**: Run from project root or pass the correct `--config` path.
- **Loss NaN**: Lower learning rate or check that labels are in [0, num_classes-1].
- **Spatial size mismatch**: Ensure input height and width are divisible by 2^depth for aligned skip connections.

## Contributing

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Follow PEP 8 and the project docstring/type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request with a clear description.

## License

See repository license.
