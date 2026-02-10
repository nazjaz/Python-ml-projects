# YOLO Object Detection

A Python implementation of object detection using a YOLO-style architecture with bounding box regression and class prediction. The model uses a convolutional backbone for feature extraction and a detection head that predicts per grid cell bounding box coordinates (center and size), objectness scores, and class logits. Training uses a combined loss: coordinate regression (MSE), objectness (binary cross-entropy), and class cross-entropy. Inference supports decoding raw outputs to boxes and non-maximum suppression.

## Project Title and Description

This project provides a self-contained YOLO-style object detector implemented in PyTorch. It solves the problem of detecting multiple objects in an image and assigning each a bounding box and class label. The architecture divides the image into a grid; each cell predicts several boxes with regression parameters (tx, ty, tw, th), an objectness score, and class probabilities. Bounding box regression converts these into image coordinates; class prediction is performed via a linear layer per anchor. The implementation is configurable via YAML and can be trained on custom or synthetic data.

**Target Audience**: Developers and researchers working on object detection, students learning YOLO-style architectures, and engineers integrating detection into pipelines.

## Features

- Darknet-style convolutional backbone (Conv2d, BatchNorm, LeakyReLU blocks)
- YOLO detection head with configurable grid size, anchors per cell, and number of classes
- Bounding box regression: center offsets (sigmoid) and log-scale width/height (exp) relative to grid cells
- Class prediction via linear layer with softmax over classes
- Combined YOLO loss: coordinate MSE, objectness BCE, and class cross-entropy with configurable weights
- Decoding of raw predictions to normalized xyxy boxes, objectness, and class scores
- Non-maximum suppression (NMS) for post-processing detections
- Synthetic data generation for training and testing without external datasets
- Config-driven architecture and training (YAML)
- Logging to file and console

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU for faster training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/yolo-object-detection
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
  input_height: 416
  input_width: 416
  grid_size: 13
  num_anchors: 5
  num_classes: 20
  backbone_channels: [32, 64, 128, 256, 512]
  backbone_blocks: [1, 2, 2, 2]

training:
  epochs: 30
  learning_rate: 0.001
  batch_size: 8
  lambda_coord: 5.0
  lambda_noobj: 0.5
  confidence_threshold: 0.5
  nms_threshold: 0.45

data:
  random_seed: 42
  num_train: 500
  num_val: 100
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set:

- `RANDOM_SEED`: Integer seed for reproducibility
- `DATA_ROOT`: Path to dataset root (optional; script runs with synthetic data if not provided)

No API keys or credentials are required for synthetic training.

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
from src.main import (
    YOLONet,
    decode_predictions,
    yolo_loss,
    non_max_suppression,
    generate_synthetic_batch,
)

model = YOLONet(
    in_channels=3,
    grid_size=13,
    num_anchors=5,
    num_classes=20,
)
images = ...  # (N, 3, 416, 416)
predictions = model(images)
boxes, objectness, class_scores = decode_predictions(
    predictions, grid_size=13, image_size=(416, 416)
)
keep_indices = non_max_suppression(boxes[0:1], objectness[0:1], iou_threshold=0.45)
```

### Common Use Cases

- Train a small detector on synthetic data to verify the pipeline
- Replace synthetic data with a custom DataLoader and target format (S, S, B, 5+C)
- Export decoded boxes and class IDs after NMS for downstream tasks

## Project Structure

```
yolo-object-detection/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/
│   └── main.py          # YOLO model, loss, decoding, NMS, training
├── tests/
│   └── test_main.py
├── docs/
│   └── API.md
└── logs/
    └── .gitkeep
```

- `src/main.py`: Backbone, YOLODetectionHead, YOLONet, decode_predictions, yolo_loss, non_max_suppression, synthetic batch generation, and training loop
- `config.yaml`: Model dimensions, training hyperparameters, logging, and data settings
- `tests/test_main.py`: Unit tests for model, decoding, loss, and NMS
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

Tests cover: backbone and head output shapes, YOLONet forward, decode_predictions shapes and value ranges, yolo_loss computation, NMS behavior, and synthetic batch generation.

## Troubleshooting

- **CUDA out of memory**: Reduce `batch_size` in `config.yaml` or use a smaller input size.
- **Configuration file not found**: Ensure you run from the project root or pass the correct `--config` path.
- **Loss NaN**: Lower learning rate or check that targets use the same grid size and num_anchors as the model.
- **No detections after NMS**: Lower `confidence_threshold` or `nms_threshold` in config; ensure objectness and class scores are trained.

## Contributing

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Follow PEP 8 and the project docstring/type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request with a clear description.

## License

See repository license.
