# Neural Style Transfer

This project implements neural style transfer with content and style loss optimization. A small feature extractor (Conv2D + ReLU) in NumPy provides feature maps; content loss is the L2 distance between features of the generated and content images, and style loss is the L2 distance between Gram matrices of the generated and style images. The generated image is optimized via gradient descent to minimize a weighted sum of both losses.

### Description

Style transfer produces an image that preserves the content of one image and the style (texture, color statistics) of another. This implementation uses a single-layer feature extractor and explicit content and style loss terms, following the classic Gatys et al. formulation.

**Target audience**: Developers and students learning style transfer and loss-based image optimization.

### Features

- **Content loss**: MSE between the feature map of the generated image and the content image (same feature extractor).
- **Style loss**: MSE between the Gram matrix of the generated image features and the style image features (captures feature correlations).
- **Feature extractor**: One Conv2D layer plus ReLU; fixed weights (random init). Input and output are small grayscale images (e.g. 8x8).
- **Optimization**: Gradient descent on pixel values; gradients flow through the feature extractor. Total loss = content_weight * content_loss + style_weight * style_loss.
- **Data**: Content and style images from sklearn digits or synthetic 8x8 arrays in [-1, 1].
- **Config and CLI**: YAML config for steps, weights, learning rate, image size; optional JSON output.

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd Python-ml-projects/neural-style-transfer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Default: `config.yaml`

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

style_transfer:
  num_steps: 200
  content_weight: 1.0
  style_weight: 1000.0
  learning_rate: 1.0
  extractor_channels: 16
  image_size: 8
  content_index: 0
  style_index: 1
  random_seed: 0
```

- **content_weight**, **style_weight**: Balance between preserving content and matching style.
- **content_index**, **style_index**: Indices into the digit dataset for content and style images.

### Usage

```bash
python src/main.py
python src/main.py --config path/to/config.yaml --output results.json
```

Output includes final content loss, style loss, total loss, and generated image shape.

### Project structure

```
neural-style-transfer/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/main.py
├── tests/test_main.py
├── docs/API.md
└── logs/.gitkeep
```

### Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

### Troubleshooting

- **Style dominates**: Reduce style_weight or increase content_weight.
- **Content dominates**: Increase style_weight.
- **Instability**: Lower learning_rate or reduce num_steps.

### License

Part of Python ML Projects; see repository license.
