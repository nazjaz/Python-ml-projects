# Contrastive Learning with SimCLR

This project implements SimCLR-style contrastive learning for self-supervised representation learning: two augmented views per sample, a shared encoder and projection head, and the NT-Xent (InfoNCE) contrastive loss. Implemented in NumPy; uses a small input space (e.g. flattened digits) with simple augmentations (noise and scaling).

### Description

SimCLR learns representations by contrasting positive pairs (two augmentations of the same sample) against negatives (other samples in the batch). No labels are used; the encoder can be used for downstream tasks. This code provides a minimal but complete pipeline: augmentations, encoder, projection head, and NT-Xent loss with temperature scaling.

**Target audience**: Developers and students learning contrastive and self-supervised learning.

### Features

- **Two-view augmentation**: Each sample is augmented twice (random scale and Gaussian noise) to form positive pairs.
- **Encoder and projection head**: Shared encoder (MLP) maps input to representation; projection head (MLP) maps to projection space for the contrastive loss.
- **NT-Xent (InfoNCE) loss**: Contrastive loss with temperature scaling; L2-normalized projections; gradient through normalization.
- **Self-supervised**: No labels; training minimizes the contrastive loss on augmented views.
- **Config and CLI**: YAML config for dimensions, epochs, batch size, temperature, augmentation strength; optional JSON output.

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd Python-ml-projects/simclr-contrastive-learning
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

simclr:
  repr_dim: 128
  proj_dim: 64
  encoder_hidden: 256
  proj_hidden: 128
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  temperature: 0.5
  noise_std: 0.1
  scale_low: 0.8
  scale_high: 1.2
  max_samples: null
  random_seed: 0
```

- **temperature**: Scaling in NT-Xent (e.g. 0.5).
- **noise_std**, **scale_low**, **scale_high**: Augmentation parameters for the two views.

### Usage

```bash
python src/main.py
python src/main.py --config path/to/config.yaml --output results.json
```

Output includes final contrastive loss and data/config summary.

### Project structure

```
simclr-contrastive-learning/
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

- **Loss not decreasing**: Lower learning rate; increase batch size; adjust temperature.
- **ImportError for sklearn**: Install scikit-learn for digits; otherwise synthetic data is used.

### License

Part of Python ML Projects; see repository license.
