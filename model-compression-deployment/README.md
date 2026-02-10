# Model Compression for Efficient Deployment

A Python implementation of model compression using quantization, pruning, and knowledge distillation. The script trains a teacher MLP, optionally applies dynamic quantization and L1 pruning, and trains a smaller student via distillation for efficient deployment.

## Project Title and Description

This project demonstrates three compression techniques: (1) post-training dynamic quantization to int8 for linear layers, reducing memory and compute; (2) magnitude-based pruning of a fraction of weights; (3) knowledge distillation from a teacher to a smaller student using soft targets and temperature scaling. Together they support smaller, faster models suitable for deployment.

**Target Audience**: Engineers and researchers deploying models under resource constraints and students learning compression techniques.

## Features

- **Quantization**: Dynamic quantization (torch.quantization.quantize_dynamic) for Linear layers
- **Pruning**: L1 unstructured pruning with configurable amount; optional permanent mask
- **Distillation**: Temperature-scaled soft targets plus hard label cross-entropy; configurable alpha
- MLP teacher and student with configurable hidden layers
- Synthetic data and training loop; accuracy reporting for teacher, quantized, pruned, and student
- Config-driven via YAML

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA for faster training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/model-compression-deployment
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

Configure via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  input_dim: 16
  teacher_hidden: [64, 64]
  student_hidden: [32]
  num_classes: 3

compression:
  quantization:
    enabled: true
    dtype: "qint8"
  pruning:
    enabled: true
    amount: 0.3
  distillation:
    temperature: 4.0
    alpha: 0.7

training:
  teacher_epochs: 20
  student_epochs: 25
  batch_size: 32
  learning_rate: 0.001

data:
  random_seed: 42
  num_train: 600
  num_val: 150
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set `RANDOM_SEED`.

## Usage

### Command-Line

```bash
python src/main.py
python src/main.py --config config.yaml
```

### Programmatic Usage

```python
from src.main import (
    MLP,
    apply_quantization,
    apply_pruning,
    distillation_loss,
    generate_synthetic_data,
)

teacher = MLP(16, [64, 64], 3)
teacher_quant = apply_quantization(teacher, dtype="qint8")
apply_pruning(teacher, amount=0.3)
# Train student with distillation_loss(student_logits, teacher_logits, labels, T, alpha)
```

## Project Structure

```
model-compression-deployment/
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

- **Quantization on CUDA**: Dynamic quantization is often CPU-oriented; run quantized model on CPU if needed.
- **Pruning accuracy drop**: Reduce pruning amount.
- **Student underperforms**: Increase student capacity, temperature, or distillation epochs.

## Contributing

1. Create a virtual environment and install dependencies.
2. Follow PEP 8 and project docstring and type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request.

## License

See repository license.
