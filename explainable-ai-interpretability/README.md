# Explainable AI: Integrated Gradients, LIME, and Attention Visualization

A Python implementation of explainable AI for model interpretability: integrated gradients for input attribution, LIME for local linear explanations, and attention visualization for models that expose attention weights. Includes a simple classifier and an attention-based model for demonstration.

## Project Title and Description

This project provides three interpretability methods in one script. Integrated gradients attribute predictions to input features by integrating gradients along a path from a baseline to the input. LIME approximates the model locally with a weighted linear model to produce feature importance. Attention visualization exposes and formats attention weights from attention-based models for inspection. A small synthetic training loop demonstrates all three on a trained model.

**Target Audience**: Practitioners and students working on explainable AI and model interpretability.

## Features

- **Integrated gradients**: Path-based attribution from baseline to input; configurable steps and baseline
- **LIME**: Perturbed sampling, exponential kernel weighting, weighted linear fit for feature importance
- **Attention visualization**: ModelWithAttention stores last attention; format_attention_for_visualization for logging
- SimpleClassifier (MLP) for IG and LIME; ModelWithAttention for attention demo
- Config-driven via YAML; synthetic data and training for demo

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: CUDA-capable GPU

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/explainable-ai-interpretability
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
  input_dim: 8
  hidden_dim: 32
  num_classes: 2

interpretability:
  integrated_gradients:
    steps: 50
    baseline: "zero"
  lime:
    num_samples: 500
    num_features: 8
    kernel_width: 0.25
  attention:
    num_heads: 2
    embed_dim: 16

data:
  random_seed: 42
  num_train: 400
  num_val: 100

training:
  epochs: 15
  batch_size: 32
  learning_rate: 0.001
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
    SimpleClassifier,
    ModelWithAttention,
    integrated_gradients,
    lime_explain,
    get_attention_weights,
    format_attention_for_visualization,
)

model = SimpleClassifier(8, 32, 2)
x = torch.randn(1, 8)
attr = integrated_gradients(model, x, target_class=0, steps=30)
importance = lime_explain(model, x, num_samples=300, num_features=8)

attn_model = ModelWithAttention(8, 16, 2, 2)
_ = attn_model(x)
weights = get_attention_weights(attn_model)
if weights is not None:
    print(format_attention_for_visualization(weights[0]))
```

## Project Structure

```
explainable-ai-interpretability/
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

- **IG attribution all zeros**: Ensure input requires_grad and model is differentiable; try more steps.
- **LIME importance unstable**: Increase num_samples or adjust kernel_width.
- **No attention weights**: Use ModelWithAttention or a model that sets last_attention.

## Contributing

1. Create a virtual environment and install dependencies.
2. Follow PEP 8 and project docstring and type-hint conventions.
3. Add tests for new public functions and classes.
4. Submit changes via pull request.

## License

See repository license.
