# CNN from Scratch for Image Classification

A Python implementation of a convolutional neural network (CNN) built from scratch using only NumPy. Implements convolution layers, max pooling layers, ReLU activation, and fully connected layers with backpropagation for image classification tasks.

## Project Title and Description

This tool provides a complete CNN implementation from scratch without relying on deep learning frameworks. It demonstrates how convolution extracts spatial features, how max pooling reduces dimensionality, and how backpropagation flows through convolutional and pooling layers. The implementation uses the MNIST handwritten digits dataset for image classification.

**Target Audience**: Machine learning students, developers learning CNNs from first principles, and anyone needing an educational reference implementation without PyTorch or TensorFlow dependencies.

## Features

- Conv2D layer with configurable filters, stride, and padding
- MaxPool2D layer for spatial downsampling
- ReLU activation function
- Flatten and Dense (fully connected) layers
- Softmax output with cross-entropy loss
- Backpropagation through convolution and pooling
- MNIST dataset support (with synthetic fallback)
- Configurable architecture via YAML
- Command-line interface with config and output options
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/cnn-image-classification
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

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

data:
  n_train: 5000
  n_test: 1000
  random_seed: 42

model:
  conv_layers:
    - filters: 32
      kernel_size: 3
      stride: 1
      padding: 1
    - filters: 64
      kernel_size: 3
      stride: 1
      padding: 1
  dense_units: 128

training:
  epochs: 10
  learning_rate: 0.001
  batch_size: 32
```

### Environment Variables

Copy `.env.example` to `.env` and adjust as needed. Optional variables:

- `RANDOM_SEED`: Seed for reproducibility

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

## Project Structure

```
cnn-image-classification/
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

- `src/main.py`: CNN implementation with Conv2D, MaxPool2D, Flatten, Dense layers
- `config.yaml`: Model and training configuration
- `tests/test_main.py`: Unit tests for layers and model

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Troubleshooting

### MNIST Download Fails

If `fetch_openml` fails (e.g., offline), the script automatically falls back to synthetic image data. Training will still run and demonstrate the CNN.

### Slow Training

NumPy-based CNNs are slower than GPU-accelerated frameworks. Reduce `n_train` or `epochs` in config for faster iteration.

### Out of Memory

Reduce `batch_size` and `n_train` in config if you encounter memory issues.

## Contributing

1. Create a branch from main
2. Follow PEP 8 and project code style
3. Add tests for new functionality
4. Submit a pull request with a clear description

## License

See LICENSE file in the repository root.
