# Neural Network from Scratch with Backpropagation

A complete implementation of a feedforward neural network built from scratch using only NumPy. Supports multiple hidden layers, configurable activation functions, weight initialization strategies, and training via mini-batch gradient descent with full backpropagation.

## Project Title and Description

This project implements a neural network framework without relying on deep learning libraries such as TensorFlow or PyTorch. Every component - forward propagation, backpropagation, activation functions, weight initialization, and loss computation - is implemented from first principles using only NumPy for matrix operations.

The primary goal is to provide a transparent, educational, and production-quality implementation that demonstrates how neural networks learn through gradient-based optimization. This is useful for understanding the internals of deep learning, for research purposes where full control over the training process is needed, and as a reference implementation for custom model development.

**Target Audience**: Machine learning students studying neural network fundamentals, researchers implementing custom architectures, and developers building ML frameworks from scratch.

## Features

- Feedforward neural network with arbitrary number of hidden layers
- Six activation functions: sigmoid, tanh, ReLU, Leaky ReLU, softmax, linear
- Three weight initialization strategies: Xavier (Glorot), He, random normal
- Three loss functions: cross-entropy, binary cross-entropy, mean squared error
- Mini-batch stochastic gradient descent with configurable batch size
- Full backpropagation with chain rule gradient computation
- Numerically stable softmax and sigmoid implementations
- Combined softmax + cross-entropy gradient optimization
- One-hot encoding for multi-class classification labels
- Feature normalization using training set statistics
- Training history tracking with per-epoch loss and accuracy
- Validation monitoring during training
- Configurable architecture via YAML file
- Command-line interface for running experiments
- Synthetic data generation for classification and regression
- Network architecture summary with parameter counts
- Comprehensive logging with rotating file handler

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/neural-network-backpropagation
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
python src/main.py --task classification
```

## Configuration

### Configuration File Structure

The project is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

data:
  n_samples: 1000
  n_features: 20
  n_classes: 3
  random_seed: 42

network:
  hidden_layers: [128, 64, 32]
  hidden_activations: ["relu", "relu", "relu"]
  output_activation: "softmax"
  weight_init: "he"

training:
  epochs: 100
  learning_rate: 0.01
  batch_size: 32
```

### Configuration Parameters

- `logging.level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to the rotating log file
- `data.n_samples`: Total number of samples in synthetic dataset
- `data.n_features`: Number of input features
- `data.n_classes`: Number of target classes (classification only)
- `data.random_seed`: Seed for reproducible data generation
- `network.hidden_layers`: List of integers specifying hidden layer sizes
- `network.hidden_activations`: Activation function for each hidden layer
- `network.output_activation`: Activation function for the output layer
- `network.weight_init`: Weight initialization strategy ("xavier", "he", "random_normal")
- `training.epochs`: Number of training passes over the dataset
- `training.learning_rate`: Step size for gradient descent
- `training.batch_size`: Number of samples per mini-batch

### Environment Variables

Copy `.env.example` to `.env` and configure:

- `LOG_LEVEL`: Override logging level
- `RANDOM_SEED`: Override random seed for reproducibility

## Usage

### Command-Line Interface

#### Run Classification Experiment

```bash
# Using default configuration
python src/main.py --task classification

# Using custom configuration
python src/main.py --task classification --config config.yaml

# Save results to JSON
python src/main.py --task classification --output results.json
```

#### Run Regression Experiment

```bash
python src/main.py --task regression
python src/main.py --task regression --output regression_results.json
```

### Programmatic Usage

#### Building and Training a Classification Network

```python
import numpy as np
from src.main import NeuralNetwork, generate_classification_data, normalize_features

x_train, x_test, y_train, y_test = generate_classification_data(
    n_samples=1000, n_features=20, n_classes=3
)
x_train, x_test = normalize_features(x_train, x_test)

network = NeuralNetwork(
    layer_sizes=[20, 128, 64, 32, 3],
    activations=["relu", "relu", "relu", "softmax"],
    loss="cross_entropy",
    weight_init="he",
)

print(network.summary())

history = network.train(
    x_train, y_train,
    epochs=100,
    learning_rate=0.01,
    batch_size=32,
    validation_data=(x_test, y_test),
)

results = network.evaluate(x_test, y_test)
print(f"Test Loss: {results['loss']:.6f}")
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

#### Building a Regression Network

```python
import numpy as np
from src.main import NeuralNetwork, generate_regression_data, normalize_features

x_train, x_test, y_train, y_test = generate_regression_data(
    n_samples=500, n_features=10
)
x_train, x_test = normalize_features(x_train, x_test)

network = NeuralNetwork(
    layer_sizes=[10, 64, 32, 1],
    activations=["relu", "relu", "linear"],
    loss="mse",
    weight_init="he",
)

history = network.train(
    x_train, y_train,
    epochs=200,
    learning_rate=0.001,
    batch_size=32,
)

predictions = network.predict(x_test)
```

#### Using Different Activation Functions

```python
from src.main import NeuralNetwork

# Network with mixed activations
network = NeuralNetwork(
    layer_sizes=[20, 64, 32, 16, 3],
    activations=["tanh", "relu", "leaky_relu", "softmax"],
    loss="cross_entropy",
    weight_init="xavier",
)
```

#### Binary Classification

```python
from src.main import NeuralNetwork

network = NeuralNetwork(
    layer_sizes=[10, 32, 16, 1],
    activations=["relu", "relu", "sigmoid"],
    loss="binary_cross_entropy",
)
```

### Common Use Cases

1. **Learning Neural Network Internals**: Step through forward and backward passes to understand gradient flow
2. **Custom Architecture Experiments**: Test different layer configurations and activation combinations
3. **Benchmarking**: Compare from-scratch implementation against framework-based models
4. **Research Prototyping**: Modify internal components (loss functions, optimizers) without framework constraints
5. **Teaching Material**: Use as reference code for neural network courses and workshops

## Project Structure

```
neural-network-backpropagation/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies with versions
├── config.yaml              # Network and training configuration
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py             # Neural network implementation
├── tests/
│   └── test_main.py        # Comprehensive unit tests
├── docs/
│   └── (reserved)          # Additional documentation
└── logs/
    └── .gitkeep            # Keep logs directory in version control
```

### File Descriptions

- `src/main.py`: Complete neural network implementation containing:
  - `ActivationFunction`: Six activation functions with analytical derivatives
  - `WeightInitializer`: Xavier, He, and random normal initialization
  - `DenseLayer`: Fully connected layer with forward/backward pass
  - `NeuralNetwork`: Multi-layer network with training and evaluation
  - `NetworkRunner`: Configuration-driven experiment orchestrator
  - `generate_classification_data()`: Synthetic classification data generator
  - `generate_regression_data()`: Synthetic regression data generator
  - `normalize_features()`: Z-score feature normalization

- `tests/test_main.py`: Test suite covering:
  - Activation function outputs, derivatives, and edge cases
  - Weight initializer shapes and variance properties
  - Dense layer forward/backward pass correctness
  - Network creation, training, and evaluation
  - Numerical gradient verification against analytical gradients
  - Data generation and normalization utilities

- `config.yaml`: YAML configuration for data, network architecture, and training

## Testing

### Run All Tests

```bash
pytest tests/test_main.py -v
```

### Run Tests with Coverage

```bash
pytest tests/test_main.py --cov=src --cov-report=html
```

### Run Specific Test Class

```bash
pytest tests/test_main.py::TestActivationFunction -v
pytest tests/test_main.py::TestNeuralNetwork -v
pytest tests/test_main.py::TestGradientNumericalVerification -v
```

### Test Coverage Information

The test suite includes:
- Activation function output ranges and boundary values
- Numerical stability tests for extreme inputs
- Weight initialization variance verification
- Layer dimension validation and error handling
- End-to-end training convergence tests
- Gradient accuracy verification using finite differences
- Data generation reproducibility tests
- Normalization correctness tests

Current test coverage: >90% of code paths

## Mathematical Background

### Forward Propagation

For each layer l with weights W, biases b, and activation function g:

```
z^(l) = a^(l-1) * W^(l) + b^(l)
a^(l) = g(z^(l))
```

### Backpropagation

The gradient of the loss L with respect to weights in layer l:

```
delta^(l) = upstream_gradient * g'(z^(l))
dL/dW^(l) = (a^(l-1))^T * delta^(l) / n
dL/db^(l) = mean(delta^(l), axis=0)
gradient_to_prev = delta^(l) * (W^(l))^T
```

### Softmax + Cross-Entropy Optimization

When using softmax activation with cross-entropy loss, the combined gradient simplifies to:

```
dL/dz = softmax(z) - y_true
```

This avoids computing the full Jacobian of softmax, improving both numerical stability and computational efficiency.

### Weight Initialization

- **Xavier**: W ~ N(0, 2/(fan_in + fan_out)) - optimal for sigmoid and tanh
- **He**: W ~ N(0, 2/fan_in) - optimal for ReLU and variants

## Troubleshooting

### Common Issues and Solutions

#### Issue: Loss Not Decreasing

**Possible Causes**: Learning rate too high or too low, poor weight initialization, vanishing/exploding gradients.

**Solution**: Try reducing learning rate by factor of 10. Ensure features are normalized. Use He initialization with ReLU, Xavier with tanh/sigmoid.

#### Issue: NaN Values During Training

**Possible Causes**: Learning rate too high causing gradient explosion, unnormalized input features.

**Solution**: Reduce learning rate. Normalize input features using `normalize_features()`. Check for extreme values in input data.

#### Issue: Low Accuracy on Classification

**Possible Causes**: Insufficient model capacity, too few epochs, learning rate mismatch.

**Solution**: Add more hidden layers or increase layer sizes. Train for more epochs. Experiment with different learning rates.

#### Issue: Shape Mismatch Error

**Error**: `ValueError: Input dimension X does not match layer input size Y`

**Solution**: Verify that input data dimensions match the first value in `layer_sizes`. Ensure all layer sizes are consistent.

### Error Message Explanations

- **"at least an input and output layer"**: `layer_sizes` must have at least 2 elements
- **"Expected N activations"**: Number of activation functions must equal number of layer transitions
- **"Unknown activation"**: Activation name not in registry (use sigmoid, tanh, relu, leaky_relu, softmax, linear)
- **"Forward pass must be called"**: Attempted backward pass without a preceding forward pass

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment and install dependencies
3. Create a feature branch: `git checkout -b feature/new-optimizer`

### Code Style Guidelines

- Follow PEP 8 with 88 character line length
- Type hints required for all function signatures
- Google-style docstrings for all public functions and classes
- Verify gradients using numerical differentiation

### Pull Request Process

1. Ensure all tests pass: `pytest tests/test_main.py -v`
2. Check code coverage: `pytest --cov=src`
3. Update documentation for new features
4. Submit pull request with a clear description

## License

This project is part of the Python ML Projects collection. Please refer to the main repository license.

## Additional Resources

- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)
- [Neural Network Initialization](https://proceedings.mlr.press/v9/glorot10a.html)
- [He Initialization Paper](https://arxiv.org/abs/1502.01852)
- [NumPy Documentation](https://numpy.org/doc/stable/)
