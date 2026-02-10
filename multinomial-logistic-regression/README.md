# Multinomial Logistic Regression for Multi-Class Classification

A Python implementation of multinomial logistic regression from scratch for multi-class classification with softmax activation and cross-entropy cost function optimization. This is the twenty-fourth project in the ML learning series, focusing on understanding multinomial logistic regression and multi-class classification algorithms.

## Project Title and Description

The Multinomial Logistic Regression tool provides a complete implementation of multinomial logistic regression from scratch for multi-class classification, including softmax activation function and cross-entropy cost function optimization using gradient descent. It helps users understand how multinomial logistic regression works and how it extends binary logistic regression to handle multiple classes.

This tool solves the problem of learning multi-class classification fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates softmax activation, cross-entropy cost function, and gradient descent optimization for multi-class classification tasks.

**Target Audience**: Beginners learning machine learning, students studying classification algorithms, and anyone who needs to understand multinomial logistic regression and multi-class classification from scratch.

## Features

- Multinomial logistic regression implementation from scratch
- Softmax activation function
- Cross-entropy cost function
- Gradient descent optimization
- Multi-class classification predictions
- Probability predictions for all classes
- Feature scaling (standardization)
- Cost function tracking
- Convergence detection
- Training history visualization
- Accuracy score calculation
- Support for multiple features
- Optional intercept term
- One-hot encoding for labels
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas DataFrames)

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/multinomial-logistic-regression
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
python src/main.py --input sample.csv --target label --plot
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  learning_rate: 0.01
  max_iterations: 1000
  tolerance: 1e-6
  fit_intercept: true
  scale_features: true
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.learning_rate`: Initial learning rate (default: 0.01)
- `model.max_iterations`: Maximum number of iterations (default: 1000)
- `model.tolerance`: Convergence tolerance (default: 1e-6)
- `model.fit_intercept`: Whether to fit intercept term (default: true)
- `model.scale_features`: Whether to scale features (default: true)

## Usage

### Basic Usage

```python
from src.main import MultinomialLogisticRegression
import numpy as np

# Create sample data with multiple classes
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 1, 1, 2, 2])

# Initialize and fit model
model = MultinomialLogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Get probabilities for all classes
probabilities = model.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")

# Calculate accuracy
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Multiple Features

```python
from src.main import MultinomialLogisticRegression
import numpy as np

# Multiple features
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 1, 1, 2, 2])

model = MultinomialLogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

print(f"Classes: {model.classes}")
print(f"Intercept (per class): {model.intercept}")
print(f"Weights shape: {model.weights.shape}")
```

### With Pandas DataFrame

```python
from src.main import MultinomialLogisticRegression
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

model = MultinomialLogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

### Command-Line Usage

Basic training:

```bash
python src/main.py --input data.csv --target label
```

Without feature scaling:

```bash
python src/main.py --input data.csv --target label --no-scale
```

Plot training history:

```bash
python src/main.py --input data.csv --target label --plot
```

Save predictions with probabilities:

```bash
python src/main.py --input data.csv --target label --output predictions.csv
```

### Complete Example

```python
from src.main import MultinomialLogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with 3 classes
np.random.seed(42)
n_samples = 150
X = np.random.randn(n_samples, 2) * 2

# Create 3 distinct clusters
y = np.zeros(n_samples, dtype=int)
y[50:100] = 1
y[100:] = 2

# Initialize and fit model
model = MultinomialLogisticRegression(
    learning_rate=0.1,
    max_iterations=1000,
    scale_features=True,
)

model.fit(X, y)

# Print results
print(f"Classes: {model.classes}")
print(f"Intercept (per class): {model.intercept}")
print(f"Weights shape: {model.weights.shape}")
print(f"Accuracy: {model.score(X, y):.4f}")
print(f"Final cost: {model.cost_history[-1]:.6f}")

# Get predictions and probabilities
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Plot training history
model.plot_training_history()

# Plot decision boundaries (for 2D data)
if X.shape[1] == 2:
    plt.figure(figsize=(12, 5))

    # Plot 1: Actual classes
    plt.subplot(1, 2, 1)
    colors = ["blue", "red", "green"]
    for i, cls in enumerate(model.classes):
        mask = y == cls
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[i],
            label=f"Class {cls}",
            alpha=0.6,
        )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Actual Classes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Predicted classes
    plt.subplot(1, 2, 2)
    for i, cls in enumerate(model.classes):
        mask = predictions == cls
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[i],
            label=f"Class {cls}",
            alpha=0.6,
        )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Predicted Classes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

## Project Structure

```
multinomial-logistic-regression/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py              # Main implementation
├── tests/
│   └── test_main.py         # Unit tests
├── docs/
│   └── API.md               # API documentation
└── logs/
    └── .gitkeep             # Log directory
```

### File Descriptions

- `src/main.py`: Core implementation with `MultinomialLogisticRegression` class
- `config.yaml`: Configuration file for model and logging settings
- `tests/test_main.py`: Comprehensive unit tests
- `docs/API.md`: Detailed API documentation
- `logs/`: Directory for application logs

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

Tests cover:
- Softmax activation function
- One-hot encoding
- Model initialization
- Training and fitting
- Probability prediction
- Class prediction
- Accuracy calculation
- Cost history
- Convergence
- Multiple features
- Error handling
- Different input types

## Understanding Multinomial Logistic Regression

### Multinomial Logistic Regression

Multinomial logistic regression extends binary logistic regression to handle multiple classes. It models the probability of each class using the softmax function:

```
P(y=k|x) = exp(z_k) / Σ exp(z_j)
```

Where:
- `z_k = w_k · x + b_k` is the score for class k
- `w_k` is the weight vector for class k
- `b_k` is the intercept for class k
- Output is probability distribution over all classes

### Softmax Activation Function

The softmax function converts raw scores to probabilities:

```
softmax(z)_i = exp(z_i) / Σ exp(z_j)
```

**Properties:**
- Outputs probability distribution (sums to 1)
- All outputs are between 0 and 1
- Smooth and differentiable
- Numerical stability: subtract max before exp

### Cross-Entropy Cost Function

Multinomial logistic regression uses cross-entropy loss:

```
J(θ) = -(1/m) * Σ Σ y_ij * log(h_ij)
```

Where:
- `h_ij` is predicted probability of class j for sample i
- `y_ij` is one-hot encoded true label
- `m` is number of samples

**Properties:**
- Penalizes confident wrong predictions
- Convex function
- Well-suited for probability outputs

### Gradient Descent Update

Gradient for multinomial logistic regression:

```
∇J(θ_k) = (1/m) * X.T @ (h_k - y_k)
```

Weight update for each class:
```
θ_k = θ_k - α * ∇J(θ_k)
```

Where `α` is the learning rate.

## Troubleshooting

### Common Issues

**Issue**: Model not converging

**Solution**: 
- Scale features (enabled by default)
- Reduce learning rate
- Increase max_iterations
- Check data quality

**Issue**: Predictions always same class

**Solution**: 
- Check feature scaling
- Verify data balance
- Increase iterations
- Check for class imbalance

**Issue**: Cost not decreasing

**Solution**: 
- Learning rate too high or too low
- Features need scaling
- Check for numerical issues

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: At least 2 classes required`: Need multiple classes for classification
- `ValueError: Length mismatch`: X and y have different lengths

## Best Practices

1. **Always scale features**: Essential for gradient descent convergence
2. **Use appropriate learning rate**: Start with 0.01, adjust based on cost history
3. **Monitor cost history**: Ensure cost decreases during training
4. **Check class balance**: Imbalanced classes may need special handling
5. **Use probabilities**: `predict_proba()` provides more information than `predict()`
6. **Handle many classes**: Softmax works well even with many classes

## Real-World Applications

- Multi-class classification problems (image classification, text categorization)
- Handwritten digit recognition
- Species classification
- Product categorization
- Sentiment analysis (positive, neutral, negative)
- Educational purposes for learning multi-class classification algorithms

## Contributing

### Development Setup

1. Clone the repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Run tests: `pytest tests/`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Include docstrings for all public functions and classes
- Write tests for all new functionality

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. See LICENSE file in parent directory for details.
