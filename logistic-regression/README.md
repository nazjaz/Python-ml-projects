# Logistic Regression for Binary Classification

A Python implementation of logistic regression from scratch for binary classification with sigmoid activation and cost function optimization. This is the twenty-third project in the ML learning series, focusing on understanding logistic regression and binary classification algorithms.

## Project Title and Description

The Logistic Regression tool provides a complete implementation of logistic regression from scratch for binary classification, including sigmoid activation function and logistic cost function optimization using gradient descent. It helps users understand how logistic regression works at a fundamental level and how it differs from linear regression.

This tool solves the problem of learning binary classification fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates sigmoid activation, logistic cost function, and gradient descent optimization for classification tasks.

**Target Audience**: Beginners learning machine learning, students studying classification algorithms, and anyone who needs to understand logistic regression and binary classification from scratch.

## Features

- Logistic regression implementation from scratch
- Sigmoid activation function
- Logistic cost function (log loss)
- Gradient descent optimization
- Binary classification predictions
- Probability predictions
- Feature scaling (standardization)
- Cost function tracking
- Convergence detection
- Training history visualization
- Accuracy score calculation
- Support for multiple features
- Optional intercept term
- Configurable decision threshold
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
cd /path/to/Python-ml-projects/logistic-regression
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
from src.main import LogisticRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Initialize and fit model
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Get probabilities
probabilities = model.predict_proba(X)
print(f"Probabilities: {probabilities}")

# Calculate accuracy
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Multiple Features

```python
from src.main import LogisticRegression
import numpy as np

# Multiple features
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

print(f"Intercept: {model.intercept:.4f}")
print(f"Weights: {model.weights}")
```

### With Custom Threshold

```python
from src.main import LogisticRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

# Lower threshold (more positive predictions)
predictions_low = model.predict(X, threshold=0.3)

# Higher threshold (fewer positive predictions)
predictions_high = model.predict(X, threshold=0.7)
```

### With Pandas DataFrame

```python
from src.main import LogisticRegression
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
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

With custom threshold:

```bash
python src/main.py --input data.csv --target label --threshold 0.7
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
from src.main import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2) * 2
y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

# Initialize and fit model
model = LogisticRegression(
    learning_rate=0.1,
    max_iterations=1000,
    scale_features=True,
)

model.fit(X, y)

# Print results
print(f"Intercept: {model.intercept:.4f}")
print(f"Weights: {model.weights}")
print(f"Accuracy: {model.score(X, y):.4f}")
print(f"Final cost: {model.cost_history[-1]:.6f}")

# Get predictions and probabilities
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Plot training history
model.plot_training_history()

# Plot decision boundary (for 2D data)
if X.shape[1] == 2:
    plt.figure(figsize=(10, 8))
    plt.scatter(
        X[y == 0, 0], X[y == 0, 1], c="blue", label="Class 0", alpha=0.6
    )
    plt.scatter(
        X[y == 1, 0], X[y == 1, 1], c="red", label="Class 1", alpha=0.6
    )

    # Decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.show()
```

## Project Structure

```
logistic-regression/
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

- `src/main.py`: Core implementation with `LogisticRegression` class
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
- Sigmoid activation function
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

## Understanding Logistic Regression

### Logistic Regression

Logistic regression models the probability of a binary outcome using the sigmoid function:

```
P(y=1|x) = σ(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

Where:
- `σ(z) = 1 / (1 + e^(-z))` is the sigmoid function
- `w₁, w₂, ..., wₙ` are weights
- `b` is the intercept
- Output is probability between 0 and 1

### Sigmoid Activation Function

The sigmoid function maps any real number to (0, 1):

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties:**
- Output range: (0, 1)
- S-shaped curve
- Smooth and differentiable
- σ(0) = 0.5

### Logistic Cost Function

Logistic regression uses log loss (cross-entropy):

```
J(θ) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
```

Where:
- `h = σ(X @ θ)` is the predicted probability
- `y` is the true label (0 or 1)
- `m` is the number of samples

**Properties:**
- Penalizes confident wrong predictions heavily
- Convex function (guarantees global minimum)
- Well-suited for probability outputs

### Gradient Descent Update

Gradient for logistic regression:

```
∇J(θ) = (1/m) * X.T @ (h - y)
```

Weight update:
```
θ = θ - α * ∇J(θ)
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
- Adjust decision threshold
- Check feature scaling
- Verify data balance
- Increase iterations

**Issue**: Cost not decreasing

**Solution**: 
- Learning rate too high or too low
- Features need scaling
- Check for numerical issues

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: y must contain only 0 and 1`: Labels must be binary
- `ValueError: Length mismatch`: X and y have different lengths

## Best Practices

1. **Always scale features**: Essential for gradient descent convergence
2. **Use appropriate learning rate**: Start with 0.01, adjust based on cost history
3. **Monitor cost history**: Ensure cost decreases during training
4. **Tune decision threshold**: Default 0.5 may not be optimal for imbalanced data
5. **Check class balance**: Imbalanced classes may need threshold adjustment
6. **Use probabilities**: `predict_proba()` provides more information than `predict()`

## Real-World Applications

- Binary classification problems (spam detection, fraud detection)
- Medical diagnosis (disease prediction)
- Customer churn prediction
- Credit approval
- Email classification
- Educational purposes for learning classification algorithms

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
