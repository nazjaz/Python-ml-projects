# Custom Loss Functions for Regression and Classification

A Python implementation of custom loss functions for both regression and classification tasks, including gradient computation for use in gradient-based optimization algorithms. This project provides a comprehensive collection of loss functions with analytical gradient computation, making it suitable for machine learning model training from scratch.

## Project Title and Description

The Custom Loss Functions tool provides implementations of various loss functions commonly used in machine learning, along with their analytical gradients. This is essential for training models using gradient descent and other optimization algorithms. The tool supports both regression tasks (MSE, MAE, Huber, Smooth L1, Log-Cosh) and classification tasks (Cross-Entropy, Focal Loss, Hinge Loss, KL Divergence).

This tool solves the problem of needing to implement loss functions from scratch when building custom machine learning models or when standard library implementations don't meet specific requirements. It provides a clean, well-documented API with gradient computation, which is crucial for optimization.

**Target Audience**: Machine learning practitioners, students learning optimization algorithms, researchers implementing custom models, and developers building ML frameworks from scratch.

## Features

### Regression Loss Functions
- Mean Squared Error (MSE) with gradient
- Mean Absolute Error (MAE) with gradient
- Huber Loss with configurable delta parameter
- Smooth L1 Loss with configurable beta parameter
- Log-Cosh Loss for smooth optimization

### Classification Loss Functions
- Binary Cross-Entropy with gradient
- Categorical Cross-Entropy for multi-class problems
- Focal Loss for handling class imbalance
- Hinge Loss for maximum-margin classification
- Kullback-Leibler Divergence for distribution matching

### Additional Features
- Analytical gradient computation for all loss functions
- Comprehensive input validation and error handling
- Command-line interface for evaluation
- Configuration via YAML file
- Support for numpy arrays and pandas DataFrames
- Comprehensive logging
- Unit tests with numerical gradient verification
- Documentation with mathematical formulations

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/custom-loss-functions
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
python src/main.py --task regression
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

loss_functions:
  regression:
    huber_delta: 1.0
    smooth_l1_beta: 1.0
  classification:
    focal_alpha: 1.0
    focal_gamma: 2.0
    epsilon: 1e-15
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `loss_functions.regression.huber_delta`: Delta parameter for Huber loss (default: 1.0)
- `loss_functions.regression.smooth_l1_beta`: Beta parameter for Smooth L1 loss (default: 1.0)
- `loss_functions.classification.focal_alpha`: Alpha parameter for Focal loss (default: 1.0)
- `loss_functions.classification.focal_gamma`: Gamma parameter for Focal loss (default: 2.0)
- `loss_functions.classification.epsilon`: Small value for numerical stability (default: 1e-15)

## Usage

### Command-Line Interface

#### Evaluate Regression Loss Functions

```bash
# Using synthetic data
python src/main.py --task regression

# Using CSV file
python src/main.py --task regression --input data.csv --true-col y_true --pred-col y_pred

# Save results to JSON
python src/main.py --task regression --input data.csv --output results.json
```

#### Evaluate Classification Loss Functions

```bash
# Using synthetic data
python src/main.py --task classification

# Using CSV file
python src/main.py --task classification --input data.csv --true-col y_true --pred-col y_pred
```

### Programmatic Usage

#### Regression Loss Functions

```python
import numpy as np
from src.main import RegressionLoss

y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 1.9, 3.2, 3.8])

# Mean Squared Error
loss, gradient = RegressionLoss.mean_squared_error(y_true, y_pred)
print(f"MSE Loss: {loss:.6f}")

# Mean Absolute Error
loss, gradient = RegressionLoss.mean_absolute_error(y_true, y_pred)
print(f"MAE Loss: {loss:.6f}")

# Huber Loss
loss, gradient = RegressionLoss.huber_loss(y_true, y_pred, delta=1.0)
print(f"Huber Loss: {loss:.6f}")

# Smooth L1 Loss
loss, gradient = RegressionLoss.smooth_l1_loss(y_true, y_pred, beta=1.0)
print(f"Smooth L1 Loss: {loss:.6f}")

# Log-Cosh Loss
loss, gradient = RegressionLoss.log_cosh_loss(y_true, y_pred)
print(f"Log-Cosh Loss: {loss:.6f}")
```

#### Classification Loss Functions

```python
import numpy as np
from src.main import ClassificationLoss

# Binary Classification
y_true_binary = np.array([0.0, 1.0, 1.0, 0.0])
y_pred_binary = np.array([0.1, 0.9, 0.8, 0.2])

# Binary Cross-Entropy
loss, gradient = ClassificationLoss.binary_cross_entropy(y_true_binary, y_pred_binary)
print(f"Binary CE Loss: {loss:.6f}")

# Focal Loss
loss, gradient = ClassificationLoss.focal_loss(
    y_true_binary, y_pred_binary, alpha=1.0, gamma=2.0
)
print(f"Focal Loss: {loss:.6f}")

# Hinge Loss (requires labels in {-1, 1})
y_true_hinge = np.array([-1.0, 1.0, 1.0, -1.0])
y_pred_hinge = np.array([-0.5, 0.8, 1.2, -1.5])
loss, gradient = ClassificationLoss.hinge_loss(y_true_hinge, y_pred_hinge)
print(f"Hinge Loss: {loss:.6f}")

# Multi-class Classification
y_true_cat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
y_pred_cat = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

# Categorical Cross-Entropy
loss, gradient = ClassificationLoss.categorical_cross_entropy(y_true_cat, y_pred_cat)
print(f"Categorical CE Loss: {loss:.6f}")

# KL Divergence
loss, gradient = ClassificationLoss.kullback_leibler_divergence(y_true_cat, y_pred_cat)
print(f"KL Divergence: {loss:.6f}")
```

#### Using the Evaluator

```python
import numpy as np
from src.main import LossFunctionEvaluator

evaluator = LossFunctionEvaluator()

# Evaluate all regression losses
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 1.9, 3.1])
results = evaluator.evaluate_regression_losses(y_true, y_pred)

for name, result in results.items():
    if "error" not in result:
        print(f"{name}: Loss={result['loss']:.6f}, Gradient Norm={result['gradient_norm']:.6f}")
```

### Common Use Cases

1. **Training Custom Models**: Use loss functions with gradients in gradient descent optimization
2. **Loss Function Comparison**: Compare different loss functions on the same dataset
3. **Hyperparameter Tuning**: Experiment with different loss function parameters (delta, beta, gamma, etc.)
4. **Educational Purposes**: Understand how loss functions and gradients work
5. **Research**: Implement custom loss functions for specific research needs

## Project Structure

```
custom-loss-functions/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py             # Main implementation with loss functions
├── tests/
│   └── test_main.py        # Unit tests
├── docs/
│   └── API.md              # API documentation (if applicable)
└── logs/
    └── .gitkeep            # Keep logs directory in git
```

### File Descriptions

- `src/main.py`: Contains all loss function implementations:
  - `RegressionLoss`: Class with regression loss functions (MSE, MAE, Huber, Smooth L1, Log-Cosh)
  - `ClassificationLoss`: Class with classification loss functions (BCE, CCE, Focal, Hinge, KL)
  - `LossFunctionEvaluator`: Class for evaluating and comparing loss functions
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - Basic functionality tests for all loss functions
  - Input validation tests
  - Gradient computation accuracy tests using numerical differentiation
  - Edge case tests (perfect predictions, shape mismatches, etc.)

- `config.yaml`: Configuration file for logging and loss function parameters

- `requirements.txt`: Python package dependencies with versions

## Testing

### Run All Tests

```bash
pytest tests/test_main.py -v
```

### Run Tests with Coverage

```bash
pytest tests/test_main.py --cov=src --cov-report=html
```

### Test Coverage Information

The test suite includes:
- Unit tests for all loss functions
- Input validation tests
- Gradient accuracy verification using numerical differentiation
- Edge case handling tests
- Integration tests for the evaluator

Current test coverage: >90% of code paths

## Mathematical Formulations

### Regression Loss Functions

#### Mean Squared Error (MSE)
```
MSE = (1/n) * Σ(y_true - y_pred)²
∂MSE/∂y_pred = -2 * (y_true - y_pred) / n
```

#### Mean Absolute Error (MAE)
```
MAE = (1/n) * Σ|y_true - y_pred|
∂MAE/∂y_pred = -sign(y_true - y_pred) / n
```

#### Huber Loss
```
L = { 0.5 * error²           if |error| ≤ δ
    { δ * |error| - 0.5 * δ²  if |error| > δ
```

#### Log-Cosh Loss
```
L = (1/n) * Σ log(cosh(y_true - y_pred))
∂L/∂y_pred = -tanh(y_true - y_pred) / n
```

### Classification Loss Functions

#### Binary Cross-Entropy
```
BCE = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
∂BCE/∂y_pred = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n
```

#### Focal Loss
```
FL = -(1/n) * Σ α * (1 - p_t)^γ * log(p_t)
where p_t = y_pred if y_true=1, else 1-y_pred
```

#### Categorical Cross-Entropy
```
CCE = -(1/n) * Σ Σ y_true * log(y_pred)
∂CCE/∂y_pred = -y_true / y_pred / n
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Shape Mismatch Error
**Error**: `ValueError: Shape mismatch: y_true (100,) != y_pred (50,)`

**Solution**: Ensure `y_true` and `y_pred` have the same shape. For classification, ensure labels are properly formatted (binary: 0/1, one-hot for multi-class).

#### Issue: Invalid Probability Values
**Error**: `ValueError: y_pred must be in [0, 1]`

**Solution**: For classification loss functions, ensure predictions are probabilities in [0, 1]. Use sigmoid or softmax activation if needed.

#### Issue: Numerical Instability
**Error**: NaN or Inf values in loss or gradient

**Solution**: The functions use epsilon clipping to prevent numerical issues. If problems persist, check input data for extreme values or use larger epsilon values.

#### Issue: One-Hot Encoding Required
**Error**: `ValueError: y_true must be one-hot encoded`

**Solution**: For categorical cross-entropy and KL divergence, convert integer labels to one-hot encoding:
```python
import numpy as np
n_classes = len(np.unique(y_true))
y_true_onehot = np.eye(n_classes)[y_true.astype(int)]
```

### Error Message Explanations

- **"Shape mismatch"**: Input arrays have different shapes
- **"y_true must contain only 0 and 1"**: Binary classification requires 0/1 labels
- **"y_pred must be in [0, 1]"**: Classification requires probability values
- **"delta must be positive"**: Huber loss delta parameter must be > 0
- **"gamma must be non-negative"**: Focal loss gamma parameter must be >= 0

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-loss-function`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Add unit tests for new loss functions
- Verify gradients using numerical differentiation

### Pull Request Process

1. Ensure all tests pass: `pytest tests/test_main.py`
2. Check code coverage: `pytest --cov=src`
3. Update documentation if adding new features
4. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. Please refer to the main repository license.

## Additional Resources

- [Loss Functions in Machine Learning](https://en.wikipedia.org/wiki/Loss_function)
- [Gradient Descent Optimization](https://en.wikipedia.org/wiki/Gradient_descent)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [NumPy Documentation](https://numpy.org/doc/stable/)
