# Linear Regression with Gradient Descent

A Python implementation of linear regression from scratch using gradient descent optimization with learning rate scheduling. This is the twenty-first project in the ML learning series, focusing on understanding the fundamentals of linear regression and optimization algorithms.

## Project Title and Description

The Linear Regression with Gradient Descent tool provides a complete implementation of linear regression from scratch, including gradient descent optimization and multiple learning rate scheduling strategies. It helps users understand how linear regression works at a fundamental level and how optimization algorithms converge to find optimal parameters.

This tool solves the problem of learning linear regression fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates gradient descent optimization and various learning rate scheduling techniques.

**Target Audience**: Beginners learning machine learning, students studying optimization algorithms, and anyone who needs to understand linear regression and gradient descent from scratch.

## Features

- Linear regression implementation from scratch
- Gradient descent optimization
- Multiple learning rate scheduling strategies:
  - Constant learning rate
  - Exponential decay
  - Step decay
  - Polynomial decay
- Cost function tracking (MSE)
- Convergence detection
- Training history visualization
- R-squared score calculation
- Support for multiple features
- Optional intercept term
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
cd /path/to/Python-ml-projects/linear-regression-gd
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
python src/main.py --input sample.csv --target price --plot
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
  scheduler: "constant"
  scheduler_params:
    decay_rate: 0.95
    drop_rate: 0.5
    epochs_drop: 10
    end_lr: 0.001
    power: 1.0
    max_epochs: 1000
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.learning_rate`: Initial learning rate (default: 0.01)
- `model.max_iterations`: Maximum number of iterations (default: 1000)
- `model.tolerance`: Convergence tolerance (default: 1e-6)
- `model.fit_intercept`: Whether to fit intercept term (default: true)
- `model.scheduler`: Learning rate scheduler type (constant, exponential, step, polynomial)
- `model.scheduler_params`: Parameters for scheduler

## Usage

### Basic Usage

```python
from src.main import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Initialize and fit model
model = LinearRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Calculate R-squared
score = model.score(X, y)
print(f"R-squared: {score:.4f}")
```

### With Learning Rate Scheduling

```python
from src.main import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Exponential decay scheduler
model = LinearRegression(
    learning_rate=0.01,
    max_iterations=1000,
    scheduler="exponential",
    scheduler_params={"decay_rate": 0.95},
)
model.fit(X, y)

# Plot training history
model.plot_training_history()
```

### Multiple Features

```python
from src.main import LinearRegression
import numpy as np

# Multiple features
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([5, 8, 11, 14, 17])

model = LinearRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

print(f"Intercept: {model.intercept:.4f}")
print(f"Weights: {model.weights}")
```

### With Pandas DataFrame

```python
from src.main import LinearRegression
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2"]]
y = df["target"]

model = LinearRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

predictions = model.predict(X)
```

### Command-Line Usage

Basic training:

```bash
python src/main.py --input data.csv --target price
```

With specific features:

```bash
python src/main.py --input data.csv --target price --features "feature1,feature2"
```

With exponential decay scheduler:

```bash
python src/main.py --input data.csv --target price --scheduler exponential
```

Plot training history:

```bash
python src/main.py --input data.csv --target price --plot
```

Save training history plot:

```bash
python src/main.py --input data.csv --target price --save-plot history.png
```

Save predictions:

```bash
python src/main.py --input data.csv --target price --output predictions.csv
```

Custom learning rate and iterations:

```bash
python src/main.py --input data.csv --target price --learning-rate 0.05 --max-iterations 500
```

### Complete Example

```python
from src.main import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 3 * X.flatten() + 2 + np.random.randn(100) * 2

# Initialize model with step decay scheduler
model = LinearRegression(
    learning_rate=0.1,
    max_iterations=1000,
    scheduler="step",
    scheduler_params={"drop_rate": 0.5, "epochs_drop": 100},
)

# Fit model
model.fit(X, y)

# Print results
print(f"Intercept: {model.intercept:.4f}")
print(f"Weight: {model.weights[0]:.4f}")
print(f"R-squared: {model.score(X, y):.4f}")
print(f"Final cost: {model.cost_history[-1]:.6f}")

# Plot training history
model.plot_training_history()

# Make predictions
predictions = model.predict(X)

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label="Actual")
plt.plot(X, predictions, "r-", linewidth=2, label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Predictions")
plt.show()
```

## Project Structure

```
linear-regression-gd/
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

- `src/main.py`: Core implementation with `LinearRegression` class and `LearningRateScheduler`
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
- Learning rate schedulers
- Model initialization
- Training and fitting
- Prediction
- Score calculation
- Cost history
- Convergence
- Multiple features
- Error handling
- Different input types

## Understanding Linear Regression and Gradient Descent

### Linear Regression

Linear regression models the relationship between features and target using a linear equation:

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Where:
- `w₁, w₂, ..., wₙ` are weights (coefficients)
- `b` is the intercept (bias term)
- `x₁, x₂, ..., xₙ` are features
- `y` is the target

### Gradient Descent

Gradient descent is an optimization algorithm that finds optimal parameters by iteratively moving in the direction of steepest descent of the cost function.

**Algorithm:**
1. Initialize weights randomly (or to zero)
2. Compute cost (MSE)
3. Compute gradient of cost function
4. Update weights: `θ = θ - α * ∇J(θ)`
5. Repeat until convergence

Where:
- `α` is the learning rate
- `∇J(θ)` is the gradient of the cost function

### Learning Rate Scheduling

Different strategies for adjusting learning rate during training:

**Constant**: Learning rate remains fixed throughout training.

**Exponential Decay**: Learning rate decreases exponentially:
```
lr(t) = lr₀ * decay_rate^t
```

**Step Decay**: Learning rate drops by a factor at regular intervals:
```
lr(t) = lr₀ * drop_rate^(t // epochs_drop)
```

**Polynomial Decay**: Learning rate decreases polynomially:
```
lr(t) = (lr₀ - lr_end) * (1 - t/max_epochs)^power + lr_end
```

## Troubleshooting

### Common Issues

**Issue**: Model not converging

**Solution**: 
- Reduce learning rate
- Increase max_iterations
- Use learning rate scheduling
- Check data scaling

**Issue**: Cost increasing instead of decreasing

**Solution**: Learning rate too high. Reduce learning rate.

**Issue**: Convergence too slow

**Solution**: 
- Increase learning rate (carefully)
- Use adaptive learning rate scheduling
- Check data preprocessing

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: Length mismatch`: X and y have different lengths
- `ValueError: Input data cannot be empty`: Empty input data

## Best Practices

1. **Scale features**: Normalize or standardize features for better convergence
2. **Choose appropriate learning rate**: Start with 0.01 and adjust
3. **Use learning rate scheduling**: Helps with convergence and fine-tuning
4. **Monitor cost history**: Plot to visualize training progress
5. **Set convergence tolerance**: Prevents unnecessary iterations
6. **Use cross-validation**: Evaluate model performance properly

## Real-World Applications

- Predicting continuous values (price, temperature, sales)
- Understanding optimization algorithms
- Educational purposes for learning ML fundamentals
- Feature importance analysis
- Baseline model for regression tasks
- Understanding gradient descent behavior

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
