# Multiple Linear Regression with Feature Scaling and Regularization

A Python implementation of multiple linear regression from scratch with feature scaling and regularization (Ridge and Lasso). This is the twenty-second project in the ML learning series, focusing on understanding multiple linear regression, feature scaling, and regularization techniques.

## Project Title and Description

The Multiple Linear Regression tool provides a complete implementation of multiple linear regression from scratch, including feature scaling (standardization and normalization) and regularization techniques (Ridge and Lasso). It helps users understand how to handle multiple features, scale data properly, and use regularization to prevent overfitting.

This tool solves the problem of implementing multiple linear regression with proper preprocessing and regularization by providing a clear, educational implementation without relying on external ML libraries. It demonstrates feature scaling importance and regularization effects.

**Target Audience**: Beginners learning machine learning, students studying regression techniques, and anyone who needs to understand multiple linear regression, feature scaling, and regularization from scratch.

## Features

- Multiple linear regression implementation from scratch
- Feature scaling (standardization and normalization)
- Ridge regression (L2 regularization)
- Lasso regression (L1 regularization)
- Gradient descent optimization
- Cost function tracking with regularization penalties
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
cd /path/to/Python-ml-projects/multiple-linear-regression
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
python src/main.py --input sample.csv --target price
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
  regularization: null
  alpha: 0.1
  scale_features: true
  scaling_method: "standardize"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.learning_rate`: Initial learning rate (default: 0.01)
- `model.max_iterations`: Maximum number of iterations (default: 1000)
- `model.tolerance`: Convergence tolerance (default: 1e-6)
- `model.fit_intercept`: Whether to fit intercept term (default: true)
- `model.regularization`: Regularization type. Options: null, "ridge", "lasso" (default: null)
- `model.alpha`: Regularization strength (default: 0.1)
- `model.scale_features`: Whether to scale features (default: true)
- `model.scaling_method`: Feature scaling method. Options: "standardize", "normalize" (default: "standardize")

## Usage

### Basic Usage

```python
from src.main import MultipleLinearRegression
import numpy as np

# Create sample data with multiple features
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([5, 8, 11, 14, 17])

# Initialize and fit model
model = MultipleLinearRegression(
    learning_rate=0.01, max_iterations=1000
)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Calculate R-squared
score = model.score(X, y)
print(f"R-squared: {score:.4f}")
```

### With Feature Scaling

```python
from src.main import MultipleLinearRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([5, 8, 11, 14, 17])

# With standardization (default)
model = MultipleLinearRegression(
    learning_rate=0.01,
    max_iterations=1000,
    scale_features=True,
    scaling_method="standardize",
)
model.fit(X, y)

# With normalization
model = MultipleLinearRegression(
    learning_rate=0.01,
    max_iterations=1000,
    scale_features=True,
    scaling_method="normalize",
)
model.fit(X, y)
```

### With Ridge Regularization

```python
from src.main import MultipleLinearRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([5, 8, 11, 14, 17])

# Ridge regression (L2 regularization)
model = MultipleLinearRegression(
    learning_rate=0.01,
    max_iterations=1000,
    regularization="ridge",
    alpha=0.1,
)
model.fit(X, y)

print(f"Weights: {model.weights}")
print(f"Intercept: {model.intercept:.4f}")
```

### With Lasso Regularization

```python
from src.main import MultipleLinearRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([5, 8, 11, 14, 17])

# Lasso regression (L1 regularization)
model = MultipleLinearRegression(
    learning_rate=0.01,
    max_iterations=1000,
    regularization="lasso",
    alpha=0.1,
)
model.fit(X, y)

print(f"Weights: {model.weights}")
print(f"Intercept: {model.intercept:.4f}")
```

### With Pandas DataFrame

```python
from src.main import MultipleLinearRegression
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

model = MultipleLinearRegression(
    learning_rate=0.01,
    max_iterations=1000,
    regularization="ridge",
    alpha=0.1,
)
model.fit(X, y)

predictions = model.predict(X)
```

### Command-Line Usage

Basic training:

```bash
python src/main.py --input data.csv --target price
```

With Ridge regularization:

```bash
python src/main.py --input data.csv --target price --regularization ridge --alpha 0.1
```

With Lasso regularization:

```bash
python src/main.py --input data.csv --target price --regularization lasso --alpha 0.1
```

With feature normalization:

```bash
python src/main.py --input data.csv --target price --scaling normalize
```

Without feature scaling:

```bash
python src/main.py --input data.csv --target price --scaling none
```

Plot training history:

```bash
python src/main.py --input data.csv --target price --plot
```

Save predictions:

```bash
python src/main.py --input data.csv --target price --output predictions.csv
```

### Complete Example

```python
from src.main import MultipleLinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with multiple features
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 3) * 10
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5 + np.random.randn(n_samples) * 2

# Compare different models
models = {
    "No Regularization": MultipleLinearRegression(
        learning_rate=0.01, max_iterations=1000, regularization=None
    ),
    "Ridge (α=0.1)": MultipleLinearRegression(
        learning_rate=0.01,
        max_iterations=1000,
        regularization="ridge",
        alpha=0.1,
    ),
    "Lasso (α=0.1)": MultipleLinearRegression(
        learning_rate=0.01,
        max_iterations=1000,
        regularization="lasso",
        alpha=0.1,
    ),
}

for name, model in models.items():
    model.fit(X, y)
    score = model.score(X, y)
    print(f"{name}: R² = {score:.4f}, Weights = {model.weights}")

# Plot training history for one model
models["Ridge (α=0.1)"].plot_training_history()
```

## Project Structure

```
multiple-linear-regression/
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

- `src/main.py`: Core implementation with `MultipleLinearRegression` class and `FeatureScaler`
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
- Feature scaling (standardization and normalization)
- Model initialization
- Training with and without regularization
- Ridge regression
- Lasso regression
- Prediction
- Score calculation
- Cost history
- Error handling
- Different input types

## Understanding Multiple Linear Regression

### Multiple Linear Regression

Multiple linear regression models the relationship between multiple features and target:

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Where:
- `w₁, w₂, ..., wₙ` are weights for each feature
- `b` is the intercept
- `x₁, x₂, ..., xₙ` are features
- `y` is the target

### Feature Scaling

**Standardization (Z-score normalization):**
- Transforms features to have zero mean and unit variance
- Formula: `x_scaled = (x - mean) / std`
- Useful when features have different scales

**Normalization (Min-Max scaling):**
- Transforms features to [0, 1] range
- Formula: `x_scaled = (x - min) / (max - min)`
- Useful when you need bounded values

### Regularization

**Ridge Regression (L2 Regularization):**
- Adds penalty: `α * Σw²`
- Shrinks weights towards zero
- Prevents overfitting
- Keeps all features (doesn't eliminate any)

**Lasso Regression (L1 Regularization):**
- Adds penalty: `α * Σ|w|`
- Can set weights to exactly zero
- Performs feature selection
- Useful for sparse models

## Troubleshooting

### Common Issues

**Issue**: Model not converging

**Solution**: 
- Scale features (use standardization or normalization)
- Reduce learning rate
- Increase max_iterations
- Check data quality

**Issue**: Weights too large

**Solution**: Use regularization (Ridge or Lasso) with appropriate alpha

**Issue**: Poor predictions

**Solution**: 
- Ensure features are scaled
- Check for feature importance
- Try different regularization strengths
- Verify data preprocessing

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: Length mismatch`: X and y have different lengths
- `ValueError: Unknown scaling method`: Invalid scaling method specified

## Best Practices

1. **Always scale features**: Especially important for multiple features with different scales
2. **Use standardization by default**: Works well for most cases
3. **Use Ridge for multicollinearity**: Helps when features are correlated
4. **Use Lasso for feature selection**: When you want to eliminate irrelevant features
5. **Tune alpha parameter**: Balance between fit and regularization
6. **Monitor cost history**: Ensure model is converging
7. **Compare models**: Try different regularization types and strengths

## Real-World Applications

- Predicting continuous values with multiple features
- Understanding feature importance
- Handling multicollinearity
- Feature selection
- Preventing overfitting
- Educational purposes for learning regression techniques

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
