# Polynomial Regression with Degree Selection and Regularization

A Python implementation of Polynomial Regression from scratch with cross-validation for degree selection and regularization support. This is the thirty-second project in the ML learning series, focusing on understanding polynomial regression, model selection, and regularization techniques.

## Project Title and Description

The Polynomial Regression tool provides a complete implementation of polynomial regression from scratch, including automatic degree selection using cross-validation, L1/L2 regularization (Ridge/Lasso), and comprehensive model evaluation. It helps users understand how polynomial regression works, how to select optimal polynomial degrees, and how regularization prevents overfitting.

This tool solves the problem of learning polynomial regression fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates feature engineering, cross-validation, regularization techniques, and model selection from scratch.

**Target Audience**: Beginners learning machine learning, students studying regression techniques, and anyone who needs to understand polynomial regression, cross-validation, and regularization from scratch.

## Features

- Polynomial regression implementation from scratch
- Automatic degree selection using cross-validation
- Multiple regularization types: L1 (Lasso), L2 (Ridge)
- Flexible polynomial degree configuration
- R-squared and MSE evaluation metrics
- Cross-validation with configurable folds
- Model visualization for 1D features
- CV results visualization
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas DataFrames)
- Multiple feature support

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/polynomial-regression
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
python src/main.py --input sample.csv --target y --degree 2
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  degree: 2
  regularization: null
  alpha: 1.0
  fit_intercept: true

cross_validation:
  cv: 5
  degree_range: [1, 10]
  scoring: "mse"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.degree`: Polynomial degree (default: 2)
- `model.regularization`: Regularization type. Options: null, "l1", "l2", "ridge", "lasso" (default: null)
- `model.alpha`: Regularization strength (default: 1.0)
- `model.fit_intercept`: Whether to fit intercept term (default: true)
- `cross_validation.cv`: Number of CV folds (default: 5)
- `cross_validation.degree_range`: Degree range for CV [min, max] (default: [1, 10])
- `cross_validation.scoring`: Scoring metric. Options: "mse", "r2" (default: "mse")

## Usage

### Basic Usage

```python
from src.main import PolynomialRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Initialize and fit model
model = PolynomialRegression(degree=2)
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
r2 = model.score(X, y)
mse = model.mse(X, y)
print(f"R-squared: {r2:.4f}, MSE: {mse:.4f}")
```

### With Regularization

```python
from src.main import PolynomialRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# L2 regularization (Ridge)
model = PolynomialRegression(degree=3, regularization="l2", alpha=0.1)
model.fit(X, y)

# L1 regularization (Lasso)
model_lasso = PolynomialRegression(degree=3, regularization="lasso", alpha=0.1)
model_lasso.fit(X, y)
```

### Degree Selection with Cross-Validation

```python
from src.main import cross_validate_degree, select_best_degree, PolynomialRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

# Perform cross-validation
cv_results = cross_validate_degree(
    X, y, degree_range=(1, 5), cv=5, scoring="mse"
)

# Select best degree
best_degree = select_best_degree(cv_results, scoring="mse")
print(f"Best degree: {best_degree}")

# Fit model with best degree
model = PolynomialRegression(degree=best_degree)
model.fit(X, y)
```

### With Multiple Features

```python
from src.main import PolynomialRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 11])

model = PolynomialRegression(degree=2)
model.fit(X, y)

predictions = model.predict(X)
```

### Get Coefficients

```python
from src.main import PolynomialRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = PolynomialRegression(degree=2)
model.fit(X, y)

coefficients = model.get_coefficients()
for name, coef in coefficients.items():
    print(f"{name}: {coef:.6f}")
```

### With Pandas DataFrame

```python
from src.main import PolynomialRegression
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2"]].values
y = df["target"].values

model = PolynomialRegression(degree=2)
model.fit(X, y)

predictions = model.predict(X)
```

### Command-Line Usage

Basic polynomial regression:

```bash
python src/main.py --input data.csv --target y --degree 2
```

With regularization:

```bash
python src/main.py --input data.csv --target y --degree 3 --regularization l2 --alpha 0.1
```

Degree selection with cross-validation:

```bash
python src/main.py --input data.csv --target y --select-degree --degree-range 1,10 --cv 5
```

Plot predictions:

```bash
python src/main.py --input data.csv --target y --degree 2 --plot
```

Plot CV results:

```bash
python src/main.py --input data.csv --target y --select-degree --plot-cv
```

Save predictions:

```bash
python src/main.py --input data.csv --target y --degree 2 --output predictions.csv
```

Make predictions on new data:

```bash
python src/main.py --input train.csv --target y --degree 2 --predict test.csv --output predictions.csv
```

### Complete Example

```python
from src.main import (
    PolynomialRegression,
    cross_validate_degree,
    select_best_degree,
    plot_cv_results,
    plot_predictions,
)
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X[:, 0] ** 2 + 3 * X[:, 0] + 1 + np.random.normal(0, 2, 50)

# Perform cross-validation
cv_results = cross_validate_degree(
    X, y, degree_range=(1, 5), cv=5, scoring="mse"
)

# Select best degree
best_degree = select_best_degree(cv_results, scoring="mse")
print(f"Best degree: {best_degree}")

# Fit model with best degree
model = PolynomialRegression(degree=best_degree, regularization="l2", alpha=0.1)
model.fit(X, y)

# Evaluate
r2 = model.score(X, y)
mse = model.mse(X, y)
print(f"R-squared: {r2:.4f}, MSE: {mse:.4f}")

# Visualize
plot_cv_results(cv_results, scoring="mse")
plot_predictions(X, y, model)
```

## Project Structure

```
polynomial-regression/
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

- `src/main.py`: Core implementation with `PolynomialRegression` class and CV functions
- `config.yaml`: Configuration file for model and CV settings
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
- Model initialization
- Fitting with different degrees
- Prediction
- Regularization (L1, L2, Ridge, Lasso)
- Evaluation metrics (R-squared, MSE)
- Cross-validation
- Degree selection
- Error handling
- Different input types
- Multiple features

## Understanding Polynomial Regression

### Polynomial Regression

Polynomial regression extends linear regression by adding polynomial features:

**Linear Regression:**
```
y = β₀ + β₁x + ε
```

**Polynomial Regression (degree 2):**
```
y = β₀ + β₁x + β₂x² + ε
```

**Polynomial Regression (degree n):**
```
y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε
```

### Feature Engineering

For degree `d` and `n` features, polynomial features include:
- Intercept term (if `fit_intercept=True`)
- All features to powers 1 through `d`
- Total features: `1 + n*d` (with intercept) or `n*d` (without intercept)

### Regularization

**L2 Regularization (Ridge):**
- Penalizes large coefficients
- Loss: `MSE + α * Σβᵢ²`
- Prevents overfitting by shrinking coefficients

**L1 Regularization (Lasso):**
- Penalizes absolute values of coefficients
- Loss: `MSE + α * Σ|βᵢ|`
- Can set coefficients to exactly zero (feature selection)

### Cross-Validation for Degree Selection

**Process:**
1. Split data into k folds
2. For each degree in range:
   - Train on k-1 folds, validate on 1 fold
   - Repeat for all folds
   - Calculate mean and std of scores
3. Select degree with best mean score

**Scoring Metrics:**
- MSE: Lower is better (minimize)
- R²: Higher is better (maximize)

### Choosing Regularization Strength

**Alpha (α) values:**
- Small α (0.01-0.1): Light regularization
- Medium α (0.1-1.0): Moderate regularization
- Large α (1.0-10.0): Strong regularization

**Guidelines:**
- Start with small α and increase if overfitting
- Use cross-validation to select optimal α
- L1 (Lasso) typically needs smaller α than L2 (Ridge)

## Troubleshooting

### Common Issues

**Issue**: Overfitting with high degree

**Solution**: 
- Use regularization (L2 or L1)
- Reduce polynomial degree
- Use cross-validation to select optimal degree
- Increase regularization strength (alpha)

**Issue**: Underfitting with low degree

**Solution**: 
- Increase polynomial degree
- Use cross-validation to find optimal degree
- Check if data is truly non-linear

**Issue**: Singular matrix error

**Solution**: 
- Use regularization
- Reduce polynomial degree
- Check for linearly dependent features
- Ensure sufficient samples

**Issue**: Poor cross-validation results

**Solution**: 
- Increase number of CV folds
- Check data quality
- Try different degree ranges
- Use regularization

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: Degree must be at least 1`: Use degree >= 1
- `ValueError: Alpha must be non-negative`: Use alpha >= 0
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Use cross-validation**: Always use CV for degree selection
2. **Start with low degrees**: Begin with degree 1-3, increase if needed
3. **Use regularization**: Especially for high degrees or small datasets
4. **Visualize results**: Plot predictions to understand model behavior
5. **Check CV results**: Visualize CV scores to see degree impact
6. **Balance complexity**: Higher degree = more flexible but risk of overfitting
7. **Regularization strength**: Tune alpha using cross-validation
8. **Multiple features**: Be careful with high degrees and many features (feature explosion)

## Real-World Applications

- Curve fitting
- Trend analysis
- Non-linear relationship modeling
- Time series forecasting
- Scientific data analysis
- Engineering applications
- Educational purposes for learning regression

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
