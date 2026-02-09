# Regression Metrics Calculator

A Python tool for calculating evaluation metrics for regression tasks including MAE, MSE, RMSE, and R-squared with detailed reporting. This is the seventeenth project in the ML learning series, focusing on understanding and implementing fundamental regression evaluation metrics.

## Project Title and Description

The Regression Metrics Calculator provides automated calculation of essential evaluation metrics for regression models. It implements Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared from scratch, helping users understand how these metrics work and when to use them.

This tool solves the problem of evaluating regression model performance by providing a clear, educational implementation of standard metrics with detailed reporting capabilities. It includes comprehensive residual analysis and statistical summaries.

**Target Audience**: Beginners learning machine learning, data scientists evaluating regression models, and anyone who needs to understand or implement regression metrics from scratch.

## Features

- Calculate Mean Absolute Error (MAE)
- Calculate Mean Squared Error (MSE)
- Calculate Root Mean Squared Error (RMSE)
- Calculate R-squared (Coefficient of Determination)
- Detailed reporting with residual statistics
- Statistical summaries (mean, std, min, max, median, percentiles)
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas Series)

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/regression-metrics
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
python src/main.py --y-true "3.0,-0.5,2.0,7.0" --y-pred "2.5,0.0,2.0,8.0"
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import RegressionMetrics

metrics = RegressionMetrics()

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

# Calculate individual metrics
mae = metrics.mae(y_true, y_pred)
mse = metrics.mse(y_true, y_pred)
rmse = metrics.rmse(y_true, y_pred)
r2 = metrics.r_squared(y_true, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
```

### Calculate All Metrics

```python
from src.main import RegressionMetrics

metrics = RegressionMetrics()
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

results = metrics.calculate_all_metrics(y_true, y_pred)
print(results)
```

### Detailed Report

```python
from src.main import RegressionMetrics

metrics = RegressionMetrics()
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

# Generate detailed report
report = metrics.generate_detailed_report(y_true, y_pred)
print(report)

# Print formatted report
metrics.print_report(y_true, y_pred)
```

### Command-Line Usage

Basic usage with comma-separated values:

```bash
python src/main.py --y-true "3.0,-0.5,2.0,7.0" --y-pred "2.5,0.0,2.0,8.0"
```

Using CSV files:

```bash
python src/main.py --y-true true_values.csv --y-pred pred_values.csv --column value
```

Print detailed formatted report:

```bash
python src/main.py --y-true "3.0,-0.5,2.0,7.0" --y-pred "2.5,0.0,2.0,8.0" --report
```

Save detailed report to JSON:

```bash
python src/main.py --y-true "3.0,-0.5,2.0,7.0" --y-pred "2.5,0.0,2.0,8.0" --output report.json
```

### Complete Example

```python
from src.main import RegressionMetrics
import numpy as np

# Initialize metrics calculator
metrics = RegressionMetrics()

# Example regression predictions
y_true = np.array([10.5, 20.3, 15.7, 8.2, 12.1, 18.9, 22.4, 14.6])
y_pred = np.array([10.2, 20.8, 15.1, 8.5, 12.3, 18.5, 22.1, 14.9])

# Calculate all metrics
results = metrics.calculate_all_metrics(y_true, y_pred)

print("=== Regression Metrics ===")
print(f"MAE:       {results['mae']:.6f}")
print(f"MSE:       {results['mse']:.6f}")
print(f"RMSE:      {results['rmse']:.6f}")
print(f"R-squared: {results['r_squared']:.6f}")

# Generate detailed report
report = metrics.generate_detailed_report(y_true, y_pred)
print("\n=== Residual Statistics ===")
print(f"Mean:   {report['residuals']['mean']:.6f}")
print(f"Std:    {report['residuals']['std']:.6f}")
print(f"Min:    {report['residuals']['min']:.6f}")
print(f"Max:    {report['residuals']['max']:.6f}")
print(f"Median: {report['residuals']['median']:.6f}")

# Print formatted report
metrics.print_report(y_true, y_pred)
```

## Project Structure

```
regression-metrics/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── .env.example             # Environment variables template
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

- `src/main.py`: Core implementation with `RegressionMetrics` class
- `config.yaml`: Configuration file for logging settings
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
- MAE calculation
- MSE calculation
- RMSE calculation
- R-squared calculation
- Detailed report generation
- Input validation
- Error handling
- Different input types (lists, numpy arrays, pandas Series)
- Edge cases (perfect predictions, poor models, constant values)

## Understanding the Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values:

```
MAE = (1/n) * Σ|y_true - y_pred|
```

**When to use**: Good for understanding average error magnitude. Less sensitive to outliers than MSE.

**Interpretation**: Lower is better. Value is in the same units as the target variable.

### Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values:

```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**When to use**: Penalizes larger errors more than MAE. Useful when large errors are particularly undesirable.

**Interpretation**: Lower is better. Value is in squared units of the target variable.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE:

```
RMSE = √MSE
```

**When to use**: Provides error in the same units as the target variable while still penalizing large errors.

**Interpretation**: Lower is better. Value is in the same units as the target variable.

### R-squared (R²)

R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variable(s):

```
R² = 1 - (SS_res / SS_tot)
```

Where:
- SS_res = Σ(y_true - y_pred)² (sum of squares of residuals)
- SS_tot = Σ(y_true - mean(y_true))² (total sum of squares)

**When to use**: Good for understanding how well the model explains variance. Can be negative for poor models.

**Interpretation**: Higher is better. Range: -∞ to 1.0. Values closer to 1.0 indicate better fit.

## Detailed Reporting

The detailed report includes:

### Performance Metrics
- MAE, MSE, RMSE, R-squared

### Residual Statistics
- Mean residual
- Standard deviation of residuals
- Minimum and maximum residuals
- Median residual
- 25th and 75th percentiles

### Dataset Information
- Sample size

## Troubleshooting

### Common Issues

**Issue**: Length mismatch error

**Solution**: Ensure `y_true` and `y_pred` have the same length.

**Issue**: NaN values in results

**Solution**: Check input data for NaN values. The tool will warn but may still produce results.

**Issue**: R-squared is negative

**Solution**: This is normal for poor models. R-squared can be negative when the model performs worse than simply predicting the mean.

### Error Messages

- `ValueError: Length mismatch`: Input arrays have different lengths
- `ValueError: Input arrays cannot be empty`: At least one input is empty
- `Warning: Input contains NaN values`: Input data contains NaN values

## Best Practices

1. **Use appropriate metrics**: MAE for average error, RMSE for penalizing large errors, R-squared for variance explanation
2. **Consider units**: MAE and RMSE are in the same units as the target, MSE is in squared units
3. **Interpret R-squared carefully**: Negative R-squared indicates poor model, but values close to 1.0 don't guarantee good predictions
4. **Examine residuals**: Detailed report helps identify patterns in errors
5. **Compare metrics**: Use multiple metrics to get a complete picture of model performance

## Real-World Applications

- Evaluating regression models (linear regression, polynomial regression, etc.)
- Model comparison and selection
- Performance monitoring in production systems
- Feature engineering evaluation
- Hyperparameter tuning
- Educational purposes for understanding metrics

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
