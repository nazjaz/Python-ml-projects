# Regression Metrics API Documentation

This document provides detailed API documentation for the Regression Metrics Calculator.

## RegressionMetrics Class

Main class for calculating regression evaluation metrics.

### Constructor

```python
RegressionMetrics(config_path: str = "config.yaml") -> None
```

Initialize RegressionMetrics with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file. Default: "config.yaml"

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
from src.main import RegressionMetrics

metrics = RegressionMetrics()
```

---

## Methods

### mae

```python
mae(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> float
```

Calculate Mean Absolute Error (MAE).

MAE measures the average absolute difference between predicted and actual values. Lower values indicate better performance.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `float`: MAE score (non-negative)

**Example:**
```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
mae = metrics.mae(y_true, y_pred)
# Returns: 0.5
```

---

### mse

```python
mse(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> float
```

Calculate Mean Squared Error (MSE).

MSE measures the average squared difference between predicted and actual values. It penalizes larger errors more than MAE. Lower values indicate better performance.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `float`: MSE score (non-negative)

**Example:**
```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
mse = metrics.mse(y_true, y_pred)
# Returns: 0.375
```

---

### rmse

```python
rmse(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> float
```

Calculate Root Mean Squared Error (RMSE).

RMSE is the square root of MSE, providing error in the same units as the target variable. Lower values indicate better performance.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `float`: RMSE score (non-negative)

**Example:**
```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
rmse = metrics.rmse(y_true, y_pred)
# Returns: 0.6123724356957945
```

---

### r_squared

```python
r_squared(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> float
```

Calculate R-squared (Coefficient of Determination).

R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). Higher values (closer to 1.0) indicate better fit. Can be negative for poor models.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `float`: R-squared score (can be negative for poor models)

**Example:**
```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
r2 = metrics.r_squared(y_true, y_pred)
# Returns: 0.9486081370449679
```

---

### calculate_all_metrics

```python
calculate_all_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> Dict[str, float]
```

Calculate all regression metrics at once.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `Dict[str, float]`: Dictionary containing:
  - `"mae"`: Mean Absolute Error
  - `"mse"`: Mean Squared Error
  - `"rmse"`: Root Mean Squared Error
  - `"r_squared"`: R-squared score

**Example:**
```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
results = metrics.calculate_all_metrics(y_true, y_pred)
# Returns: {
#     "mae": 0.5,
#     "mse": 0.375,
#     "rmse": 0.6123724356957945,
#     "r_squared": 0.9486081370449679
# }
```

---

### generate_detailed_report

```python
generate_detailed_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> Dict[str, Union[float, Dict[str, float]]]
```

Generate detailed regression metrics report.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `Dict[str, Union[float, Dict[str, float]]]`: Dictionary containing:
  - `"metrics"`: All calculated metrics (MAE, MSE, RMSE, R-squared)
  - `"statistics"`: Statistical summary of errors
  - `"residuals"`: Detailed residual statistics
  - `"sample_size"`: Number of samples

**Example:**
```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
report = metrics.generate_detailed_report(y_true, y_pred)
# Returns: {
#     "metrics": {...},
#     "statistics": {...},
#     "residuals": {...},
#     "sample_size": 4
# }
```

---

### print_report

```python
print_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> None
```

Print formatted detailed report to console.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- `None`: Prints formatted report to console

**Example:**
```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
metrics.print_report(y_true, y_pred)
# Prints formatted report to console
```

---

## Input Types

All methods accept the following input types for `y_true` and `y_pred`:

- `List`: Python list of numeric values
- `np.ndarray`: NumPy array of numeric values
- `pd.Series`: Pandas Series of numeric values

**Example:**
```python
import numpy as np
import pandas as pd

# List
y_true = [1.0, 2.0, 3.0]
y_pred = [1.5, 2.5, 2.5]

# NumPy array
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.5, 2.5, 2.5])

# Pandas Series
y_true = pd.Series([1.0, 2.0, 3.0])
y_pred = pd.Series([1.5, 2.5, 2.5])
```

---

## Error Handling

### ValueError: Length mismatch

Raised when `y_true` and `y_pred` have different lengths.

```python
y_true = [1.0, 2.0, 3.0]
y_pred = [1.0, 2.0]  # Different length
metrics.mae(y_true, y_pred)  # Raises ValueError
```

### ValueError: Input arrays cannot be empty

Raised when input arrays are empty.

```python
y_true = []
y_pred = []
metrics.mae(y_true, y_pred)  # Raises ValueError
```

### Warning: Input contains NaN values

Warning logged when input contains NaN values. Results may be invalid.

```python
y_true = [1.0, np.nan, 3.0]
y_pred = [1.0, 2.0, 3.0]
metrics.mae(y_true, y_pred)  # Logs warning
```

---

## Notes

- All error metrics (MAE, MSE, RMSE) return non-negative values
- Lower values indicate better performance for MAE, MSE, and RMSE
- R-squared can be negative for poor models (worse than predicting the mean)
- R-squared ranges from -∞ to 1.0, with 1.0 indicating perfect fit
- RMSE is always greater than or equal to MAE for the same predictions
- RMSE is the square root of MSE
- Detailed report includes comprehensive residual analysis

---

## Metric Relationships

### MAE vs MSE vs RMSE

- **MAE**: Average absolute error, less sensitive to outliers
- **MSE**: Average squared error, penalizes large errors more
- **RMSE**: Square root of MSE, same units as target variable

For the same predictions:
- `RMSE ≥ MAE` (always)
- `RMSE = √MSE` (always)
- `MSE ≥ MAE²` (always)

### R-squared Interpretation

- **R² = 1.0**: Perfect predictions
- **R² > 0.9**: Excellent fit
- **R² > 0.7**: Good fit
- **R² > 0.5**: Moderate fit
- **R² > 0**: Model is better than predicting the mean
- **R² < 0**: Model is worse than predicting the mean
