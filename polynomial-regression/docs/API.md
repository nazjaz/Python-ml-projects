# Polynomial Regression API Documentation

## PolynomialRegression Class

### `PolynomialRegression(degree=2, regularization=None, alpha=1.0, fit_intercept=True)`

Polynomial Regression model with regularization support.

#### Parameters

- `degree` (int): Polynomial degree (default: 2).
- `regularization` (str, optional): Type of regularization. Options: None, "l1", "l2", "ridge", "lasso" (default: None).
- `alpha` (float): Regularization strength (default: 1.0).
- `fit_intercept` (bool): Whether to fit intercept term (default: True).

#### Attributes

- `coefficients` (ndarray): Model coefficients.
- `intercept` (float): Intercept term.
- `feature_names` (list): Names of polynomial features.

### Methods

#### `fit(X, y)`

Fit the polynomial regression model.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target values of shape (n_samples,).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
model = PolynomialRegression(degree=2)
model.fit(X, y)
```

#### `predict(X)`

Predict target values.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Predicted values of shape (n_samples,).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
predictions = model.predict(X)
```

#### `score(X, y)`

Calculate R-squared score.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True target values.

**Returns:**
- `float`: R-squared score (0-1).

**Example:**
```python
r2 = model.score(X, y)
```

#### `mse(X, y)`

Calculate mean squared error.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True target values.

**Returns:**
- `float`: Mean squared error.

**Example:**
```python
mse = model.mse(X, y)
```

#### `get_coefficients()`

Get model coefficients as dictionary.

**Returns:**
- `dict`: Dictionary mapping feature names to coefficients.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
coefficients = model.get_coefficients()
```

## Cross-Validation Functions

### `cross_validate_degree(X, y, degree_range=(1, 10), cv=5, regularization=None, alpha=1.0, scoring="mse", random_state=None)`

Perform cross-validation to select optimal polynomial degree.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): Target values.
- `degree_range` (tuple): Tuple of (min_degree, max_degree) (default: (1, 10)).
- `cv` (int): Number of cross-validation folds (default: 5).
- `regularization` (str, optional): Type of regularization (default: None).
- `alpha` (float): Regularization strength (default: 1.0).
- `scoring` (str): Scoring metric. Options: "mse", "r2" (default: "mse").
- `random_state` (int, optional): Random seed for shuffling (default: None).

**Returns:**
- `dict`: Dictionary mapping degrees to scores.

**Example:**
```python
cv_results = cross_validate_degree(X, y, degree_range=(1, 5), cv=5)
```

### `select_best_degree(cv_results, scoring="mse")`

Select best degree from cross-validation results.

**Parameters:**
- `cv_results` (dict): Cross-validation results from `cross_validate_degree`.
- `scoring` (str): Scoring metric used. Options: "mse", "r2" (default: "mse").

**Returns:**
- `int`: Best degree.

**Example:**
```python
best_degree = select_best_degree(cv_results, scoring="mse")
```

### `plot_cv_results(cv_results, scoring="mse", save_path=None, show=True)`

Plot cross-validation results.

**Parameters:**
- `cv_results` (dict): Cross-validation results.
- `scoring` (str): Scoring metric used (default: "mse").
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).

**Example:**
```python
plot_cv_results(cv_results, scoring="mse", save_path="cv_plot.png")
```

### `plot_predictions(X, y, model, save_path=None, show=True)`

Plot model predictions.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True target values.
- `model` (PolynomialRegression): Fitted model.
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).

**Note:** Only works for 1D features.

**Example:**
```python
plot_predictions(X, y, model, save_path="predictions.png")
```

## Usage Examples

### Basic Polynomial Regression

```python
from src.main import PolynomialRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = PolynomialRegression(degree=2)
model.fit(X, y)
predictions = model.predict(X)
```

### With Regularization

```python
from src.main import PolynomialRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# L2 regularization
model = PolynomialRegression(degree=3, regularization="l2", alpha=0.1)
model.fit(X, y)

# L1 regularization
model_lasso = PolynomialRegression(degree=3, regularization="lasso", alpha=0.1)
model_lasso.fit(X, y)
```

### Degree Selection

```python
from src.main import (
    cross_validate_degree,
    select_best_degree,
    PolynomialRegression,
)
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

# Cross-validation
cv_results = cross_validate_degree(
    X, y, degree_range=(1, 5), cv=5, scoring="mse"
)

# Select best degree
best_degree = select_best_degree(cv_results, scoring="mse")

# Fit with best degree
model = PolynomialRegression(degree=best_degree)
model.fit(X, y)
```

### Multiple Features

```python
from src.main import PolynomialRegression
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 11])

model = PolynomialRegression(degree=2)
model.fit(X, y)
predictions = model.predict(X)
```
