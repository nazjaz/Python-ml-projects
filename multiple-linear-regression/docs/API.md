# Multiple Linear Regression API Documentation

This document provides detailed API documentation for the Multiple Linear Regression with Feature Scaling and Regularization implementation.

## FeatureScaler Class

Static methods for feature scaling operations.

### standardize

```python
standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Standardize features (zero mean, unit variance).

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Tuple of (scaled_X, mean, std)

**Example:**
```python
X = np.array([[1, 2], [3, 4], [5, 6]])
scaled_X, mean, std = FeatureScaler.standardize(X)
```

---

### normalize

```python
normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Normalize features to [0, 1] range.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Tuple of (scaled_X, min, max)

**Example:**
```python
X = np.array([[1, 2], [3, 4], [5, 6]])
scaled_X, min_val, max_val = FeatureScaler.normalize(X)
```

---

### apply_standardization

```python
apply_standardization(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray
```

Apply standardization using pre-computed mean and std.

**Parameters:**
- `X`: Feature matrix
- `mean`: Mean values
- `std`: Standard deviation values

**Returns:**
- Scaled feature matrix

**Example:**
```python
X_train, mean, std = FeatureScaler.standardize(X_train)
X_test = FeatureScaler.apply_standardization(X_test, mean, std)
```

---

### apply_normalization

```python
apply_normalization(
    X: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray
) -> np.ndarray
```

Apply normalization using pre-computed min and max.

**Parameters:**
- `X`: Feature matrix
- `min_val`: Minimum values
- `max_val`: Maximum values

**Returns:**
- Scaled feature matrix

**Example:**
```python
X_train, min_val, max_val = FeatureScaler.normalize(X_train)
X_test = FeatureScaler.apply_normalization(X_test, min_val, max_val)
```

---

## MultipleLinearRegression Class

Multiple linear regression model with feature scaling and regularization.

### Constructor

```python
MultipleLinearRegression(
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    regularization: Optional[str] = None,
    alpha: float = 0.1,
    scale_features: bool = True,
    scaling_method: str = "standardize",
    fit_intercept: bool = True
) -> None
```

Initialize MultipleLinearRegression.

**Parameters:**
- `learning_rate`: Initial learning rate (default: 0.01)
- `max_iterations`: Maximum number of iterations (default: 1000)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `regularization`: Regularization type. Options: None, 'ridge', 'lasso' (default: None)
- `alpha`: Regularization strength (default: 0.1)
- `scale_features`: Whether to scale features (default: True)
- `scaling_method`: Scaling method. Options: 'standardize', 'normalize' (default: 'standardize')
- `fit_intercept`: Whether to fit intercept term (default: True)

**Example:**
```python
model = MultipleLinearRegression(
    learning_rate=0.01,
    regularization="ridge",
    alpha=0.1,
    scale_features=True,
    scaling_method="standardize"
)
```

---

### fit

```python
fit(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> "MultipleLinearRegression"
```

Fit multiple linear regression model using gradient descent.

**Parameters:**
- `X`: Feature matrix
- `y`: Target values

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([5, 8, 11])
model.fit(X, y)
```

---

### predict

```python
predict(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Make predictions using fitted model.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Predicted values as numpy array

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[4, 5], [5, 6]])
predictions = model.predict(X_test)
```

---

### score

```python
score(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> float
```

Calculate R-squared score.

**Parameters:**
- `X`: Feature matrix
- `y`: True target values

**Returns:**
- R-squared score (float between -∞ and 1.0)

**Example:**
```python
r2 = model.score(X, y)
print(f"R-squared: {r2:.4f}")
```

---

### get_cost_history

```python
get_cost_history() -> List[float]
```

Get cost history from training.

**Returns:**
- List of cost values per iteration

**Example:**
```python
cost_history = model.get_cost_history()
print(f"Initial cost: {cost_history[0]:.6f}")
print(f"Final cost: {cost_history[-1]:.6f}")
```

---

### plot_training_history

```python
plot_training_history(
    save_path: Optional[str] = None,
    show: bool = True
) -> None
```

Plot training history (cost).

**Parameters:**
- `save_path`: Optional path to save figure
- `show`: Whether to display plot (default: True)

**Returns:**
- None (displays or saves plot)

**Example:**
```python
model.plot_training_history(save_path="history.png", show=False)
```

---

## Attributes

### weights

Model weights (coefficients) after fitting. None before fitting.

**Type:** `Optional[np.ndarray]`

### intercept

Model intercept (bias term) after fitting. None before fitting.

**Type:** `Optional[float]`

### scale_params

Scaling parameters (mean/std or min/max) after fitting. None if scaling disabled.

**Type:** `Optional[Dict]`

### cost_history

List of cost values during training.

**Type:** `List[float]`

---

## Input Types

All methods accept the following input types for `X` and `y`:

- `List`: Python list
- `np.ndarray`: NumPy array
- `pd.DataFrame`: Pandas DataFrame (for X)
- `pd.Series`: Pandas Series (for y)

**Example:**
```python
import numpy as np
import pandas as pd

# List
X = [[1, 2], [2, 3], [3, 4]]
y = [5, 8, 11]

# NumPy array
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([5, 8, 11])

# Pandas
X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [2, 3, 4]})
y = pd.Series([5, 8, 11])
```

---

## Error Handling

### ValueError: Model must be fitted before prediction

Raised when `predict()` is called before `fit()`.

```python
model = MultipleLinearRegression()
model.predict(X)  # Raises ValueError
```

### ValueError: Length mismatch

Raised when X and y have different lengths.

```python
X = np.array([[1], [2], [3]])
y = np.array([1, 2])
model.fit(X, y)  # Raises ValueError
```

### ValueError: Unknown scaling method

Raised when invalid scaling method is specified.

```python
model = MultipleLinearRegression(scaling_method="unknown")
model.fit(X, y)  # Raises ValueError
```

---

## Notes

- Model uses Mean Squared Error (MSE) as cost function
- Regularization penalty is added to cost function
- Feature scaling is applied automatically during fit and predict
- Ridge regularization uses L2 penalty (sum of squared weights)
- Lasso regularization uses L1 penalty (sum of absolute weights)
- Intercept term is not regularized

---

## Regularization Details

### Ridge Regression (L2)

Cost function:
```
J(θ) = MSE + α * Σw²
```

Gradient:
```
∇J(θ) = (1/m) * X.T @ error + 2α * w
```

**Properties:**
- Shrinks weights towards zero
- Keeps all features
- Good for multicollinearity
- Smooth penalty

### Lasso Regression (L1)

Cost function:
```
J(θ) = MSE + α * Σ|w|
```

Gradient:
```
∇J(θ) = (1/m) * X.T @ error + α * sign(w)
```

**Properties:**
- Can set weights to exactly zero
- Performs feature selection
- Good for sparse models
- Non-smooth penalty

---

## Feature Scaling Details

### Standardization

Transforms features to have zero mean and unit variance:
```
x_scaled = (x - mean) / std
```

**When to use:**
- Features have different scales
- Normal distribution assumed
- Most common choice

### Normalization

Transforms features to [0, 1] range:
```
x_scaled = (x - min) / (max - min)
```

**When to use:**
- Need bounded values
- Non-normal distributions
- Neural networks (sometimes)

---

## Best Practices

1. **Always scale features**: Essential for multiple features with different scales
2. **Use standardization by default**: Works well for most cases
3. **Choose regularization type**: Ridge for multicollinearity, Lasso for feature selection
4. **Tune alpha parameter**: Balance between fit and regularization
5. **Monitor cost history**: Ensure model is converging
6. **Compare models**: Try different regularization types and strengths
