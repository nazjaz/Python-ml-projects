# Logistic Regression API Documentation

This document provides detailed API documentation for the Logistic Regression for Binary Classification implementation.

## LogisticRegression Class

Logistic regression model for binary classification.

### Constructor

```python
LogisticRegression(
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    scale_features: bool = True,
    fit_intercept: bool = True
) -> None
```

Initialize LogisticRegression.

**Parameters:**
- `learning_rate`: Initial learning rate (default: 0.01)
- `max_iterations`: Maximum number of iterations (default: 1000)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `scale_features`: Whether to scale features (default: True)
- `fit_intercept`: Whether to fit intercept term (default: True)

**Example:**
```python
model = LogisticRegression(
    learning_rate=0.01,
    max_iterations=1000,
    scale_features=True
)
```

---

## Methods

### fit

```python
fit(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> "LogisticRegression"
```

Fit logistic regression model using gradient descent.

**Parameters:**
- `X`: Feature matrix
- `y`: Target labels (0 or 1)

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid or labels not binary

**Example:**
```python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])
model.fit(X, y)
```

---

### predict_proba

```python
predict_proba(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict class probabilities.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Probability of class 1 for each sample (array of values between 0 and 1)

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[6], [7], [8]])
probabilities = model.predict_proba(X_test)
# Returns: array([0.85, 0.92, 0.96])
```

---

### predict

```python
predict(
    X: Union[List, np.ndarray, pd.DataFrame],
    threshold: float = 0.5
) -> np.ndarray
```

Predict class labels.

**Parameters:**
- `X`: Feature matrix
- `threshold`: Decision threshold (default: 0.5)

**Returns:**
- Predicted class labels (0 or 1) as numpy array

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[6], [7], [8]])
predictions = model.predict(X_test, threshold=0.5)
# Returns: array([1, 1, 1])
```

---

### score

```python
score(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> float
```

Calculate accuracy score.

**Parameters:**
- `X`: Feature matrix
- `y`: True target labels

**Returns:**
- Accuracy score (float between 0 and 1)

**Example:**
```python
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
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

Scaling parameters (mean and std) after fitting. None if scaling disabled.

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
X = [[1], [2], [3]]
y = [0, 1, 1]

# NumPy array
X = np.array([[1], [2], [3]])
y = np.array([0, 1, 1])

# Pandas
X = pd.DataFrame({"feature": [1, 2, 3]})
y = pd.Series([0, 1, 1])
```

---

## Error Handling

### ValueError: Model must be fitted before prediction

Raised when `predict()` or `predict_proba()` is called before `fit()`.

```python
model = LogisticRegression()
model.predict(X)  # Raises ValueError
```

### ValueError: y must contain only 0 and 1

Raised when target labels are not binary.

```python
X = np.array([[1], [2], [3]])
y = np.array([0, 1, 2])
model.fit(X, y)  # Raises ValueError
```

### ValueError: Length mismatch

Raised when X and y have different lengths.

```python
X = np.array([[1], [2], [3]])
y = np.array([0, 1])
model.fit(X, y)  # Raises ValueError
```

---

## Notes

- Model uses logistic cost function (log loss / cross-entropy)
- Sigmoid activation ensures outputs are probabilities
- Feature scaling is applied automatically (standardization)
- Decision threshold of 0.5 is default but can be customized
- Model supports multiple features
- Intercept term is optional

---

## Sigmoid Function

The sigmoid function maps any real number to (0, 1):

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties:**
- Output range: (0, 1)
- σ(0) = 0.5
- Smooth and differentiable
- S-shaped curve

**Implementation:**
- Values are clipped to [-500, 500] to prevent overflow
- Returns values strictly between 0 and 1

---

## Logistic Cost Function

Logistic regression uses log loss (cross-entropy):

```
J(θ) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
```

Where:
- `h = σ(X @ θ)` is predicted probability
- `y` is true label (0 or 1)
- `m` is number of samples

**Properties:**
- Penalizes confident wrong predictions
- Convex function
- Well-suited for probability outputs
- Small epsilon (1e-15) added to prevent log(0)

---

## Gradient Descent

Gradient for logistic regression:

```
∇J(θ) = (1/m) * X.T @ (h - y)
```

Where:
- `h = σ(X @ θ)` is predicted probability
- `y` is true label
- `m` is number of samples

Weight update:
```
θ = θ - α * ∇J(θ)
```

Where `α` is the learning rate.

---

## Decision Threshold

Predictions are made by comparing probabilities to a threshold:

```
prediction = 1 if P(y=1|x) >= threshold else 0
```

**Default threshold:** 0.5

**Adjusting threshold:**
- Lower threshold: More positive predictions (higher recall, lower precision)
- Higher threshold: Fewer positive predictions (lower recall, higher precision)
- Useful for imbalanced datasets

---

## Best Practices

1. **Always scale features**: Essential for gradient descent convergence
2. **Use appropriate learning rate**: Start with 0.01, adjust based on cost history
3. **Monitor cost history**: Ensure cost decreases during training
4. **Tune decision threshold**: Default 0.5 may not be optimal
5. **Use probabilities**: `predict_proba()` provides more information
6. **Check class balance**: Imbalanced classes may need threshold adjustment
