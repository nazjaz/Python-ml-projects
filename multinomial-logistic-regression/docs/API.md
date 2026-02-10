# Multinomial Logistic Regression API Documentation

This document provides detailed API documentation for the Multinomial Logistic Regression for Multi-Class Classification implementation.

## MultinomialLogisticRegression Class

Multinomial logistic regression model for multi-class classification.

### Constructor

```python
MultinomialLogisticRegression(
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    scale_features: bool = True,
    fit_intercept: bool = True
) -> None
```

Initialize MultinomialLogisticRegression.

**Parameters:**
- `learning_rate`: Initial learning rate (default: 0.01)
- `max_iterations`: Maximum number of iterations (default: 1000)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `scale_features`: Whether to scale features (default: True)
- `fit_intercept`: Whether to fit intercept term (default: True)

**Example:**
```python
model = MultinomialLogisticRegression(
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
) -> "MultinomialLogisticRegression"
```

Fit multinomial logistic regression model using gradient descent.

**Parameters:**
- `X`: Feature matrix
- `y`: Target class labels

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid or less than 2 classes

**Example:**
```python
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 1, 1, 2, 2])
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
- Probability matrix (shape: [n_samples, n_classes]) with probabilities for each class

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[7], [8], [9]])
probabilities = model.predict_proba(X_test)
# Returns: array([[0.1, 0.3, 0.6],
#                 [0.05, 0.25, 0.7],
#                 [0.02, 0.18, 0.8]])
```

---

### predict

```python
predict(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict class labels.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Predicted class labels as numpy array

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[7], [8], [9]])
predictions = model.predict(X_test)
# Returns: array([2, 2, 2])
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

Model weights (coefficients) after fitting. Shape: [n_features, n_classes]. None before fitting.

**Type:** `Optional[np.ndarray]`

### intercept

Model intercept (bias term) after fitting. Shape: [n_classes]. None before fitting.

**Type:** `Optional[np.ndarray]`

### classes

Unique class labels found during fitting. None before fitting.

**Type:** `Optional[np.ndarray]`

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
y = [0, 1, 2]

# NumPy array
X = np.array([[1], [2], [3]])
y = np.array([0, 1, 2])

# Pandas
X = pd.DataFrame({"feature": [1, 2, 3]})
y = pd.Series([0, 1, 2])
```

---

## Error Handling

### ValueError: Model must be fitted before prediction

Raised when `predict()` or `predict_proba()` is called before `fit()`.

```python
model = MultinomialLogisticRegression()
model.predict(X)  # Raises ValueError
```

### ValueError: At least 2 classes required

Raised when target has less than 2 unique classes.

```python
X = np.array([[1], [2], [3]])
y = np.array([0, 0, 0])
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

- Model uses cross-entropy cost function
- Softmax activation ensures outputs are probability distributions
- Feature scaling is applied automatically (standardization)
- One-hot encoding is used internally for multi-class labels
- Model supports any number of classes (≥ 2)
- Weight matrix has shape [n_features, n_classes]
- Intercept vector has shape [n_classes]

---

## Softmax Function

The softmax function converts raw scores to probabilities:

```
softmax(z)_i = exp(z_i) / Σ exp(z_j)
```

**Properties:**
- Outputs probability distribution (sums to 1 for each sample)
- All outputs are between 0 and 1
- Smooth and differentiable
- Numerical stability: subtract max before exp to prevent overflow

**Implementation:**
- Values are shifted by max to prevent numerical overflow
- Returns probability distribution for each sample

---

## Cross-Entropy Cost Function

Multinomial logistic regression uses cross-entropy loss:

```
J(θ) = -(1/m) * Σ Σ y_ij * log(h_ij)
```

Where:
- `h_ij` is predicted probability of class j for sample i
- `y_ij` is one-hot encoded true label (1 if true class, 0 otherwise)
- `m` is number of samples

**Properties:**
- Penalizes confident wrong predictions
- Convex function
- Well-suited for probability outputs
- Small epsilon (1e-15) added to prevent log(0)

---

## Gradient Descent

Gradient for multinomial logistic regression (for each class k):

```
∇J(θ_k) = (1/m) * X.T @ (h_k - y_k)
```

Where:
- `h_k` is predicted probabilities for class k
- `y_k` is one-hot encoded labels for class k
- `m` is number of samples

Weight update for each class:
```
θ_k = θ_k - α * ∇J(θ_k)
```

Where `α` is the learning rate.

---

## One-Hot Encoding

Labels are converted to one-hot encoding internally:

- Class 0: [1, 0, 0]
- Class 1: [0, 1, 0]
- Class 2: [0, 0, 1]

This allows the model to predict probabilities for all classes simultaneously.

---

## Best Practices

1. **Always scale features**: Essential for gradient descent convergence
2. **Use appropriate learning rate**: Start with 0.01, adjust based on cost history
3. **Monitor cost history**: Ensure cost decreases during training
4. **Check class balance**: Imbalanced classes may need special handling
5. **Use probabilities**: `predict_proba()` provides more information than `predict()`
6. **Handle many classes**: Softmax works well even with many classes
