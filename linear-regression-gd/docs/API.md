# Linear Regression with Gradient Descent API Documentation

This document provides detailed API documentation for the Linear Regression with Gradient Descent implementation.

## LearningRateScheduler Class

Static methods for creating learning rate scheduling functions.

### constant

```python
constant(initial_lr: float, **kwargs) -> Callable[[int], float]
```

Constant learning rate.

**Parameters:**
- `initial_lr`: Initial learning rate

**Returns:**
- Function that returns constant learning rate

**Example:**
```python
scheduler = LearningRateScheduler.constant(initial_lr=0.01)
lr = scheduler(100)  # Returns 0.01
```

---

### exponential_decay

```python
exponential_decay(
    initial_lr: float,
    decay_rate: float = 0.95,
    **kwargs
) -> Callable[[int], float]
```

Exponential decay learning rate.

**Parameters:**
- `initial_lr`: Initial learning rate
- `decay_rate`: Decay rate per epoch (default: 0.95)

**Returns:**
- Function that returns exponentially decaying learning rate

**Example:**
```python
scheduler = LearningRateScheduler.exponential_decay(
    initial_lr=0.01, decay_rate=0.95
)
lr = scheduler(10)  # Returns 0.01 * 0.95^10
```

---

### step_decay

```python
step_decay(
    initial_lr: float,
    drop_rate: float = 0.5,
    epochs_drop: int = 10,
    **kwargs
) -> Callable[[int], float]
```

Step decay learning rate.

**Parameters:**
- `initial_lr`: Initial learning rate
- `drop_rate`: Factor to drop learning rate (default: 0.5)
- `epochs_drop`: Number of epochs before dropping (default: 10)

**Returns:**
- Function that returns step-decaying learning rate

**Example:**
```python
scheduler = LearningRateScheduler.step_decay(
    initial_lr=0.01, drop_rate=0.5, epochs_drop=10
)
lr = scheduler(20)  # Returns 0.01 * 0.5^2 = 0.0025
```

---

### polynomial_decay

```python
polynomial_decay(
    initial_lr: float,
    end_lr: float = 0.001,
    power: float = 1.0,
    max_epochs: int = 100,
    **kwargs
) -> Callable[[int], float]
```

Polynomial decay learning rate.

**Parameters:**
- `initial_lr`: Initial learning rate
- `end_lr`: Final learning rate (default: 0.001)
- `power`: Power of polynomial (default: 1.0)
- `max_epochs`: Maximum number of epochs (default: 100)

**Returns:**
- Function that returns polynomially decaying learning rate

**Example:**
```python
scheduler = LearningRateScheduler.polynomial_decay(
    initial_lr=0.01, end_lr=0.001, max_epochs=100
)
lr = scheduler(50)  # Returns interpolated value
```

---

## LinearRegression Class

Linear regression model implemented from scratch using gradient descent.

### Constructor

```python
LinearRegression(
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    scheduler: Optional[str] = None,
    scheduler_params: Optional[Dict] = None,
    fit_intercept: bool = True
) -> None
```

Initialize LinearRegression.

**Parameters:**
- `learning_rate`: Initial learning rate (default: 0.01)
- `max_iterations`: Maximum number of iterations (default: 1000)
- `tolerance`: Convergence tolerance (default: 1e-6)
- `scheduler`: Learning rate scheduler type. Options: 'constant', 'exponential', 'step', 'polynomial' (default: None, uses constant)
- `scheduler_params`: Parameters for scheduler (default: None)
- `fit_intercept`: Whether to fit intercept term (default: True)

**Example:**
```python
model = LinearRegression(
    learning_rate=0.01,
    max_iterations=1000,
    scheduler="exponential",
    scheduler_params={"decay_rate": 0.95}
)
```

---

### fit

```python
fit(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> "LinearRegression"
```

Fit linear regression model using gradient descent.

**Parameters:**
- `X`: Feature matrix
- `y`: Target values

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
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
X_test = np.array([[6], [7], [8]])
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

### get_lr_history

```python
get_lr_history() -> List[float]
```

Get learning rate history from training.

**Returns:**
- List of learning rate values per iteration

**Example:**
```python
lr_history = model.get_lr_history()
print(f"Initial LR: {lr_history[0]:.6f}")
print(f"Final LR: {lr_history[-1]:.6f}")
```

---

### plot_training_history

```python
plot_training_history(
    save_path: Optional[str] = None,
    show: bool = True
) -> None
```

Plot training history (cost and learning rate).

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

### cost_history

List of cost values during training.

**Type:** `List[float]`

### lr_history

List of learning rate values during training.

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
y = [2, 4, 6]

# NumPy array
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

# Pandas
X = pd.DataFrame({"feature": [1, 2, 3]})
y = pd.Series([2, 4, 6])
```

---

## Error Handling

### ValueError: Model must be fitted before prediction

Raised when `predict()` is called before `fit()`.

```python
model = LinearRegression()
model.predict(X)  # Raises ValueError
```

### ValueError: Length mismatch

Raised when X and y have different lengths.

```python
X = np.array([[1], [2], [3]])
y = np.array([1, 2])
model.fit(X, y)  # Raises ValueError
```

### ValueError: Input data cannot be empty

Raised when input data is empty.

```python
X = np.array([]).reshape(0, 1)
y = np.array([])
model.fit(X, y)  # Raises ValueError
```

---

## Notes

- Model uses Mean Squared Error (MSE) as cost function
- Gradient descent updates weights iteratively
- Convergence is detected when cost change < tolerance
- Learning rate scheduling helps with convergence
- Model supports both single and multiple features
- Intercept term is optional

---

## Gradient Descent Algorithm

The model uses batch gradient descent:

1. Initialize weights to zero
2. For each iteration:
   - Compute predictions: `y_pred = X @ theta`
   - Compute error: `error = y_pred - y`
   - Compute gradient: `gradient = (1/m) * X.T @ error`
   - Update weights: `theta = theta - lr * gradient`
   - Compute cost: `cost = mean((y_pred - y)^2)`
   - Check convergence

---

## Learning Rate Scheduling

### Constant
- Learning rate remains fixed
- Simple but may not converge optimally

### Exponential Decay
- Learning rate decreases exponentially
- Good for fine-tuning
- Formula: `lr(t) = lr₀ * decay_rate^t`

### Step Decay
- Learning rate drops at intervals
- Good for staged training
- Formula: `lr(t) = lr₀ * drop_rate^(t // epochs_drop)`

### Polynomial Decay
- Learning rate decreases polynomially
- Smooth transition
- Formula: `lr(t) = (lr₀ - lr_end) * (1 - t/max)^power + lr_end`

---

## Best Practices

1. **Scale features**: Normalize or standardize for better convergence
2. **Choose learning rate**: Start with 0.01, adjust based on cost history
3. **Use scheduling**: Helps with convergence and fine-tuning
4. **Monitor training**: Plot cost history to visualize progress
5. **Set tolerance**: Prevents unnecessary iterations
6. **Check convergence**: Ensure cost decreases during training
