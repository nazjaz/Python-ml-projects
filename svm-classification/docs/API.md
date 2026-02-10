# SVM API Documentation

## SVM Class

### `SVM(kernel='linear', C=1.0, degree=3, gamma=None, coef0=0.0, tol=1e-3, max_iter=1000)`

Support Vector Machine for classification.

#### Parameters

- `kernel` (str): Kernel type. Options: "linear", "poly", "rbf" (default: "linear").
- `C` (float): Regularization parameter (default: 1.0).
- `degree` (int): Degree for polynomial kernel (default: 3).
- `gamma` (float, optional): Kernel coefficient for RBF and polynomial. If None, uses 1/n_features (default: None).
- `coef0` (float): Independent term for polynomial kernel (default: 0.0).
- `tol` (float): Tolerance for stopping criterion (default: 1e-3).
- `max_iter` (int): Maximum number of iterations (default: 1000).

#### Attributes

- `support_vectors_` (ndarray): Support vectors.
- `support_vector_labels_` (ndarray): Labels of support vectors.
- `support_vector_alphas_` (ndarray): Lagrange multipliers for support vectors.
- `bias_` (float): Bias term.
- `X_train_` (ndarray): Training feature matrix.
- `y_train_` (ndarray): Training labels.
- `n_features_` (int): Number of features.
- `classes_` (ndarray): Unique class labels.

### Methods

#### `fit(X, y)`

Fit the SVM model.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,) (binary: -1, 1 or 0, 1).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
svm = SVM(kernel="linear", C=1.0)
svm.fit(X, y)
```

#### `predict(X)`

Predict class labels.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Predicted class labels of shape (n_samples,).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
predictions = svm.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Class probabilities of shape (n_samples, 2).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
probabilities = svm.predict_proba(X)
```

#### `score(X, y)`

Calculate classification accuracy.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True labels.

**Returns:**
- `float`: Classification accuracy (0-1).

**Example:**
```python
accuracy = svm.score(X, y)
```

#### `plot_decision_boundary(X=None, y=None, save_path=None, show=True)`

Plot decision boundary (for 2D features only).

**Parameters:**
- `X` (array-like, optional): Feature matrix for visualization.
- `y` (array-like, optional): Target labels for visualization.
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).

**Example:**
```python
svm.plot_decision_boundary(X=X, y=y, save_path="boundary.png")
```

## Usage Examples

### Basic SVM

```python
from src.main import SVM
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

svm = SVM(kernel="linear", C=1.0)
svm.fit(X, y)
predictions = svm.predict(X)
```

### With Different Kernels

```python
# Linear kernel
svm_linear = SVM(kernel="linear", C=1.0)
svm_linear.fit(X, y)

# Polynomial kernel
svm_poly = SVM(kernel="poly", C=1.0, degree=3, gamma=0.1)
svm_poly.fit(X, y)

# RBF kernel
svm_rbf = SVM(kernel="rbf", C=1.0, gamma=0.1)
svm_rbf.fit(X, y)
```

### Class Probabilities

```python
from src.main import SVM
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

svm = SVM(kernel="linear")
svm.fit(X, y)

probabilities = svm.predict_proba(X)
```

### Decision Boundary Visualization

```python
from src.main import SVM
import numpy as np

# 2D features required
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

svm = SVM(kernel="rbf", C=1.0, gamma=0.1)
svm.fit(X, y)

svm.plot_decision_boundary(X=X, y=y)
```

### Tuning Hyperparameters

```python
from src.main import SVM
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Try different C values
for C in [0.1, 1.0, 10.0]:
    svm = SVM(kernel="linear", C=C)
    svm.fit(X, y)
    accuracy = svm.score(X, y)
    print(f"C={C}: accuracy={accuracy:.4f}, support_vectors={len(svm.support_vectors_)}")
```
