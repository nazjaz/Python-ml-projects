# LDA API Documentation

## LDA Class

### `LDA(n_components=None, solver='eigen', shrinkage=None)`

Linear Discriminant Analysis for dimensionality reduction and classification.

#### Parameters

- `n_components` (int, optional): Number of components to keep. If None, keeps min(n_features, n_classes - 1) components. Default: None.
- `solver` (str): Solver to use. Options: "eigen" (default), "svd". Default: "eigen".
- `shrinkage` (float, optional): Shrinkage parameter for regularization (0-1). If None, no shrinkage is applied. Default: None.

#### Attributes

- `scalings` (ndarray): Transformation matrix (eigenvectors).
- `xbar_` (ndarray): Overall mean of training data.
- `classes_` (ndarray): Unique class labels.
- `priors_` (ndarray): Prior probabilities for each class.
- `means_` (ndarray): Class means.
- `covariance_` (ndarray): Within-class covariance matrix.
- `n_components_` (int): Actual number of components used.
- `explained_variance_ratio_` (ndarray): Explained variance ratio for each component.

### Methods

#### `fit(X, y)`

Fit the LDA model.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid (empty data, insufficient classes, mismatched lengths).

**Example:**
```python
lda = LDA(n_components=2)
lda.fit(X, y)
```

#### `transform(X)`

Transform data to lower-dimensional space.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Transformed data of shape (n_samples, n_components_).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
X_transformed = lda.transform(X)
```

#### `fit_transform(X, y)`

Fit model and transform data.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,).

**Returns:**
- `ndarray`: Transformed data of shape (n_samples, n_components_).

**Example:**
```python
X_transformed = lda.fit_transform(X, y)
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
predictions = lda.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Class probabilities of shape (n_samples, n_classes).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
probabilities = lda.predict_proba(X)
```

#### `score(X, y)`

Calculate classification accuracy.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): True labels of shape (n_samples,).

**Returns:**
- `float`: Classification accuracy (0-1).

**Example:**
```python
accuracy = lda.score(X, y)
```

#### `get_explained_variance_ratio()`

Get explained variance ratio for each component.

**Returns:**
- `ndarray`: Explained variance ratio array.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
variance_ratio = lda.get_explained_variance_ratio()
```

#### `plot_components(X=None, y=None, save_path=None, show=True)`

Plot LDA components with data visualization.

**Parameters:**
- `X` (array-like, optional): Feature matrix for visualization.
- `y` (array-like, optional): Target labels for visualization.
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot. Default: True.

**Example:**
```python
lda.plot_components(X=X, y=y, save_path="plot.png")
```

## Usage Examples

### Basic Dimensionality Reduction

```python
from src.main import LDA
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

lda = LDA(n_components=1)
lda.fit(X, y)
X_transformed = lda.transform(X)
```

### Classification

```python
from src.main import LDA
import numpy as np

X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y_train = np.array([0, 0, 0, 1, 1, 1])

lda = LDA(n_components=1)
lda.fit(X_train, y_train)

X_test = np.array([[2.5, 3.5], [5.5, 6.5]])
predictions = lda.predict(X_test)
probabilities = lda.predict_proba(X_test)
```

### With Different Solvers

```python
# Eigen solver (default)
lda_eigen = LDA(n_components=2, solver="eigen")
lda_eigen.fit(X, y)

# SVD solver
lda_svd = LDA(n_components=2, solver="svd")
lda_svd.fit(X, y)
```

### With Shrinkage Regularization

```python
lda = LDA(n_components=2, solver="eigen", shrinkage=0.5)
lda.fit(X, y)
```

### Multiclass Classification

```python
X = np.array([
    [1, 2], [2, 3], [3, 4],
    [4, 5], [5, 6], [6, 7],
    [7, 8], [8, 9], [9, 10]
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

lda = LDA(n_components=2)
lda.fit(X, y)
predictions = lda.predict(X)
```
