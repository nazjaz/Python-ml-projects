# K-Nearest Neighbors API Documentation

This document provides detailed API documentation for the K-Nearest Neighbors Classifier implementation.

## DistanceMetrics Class

Static methods for distance metric calculations.

### euclidean

```python
euclidean(x1: np.ndarray, x2: np.ndarray) -> float
```

Calculate Euclidean distance.

**Parameters:**
- `x1`: First feature vector
- `x2`: Second feature vector

**Returns:**
- Euclidean distance (float)

**Example:**
```python
x1 = np.array([0, 0])
x2 = np.array([3, 4])
distance = DistanceMetrics.euclidean(x1, x2)  # Returns 5.0
```

---

### manhattan

```python
manhattan(x1: np.ndarray, x2: np.ndarray) -> float
```

Calculate Manhattan (L1) distance.

**Parameters:**
- `x1`: First feature vector
- `x2`: Second feature vector

**Returns:**
- Manhattan distance (float)

**Example:**
```python
x1 = np.array([0, 0])
x2 = np.array([3, 4])
distance = DistanceMetrics.manhattan(x1, x2)  # Returns 7.0
```

---

### minkowski

```python
minkowski(x1: np.ndarray, x2: np.ndarray, p: float = 2.0) -> float
```

Calculate Minkowski distance.

**Parameters:**
- `x1`: First feature vector
- `x2`: Second feature vector
- `p`: Power parameter (p=1: Manhattan, p=2: Euclidean)

**Returns:**
- Minkowski distance (float)

**Example:**
```python
x1 = np.array([0, 0])
x2 = np.array([3, 4])
distance = DistanceMetrics.minkowski(x1, x2, p=3.0)
```

---

### hamming

```python
hamming(x1: np.ndarray, x2: np.ndarray) -> float
```

Calculate Hamming distance (for categorical data).

**Parameters:**
- `x1`: First feature vector
- `x2`: Second feature vector

**Returns:**
- Hamming distance (proportion of differing elements, float)

**Example:**
```python
x1 = np.array([0, 1, 0, 1])
x2 = np.array([1, 1, 0, 0])
distance = DistanceMetrics.hamming(x1, x2)  # Returns 0.5
```

---

### cosine

```python
cosine(x1: np.ndarray, x2: np.ndarray) -> float
```

Calculate cosine distance.

**Parameters:**
- `x1`: First feature vector
- `x2`: Second feature vector

**Returns:**
- Cosine distance (1 - cosine similarity, float)

**Example:**
```python
x1 = np.array([1, 0])
x2 = np.array([0, 1])
distance = DistanceMetrics.cosine(x1, x2)  # Returns 1.0
```

---

## KNNClassifier Class

K-nearest neighbors classifier.

### Constructor

```python
KNNClassifier(
    k: int = 5,
    distance_metric: str = "euclidean",
    metric_params: Optional[Dict] = None,
    scale_features: bool = True
) -> None
```

Initialize KNNClassifier.

**Parameters:**
- `k`: Number of neighbors to consider (default: 5)
- `distance_metric`: Distance metric. Options: "euclidean", "manhattan", "minkowski", "hamming", "cosine" (default: "euclidean")
- `metric_params`: Additional parameters for distance metric (e.g., {'p': 3} for Minkowski) (default: None)
- `scale_features`: Whether to scale features (default: True)

**Example:**
```python
knn = KNNClassifier(
    k=5,
    distance_metric="manhattan",
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
) -> "KNNClassifier"
```

Fit KNN classifier (stores training data).

**Parameters:**
- `X`: Feature matrix
- `y`: Target labels

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid or k is too large

**Example:**
```python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])
knn.fit(X, y)
```

---

### predict

```python
predict(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict class labels for test samples.

**Parameters:**
- `X`: Test feature matrix

**Returns:**
- Predicted class labels as numpy array

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[1.5], [3.5]])
predictions = knn.predict(X_test)
```

---

### predict_proba

```python
predict_proba(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict class probabilities for test samples.

**Parameters:**
- `X`: Test feature matrix

**Returns:**
- Probability matrix (shape: [n_samples, n_classes])

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[1.5], [3.5]])
probabilities = knn.predict_proba(X_test)
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
accuracy = knn.score(X, y)
```

---

## Attributes

### X_train

Training feature matrix. None before fitting.

**Type:** `Optional[np.ndarray]`

### y_train

Training target labels. None before fitting.

**Type:** `Optional[np.ndarray]`

### scale_params

Scaling parameters (mean and std) after fitting. None if scaling disabled.

**Type:** `Optional[Dict]`

---

## KNNOptimizer Class

Optimize k-value for KNN classifier using cross-validation.

### Constructor

```python
KNNOptimizer(
    k_range: List[int],
    distance_metric: str = "euclidean",
    metric_params: Optional[Dict] = None,
    cv_folds: int = 5,
    scale_features: bool = True
) -> None
```

Initialize KNNOptimizer.

**Parameters:**
- `k_range`: List of k values to test
- `distance_metric`: Distance metric to use (default: "euclidean")
- `metric_params`: Additional parameters for distance metric (default: None)
- `cv_folds`: Number of cross-validation folds (default: 5)
- `scale_features`: Whether to scale features (default: True)

**Example:**
```python
optimizer = KNNOptimizer(
    k_range=[1, 3, 5, 7, 9],
    cv_folds=5,
    scale_features=True
)
```

---

## Methods

### optimize

```python
optimize(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> Dict[str, Union[int, float, Dict]]
```

Optimize k-value using cross-validation.

**Parameters:**
- `X`: Feature matrix
- `y`: Target labels

**Returns:**
- Dictionary containing:
  - `best_k`: Optimal k value (int)
  - `best_score`: Best cross-validation score (float)
  - `results`: Dictionary with scores for each k value (Dict)

**Example:**
```python
results = optimizer.optimize(X, y)
print(f"Best k: {results['best_k']}")
print(f"Best score: {results['best_score']:.4f}")
```

---

### plot_optimization_results

```python
plot_optimization_results(
    save_path: Optional[str] = None,
    show: bool = True
) -> None
```

Plot k-value optimization results.

**Parameters:**
- `save_path`: Optional path to save figure
- `show`: Whether to display plot (default: True)

**Returns:**
- None (displays or saves plot)

**Example:**
```python
optimizer.plot_optimization_results(save_path="optimization.png", show=False)
```

---

## Attributes

### results

Optimization results dictionary. None before calling `optimize()`.

**Type:** `Optional[Dict]`

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
y = [0, 1, 0]

# NumPy array
X = np.array([[1], [2], [3]])
y = np.array([0, 1, 0])

# Pandas
X = pd.DataFrame({"feature": [1, 2, 3]})
y = pd.Series([0, 1, 0])
```

---

## Error Handling

### ValueError: Model must be fitted before prediction

Raised when `predict()` or `predict_proba()` is called before `fit()`.

```python
knn = KNNClassifier(k=5)
knn.predict(X)  # Raises ValueError
```

### ValueError: k cannot be greater than number of samples

Raised when k exceeds the number of training samples.

```python
X = np.array([[1], [2], [3]])
y = np.array([0, 1, 0])
knn = KNNClassifier(k=5)
knn.fit(X, y)  # Raises ValueError
```

### ValueError: k must be at least 1

Raised when k is less than 1.

```python
knn = KNNClassifier(k=0)
knn.fit(X, y)  # Raises ValueError
```

---

## Notes

- KNN is a lazy learning algorithm (no training phase, just stores data)
- Distance calculations are done during prediction
- Feature scaling is important for distance-based algorithms
- K-value optimization uses cross-validation to prevent overfitting
- Probability predictions are based on proportion of k neighbors in each class

---

## Distance Metric Selection Guide

- **Euclidean**: Default choice, works well for most continuous data
- **Manhattan**: Less sensitive to outliers, good for high-dimensional data
- **Minkowski**: Flexible, can interpolate between Euclidean and Manhattan
- **Hamming**: For categorical or binary features
- **Cosine**: For high-dimensional sparse data, text classification

---

## Best Practices

1. **Always scale features**: Distance metrics are scale-sensitive
2. **Optimize k-value**: Use cross-validation to find optimal k
3. **Try different metrics**: Different metrics work better for different data
4. **Use odd k for binary classification**: Avoid ties
5. **Consider computational cost**: KNN is slow for large datasets
6. **Handle class imbalance**: May need special handling
