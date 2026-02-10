# K-Means Clustering API Documentation

This document provides detailed API documentation for the K-Means Clustering implementation.

## KMeans Class

K-means clustering algorithm.

### Constructor

```python
KMeans(
    n_clusters: int = 3,
    max_iterations: int = 300,
    tolerance: float = 1e-4,
    init: str = "random",
    random_state: Optional[int] = None
) -> None
```

Initialize KMeans.

**Parameters:**
- `n_clusters`: Number of clusters (default: 3)
- `max_iterations`: Maximum number of iterations (default: 300)
- `tolerance`: Convergence tolerance (default: 1e-4)
- `init`: Initialization method. Options: "random", "k-means++" (default: "random")
- `random_state`: Random seed for reproducibility (default: None)

**Example:**
```python
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    random_state=42
)
```

---

## Methods

### fit

```python
fit(X: Union[List, np.ndarray, pd.DataFrame]) -> "KMeans"
```

Fit k-means clustering model.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
kmeans.fit(X)
```

---

### predict

```python
predict(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict cluster labels for new samples.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Cluster labels as numpy array

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_new = np.array([[1.5, 2.5], [8.5, 9.5]])
predictions = kmeans.predict(X_new)
```

---

### fit_predict

```python
fit_predict(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Fit model and predict cluster labels.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Cluster labels as numpy array

**Example:**
```python
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
labels = kmeans.fit_predict(X)
```

---

## Attributes

### centroids

Cluster centroids after fitting. None before fitting.

**Type:** `Optional[np.ndarray]` (shape: [n_clusters, n_features])

### labels

Cluster labels for training data. None before fitting.

**Type:** `Optional[np.ndarray]`

### inertia

Within-cluster sum of squares. None before fitting.

**Type:** `Optional[float]`

### n_iterations

Number of iterations until convergence. 0 before fitting.

**Type:** `int`

---

## ElbowMethod Class

Elbow method for optimal cluster number selection.

### Constructor

```python
ElbowMethod(
    k_range: List[int],
    max_iterations: int = 300,
    tolerance: float = 1e-4,
    init: str = "random",
    random_state: Optional[int] = None,
    n_runs: int = 1
) -> None
```

Initialize ElbowMethod.

**Parameters:**
- `k_range`: List of k values to test
- `max_iterations`: Maximum iterations for each k-means run (default: 300)
- `tolerance`: Convergence tolerance (default: 1e-4)
- `init`: Initialization method (default: "random")
- `random_state`: Random seed for reproducibility (default: None)
- `n_runs`: Number of runs per k value for averaging (default: 1)

**Example:**
```python
elbow = ElbowMethod(
    k_range=[2, 3, 4, 5, 6],
    init="k-means++",
    random_state=42
)
```

---

## Methods

### fit

```python
fit(X: Union[List, np.ndarray, pd.DataFrame]) -> Dict
```

Find optimal k using elbow method.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Dictionary containing:
  - `k_range`: List of tested k values
  - `inertias`: List of inertia values
  - `optimal_k`: Optimal k value (if elbow detected)
  - `results`: Detailed results for each k

**Example:**
```python
results = elbow.fit(X)
print(f"Optimal k: {results['optimal_k']}")
```

---

### plot_elbow

```python
plot_elbow(
    save_path: Optional[str] = None,
    show: bool = True
) -> None
```

Plot elbow curve.

**Parameters:**
- `save_path`: Optional path to save figure
- `show`: Whether to display plot (default: True)

**Returns:**
- None (displays or saves plot)

**Example:**
```python
elbow.plot_elbow(save_path="elbow.png", show=False)
```

---

## Attributes

### inertias

List of inertia values for each k. None before calling `fit()`.

**Type:** `Optional[List[float]]`

### results

Detailed results dictionary. None before calling `fit()`.

**Type:** `Optional[Dict]`

---

## Input Types

All methods accept the following input types for `X`:

- `List`: Python list
- `np.ndarray`: NumPy array
- `pd.DataFrame`: Pandas DataFrame

**Example:**
```python
import numpy as np
import pandas as pd

# List
X = [[1, 2], [2, 3], [8, 9]]

# NumPy array
X = np.array([[1, 2], [2, 3], [8, 9]])

# Pandas
X = pd.DataFrame({"feature1": [1, 2, 8], "feature2": [2, 3, 9]})
```

---

## Error Handling

### ValueError: Model must be fitted before prediction

Raised when `predict()` is called before `fit()`.

```python
kmeans = KMeans(n_clusters=3)
kmeans.predict(X)  # Raises ValueError
```

### ValueError: n_clusters cannot be greater than number of samples

Raised when n_clusters exceeds number of samples.

```python
X = np.array([[1], [2], [3]])
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)  # Raises ValueError
```

### ValueError: n_clusters must be at least 1

Raised when n_clusters is less than 1.

```python
kmeans = KMeans(n_clusters=0)
kmeans.fit(X)  # Raises ValueError
```

---

## Notes

- K-means uses Euclidean distance
- Algorithm converges when centroid shift is below tolerance
- Inertia decreases as k increases (more clusters = lower inertia)
- Elbow method finds k where inertia decrease slows down
- K-means++ initialization is generally better than random
- Algorithm may converge to local minima (run multiple times)

---

## Initialization Methods

### Random Initialization

Centroids are chosen randomly from data points.

**Pros:**
- Simple and fast
- No additional computation

**Cons:**
- Can lead to poor initial centroids
- May require more iterations
- Less stable results

### K-Means++ Initialization

Centroids are chosen to maximize distance from existing centroids.

**Pros:**
- Better initial centroids
- Faster convergence
- More stable results
- Often better final clustering

**Cons:**
- Slightly more computation
- Still not guaranteed optimal

---

## Best Practices

1. **Use k-means++ initialization**: Better than random
2. **Use elbow method**: Find optimal k value
3. **Scale features**: K-means is sensitive to feature scale
4. **Run multiple times**: K-means can converge to local minima
5. **Check cluster sizes**: Avoid very small or very large clusters
6. **Visualize results**: Especially for 2D/3D data
