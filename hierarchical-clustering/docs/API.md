# Hierarchical Clustering API Documentation

This document provides detailed API documentation for the Hierarchical Clustering implementation.

## HierarchicalClustering Class

Hierarchical clustering with different linkage methods.

### Constructor

```python
HierarchicalClustering(
    n_clusters: Optional[int] = None,
    linkage: str = "average",
    distance_metric: str = "euclidean"
) -> None
```

Initialize HierarchicalClustering.

**Parameters:**
- `n_clusters`: Number of clusters to form. If None, returns full dendrogram without cutting (default: None)
- `linkage`: Linkage method. Options: "single", "complete", "average" (default: "average")
- `distance_metric`: Distance metric. Currently only "euclidean" is supported (default: "euclidean")

**Example:**
```python
model = HierarchicalClustering(
    n_clusters=3,
    linkage="average"
)
```

---

## Methods

### fit

```python
fit(X: Union[List, np.ndarray, pd.DataFrame]) -> "HierarchicalClustering"
```

Fit hierarchical clustering model.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
model.fit(X)
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

**Raises:**
- `ValueError`: If n_clusters not set

**Example:**
```python
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
labels = model.fit_predict(X)
```

---

### plot_dendrogram

```python
plot_dendrogram(
    save_path: Optional[str] = None,
    show: bool = True,
    truncate_mode: Optional[str] = None,
    p: Optional[int] = None
) -> None
```

Plot dendrogram.

**Parameters:**
- `save_path`: Optional path to save figure
- `show`: Whether to display plot (default: True)
- `truncate_mode`: Truncation mode (not implemented yet, default: None)
- `p`: Number of levels to show if truncate_mode is set (default: None)

**Returns:**
- None (displays or saves plot)

**Example:**
```python
model.plot_dendrogram(save_path="dendrogram.png", show=False)
```

---

## Attributes

### labels

Cluster labels after fitting (if n_clusters is set). None before fitting or if n_clusters is None.

**Type:** `Optional[np.ndarray]`

### linkage_matrix

Linkage matrix containing cluster merge information. None before fitting.

**Type:** `Optional[np.ndarray]` (shape: [n_samples-1, 4])

Each row contains:
- Column 0: First cluster ID
- Column 1: Second cluster ID
- Column 2: Distance between clusters
- Column 3: Size of merged cluster

### dendrogram_data

List of dictionaries containing dendrogram information. None before fitting.

**Type:** `Optional[List[Dict]]`

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

### ValueError: Need at least 2 samples for clustering

Raised when input has less than 2 samples.

```python
X = np.array([[1, 2]])
model.fit(X)  # Raises ValueError
```

### ValueError: n_clusters must be set for fit_predict

Raised when fit_predict is called without n_clusters.

```python
model = HierarchicalClustering(n_clusters=None)
model.fit_predict(X)  # Raises ValueError
```

### ValueError: Unknown linkage method

Raised when invalid linkage method is specified.

```python
model = HierarchicalClustering(linkage="invalid")
model.fit(X)  # Raises ValueError
```

---

## Notes

- Hierarchical clustering has O(n³) time complexity
- Dendrogram shows complete merge history
- Linkage method significantly affects results
- Single linkage can create chaining effect
- Complete linkage creates compact clusters
- Average linkage is a balanced approach

---

## Linkage Method Details

### Single Linkage

Uses minimum distance between any two points in different clusters.

**Formula:**
```
d(C₁, C₂) = min{d(x, y) : x ∈ C₁, y ∈ C₂}
```

**Characteristics:**
- Tends to create elongated clusters
- Sensitive to outliers
- Can create "chaining" effect
- Good for non-spherical clusters

### Complete Linkage

Uses maximum distance between any two points in different clusters.

**Formula:**
```
d(C₁, C₂) = max{d(x, y) : x ∈ C₁, y ∈ C₂}
```

**Characteristics:**
- Tends to create compact, spherical clusters
- Less sensitive to outliers
- Prevents chaining
- Good for well-separated clusters

### Average Linkage

Uses average distance between all pairs of points in different clusters.

**Formula:**
```
d(C₁, C₂) = (1/|C₁||C₂|) Σ d(x, y) for x ∈ C₁, y ∈ C₂
```

**Characteristics:**
- Balanced approach
- Good general-purpose choice
- Less sensitive to outliers than single
- More flexible than complete

---

## Best Practices

1. **Choose appropriate linkage**: Average is often a good default
2. **Use single linkage for elongated clusters**: When clusters are chain-like
3. **Use complete linkage for compact clusters**: When clusters are spherical
4. **Visualize dendrogram**: Helps understand cluster structure
5. **Consider computational cost**: O(n³) complexity for large datasets
6. **Use n_clusters for specific number**: Cut dendrogram at desired level
