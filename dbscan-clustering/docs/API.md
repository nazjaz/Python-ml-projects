# DBSCAN Clustering API Documentation

This document provides detailed API documentation for the DBSCAN Clustering implementation.

## DBSCAN Class

DBSCAN clustering algorithm for density-based clustering.

### Constructor

```python
DBSCAN(
    eps: float = 0.5,
    min_samples: int = 5,
    distance_metric: str = "euclidean"
) -> None
```

Initialize DBSCAN.

**Parameters:**
- `eps`: Maximum distance between two samples for one to be considered in the neighborhood of the other (default: 0.5)
- `min_samples`: Minimum number of samples in a neighborhood for a point to be considered a core point (default: 5)
- `distance_metric`: Distance metric. Currently only "euclidean" is supported (default: "euclidean")

**Example:**
```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
```

---

## Methods

### fit

```python
fit(X: Union[List, np.ndarray, pd.DataFrame]) -> "DBSCAN"
```

Fit DBSCAN clustering model.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
dbscan.fit(X)
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
- Cluster labels as numpy array (-1 for noise)

**Example:**
```python
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
labels = dbscan.fit_predict(X)
```

---

### get_core_samples

```python
get_core_samples() -> np.ndarray
```

Get indices of core samples.

**Returns:**
- Array of core sample indices

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
core_samples = dbscan.get_core_samples()
print(f"Core samples: {core_samples}")
```

---

### get_noise_samples

```python
get_noise_samples() -> np.ndarray
```

Get indices of noise samples.

**Returns:**
- Array of noise sample indices

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
noise_samples = dbscan.get_noise_samples()
print(f"Noise samples: {noise_samples}")
```

---

### get_cluster_info

```python
get_cluster_info() -> Dict
```

Get information about clusters.

**Returns:**
- Dictionary containing:
  - `n_clusters`: Number of clusters (int)
  - `n_noise`: Number of noise points (int)
  - `n_core_samples`: Number of core points (int)
  - `cluster_sizes`: Dictionary mapping cluster ID to size (Dict[int, int])
  - `labels`: Cluster labels array (np.ndarray)

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
info = dbscan.get_cluster_info()
print(f"Clusters: {info['n_clusters']}")
print(f"Noise: {info['n_noise']}")
```

---

### plot_clusters

```python
plot_clusters(
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> None
```

Plot clusters and noise points.

**Parameters:**
- `save_path`: Optional path to save figure
- `show`: Whether to display plot (default: True)
- `title`: Optional plot title (default: None)

**Returns:**
- None (displays or saves plot)

**Note:** Only works for 2D data

**Example:**
```python
dbscan.plot_clusters(save_path="clusters.png", show=False)
```

---

## Attributes

### labels

Cluster labels after fitting. -1 indicates noise. None before fitting.

**Type:** `Optional[np.ndarray]`

### core_samples

Indices of core samples. None before fitting.

**Type:** `Optional[np.ndarray]`

### noise_samples

Indices of noise samples. None before fitting.

**Type:** `Optional[np.ndarray]`

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

### ValueError: eps must be greater than 0

Raised when eps is zero or negative.

```python
dbscan = DBSCAN(eps=0, min_samples=5)
dbscan.fit(X)  # Raises ValueError
```

### ValueError: min_samples must be at least 1

Raised when min_samples is less than 1.

```python
dbscan = DBSCAN(eps=0.5, min_samples=0)
dbscan.fit(X)  # Raises ValueError
```

### ValueError: Model must be fitted before...

Raised when methods are called before `fit()`.

```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.get_core_samples()  # Raises ValueError
```

---

## Notes

- DBSCAN uses density-based clustering
- Noise points are labeled as -1
- Clusters are numbered starting from 0
- Algorithm automatically determines number of clusters
- Can find clusters of arbitrary shape
- Sensitive to eps and min_samples parameters

---

## Point Classification

### Core Point

A point is a core point if it has at least `min_samples` neighbors within `eps` distance.

**Properties:**
- Forms backbone of clusters
- Can expand clusters
- Has sufficient density

### Border Point

A point is a border point if:
- It is a neighbor of a core point
- It doesn't have enough neighbors to be core itself

**Properties:**
- Belongs to a cluster
- Doesn't expand clusters
- On the edge of clusters

### Noise Point

A point is a noise point if:
- It is not a core point
- It is not a neighbor of any core point

**Properties:**
- Labeled as -1
- Considered an outlier
- Doesn't belong to any cluster

---

## Parameter Selection

### eps Selection

**k-Distance Graph Method:**
1. For each point, find distance to k-th nearest neighbor
2. Sort these distances
3. Look for "knee" in the plot
4. Use that distance as eps

**Rule of Thumb:**
- Start with small eps and increase
- Consider data scale
- Use domain knowledge

### min_samples Selection

**Rule of Thumb:**
- `min_samples = 2 * dimensions` (good default)
- For 2D: min_samples = 4
- For higher dimensions: increase accordingly

**Considerations:**
- Too small: Many clusters, sensitive to noise
- Too large: Few clusters, may miss small clusters

---

## Best Practices

1. **Scale features**: DBSCAN is sensitive to feature scale
2. **Tune eps carefully**: Use k-distance graph to find good eps
3. **Start with min_samples = 2*dimensions**: Good default
4. **Visualize results**: Especially for 2D data
5. **Check noise points**: Verify they are actually outliers
6. **Try different parameters**: DBSCAN is sensitive to parameter choice
