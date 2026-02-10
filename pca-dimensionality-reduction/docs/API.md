# Principal Component Analysis API Documentation

This document provides detailed API documentation for the Principal Component Analysis (PCA) implementation.

## PCA Class

Principal Component Analysis for dimensionality reduction.

### Constructor

```python
PCA(
    n_components: Optional[int] = None,
    whiten: bool = False
) -> None
```

Initialize PCA.

**Parameters:**
- `n_components`: Number of components to keep. Options:
  - `None`: Keep all components
  - `int`: Keep top n components
  - `float` (0-1): Keep components explaining at least that variance
  (default: None)
- `whiten`: Whether to whiten the components (default: False)

**Example:**
```python
# Keep top 2 components
pca = PCA(n_components=2)

# Keep components explaining 95% variance
pca = PCA(n_components=0.95)

# Keep all components
pca = PCA(n_components=None)
```

---

## Methods

### fit

```python
fit(X: Union[List, np.ndarray, pd.DataFrame]) -> "PCA"
```

Fit PCA model.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
pca.fit(X)
```

---

### transform

```python
transform(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Transform data to lower-dimensional space.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Transformed data (shape: [n_samples, n_components_])

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_transformed = pca.transform(X)
```

---

### fit_transform

```python
fit_transform(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Fit model and transform data.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Transformed data

**Example:**
```python
X_transformed = pca.fit_transform(X)
```

---

### inverse_transform

```python
inverse_transform(X_transformed: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Transform data back to original space.

**Parameters:**
- `X_transformed`: Transformed feature matrix

**Returns:**
- Data in original space (shape: [n_samples, n_features])

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_reconstructed = pca.inverse_transform(X_transformed)
```

---

### get_explained_variance

```python
get_explained_variance() -> np.ndarray
```

Get explained variance for each component.

**Returns:**
- Explained variance array (eigenvalues)

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
explained_variance = pca.get_explained_variance()
```

---

### get_explained_variance_ratio

```python
get_explained_variance_ratio() -> np.ndarray
```

Get explained variance ratio for each component.

**Returns:**
- Explained variance ratio array (normalized eigenvalues)

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
explained_variance_ratio = pca.get_explained_variance_ratio()
```

---

### get_cumulative_variance

```python
get_cumulative_variance() -> np.ndarray
```

Get cumulative explained variance ratio.

**Returns:**
- Cumulative explained variance ratio array

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
cumulative_variance = pca.get_cumulative_variance()
```

---

### plot_explained_variance

```python
plot_explained_variance(
    save_path: Optional[str] = None,
    show: bool = True,
    cumulative: bool = False
) -> None
```

Plot explained variance.

**Parameters:**
- `save_path`: Optional path to save figure
- `show`: Whether to display plot (default: True)
- `cumulative`: Whether to plot cumulative variance (default: False)

**Returns:**
- None (displays or saves plot)

**Example:**
```python
pca.plot_explained_variance(save_path="variance.png", show=False)
pca.plot_explained_variance(cumulative=True)
```

---

### plot_components

```python
plot_components(
    save_path: Optional[str] = None,
    show: bool = True,
    n_components: Optional[int] = None
) -> None
```

Plot principal components (for 2D original data).

**Parameters:**
- `save_path`: Optional path to save figure
- `show`: Whether to display plot (default: True)
- `n_components`: Number of components to plot (default: all)

**Returns:**
- None (displays or saves plot)

**Note:** Only works for 2D original data

**Example:**
```python
pca.plot_components(show=False)
```

---

## Attributes

### components

Principal components (eigenvectors). None before fitting.

**Type:** `Optional[np.ndarray]` (shape: [n_components_, n_features])

### explained_variance

Explained variance (eigenvalues) for each component. None before fitting.

**Type:** `Optional[np.ndarray]`

### explained_variance_ratio

Explained variance ratio for each component. None before fitting.

**Type:** `Optional[np.ndarray]`

### mean

Mean of training data. None before fitting.

**Type:** `Optional[np.ndarray]`

### n_components_

Actual number of components kept. None before fitting.

**Type:** `Optional[int]`

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
X = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

# NumPy array
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

# Pandas
X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [2, 3, 4]})
```

---

## Error Handling

### ValueError: Model must be fitted before transformation

Raised when `transform()` or `inverse_transform()` is called before `fit()`.

```python
pca = PCA(n_components=2)
pca.transform(X)  # Raises ValueError
```

### ValueError: n_components must be between 0 and 1 when float

Raised when float n_components is not in (0, 1] range.

```python
pca = PCA(n_components=1.5)
pca.fit(X)  # Raises ValueError
```

### ValueError: Need at least 2 samples for PCA

Raised when input has less than 2 samples.

```python
X = np.array([[1, 2]])
pca.fit(X)  # Raises ValueError
```

---

## Notes

- PCA centers data automatically (subtracts mean)
- Components are sorted by explained variance (descending)
- Explained variance ratios sum to 1.0 (for all components)
- Whitening scales components to unit variance
- Inverse transformation reconstructs data with some information loss
- PCA assumes linear relationships in data

---

## Explained Variance Interpretation

### Explained Variance

The variance captured by each principal component.

**Properties:**
- Higher variance = more important component
- Sum of all explained variances = total variance
- Components are ordered by variance (descending)

### Explained Variance Ratio

The proportion of total variance explained by each component.

**Properties:**
- Values between 0 and 1
- Sum equals 1.0 (for all components)
- First component typically explains most variance

### Cumulative Explained Variance

The cumulative proportion of variance explained by first n components.

**Usage:**
- Helps choose number of components
- Find how many components needed for X% variance
- Common thresholds: 95%, 99%

---

## Best Practices

1. **Scale features**: PCA centers data but scaling may help
2. **Use variance threshold**: More flexible than fixed number
3. **Visualize explained variance**: Helps choose optimal number
4. **Check reconstruction error**: Verify information loss is acceptable
5. **Consider whitening**: If components need unit variance
6. **Interpret components carefully**: PCA components may not be interpretable
