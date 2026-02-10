# Naive Bayes Classifier API Documentation

This document provides detailed API documentation for the Naive Bayes Classifier implementation with Gaussian, Multinomial, and Bernoulli variants.

## GaussianNB Class

Gaussian Naive Bayes classifier for continuous features.

### Constructor

```python
GaussianNB(smoothing: float = 1e-9) -> None
```

Initialize GaussianNB.

**Parameters:**
- `smoothing`: Smoothing parameter to prevent division by zero (default: 1e-9)

**Example:**
```python
model = GaussianNB(smoothing=1e-9)
```

---

## Methods

### fit

```python
fit(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> "GaussianNB"
```

Fit Gaussian Naive Bayes classifier.

**Parameters:**
- `X`: Feature matrix (continuous values)
- `y`: Target labels

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[1.5], [2.3], [3.1], [4.7], [5.2]])
y = np.array([0, 0, 1, 1, 1])
model.fit(X, y)
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
X_test = np.array([[2.0], [4.5]])
predictions = model.predict(X_test)
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
- Probability matrix (shape: [n_samples, n_classes])

**Raises:**
- `ValueError`: If model not fitted

**Example:**
```python
X_test = np.array([[2.0], [4.5]])
probabilities = model.predict_proba(X_test)
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
```

---

## Attributes

### classes

Unique class labels found during fitting. None before fitting.

**Type:** `Optional[np.ndarray]`

### class_priors

Prior probabilities for each class. None before fitting.

**Type:** `Optional[np.ndarray]`

### means

Mean values for each class and feature. None before fitting.

**Type:** `Optional[np.ndarray]` (shape: [n_classes, n_features])

### variances

Variance values for each class and feature. None before fitting.

**Type:** `Optional[np.ndarray]` (shape: [n_classes, n_features])

---

## MultinomialNB Class

Multinomial Naive Bayes classifier for count data.

### Constructor

```python
MultinomialNB(alpha: float = 1.0) -> None
```

Initialize MultinomialNB.

**Parameters:**
- `alpha`: Smoothing parameter (Laplace smoothing, default: 1.0)

**Example:**
```python
model = MultinomialNB(alpha=1.0)
```

---

## Methods

### fit

```python
fit(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> "MultinomialNB"
```

Fit Multinomial Naive Bayes classifier.

**Parameters:**
- `X`: Feature matrix (count data, non-negative integers)
- `y`: Target labels

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid or contain negative values

**Example:**
```python
X = np.array([[5, 2, 0], [3, 4, 1], [1, 6, 2]])
y = np.array([0, 0, 1])
model.fit(X, y)
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

---

### predict_proba

```python
predict_proba(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict class probabilities.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Probability matrix (shape: [n_samples, n_classes])

**Raises:**
- `ValueError`: If model not fitted

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

---

## Attributes

### classes

Unique class labels found during fitting. None before fitting.

**Type:** `Optional[np.ndarray]`

### class_priors

Prior probabilities for each class. None before fitting.

**Type:** `Optional[np.ndarray]`

### feature_counts

Count of each feature for each class. None before fitting.

**Type:** `Optional[np.ndarray]` (shape: [n_classes, n_features])

### class_counts

Total count for each class. None before fitting.

**Type:** `Optional[np.ndarray]`

---

## BernoulliNB Class

Bernoulli Naive Bayes classifier for binary features.

### Constructor

```python
BernoulliNB(alpha: float = 1.0, binarize: Optional[float] = 0.0) -> None
```

Initialize BernoulliNB.

**Parameters:**
- `alpha`: Smoothing parameter (Laplace smoothing, default: 1.0)
- `binarize`: Threshold for binarizing features. If None, assumes features are already binary (default: 0.0)

**Example:**
```python
model = BernoulliNB(alpha=1.0, binarize=0.5)
```

---

## Methods

### fit

```python
fit(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series]
) -> "BernoulliNB"
```

Fit Bernoulli Naive Bayes classifier.

**Parameters:**
- `X`: Feature matrix (binary or continuous to be binarized)
- `y`: Target labels

**Returns:**
- Self for method chaining

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
X = np.array([[0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 1])
model.fit(X, y)
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

---

### predict_proba

```python
predict_proba(X: Union[List, np.ndarray, pd.DataFrame]) -> np.ndarray
```

Predict class probabilities.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Probability matrix (shape: [n_samples, n_classes])

**Raises:**
- `ValueError`: If model not fitted

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

---

## Attributes

### classes

Unique class labels found during fitting. None before fitting.

**Type:** `Optional[np.ndarray]`

### class_priors

Prior probabilities for each class. None before fitting.

**Type:** `Optional[np.ndarray]`

### feature_probs

Probability of each feature being 1 for each class. None before fitting.

**Type:** `Optional[np.ndarray]` (shape: [n_classes, n_features])

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
X = [[1.5], [2.3], [3.1]]
y = [0, 1, 0]

# NumPy array
X = np.array([[1.5], [2.3], [3.1]])
y = np.array([0, 1, 0])

# Pandas
X = pd.DataFrame({"feature": [1.5, 2.3, 3.1]})
y = pd.Series([0, 1, 0])
```

---

## Error Handling

### ValueError: Model must be fitted before prediction

Raised when `predict()` or `predict_proba()` is called before `fit()`.

```python
model = GaussianNB()
model.predict(X)  # Raises ValueError
```

### ValueError: MultinomialNB requires non-negative feature values

Raised when MultinomialNB receives negative feature values.

```python
X = np.array([[-1, 2], [2, 3]])
y = np.array([0, 1])
model = MultinomialNB()
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

- All variants use log probabilities for numerical stability
- Probabilities are normalized to sum to 1 for each sample
- Smoothing parameters prevent zero probabilities
- Feature independence is assumed (naive assumption)
- All variants support multi-class classification

---

## Variant Selection Guide

- **GaussianNB**: Use for continuous, real-valued features
- **MultinomialNB**: Use for count data, word frequencies, non-negative integers
- **BernoulliNB**: Use for binary features, presence/absence data

---

## Best Practices

1. **Choose the right variant**: Match variant to data type
2. **Use GaussianNB for continuous data**: Assumes normal distribution
3. **Use MultinomialNB for count data**: Word frequencies, document counts
4. **Use BernoulliNB for binary data**: Presence/absence features
5. **Adjust smoothing parameters**: Alpha parameter affects predictions
6. **Handle feature independence**: Be aware of the naive assumption
