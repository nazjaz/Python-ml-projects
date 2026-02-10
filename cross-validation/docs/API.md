# Cross-Validation API Documentation

This document provides detailed API documentation for the Cross-Validation Tool.

## CrossValidator Class

Main class for performing cross-validation with various strategies.

### Constructor

```python
CrossValidator(config_path: str = "config.yaml") -> None
```

Initialize CrossValidator with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file. Default: "config.yaml"

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
from src.main import CrossValidator

cv = CrossValidator()
```

---

## Methods

### k_fold_split

```python
k_fold_split(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Optional[Union[List, np.ndarray, pd.Series]] = None,
    n_splits: Optional[int] = None,
    shuffle: Optional[bool] = None,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]
```

Perform k-fold cross-validation split.

Divides data into k folds and returns train/test indices for each fold.

**Parameters:**
- `X`: Feature data
- `y`: Target data (optional, not used for splitting)
- `n_splits`: Number of folds. Default: from config
- `shuffle`: Whether to shuffle data before splitting. Default: from config
- `random_state`: Random seed for reproducibility. Default: from config

**Returns:**
- `List[Tuple[np.ndarray, np.ndarray]]`: List of tuples (train_indices, test_indices) for each fold

**Raises:**
- `ValueError`: If n_splits < 2 or n_splits > number of samples

**Example:**
```python
X = [[1], [2], [3], [4], [5]]
splits = cv.k_fold_split(X, n_splits=3)
# Returns: [(train_idx, test_idx), ...] for 3 folds
```

---

### stratified_k_fold_split

```python
stratified_k_fold_split(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Union[List, np.ndarray, pd.Series],
    n_splits: Optional[int] = None,
    shuffle: Optional[bool] = None,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]
```

Perform stratified k-fold cross-validation split.

Divides data into k folds while maintaining class distribution in each fold. Requires target labels.

**Parameters:**
- `X`: Feature data
- `y`: Target labels (required for stratification)
- `n_splits`: Number of folds. Default: from config
- `shuffle`: Whether to shuffle data before splitting. Default: from config
- `random_state`: Random seed for reproducibility. Default: from config

**Returns:**
- `List[Tuple[np.ndarray, np.ndarray]]`: List of tuples (train_indices, test_indices) for each fold

**Raises:**
- `ValueError`: If y is None, n_splits < 2, or n_splits > samples per class

**Example:**
```python
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 1, 1, 0, 1]
splits = cv.stratified_k_fold_split(X, y, n_splits=3)
# Returns: [(train_idx, test_idx), ...] for 3 folds
```

---

### leave_one_out_split

```python
leave_one_out_split(
    X: Union[List, np.ndarray, pd.DataFrame],
    y: Optional[Union[List, np.ndarray, pd.Series]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]
```

Perform leave-one-out cross-validation split.

Creates n splits where each split uses one sample as test and the rest as training data.

**Parameters:**
- `X`: Feature data
- `y`: Target data (optional, not used for splitting)

**Returns:**
- `List[Tuple[np.ndarray, np.ndarray]]`: List of tuples (train_indices, test_indices) for each fold

**Example:**
```python
X = [[1], [2], [3], [4]]
splits = cv.leave_one_out_split(X)
# Returns: [(train_idx, test_idx), ...] for 4 folds
```

---

### get_split_summary

```python
get_split_summary(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    y: Optional[Union[List, np.ndarray, pd.Series]] = None
) -> Dict[str, Union[int, List[Dict[str, Union[int, float]]]]]
```

Get summary statistics for cross-validation splits.

**Parameters:**
- `splits`: List of (train_indices, test_indices) tuples
- `y`: Target labels (optional, for class distribution)

**Returns:**
- `Dict[str, Union[int, List[Dict[str, Union[int, float]]]]]`: Dictionary containing:
  - `n_folds`: Number of folds
  - `folds`: List of fold information dictionaries

**Example:**
```python
splits = cv.k_fold_split(X, n_splits=5)
summary = cv.get_split_summary(splits, y)
# Returns: {"n_folds": 5, "folds": [...]}
```

---

### print_summary

```python
print_summary(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    y: Optional[Union[List, np.ndarray, pd.Series]] = None,
    strategy: str = "cross-validation"
) -> None
```

Print formatted summary of cross-validation splits.

**Parameters:**
- `splits`: List of (train_indices, test_indices) tuples
- `y`: Target labels (optional, for class distribution)
- `strategy`: Name of cross-validation strategy

**Returns:**
- `None`: Prints formatted summary to console

**Example:**
```python
splits = cv.k_fold_split(X, n_splits=5)
cv.print_summary(splits, y, strategy="K-Fold")
```

---

### save_splits

```python
save_splits(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    y: Optional[Union[List, np.ndarray, pd.Series]] = None,
    output_path: str = "cv_splits.json"
) -> None
```

Save cross-validation splits to JSON file.

**Parameters:**
- `splits`: List of (train_indices, test_indices) tuples
- `y`: Target labels (optional)
- `output_path`: Path to save JSON file. Default: "cv_splits.json"

**Returns:**
- `None`: Saves splits to JSON file

**Example:**
```python
splits = cv.k_fold_split(X, n_splits=5)
cv.save_splits(splits, y, output_path="splits.json")
```

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

### ValueError: n_splits must be at least 2

Raised when n_splits is less than 2.

```python
cv.k_fold_split(X, n_splits=1)  # Raises ValueError
```

### ValueError: n_splits cannot be greater than number of samples

Raised when n_splits exceeds number of samples.

```python
cv.k_fold_split(X, n_splits=100)  # Raises ValueError if len(X) < 100
```

### ValueError: y is required for stratified k-fold

Raised when y is None for stratified k-fold.

```python
cv.stratified_k_fold_split(X, y=None)  # Raises ValueError
```

### ValueError: Length mismatch

Raised when X and y have different lengths.

```python
X = [[1], [2], [3]]
y = [0, 1]
cv._validate_inputs(X, y)  # Raises ValueError
```

---

## Notes

- K-fold and stratified k-fold support shuffling with random_state for reproducibility
- Leave-one-out does not use shuffling (deterministic)
- Stratified k-fold maintains class distribution in each fold
- All splits ensure no overlap between train and test sets
- All samples are used exactly once as test data across all folds

---

## Cross-Validation Strategy Comparison

### K-Fold
- **Use case**: General purpose, balanced datasets
- **Folds**: k folds (typically 5 or 10)
- **Class distribution**: May vary
- **Computational cost**: Low

### Stratified K-Fold
- **Use case**: Classification, imbalanced datasets
- **Folds**: k folds (typically 5 or 10)
- **Class distribution**: Maintained
- **Computational cost**: Low

### Leave-One-Out
- **Use case**: Small datasets, maximum data usage
- **Folds**: n folds (one per sample)
- **Class distribution**: Maintained (if applicable)
- **Computational cost**: High

---

## Best Practices

1. **Use stratified k-fold for classification**: Maintains class distribution
2. **Use k-fold for regression**: Stratification not applicable
3. **Use leave-one-out for small datasets**: Maximum data usage
4. **Set random_state for reproducibility**: Ensures consistent splits
5. **Shuffle data when appropriate**: Reduces order bias
6. **Choose appropriate k**: k=5 or k=10 are common choices
