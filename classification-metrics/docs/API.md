# Classification Metrics API Documentation

This document provides detailed API documentation for the Classification Metrics Calculator.

## ClassificationMetrics Class

Main class for calculating classification evaluation metrics.

### Constructor

```python
ClassificationMetrics(config_path: str = "config.yaml") -> None
```

Initialize ClassificationMetrics with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file. Default: "config.yaml"

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
from src.main import ClassificationMetrics

metrics = ClassificationMetrics()
```

---

## Methods

### accuracy

```python
accuracy(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> float
```

Calculate accuracy score.

Accuracy is the proportion of correct predictions among all predictions.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels

**Returns:**
- `float`: Accuracy score between 0 and 1

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
accuracy = metrics.accuracy(y_true, y_pred)
# Returns: 0.8
```

---

### precision

```python
precision(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    average: str = "binary",
    pos_label: Union[int, str] = 1
) -> Union[float, Dict[str, float]]
```

Calculate precision score.

Precision is the proportion of true positives among all positive predictions (TP / (TP + FP)).

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `average` (str): Averaging strategy. Options:
  - `"binary"`: Binary classification (default)
  - `"macro"`: Unweighted mean of per-class precision
  - `"micro"`: Global precision calculated from total TP and FP
  - `"weighted"`: Weighted mean of per-class precision
  - `None`: Return per-class precision as dictionary
- `pos_label`: Positive class label for binary classification. Default: 1

**Returns:**
- `float`: Precision score for binary/macro/micro/weighted averaging
- `Dict[str, float]`: Per-class precision when `average=None`

**Example:**
```python
# Binary classification
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
precision = metrics.precision(y_true, y_pred)
# Returns: 1.0

# Multiclass with macro averaging
y_true = [0, 1, 2, 0, 1]
y_pred = [0, 1, 1, 0, 1]
precision = metrics.precision(y_true, y_pred, average="macro")
# Returns: float

# Per-class precision
precision = metrics.precision(y_true, y_pred, average=None)
# Returns: {"0": 1.0, "1": 0.5, "2": 0.0}
```

---

### recall

```python
recall(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    average: str = "binary",
    pos_label: Union[int, str] = 1
) -> Union[float, Dict[str, float]]
```

Calculate recall score.

Recall is the proportion of true positives among all actual positives (TP / (TP + FN)).

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `average` (str): Averaging strategy. Options:
  - `"binary"`: Binary classification (default)
  - `"macro"`: Unweighted mean of per-class recall
  - `"micro"`: Global recall calculated from total TP and FN
  - `"weighted"`: Weighted mean of per-class recall
  - `None`: Return per-class recall as dictionary
- `pos_label`: Positive class label for binary classification. Default: 1

**Returns:**
- `float`: Recall score for binary/macro/micro/weighted averaging
- `Dict[str, float]`: Per-class recall when `average=None`

**Example:**
```python
# Binary classification
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
recall = metrics.recall(y_true, y_pred)
# Returns: 0.6666666666666666

# Multiclass with weighted averaging
y_true = [0, 1, 2, 0, 1]
y_pred = [0, 1, 1, 0, 1]
recall = metrics.recall(y_true, y_pred, average="weighted")
# Returns: float
```

---

### f1_score

```python
f1_score(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    average: str = "binary",
    pos_label: Union[int, str] = 1
) -> Union[float, Dict[str, float]]
```

Calculate F1-score.

F1-score is the harmonic mean of precision and recall:
F1 = 2 * (precision * recall) / (precision + recall)

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `average` (str): Averaging strategy. Options:
  - `"binary"`: Binary classification (default)
  - `"macro"`: Unweighted mean of per-class F1-score
  - `"micro"`: Global F1-score calculated from total TP, FP, and FN
  - `"weighted"`: Weighted mean of per-class F1-score
  - `None`: Return per-class F1-score as dictionary
- `pos_label`: Positive class label for binary classification. Default: 1

**Returns:**
- `float`: F1-score for binary/macro/micro/weighted averaging
- `Dict[str, float]`: Per-class F1-score when `average=None`

**Example:**
```python
# Binary classification
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
f1 = metrics.f1_score(y_true, y_pred)
# Returns: 0.8

# Multiclass with micro averaging
y_true = [0, 1, 2, 0, 1]
y_pred = [0, 1, 1, 0, 1]
f1 = metrics.f1_score(y_true, y_pred, average="micro")
# Returns: float
```

---

### calculate_all_metrics

```python
calculate_all_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    average: str = "binary",
    pos_label: Union[int, str] = 1
) -> Dict[str, Union[float, Dict[str, float]]]
```

Calculate all classification metrics at once.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `average` (str): Averaging strategy for precision, recall, F1. Default: "binary"
- `pos_label`: Positive class label for binary classification. Default: 1

**Returns:**
- `Dict[str, Union[float, Dict[str, float]]]`: Dictionary containing:
  - `"accuracy"`: Accuracy score (float)
  - `"precision"`: Precision score(s)
  - `"recall"`: Recall score(s)
  - `"f1_score"`: F1-score(s)

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
results = metrics.calculate_all_metrics(y_true, y_pred)
# Returns: {
#     "accuracy": 0.8,
#     "precision": 1.0,
#     "recall": 0.6666666666666666,
#     "f1_score": 0.8
# }
```

---

### confusion_matrix

```python
confusion_matrix(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series]
) -> Dict[str, Dict[str, int]]
```

Calculate confusion matrix.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels

**Returns:**
- `Dict[str, Dict[str, int]]`: Dictionary representation of confusion matrix with structure: `{true_label: {predicted_label: count}}`

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
cm = metrics.confusion_matrix(y_true, y_pred)
# Returns: {
#     "0": {"0": 2, "1": 0},
#     "1": {"0": 1, "1": 2}
# }
```

---

## Input Types

All methods accept the following input types for `y_true` and `y_pred`:

- `List`: Python list of labels
- `np.ndarray`: NumPy array of labels
- `pd.Series`: Pandas Series of labels

**Example:**
```python
import numpy as np
import pandas as pd

# List
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

# NumPy array
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])

# Pandas Series
y_true = pd.Series([0, 1, 1, 0])
y_pred = pd.Series([0, 1, 0, 0])
```

---

## Error Handling

### ValueError: Length mismatch

Raised when `y_true` and `y_pred` have different lengths.

```python
y_true = [0, 1, 1]
y_pred = [0, 1]  # Different length
metrics.accuracy(y_true, y_pred)  # Raises ValueError
```

### ValueError: Input arrays cannot be empty

Raised when input arrays are empty.

```python
y_true = []
y_pred = []
metrics.accuracy(y_true, y_pred)  # Raises ValueError
```

### ValueError: Invalid average parameter

Raised when an unsupported averaging strategy is provided.

```python
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
metrics.precision(y_true, y_pred, average="invalid")  # Raises ValueError
```

---

## Notes

- All metrics return values between 0 and 1 (or 0.0 to 1.0)
- When division by zero occurs (e.g., no positive predictions), metrics return 0.0
- Binary classification uses the positive class label (default: 1)
- Multiclass classification requires specifying an averaging strategy
- Confusion matrix keys are strings representing class labels
