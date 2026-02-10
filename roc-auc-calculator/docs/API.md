# ROC Curve and AUC Calculator API Documentation

This document provides detailed API documentation for the ROC Curve and AUC Calculator.

## ROCCalculator Class

Main class for calculating ROC curves and AUC for binary classification.

### Constructor

```python
ROCCalculator(config_path: str = "config.yaml") -> None
```

Initialize ROCCalculator with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file. Default: "config.yaml"

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
from src.main import ROCCalculator

calc = ROCCalculator()
```

---

## Methods

### roc_curve

```python
roc_curve(
    y_true: Union[List, np.ndarray, pd.Series],
    y_scores: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Calculate ROC curve (True Positive Rate vs False Positive Rate).

**Parameters:**
- `y_true`: True binary labels (0 or 1)
- `y_scores`: Predicted scores or probabilities
- `pos_label`: Label of the positive class. Default: 1

**Returns:**
- `Tuple[np.ndarray, np.ndarray, np.ndarray]`: Tuple of (fpr, tpr, thresholds):
  - `fpr`: False Positive Rate array
  - `tpr`: True Positive Rate array
  - `thresholds`: Threshold values used

**Example:**
```python
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)
```

---

### auc

```python
auc(
    y_true: Union[List, np.ndarray, pd.Series],
    y_scores: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1
) -> float
```

Calculate Area Under the ROC Curve (AUC).

Uses the trapezoidal rule to compute the area under the ROC curve. AUC ranges from 0 to 1, where 1.0 indicates perfect classification and 0.5 indicates random performance.

**Parameters:**
- `y_true`: True binary labels (0 or 1)
- `y_scores`: Predicted scores or probabilities
- `pos_label`: Label of the positive class. Default: 1

**Returns:**
- `float`: AUC score (between 0 and 1)

**Example:**
```python
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
auc_score = calc.auc(y_true, y_scores)
# Returns: 0.75
```

---

### plot_roc_curve

```python
plot_roc_curve(
    y_true: Union[List, np.ndarray, pd.Series],
    y_scores: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1,
    title: Optional[str] = None,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None
```

Plot ROC curve.

**Parameters:**
- `y_true`: True binary labels (0 or 1)
- `y_scores`: Predicted scores or probabilities
- `pos_label`: Label of the positive class. Default: 1
- `title`: Optional title for the plot. Default: auto-generated
- `label`: Optional label for the curve in legend
- `save_path`: Optional path to save the figure. Default: None
- `show`: Whether to display the plot. Default: True

**Returns:**
- `None`: Displays or saves the plot

**Example:**
```python
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
calc.plot_roc_curve(y_true, y_scores)
```

---

### calculate_all_metrics

```python
calculate_all_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_scores: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1
) -> Dict[str, Union[float, Dict[str, np.ndarray]]]
```

Calculate ROC curve and AUC.

**Parameters:**
- `y_true`: True binary labels (0 or 1)
- `y_scores`: Predicted scores or probabilities
- `pos_label`: Label of the positive class. Default: 1

**Returns:**
- `Dict[str, Union[float, Dict[str, np.ndarray]]]`: Dictionary containing:
  - `auc`: AUC score
  - `roc_curve`: Dictionary with fpr, tpr, thresholds arrays

**Example:**
```python
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
results = calc.calculate_all_metrics(y_true, y_scores)
# Returns: {
#     "auc": 0.75,
#     "roc_curve": {
#         "fpr": [...],
#         "tpr": [...],
#         "thresholds": [...]
#     }
# }
```

---

### print_report

```python
print_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_scores: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1
) -> None
```

Print formatted ROC and AUC report to console.

**Parameters:**
- `y_true`: True binary labels (0 or 1)
- `y_scores`: Predicted scores or probabilities
- `pos_label`: Label of the positive class. Default: 1

**Returns:**
- `None`: Prints formatted report to console

**Example:**
```python
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
calc.print_report(y_true, y_scores)
```

---

### save_report

```python
save_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_scores: Union[List, np.ndarray, pd.Series],
    pos_label: int = 1,
    output_path: str = "roc_auc_report.json"
) -> None
```

Save ROC curve and AUC report to JSON file.

**Parameters:**
- `y_true`: True binary labels (0 or 1)
- `y_scores`: Predicted scores or probabilities
- `pos_label`: Label of the positive class. Default: 1
- `output_path`: Path to save JSON file. Default: "roc_auc_report.json"

**Returns:**
- `None`: Saves report to JSON file

**Example:**
```python
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
calc.save_report(y_true, y_scores, output_path="report.json")
```

---

## Input Types

All methods accept the following input types for `y_true` and `y_scores`:

- `List`: Python list of values
- `np.ndarray`: NumPy array of values
- `pd.Series`: Pandas Series of values

**Example:**
```python
import numpy as np
import pandas as pd

# List
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# NumPy array
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

# Pandas Series
y_true = pd.Series([0, 0, 1, 1])
y_scores = pd.Series([0.1, 0.4, 0.35, 0.8])
```

---

## Error Handling

### ValueError: Length mismatch

Raised when `y_true` and `y_scores` have different lengths.

```python
y_true = [0, 1, 1]
y_scores = [0.5, 0.7]  # Different length
calc.roc_curve(y_true, y_scores)  # Raises ValueError
```

### ValueError: Input arrays cannot be empty

Raised when input arrays are empty.

```python
y_true = []
y_scores = []
calc.roc_curve(y_true, y_scores)  # Raises ValueError
```

### ValueError: ROC curve is only for binary classification

Raised when more than 2 unique labels are found.

```python
y_true = [0, 1, 2]
y_scores = [0.3, 0.5, 0.7]
calc.roc_curve(y_true, y_scores)  # Raises ValueError
```

### ValueError: y_true must contain only 0 and 1

Raised when labels are not binary (0 and 1).

```python
y_true = [0, 1, 2, 3]
y_scores = [0.3, 0.5, 0.7, 0.9]
calc.roc_curve(y_true, y_scores)  # Raises ValueError
```

---

## Notes

- ROC curve is only for binary classification (2 classes)
- y_true must contain only 0 and 1 values
- y_scores can be any numeric values (probabilities or scores)
- AUC ranges from 0 to 1, where 1.0 is perfect and 0.5 is random
- ROC curve always starts at (0, 0) and ends at (1, 1)
- AUC is calculated using trapezoidal rule
- Higher AUC indicates better classifier performance

---

## ROC Curve Properties

### True Positive Rate (TPR)
- Also called Sensitivity or Recall
- TPR = TP / (TP + FN)
- Measures proportion of actual positives correctly identified

### False Positive Rate (FPR)
- Also called Fall-out
- FPR = FP / (FP + TN)
- Measures proportion of actual negatives incorrectly identified as positives

### Threshold
- Decision threshold for converting scores to binary predictions
- Lower threshold: More positives predicted (higher TPR, higher FPR)
- Higher threshold: Fewer positives predicted (lower TPR, lower FPR)

---

## AUC Interpretation

- **AUC = 1.0**: Perfect classifier (all positives ranked above negatives)
- **AUC > 0.9**: Excellent classifier
- **AUC > 0.8**: Good classifier
- **AUC > 0.7**: Fair classifier
- **AUC = 0.5**: Random classifier (no discriminative ability)
- **AUC < 0.5**: Worse than random (classifier predictions are inverted)

---

## Best Practices

1. **Use for binary classification only**: ROC curve is not applicable to multiclass problems
2. **Compare multiple models**: Plot multiple ROC curves on same plot
3. **Consider class imbalance**: ROC curve is less sensitive to imbalance than accuracy
4. **Use AUC for model selection**: Single metric for comparing models
5. **Examine threshold trade-offs**: ROC curve shows TPR vs FPR trade-off
