# Classification Visualization API Documentation

This document provides detailed API documentation for the Classification Visualization Tool.

## ClassificationVisualizer Class

Main class for creating confusion matrices and classification reports with visualization capabilities.

### Constructor

```python
ClassificationVisualizer(config_path: str = "config.yaml") -> None
```

Initialize ClassificationVisualizer with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file. Default: "config.yaml"

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
from src.main import ClassificationVisualizer

viz = ClassificationVisualizer()
```

---

## Methods

### confusion_matrix

```python
confusion_matrix(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    labels: Optional[List] = None
) -> np.ndarray
```

Calculate confusion matrix.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: List of labels to include in matrix. If None, all unique labels are used.

**Returns:**
- `np.ndarray`: Confusion matrix as numpy array

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
cm = viz.confusion_matrix(y_true, y_pred)
# Returns: array([[2, 0],
#                  [1, 2]])
```

---

### classification_report

```python
classification_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, Union[Dict[str, float], float]]
```

Generate classification report with precision, recall, F1-score.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: List of labels to include in report. If None, all unique labels are used.
- `target_names`: Optional names for labels. If None, labels are used as names.

**Returns:**
- `Dict[str, Union[Dict[str, float], float]]`: Dictionary containing:
  - `per_class`: Dictionary with metrics for each class
  - `accuracy`: Overall accuracy
  - `macro_avg`: Macro-averaged metrics
  - `weighted_avg`: Weighted-averaged metrics

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
report = viz.classification_report(y_true, y_pred)
# Returns: {
#     "per_class": {...},
#     "accuracy": 0.8,
#     "macro_avg": {...},
#     "weighted_avg": {...}
# }
```

---

### plot_confusion_matrix

```python
plot_confusion_matrix(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None
```

Plot confusion matrix as heatmap.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: List of labels to include. If None, all unique labels are used.
- `target_names`: Optional names for labels. If None, labels are used as names.
- `normalize`: If True, normalize confusion matrix to show percentages. Default: False.
- `title`: Optional title for the plot. Default: auto-generated.
- `save_path`: Optional path to save the figure. Default: None.
- `show`: Whether to display the plot. Default: True.

**Returns:**
- `None`: Displays or saves the plot

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
viz.plot_confusion_matrix(y_true, y_pred)
```

---

### plot_classification_report

```python
plot_classification_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None
```

Plot classification report as heatmap.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: List of labels to include. If None, all unique labels are used.
- `target_names`: Optional names for labels. If None, labels are used as names.
- `title`: Optional title for the plot. Default: auto-generated.
- `save_path`: Optional path to save the figure. Default: None.
- `show`: Whether to display the plot. Default: True.

**Returns:**
- `None`: Displays or saves the plot

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
viz.plot_classification_report(y_true, y_pred)
```

---

### print_classification_report

```python
print_classification_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None
) -> None
```

Print formatted classification report to console.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: List of labels to include. If None, all unique labels are used.
- `target_names`: Optional names for labels. If None, labels are used as names.

**Returns:**
- `None`: Prints formatted report to console

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
viz.print_classification_report(y_true, y_pred)
```

---

### save_report

```python
save_report(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    output_path: str = "classification_report.json"
) -> None
```

Save classification report to JSON file.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: List of labels to include. If None, all unique labels are used.
- `target_names`: Optional names for labels. If None, labels are used as names.
- `output_path`: Path to save JSON file. Default: "classification_report.json"

**Returns:**
- `None`: Saves report to JSON file

**Example:**
```python
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
viz.save_report(y_true, y_pred, output_path="report.json")
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
viz.confusion_matrix(y_true, y_pred)  # Raises ValueError
```

### ValueError: Input arrays cannot be empty

Raised when input arrays are empty.

```python
y_true = []
y_pred = []
viz.confusion_matrix(y_true, y_pred)  # Raises ValueError
```

---

## Configuration

Visualization parameters can be configured via `config.yaml`:

```yaml
visualization:
  figsize: [10, 8]      # Figure size in inches
  dpi: 100              # DPI for saved figures
  colormap: "Blues"     # Colormap for confusion matrix
  fontsize: 12          # Font size for labels
  save_format: "png"    # Format for saving figures
```

---

## Notes

- Confusion matrix shows counts by default, use `normalize=True` for proportions
- Classification report includes per-class metrics and averages
- Visualizations use seaborn heatmaps for better appearance
- Saved figures use high DPI for publication quality
- JSON reports include both confusion matrix and classification report

---

## Visualization Best Practices

1. **Use normalized confusion matrix** for comparing classes with different sample sizes
2. **Customize target names** for better readability in visualizations
3. **Adjust figure size** based on number of classes
4. **Choose appropriate colormap** based on context (Blues, Greens, Reds, etc.)
5. **Save high-resolution figures** for reports and presentations
