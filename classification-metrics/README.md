# Classification Metrics Calculator

A Python tool for calculating basic evaluation metrics for classification tasks including accuracy, precision, recall, and F1-score. This is the sixteenth project in the ML learning series, focusing on understanding and implementing fundamental classification evaluation metrics.

## Project Title and Description

The Classification Metrics Calculator provides automated calculation of essential evaluation metrics for classification models. It implements accuracy, precision, recall, and F1-score from scratch, helping users understand how these metrics work and when to use them.

This tool solves the problem of evaluating classification model performance by providing a clear, educational implementation of standard metrics. It supports both binary and multiclass classification with various averaging strategies.

**Target Audience**: Beginners learning machine learning, data scientists evaluating classification models, and anyone who needs to understand or implement classification metrics from scratch.

## Features

- Calculate accuracy score
- Calculate precision score (binary and multiclass)
- Calculate recall score (binary and multiclass)
- Calculate F1-score (binary and multiclass)
- Support for multiple averaging strategies (macro, micro, weighted)
- Per-class metric calculation
- Confusion matrix generation
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/classification-metrics
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1"
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import ClassificationMetrics

metrics = ClassificationMetrics()

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Calculate individual metrics
accuracy = metrics.accuracy(y_true, y_pred)
precision = metrics.precision(y_true, y_pred)
recall = metrics.recall(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### Calculate All Metrics

```python
from src.main import ClassificationMetrics

metrics = ClassificationMetrics()
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

results = metrics.calculate_all_metrics(y_true, y_pred)
print(results)
```

### Multiclass Classification

```python
from src.main import ClassificationMetrics

metrics = ClassificationMetrics()
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 1, 2]

# Macro averaging
precision_macro = metrics.precision(y_true, y_pred, average="macro")
recall_macro = metrics.recall(y_true, y_pred, average="macro")
f1_macro = metrics.f1_score(y_true, y_pred, average="macro")

# Micro averaging
precision_micro = metrics.precision(y_true, y_pred, average="micro")
recall_micro = metrics.recall(y_true, y_pred, average="micro")
f1_micro = metrics.f1_score(y_true, y_pred, average="micro")

# Weighted averaging
precision_weighted = metrics.precision(y_true, y_pred, average="weighted")
recall_weighted = metrics.recall(y_true, y_pred, average="weighted")
f1_weighted = metrics.f1_score(y_true, y_pred, average="weighted")

# Per-class metrics
precision_per_class = metrics.precision(y_true, y_pred, average=None)
recall_per_class = metrics.recall(y_true, y_pred, average=None)
f1_per_class = metrics.f1_score(y_true, y_pred, average=None)
```

### Confusion Matrix

```python
from src.main import ClassificationMetrics

metrics = ClassificationMetrics()
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

cm = metrics.confusion_matrix(y_true, y_pred)
print(cm)
```

### Command-Line Usage

Basic usage with comma-separated values:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1"
```

Using CSV files:

```bash
python src/main.py --y-true true_labels.csv --y-pred pred_labels.csv --column label
```

Multiclass with macro averaging:

```bash
python src/main.py --y-true "0,1,2,0,1" --y-pred "0,1,1,0,1" --average macro
```

Save results to JSON:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --output results.json
```

### Complete Example

```python
from src.main import ClassificationMetrics
import numpy as np

# Initialize metrics calculator
metrics = ClassificationMetrics()

# Example binary classification
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 0])

# Calculate all metrics
results = metrics.calculate_all_metrics(y_true, y_pred)

print("=== Classification Metrics ===")
print(f"Accuracy:  {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")
print(f"F1-Score:  {results['f1_score']:.4f}")

# Get confusion matrix
cm = metrics.confusion_matrix(y_true, y_pred)
print("\n=== Confusion Matrix ===")
for true_label, pred_dict in cm.items():
    for pred_label, count in pred_dict.items():
        print(f"True={true_label}, Pred={pred_label}: {count}")
```

## Project Structure

```
classification-metrics/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── .env.example             # Environment variables template
├── src/
│   └── main.py              # Main implementation
├── tests/
│   └── test_main.py         # Unit tests
├── docs/
│   └── API.md               # API documentation
└── logs/
    └── .gitkeep             # Log directory
```

### File Descriptions

- `src/main.py`: Core implementation with `ClassificationMetrics` class
- `config.yaml`: Configuration file for logging settings
- `tests/test_main.py`: Comprehensive unit tests
- `docs/API.md`: Detailed API documentation
- `logs/`: Directory for application logs

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

Tests cover:
- Accuracy calculation
- Precision calculation (binary and multiclass)
- Recall calculation (binary and multiclass)
- F1-score calculation (binary and multiclass)
- Confusion matrix generation
- Input validation
- Error handling
- Different input types (lists, numpy arrays, pandas Series)

## Understanding the Metrics

### Accuracy

Accuracy measures the proportion of correct predictions among all predictions:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use**: Good for balanced datasets. Can be misleading for imbalanced datasets.

### Precision

Precision measures the proportion of true positives among all positive predictions:

```
Precision = TP / (TP + FP)
```

**When to use**: Important when false positives are costly (e.g., spam detection).

### Recall

Recall measures the proportion of true positives among all actual positives:

```
Recall = TP / (TP + FN)
```

**When to use**: Important when false negatives are costly (e.g., disease diagnosis).

### F1-Score

F1-score is the harmonic mean of precision and recall:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**When to use**: Good balance between precision and recall. Useful when you need a single metric.

## Averaging Strategies

### Binary

Default for binary classification. Calculates metrics for the positive class only.

### Macro

Calculates metrics for each class independently and takes the unweighted mean.

**Use when**: All classes are equally important.

### Micro

Calculates metrics globally by counting total true positives, false negatives, and false positives.

**Use when**: You want to account for class imbalance.

### Weighted

Calculates metrics for each class and takes the weighted mean based on class frequency.

**Use when**: Classes are imbalanced and you want to account for it.

### None

Returns per-class metrics as a dictionary.

**Use when**: You need detailed per-class performance.

## Troubleshooting

### Common Issues

**Issue**: Length mismatch error

**Solution**: Ensure `y_true` and `y_pred` have the same length.

**Issue**: Division by zero warnings

**Solution**: This occurs when a class has no predictions or no true samples. The metric returns 0.0 in these cases, which is correct behavior.

**Issue**: Invalid average parameter

**Solution**: Use one of: 'binary', 'macro', 'micro', 'weighted', or None.

### Error Messages

- `ValueError: Length mismatch`: Input arrays have different lengths
- `ValueError: Input arrays cannot be empty`: At least one input is empty
- `ValueError: Invalid average parameter`: Unsupported averaging strategy

## Best Practices

1. **Choose appropriate metrics**: Use precision when false positives are costly, recall when false negatives are costly
2. **Consider class imbalance**: Accuracy can be misleading for imbalanced datasets
3. **Use F1-score for balance**: F1-score provides a good balance between precision and recall
4. **Examine confusion matrix**: Provides detailed insight into model performance
5. **Use appropriate averaging**: Macro for equal importance, weighted for imbalanced classes

## Real-World Applications

- Evaluating binary classification models (spam detection, fraud detection)
- Evaluating multiclass classification models (image classification, text categorization)
- Model comparison and selection
- Performance monitoring in production systems
- Educational purposes for understanding metrics

## Contributing

### Development Setup

1. Clone the repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Run tests: `pytest tests/`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Include docstrings for all public functions and classes
- Write tests for all new functionality

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. See LICENSE file in parent directory for details.
