# ROC Curve and AUC Calculator

A Python tool for calculating ROC curves and AUC (Area Under the Curve) for binary classification model evaluation. This is the nineteenth project in the ML learning series, focusing on understanding and implementing ROC curve analysis for model evaluation.

## Project Title and Description

The ROC Curve and AUC Calculator provides automated calculation of ROC curves and AUC scores for binary classification models. It helps users understand model performance through receiver operating characteristic analysis, which is essential for evaluating classification models, especially when dealing with imbalanced datasets.

This tool solves the problem of evaluating binary classification model performance by providing clear ROC curve visualization and AUC calculation. It implements the ROC curve algorithm from scratch, helping users understand how these metrics work.

**Target Audience**: Beginners learning machine learning, data scientists evaluating binary classification models, and anyone who needs to understand or implement ROC curve analysis from scratch.

## Features

- Calculate ROC curve (True Positive Rate vs False Positive Rate)
- Calculate AUC (Area Under the ROC Curve)
- Visualize ROC curves with matplotlib
- Support for custom positive class labels
- Detailed reporting with threshold information
- Export results to JSON
- Save ROC curve plots to files
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas Series)

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/roc-auc-calculator
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
python src/main.py --y-true "0,0,1,1" --y-scores "0.1,0.4,0.35,0.8" --plot
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

visualization:
  figsize: [10, 8]
  dpi: 100
  linewidth: 2
  fontsize: 12
  save_format: "png"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `visualization.figsize`: Figure size [width, height] in inches
- `visualization.dpi`: DPI for saved figures
- `visualization.linewidth`: Line width for ROC curve
- `visualization.fontsize`: Font size for labels
- `visualization.save_format`: Format for saving figures (png, pdf, svg)

## Usage

### Basic Usage

```python
from src.main import ROCCalculator

calc = ROCCalculator()

y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# Calculate ROC curve
fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)

# Calculate AUC
auc_score = calc.auc(y_true, y_scores)
print(f"AUC: {auc_score:.4f}")
```

### Visualization

```python
from src.main import ROCCalculator

calc = ROCCalculator()
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# Plot ROC curve
calc.plot_roc_curve(y_true, y_scores)

# Save ROC curve
calc.plot_roc_curve(
    y_true, y_scores, save_path="roc_curve.png", show=False
)
```

### Calculate All Metrics

```python
from src.main import ROCCalculator

calc = ROCCalculator()
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# Calculate all metrics
results = calc.calculate_all_metrics(y_true, y_scores)
print(f"AUC: {results['auc']:.4f}")
print(f"FPR points: {len(results['roc_curve']['fpr'])}")
print(f"TPR points: {len(results['roc_curve']['tpr'])}")
```

### Print Report

```python
from src.main import ROCCalculator

calc = ROCCalculator()
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# Print formatted report
calc.print_report(y_true, y_scores)
```

### Save Report to JSON

```python
from src.main import ROCCalculator

calc = ROCCalculator()
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

# Save report to JSON
calc.save_report(y_true, y_scores, output_path="report.json")
```

### Command-Line Usage

Basic usage with comma-separated values:

```bash
python src/main.py --y-true "0,0,1,1" --y-scores "0.1,0.4,0.35,0.8"
```

Plot ROC curve:

```bash
python src/main.py --y-true "0,0,1,1" --y-scores "0.1,0.4,0.35,0.8" --plot
```

Save ROC curve plot:

```bash
python src/main.py --y-true "0,0,1,1" --y-scores "0.1,0.4,0.35,0.8" --save-plot roc.png --no-show
```

Save report to JSON:

```bash
python src/main.py --y-true "0,0,1,1" --y-scores "0.1,0.4,0.35,0.8" --save-report report.json
```

Using CSV files:

```bash
python src/main.py --y-true true_labels.csv --y-scores scores.csv --column value --plot
```

Custom positive label:

```bash
python src/main.py --y-true "1,1,0,0" --y-scores "0.1,0.4,0.35,0.8" --pos-label 0 --plot
```

### Complete Example

```python
from src.main import ROCCalculator
import numpy as np

# Initialize calculator
calc = ROCCalculator()

# Example binary classification results
y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.4, 0.85, 0.25])

# Calculate ROC curve
fpr, tpr, thresholds = calc.roc_curve(y_true, y_scores)

# Calculate AUC
auc_score = calc.auc(y_true, y_scores)

print(f"AUC Score: {auc_score:.4f}")
print(f"ROC Curve Points: {len(fpr)}")

# Plot ROC curve
calc.plot_roc_curve(
    y_true,
    y_scores,
    title="ROC Curve Example",
    label="Model Performance",
)

# Print detailed report
calc.print_report(y_true, y_scores)

# Save all results
calc.plot_roc_curve(
    y_true, y_scores, save_path="roc_curve.png", show=False
)
calc.save_report(y_true, y_scores, output_path="roc_auc_report.json")
```

## Project Structure

```
roc-auc-calculator/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
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

- `src/main.py`: Core implementation with `ROCCalculator` class
- `config.yaml`: Configuration file for logging and visualization settings
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
- ROC curve calculation
- AUC calculation
- Visualization plotting
- File saving
- Input validation
- Error handling
- Different input types (lists, numpy arrays, pandas Series)
- Edge cases (perfect classifier, random classifier)

## Understanding ROC Curve and AUC

### ROC Curve

The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

**Components:**
- **True Positive Rate (TPR)**: Also called Sensitivity or Recall
  - TPR = TP / (TP + FN)
  - Proportion of actual positives correctly identified
- **False Positive Rate (FPR)**: Also called Fall-out
  - FPR = FP / (FP + TN)
  - Proportion of actual negatives incorrectly identified as positives

**Interpretation:**
- **Top-left corner (0, 1)**: Perfect classifier (TPR=1, FPR=0)
- **Diagonal line**: Random classifier (AUC = 0.5)
- **Above diagonal**: Better than random
- **Below diagonal**: Worse than random

### AUC (Area Under the Curve)

AUC measures the area under the ROC curve, providing a single metric to evaluate classifier performance.

**Interpretation:**
- **AUC = 1.0**: Perfect classifier
- **AUC > 0.9**: Excellent classifier
- **AUC > 0.8**: Good classifier
- **AUC > 0.7**: Fair classifier
- **AUC = 0.5**: Random classifier (no discriminative ability)
- **AUC < 0.5**: Worse than random (classifier is inverted)

**Advantages:**
- Threshold-independent metric
- Works well with imbalanced datasets
- Provides single number for comparison
- Scale-invariant

## Troubleshooting

### Common Issues

**Issue**: Multiclass classification error

**Solution**: ROC curve is only for binary classification. Ensure labels are 0 and 1 only.

**Issue**: Invalid labels error

**Solution**: Ensure y_true contains only 0 and 1 values.

**Issue**: Plots not displaying

**Solution**: Ensure matplotlib backend is properly configured. Use `--no-show` flag when saving files.

### Error Messages

- `ValueError: Length mismatch`: Input arrays have different lengths
- `ValueError: Input arrays cannot be empty`: At least one input is empty
- `ValueError: ROC curve is only for binary classification`: More than 2 unique labels found
- `ValueError: y_true must contain only 0 and 1`: Invalid label values

## Best Practices

1. **Use ROC curve for binary classification**: Only applicable to binary problems
2. **Compare multiple models**: Plot multiple ROC curves on same plot for comparison
3. **Consider class imbalance**: ROC curve is less sensitive to class imbalance than accuracy
4. **Use AUC for model selection**: Single metric for comparing models
5. **Examine threshold trade-offs**: ROC curve shows trade-off between TPR and FPR

## Real-World Applications

- Evaluating binary classification models
- Model comparison and selection
- Threshold selection for classification
- Performance monitoring in production systems
- Medical diagnosis systems
- Fraud detection systems
- Spam detection systems
- Educational purposes for understanding ROC analysis

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
