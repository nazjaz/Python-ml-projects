# Classification Visualization Tool

A Python tool for creating confusion matrices and classification reports with visualization capabilities. This is the eighteenth project in the ML learning series, focusing on visualizing and understanding classification model performance.

## Project Title and Description

The Classification Visualization Tool provides automated generation of confusion matrices and classification reports with comprehensive visualization capabilities. It helps users understand model performance through visual representations and detailed metrics.

This tool solves the problem of interpreting classification model results by providing clear visualizations and structured reports. It supports both binary and multiclass classification with customizable visualizations.

**Target Audience**: Beginners learning machine learning, data scientists evaluating classification models, and anyone who needs to visualize and understand classification performance metrics.

## Features

- Generate confusion matrices
- Create detailed classification reports
- Visualize confusion matrices as heatmaps
- Visualize classification reports as heatmaps
- Support for normalized confusion matrices
- Print formatted classification reports to console
- Save visualizations to files (PNG, PDF, SVG)
- Export reports to JSON
- Support for binary and multiclass classification
- Customizable labels and target names
- Configurable visualization parameters
- Command-line interface
- Comprehensive logging
- Input validation and error handling

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/classification-visualization
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
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --plot-cm
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
  colormap: "Blues"
  fontsize: 12
  save_format: "png"
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `visualization.figsize`: Figure size [width, height] in inches
- `visualization.dpi`: DPI for saved figures
- `visualization.colormap`: Colormap for confusion matrix (e.g., "Blues", "Greens", "Reds")
- `visualization.fontsize`: Font size for labels
- `visualization.save_format`: Format for saving figures (png, pdf, svg)

## Usage

### Basic Usage

```python
from src.main import ClassificationVisualizer

viz = ClassificationVisualizer()

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Generate confusion matrix
cm = viz.confusion_matrix(y_true, y_pred)
print(cm)

# Generate classification report
report = viz.classification_report(y_true, y_pred)
print(report)
```

### Visualization

```python
from src.main import ClassificationVisualizer

viz = ClassificationVisualizer()
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Plot confusion matrix
viz.plot_confusion_matrix(y_true, y_pred)

# Plot normalized confusion matrix
viz.plot_confusion_matrix(y_true, y_pred, normalize=True)

# Plot classification report
viz.plot_classification_report(y_true, y_pred)

# Save visualizations
viz.plot_confusion_matrix(
    y_true, y_pred, save_path="confusion_matrix.png", show=False
)
viz.plot_classification_report(
    y_true, y_pred, save_path="classification_report.png", show=False
)
```

### Print Classification Report

```python
from src.main import ClassificationVisualizer

viz = ClassificationVisualizer()
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Print formatted report to console
viz.print_classification_report(y_true, y_pred)
```

### Save Report to JSON

```python
from src.main import ClassificationVisualizer

viz = ClassificationVisualizer()
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Save report to JSON
viz.save_report(y_true, y_pred, output_path="report.json")
```

### Multiclass Classification

```python
from src.main import ClassificationVisualizer

viz = ClassificationVisualizer()
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 1, 2]

# With custom target names
viz.plot_confusion_matrix(
    y_true,
    y_pred,
    target_names=["Class A", "Class B", "Class C"],
)
```

### Command-Line Usage

Basic usage with comma-separated values:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1"
```

Print classification report:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1"
```

Plot confusion matrix:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --plot-cm
```

Plot normalized confusion matrix:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --plot-cm --normalize
```

Plot classification report:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --plot-report
```

Save confusion matrix:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --save-cm cm.png --no-show
```

Save classification report plot:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --save-report-plot report.png --no-show
```

Save report to JSON:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --save-report report.json
```

Using CSV files:

```bash
python src/main.py --y-true true_labels.csv --y-pred pred_labels.csv --column label --plot-cm
```

With custom target names:

```bash
python src/main.py --y-true "0,1,1,0,1" --y-pred "0,1,0,0,1" --target-names "Negative,Positive" --plot-cm
```

### Complete Example

```python
from src.main import ClassificationVisualizer
import numpy as np

# Initialize visualizer
viz = ClassificationVisualizer()

# Example classification results
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 0])

# Print classification report
viz.print_classification_report(y_true, y_pred)

# Plot confusion matrix
viz.plot_confusion_matrix(
    y_true,
    y_pred,
    target_names=["Negative", "Positive"],
    title="Binary Classification Confusion Matrix",
)

# Plot normalized confusion matrix
viz.plot_confusion_matrix(
    y_true,
    y_pred,
    normalize=True,
    target_names=["Negative", "Positive"],
    title="Normalized Confusion Matrix",
)

# Plot classification report
viz.plot_classification_report(
    y_true,
    y_pred,
    target_names=["Negative", "Positive"],
    title="Classification Report",
)

# Save all visualizations
viz.plot_confusion_matrix(
    y_true,
    y_pred,
    save_path="confusion_matrix.png",
    show=False,
)
viz.plot_classification_report(
    y_true,
    y_pred,
    save_path="classification_report.png",
    show=False,
)

# Save report to JSON
viz.save_report(y_true, y_pred, output_path="report.json")
```

## Project Structure

```
classification-visualization/
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

- `src/main.py`: Core implementation with `ClassificationVisualizer` class
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
- Confusion matrix generation
- Classification report generation
- Visualization plotting
- File saving
- Input validation
- Error handling
- Different input types (lists, numpy arrays, pandas Series)
- Binary and multiclass classification

## Understanding the Visualizations

### Confusion Matrix

A confusion matrix shows the count of correct and incorrect predictions for each class. Rows represent true labels, columns represent predicted labels.

**Interpretation:**
- Diagonal elements: Correct predictions
- Off-diagonal elements: Misclassifications
- Normalized version: Shows proportions instead of counts

### Classification Report

A classification report provides detailed metrics for each class:
- **Precision**: Proportion of true positives among all positive predictions
- **Recall**: Proportion of true positives among all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of true samples for each class

**Averages:**
- **Macro Average**: Unweighted mean of per-class metrics
- **Weighted Average**: Weighted mean based on class frequency

## Troubleshooting

### Common Issues

**Issue**: Plots not displaying

**Solution**: Ensure matplotlib backend is properly configured. Use `--no-show` flag when saving files.

**Issue**: Figure too small or large

**Solution**: Adjust `figsize` in `config.yaml` or pass custom figure size.

**Issue**: Labels not showing correctly

**Solution**: Use `--target-names` to provide custom names for classes.

### Error Messages

- `ValueError: Length mismatch`: Input arrays have different lengths
- `ValueError: Input arrays cannot be empty`: At least one input is empty
- `FileNotFoundError`: Configuration file not found

## Best Practices

1. **Use normalized confusion matrix**: For comparing classes with different sample sizes
2. **Customize target names**: Use meaningful names instead of numeric labels
3. **Save visualizations**: Export plots for reports and presentations
4. **Examine per-class metrics**: Identify which classes are problematic
5. **Compare macro and weighted averages**: Understand overall performance

## Real-World Applications

- Evaluating classification models (binary and multiclass)
- Model comparison and selection
- Performance monitoring in production systems
- Creating reports for stakeholders
- Educational purposes for understanding classification metrics
- Debugging model performance issues

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
