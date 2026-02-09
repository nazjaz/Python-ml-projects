# Feature Selection Tool

A Python tool for performing feature selection using variance threshold, correlation analysis, and univariate statistics. This is the ninth project in the ML learning series, focusing on feature selection for improving model performance.

## Project Title and Description

The Feature Selection Tool provides automated feature selection using multiple statistical methods. It helps identify and select the most relevant features, reducing dimensionality, improving model performance, and reducing overfitting in machine learning models.

This tool solves the problem of high-dimensional datasets with irrelevant, redundant, or noisy features. By selecting only the most important features, models become more interpretable, faster to train, and often perform better.

**Target Audience**: Beginners learning machine learning, data scientists performing feature engineering, and anyone who needs to select relevant features for ML models.

## Features

- Load data from CSV files or pandas DataFrames
- Variance threshold feature selection
- Correlation-based feature selection
- Univariate statistical feature selection (f_classif, f_regression, chi2, mutual_info)
- Sequential application of all methods
- Feature selection summary and statistics
- Apply selection to data
- Save selected features to CSV
- Automatic detection of numerical columns
- Configurable thresholds and parameters
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/feature-selection
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
python src/main.py --input sample.csv --method variance
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
feature_selection:
  variance_threshold: 0.0
  correlation_threshold: 0.95
  univariate_k: 10
  univariate_score_func: "f_classif"

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `variance_threshold`: Variance threshold (features below this are removed)
- `correlation_threshold`: Correlation threshold (features above this are removed)
- `univariate_k`: Number of top features for univariate selection
- `univariate_score_func`: Score function (f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression)
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import FeatureSelector

selector = FeatureSelector()
selector.load_data(file_path="data.csv", target_column="label")

# Select features using variance threshold
selected = selector.select_variance_threshold()

# Apply selection
selected_data = selector.apply_selection()
```

### Command-Line Usage

Variance threshold selection:

```bash
python src/main.py --input data.csv --method variance
```

Correlation-based selection:

```bash
python src/main.py --input data.csv --method correlation
```

Univariate selection:

```bash
python src/main.py --input data.csv --method univariate --target label
```

Apply all methods:

```bash
python src/main.py --input data.csv --method all --target label
```

Custom thresholds:

```bash
python src/main.py --input data.csv --method variance --variance-threshold 0.1
python src/main.py --input data.csv --method correlation --correlation-threshold 0.9
python src/main.py --input data.csv --method univariate --target label --k 20
```

Save selected features:

```bash
python src/main.py --input data.csv --method all --target label --output selected_features.csv
```

### Complete Example

```python
from src.main import FeatureSelector
import pandas as pd

# Initialize selector
selector = FeatureSelector()

# Load data with target
selector.load_data(
    file_path="sales_data.csv",
    target_column="sales_category"
)

# Apply variance threshold
variance_features = selector.select_variance_threshold(threshold=0.01)

# Apply correlation analysis
correlation_features = selector.select_correlation(threshold=0.9)

# Apply univariate selection
univariate_features = selector.select_univariate(
    k=15,
    score_func="f_classif"
)

# Or apply all methods sequentially
all_features = selector.select_all(
    variance_threshold=0.01,
    correlation_threshold=0.9,
    univariate_k=15
)

# Get summary
summary = selector.get_selection_summary()
print(f"Selected {summary['selected_features']} features")
print(f"Removed {summary['removed_features']} features")

# Apply selection to data
selected_data = selector.apply_selection()

# Save selected features
selector.save_selected_data("selected_sales_data.csv")
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import FeatureSelector

df = pd.read_csv("data.csv")
selector = FeatureSelector()
selector.load_data(dataframe=df, target_column="target")

selected = selector.select_all()
selected_data = selector.apply_selection()
```

## Project Structure

```
feature-selection/
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

- `src/main.py`: Core implementation with `FeatureSelector` class
- `config.yaml`: Configuration file for selection parameters
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
- Data loading from files and DataFrames
- Variance threshold selection
- Correlation-based selection
- Univariate statistical selection
- Sequential application of methods
- Selection summary generation
- Error handling

## Troubleshooting

### Common Issues

**Issue**: No features selected

**Solution**: Adjust thresholds (lower variance threshold, higher correlation threshold, increase k for univariate).

**Issue**: Univariate selection fails

**Solution**: Ensure target column is provided and data types are appropriate for the score function.

**Issue**: Too many features removed

**Solution**: Use less aggressive thresholds or apply methods individually.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Target column required`: Provide target column for univariate selection
- `ValueError: Invalid score function`: Use valid score function name

## Selection Methods

### Variance Threshold

- **Method**: Removes features with low variance
- **Use for**: Removing constant or near-constant features
- **Pros**: Simple, fast, removes uninformative features
- **Cons**: Doesn't consider target variable

### Correlation Analysis

- **Method**: Removes highly correlated features
- **Use for**: Reducing multicollinearity
- **Pros**: Removes redundant features
- **Cons**: May remove important features if threshold too low

### Univariate Statistics

- **Methods**: f_classif, f_regression, chi2, mutual_info
- **Use for**: Selecting features most related to target
- **Pros**: Considers target variable, statistical foundation
- **Cons**: Doesn't consider feature interactions

## Score Functions

### For Classification

- **f_classif**: F-test for classification
- **chi2**: Chi-squared test (non-negative features)
- **mutual_info_classif**: Mutual information for classification

### For Regression

- **f_regression**: F-test for regression
- **mutual_info_regression**: Mutual information for regression

## Best Practices

1. **Start with variance**: Remove constant features first
2. **Check correlations**: Remove highly correlated features
3. **Use univariate**: Select top features based on target relationship
4. **Validate selection**: Test model performance with selected features
5. **Document choices**: Keep track of selected features and thresholds

## Real-World Applications

- Dimensionality reduction for ML models
- Improving model interpretability
- Reducing overfitting
- Speeding up model training
- Feature engineering pipelines
- Model performance optimization

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
