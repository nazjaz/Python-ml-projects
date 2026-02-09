# Categorical Variable Encoding Tool

A Python tool for encoding categorical variables using one-hot encoding and label encoding with comprehensive comparison analysis. This is the fourth project in the ML learning series, focusing on categorical data preprocessing.

## Project Title and Description

The Categorical Variable Encoding Tool provides automated solutions for converting categorical data into numerical formats suitable for machine learning algorithms. It supports one-hot encoding for nominal data and label encoding for ordinal data, with intelligent comparison to help choose the best method.

This tool solves the problem of categorical variables that cannot be directly used by most ML algorithms. It provides multiple encoding strategies with analysis to help select the most appropriate method based on data characteristics.

**Target Audience**: Beginners learning machine learning, data scientists preprocessing categorical data, and anyone who needs to encode categorical variables for ML models.

## Features

- Load data from CSV files or pandas DataFrames
- One-hot encoding for nominal categorical variables
- Label encoding for ordinal categorical variables
- Automatic detection of categorical columns
- Column-specific encoding support
- Comparison analysis between encoding methods
- Inverse transformation for label encoding
- Encoding parameter tracking
- Save encoded data to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/categorical-encoding
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
python src/main.py --input sample.csv --method compare
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
encoding:
  drop_first: false
  prefix: null
  prefix_sep: "_"
  inplace: false

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `drop_first`: Whether to drop first category in one-hot encoding (reduces multicollinearity)
- `prefix`: Prefix for one-hot encoded column names (None uses column name)
- `prefix_sep`: Separator for prefix in one-hot encoding
- `inplace`: Whether to modify data in place (False creates copy)
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import CategoricalEncoder

encoder = CategoricalEncoder()
encoder.load_data(file_path="data.csv")

# Apply one-hot encoding
encoded_data = encoder.one_hot_encode()

# Apply label encoding
encoded_data = encoder.label_encode()
```

### Command-Line Usage

Compare encoding methods:

```bash
python src/main.py --input data.csv --method compare
```

Apply one-hot encoding:

```bash
python src/main.py --input data.csv --method one_hot
```

Apply label encoding:

```bash
python src/main.py --input data.csv --method label
```

Apply both methods:

```bash
python src/main.py --input data.csv --method both
```

Encode specific columns:

```bash
python src/main.py --input data.csv --method one_hot --columns category type
```

Save encoded data:

```bash
python src/main.py --input data.csv --method label --output encoded_data.csv
```

### Complete Example

```python
from src.main import CategoricalEncoder
import pandas as pd

# Initialize encoder
encoder = CategoricalEncoder()

# Load data
encoder.load_data(file_path="sales_data.csv")

# Get categorical columns
categorical_cols = encoder.get_categorical_columns()
print(f"Categorical columns: {categorical_cols}")

# Compare encoding methods
comparison = encoder.compare_encodings()
for col, info in comparison.items():
    if isinstance(info, dict) and "column" in info:
        print(f"{info['column']}: {info['recommendation']}")

# Apply one-hot encoding
one_hot_data = encoder.one_hot_encode(columns=["category", "type"])

# Apply label encoding
label_data = encoder.label_encode(columns=["priority"])

# Get encoding summary
summary = encoder.get_encoding_summary()
print("Encoding summary:")
for col, info in summary.items():
    print(f"  {col}: {info['method']}")

# Inverse transform label encoding
original_data = encoder.inverse_label_encode(label_data, columns=["priority"])

# Save encoded data
encoder.save_encoded_data("encoded_sales_data.csv")
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import CategoricalEncoder

df = pd.read_csv("data.csv")
encoder = CategoricalEncoder()
encoder.load_data(dataframe=df)

encoded_df = encoder.one_hot_encode()
```

## Project Structure

```
categorical-encoding/
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

- `src/main.py`: Core implementation with `CategoricalEncoder` class
- `config.yaml`: Configuration file for encoding parameters
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
- One-hot encoding with various options
- Label encoding
- Comparison analysis
- Inverse transformation
- Column-specific encoding
- Error handling

## Troubleshooting

### Common Issues

**Issue**: One-hot encoding creates too many columns

**Solution**: Use label encoding for columns with many categories, or use `drop_first=True` to reduce columns.

**Issue**: Label encoding introduces false ordinality

**Solution**: Use one-hot encoding for nominal data (no inherent order).

**Issue**: Memory errors with one-hot encoding

**Solution**: Use label encoding for high-cardinality categorical variables.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: No label encodings found`: Encode data before inverse transform

## Encoding Methods

### One-Hot Encoding

- **Method**: Creates binary columns for each category
- **Use for**: Nominal data (no inherent order)
- **Pros**: No false ordinality, preserves all information
- **Cons**: Creates many columns, high memory usage
- **Example**: Colors (red, blue, green) -> 3 binary columns

### Label Encoding

- **Method**: Maps each category to unique integer
- **Use for**: Ordinal data (has inherent order) or high-cardinality
- **Pros**: Single column, low memory usage
- **Cons**: Introduces false ordinality for nominal data
- **Example**: Priority (low, medium, high) -> (0, 1, 2)

## When to Use Each Method

### Use One-Hot Encoding When:
- Data is nominal (no inherent order)
- Few unique categories (< 10)
- Need to preserve all category information
- Using tree-based models (can handle many features)

### Use Label Encoding When:
- Data is ordinal (has inherent order)
- Many unique categories (> 10)
- Memory is a concern
- Using linear models (fewer features preferred)

## Comparison Analysis

The tool provides automatic comparison that considers:
- Number of unique categories
- Memory implications
- Sparsity of one-hot encoding
- Recommendations based on data characteristics

## Best Practices

1. **Analyze first**: Use comparison to choose appropriate method
2. **Consider data type**: Nominal vs ordinal determines method
3. **Handle high-cardinality**: Use label encoding for many categories
4. **Track mappings**: Save encoding mappings for inverse transform
5. **Test both**: Sometimes try both methods and compare model performance

## Real-World Applications

- Preprocessing categorical features for ML models
- Converting survey data to numerical format
- Encoding product categories and types
- Handling geographic or demographic data
- Preparing data for scikit-learn models

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
