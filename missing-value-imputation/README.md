# Missing Value Imputation Tool

A Python tool for cleaning datasets by handling missing values using mean, median, and mode imputation methods. This is the second project in the ML learning series, focusing on fundamental data cleaning skills.

## Project Title and Description

The Missing Value Imputation Tool provides automated solutions for handling missing data in datasets. It supports multiple imputation strategies (mean, median, mode) and can automatically select appropriate methods based on data types. This tool is essential for preparing clean datasets for machine learning models.

This tool solves the problem of missing data that is common in real-world datasets. Missing values can cause errors in ML models and reduce their performance. This tool provides systematic approaches to handle missing values while preserving data characteristics.

**Target Audience**: Beginners learning machine learning, data scientists cleaning datasets, and anyone who needs to handle missing values in their data.

## Features

- Load data from CSV files or pandas DataFrames
- Analyze missing values with detailed statistics
- Mean imputation for numerical columns
- Median imputation for numerical columns
- Mode imputation for categorical columns
- Automatic strategy selection based on data types
- Column-specific imputation
- Imputation summary and tracking
- Save cleaned data to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/missing-value-imputation
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
python src/main.py --input sample.csv --strategy auto
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
imputation:
  default_numeric_strategy: "mean"
  default_categorical_strategy: "mode"
  inplace: false

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `default_numeric_strategy`: Default strategy for numerical columns (mean or median)
- `default_categorical_strategy`: Default strategy for categorical columns (mode)
- `inplace`: Whether to modify data in place (False creates copy)
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import MissingValueImputer

imputer = MissingValueImputer()
imputer.load_data(file_path="data.csv")

# Analyze missing values
analysis = imputer.analyze_missing_values()

# Impute all missing values automatically
cleaned_data = imputer.impute_all()
```

### Command-Line Usage

Impute missing values automatically:

```bash
python src/main.py --input data.csv --strategy auto
```

Use specific strategy:

```bash
python src/main.py --input data.csv --strategy mean
python src/main.py --input data.csv --strategy median
python src/main.py --input data.csv --strategy mode
```

Impute specific columns:

```bash
python src/main.py --input data.csv --strategy mean --columns age score
```

Save cleaned data:

```bash
python src/main.py --input data.csv --strategy auto --output cleaned_data.csv
```

### Complete Example

```python
from src.main import MissingValueImputer
import pandas as pd

# Initialize imputer
imputer = MissingValueImputer()

# Load data
imputer.load_data(file_path="sales_data.csv")

# Analyze missing values
analysis = imputer.analyze_missing_values()
print(f"Missing values: {analysis['total_missing']}")

# Impute numerical columns with mean
imputer.impute_mean()

# Impute categorical columns with mode
imputer.impute_mode()

# Or impute all automatically
cleaned_data = imputer.impute_all()

# Get imputation summary
summary = imputer.get_imputation_summary()
print("Imputation strategies used:")
for col, strategy in summary.items():
    print(f"  {col}: {strategy}")

# Save cleaned data
imputer.save_cleaned_data("cleaned_sales_data.csv")
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import MissingValueImputer

df = pd.read_csv("data.csv")
imputer = MissingValueImputer()
imputer.load_data(dataframe=df)

cleaned_df = imputer.impute_all()
```

## Project Structure

```
missing-value-imputation/
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

- `src/main.py`: Core implementation with `MissingValueImputer` class
- `config.yaml`: Configuration file for imputation strategies
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
- Missing value analysis
- Mean imputation
- Median imputation
- Mode imputation
- Automatic imputation
- Column-specific imputation
- Error handling

## Troubleshooting

### Common Issues

**Issue**: All values become NaN after imputation

**Solution**: Check that columns have valid data types. Mean/median require numeric columns, mode requires categorical columns.

**Issue**: Mode imputation fails

**Solution**: Ensure categorical columns have at least one non-null value to calculate mode.

**Issue**: Memory errors with large datasets

**Solution**: Use `inplace=False` (default) to work with copies, or process data in chunks.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: Invalid strategy`: Use 'mean', 'median', or 'mode'

## Imputation Strategies

### Mean Imputation

- **Use for**: Numerical columns with normal distribution
- **Pros**: Preserves mean, good for normally distributed data
- **Cons**: Can reduce variance, not ideal for skewed data

### Median Imputation

- **Use for**: Numerical columns with skewed distribution
- **Pros**: Robust to outliers, preserves median
- **Cons**: May not preserve mean

### Mode Imputation

- **Use for**: Categorical columns
- **Pros**: Preserves most common category
- **Cons**: May introduce bias if mode is not representative

## Best Practices

1. **Analyze first**: Always analyze missing values before imputing
2. **Choose wisely**: Select imputation strategy based on data distribution
3. **Document**: Keep track of imputation strategies used
4. **Validate**: Check imputed values make sense for your domain
5. **Consider alternatives**: Sometimes dropping rows/columns may be better

## Real-World Applications

- Data preprocessing for ML pipelines
- Cleaning survey data
- Handling sensor data gaps
- Preparing datasets for analysis
- Data quality improvement

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
