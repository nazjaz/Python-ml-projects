# Data Pipeline with Custom Transformers

A Python tool for creating data pipelines with preprocessing steps chained together using custom transformers. This is the twelfth project in the ML learning series, focusing on building reusable and composable data preprocessing pipelines.

## Project Title and Description

The Data Pipeline Tool provides a framework for chaining multiple preprocessing transformers together in a pipeline. It follows the scikit-learn transformer pattern with custom implementations, allowing users to build complex preprocessing workflows that can be easily reused and maintained.

This tool solves the problem of managing complex preprocessing workflows by providing a clean, chainable interface. It enables users to build preprocessing pipelines that can be fitted once and applied to multiple datasets, ensuring consistent transformations across training and inference.

**Target Audience**: Beginners learning machine learning, data scientists building preprocessing workflows, and anyone who needs to create reusable data transformation pipelines.

## Features

- Custom transformer base class following scikit-learn pattern
- Standard scaler transformer (z-score normalization)
- Min-max scaler transformer (normalization to [0, 1])
- Imputer transformer (handles missing values with multiple strategies)
- One-hot encoder transformer (categorical variable encoding)
- Pipeline class for chaining transformers
- Fit and transform methods for all transformers
- Pipeline information and status tracking
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/data-pipeline
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
python src/main.py --input sample.csv
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
pipeline:
  default_transformers:
    - type: "imputer"
      strategy: "mean"
    - type: "standard_scaler"
  random_state: 42

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `pipeline.default_transformers`: Default transformer configuration
- `pipeline.random_state`: Random seed for reproducibility
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import (
    DataPipeline,
    ImputerTransformer,
    StandardScalerTransformer,
)

# Create transformers
imputer = ImputerTransformer(strategy="mean")
scaler = StandardScalerTransformer()

# Create pipeline
pipeline = DataPipeline(transformers=[imputer, scaler])

# Fit and transform
transformed_df = pipeline.fit_transform(df)
```

### Command-Line Usage

Process data with default pipeline:

```bash
python src/main.py --input data.csv --output transformed_data.csv
```

### Complete Example

```python
import pandas as pd
from src.main import (
    DataPipeline,
    ImputerTransformer,
    StandardScalerTransformer,
    MinMaxScalerTransformer,
    OneHotEncoderTransformer,
)

# Load data
df = pd.read_csv("data.csv")

# Create custom transformers
imputer = ImputerTransformer(strategy="median", columns=["age", "score"])
scaler = StandardScalerTransformer(columns=["age", "score"])
encoder = OneHotEncoderTransformer(columns=["category"])

# Create pipeline
pipeline = DataPipeline(transformers=[imputer, scaler, encoder])

# Fit pipeline on training data
pipeline.fit(df_train)

# Transform training data
df_train_transformed = pipeline.transform(df_train)

# Transform test data (using fitted parameters)
df_test_transformed = pipeline.transform(df_test)
```

### Advanced Pipeline

```python
from src.main import (
    DataPipeline,
    ImputerTransformer,
    MinMaxScalerTransformer,
    OneHotEncoderTransformer,
)

# Complex pipeline with multiple steps
pipeline = DataPipeline(
    transformers=[
        ImputerTransformer(strategy="mean"),
        MinMaxScalerTransformer(feature_range=(0, 1)),
        OneHotEncoderTransformer(),
    ]
)

# Fit and transform
transformed = pipeline.fit_transform(df)

# Get pipeline information
info = pipeline.get_pipeline_info()
print(f"Transformers: {info['transformer_types']}")
```

### Using Individual Transformers

```python
from src.main import StandardScalerTransformer

# Create and fit transformer
scaler = StandardScalerTransformer(columns=["age", "score"])
scaler.fit(df)

# Transform data
scaled_df = scaler.transform(df)

# Or use fit_transform
scaled_df = scaler.fit_transform(df)
```

## Project Structure

```
data-pipeline/
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

- `src/main.py`: Core implementation with transformer classes and pipeline
- `config.yaml`: Configuration file for pipeline parameters
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
- Individual transformer functionality
- Pipeline chaining
- Fit and transform methods
- Error handling
- Edge cases

## Troubleshooting

### Common Issues

**Issue**: Transformer not fitted error

**Solution**: Always call `fit()` before `transform()`, or use `fit_transform()`.

**Issue**: Column not found error

**Solution**: Ensure specified columns exist in the DataFrame.

**Issue**: Pipeline fails on test data

**Solution**: Ensure test data has the same columns as training data.

### Error Messages

- `ValueError: Transformer not fitted`: Call `fit()` first
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: Column 'X' is not numeric`: Use appropriate transformer type

## Transformer Types

### StandardScalerTransformer

- **Purpose**: Z-score normalization (mean=0, std=1)
- **Use for**: Features with different scales
- **Parameters**: columns (optional)

### MinMaxScalerTransformer

- **Purpose**: Normalization to specified range (default [0, 1])
- **Use for**: Features that need bounded ranges
- **Parameters**: columns (optional), feature_range

### ImputerTransformer

- **Purpose**: Handle missing values
- **Strategies**: mean, median, mode, constant
- **Use for**: Datasets with missing values
- **Parameters**: strategy, columns (optional)

### OneHotEncoderTransformer

- **Purpose**: Encode categorical variables
- **Use for**: Categorical features
- **Parameters**: columns (optional)

## Best Practices

1. **Fit on training data**: Always fit pipeline on training data only
2. **Transform consistently**: Use same fitted pipeline for train/test
3. **Order matters**: Place imputation before scaling
4. **Validate columns**: Ensure test data has same columns as training
5. **Check for errors**: Monitor logs for warnings and errors

## Real-World Applications

- Preprocessing pipelines for ML models
- Data transformation workflows
- Feature engineering pipelines
- Consistent preprocessing across datasets
- Reusable preprocessing components

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
