# Feature Scaling Tool

A Python tool for scaling numerical features using min-max scaling, z-score
normalization, robust scaling, quantile transformation, and power
transformation. This project focuses on feature scaling techniques essential
for machine learning.

## Project Title and Description

The Feature Scaling Tool provides automated solutions for scaling numerical
features in datasets. It supports min-max scaling, z-score normalization,
robust scaling, quantile transformation, and power transformation, which are
common preprocessing steps for many machine learning algorithms that are
sensitive to feature scales.

This tool solves the problem of features with different scales that can cause ML algorithms to perform poorly. Features with larger scales can dominate those with smaller scales, leading to biased models. This tool ensures all features are on a similar scale.

**Target Audience**: Beginners learning machine learning, data scientists preprocessing data, and anyone who needs to scale features for ML models.

## Features

- Load data from CSV files or pandas DataFrames
- Min-max scaling (normalization) to specified range
- Z-score normalization (standardization) to mean=0, std=1
- Robust scaling using median and IQR (outlier resistant)
- Quantile transformation to uniform or normal distributions
- Power transformation (Yeo-Johnson and Box-Cox)
- Automatic detection of numerical columns
- Column-specific scaling support
- Inverse transformation to original scale
- Scaling parameter tracking
- Save scaled data to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/feature-scaling
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
python src/main.py --input sample.csv --method min_max
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
scaling:
  min_max_range: [0, 1]
  inplace: false

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `min_max_range`: Feature range for min-max scaling [min, max] (default: [0, 1])
- `inplace`: Whether to modify data in place (False creates copy)
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import FeatureScaler

scaler = FeatureScaler()
scaler.load_data(file_path="data.csv")

# Apply min-max scaling
scaled_data = scaler.min_max_scale()

# Apply z-score normalization
normalized_data = scaler.z_score_normalize()
```

### Command-Line Usage

Apply min-max scaling:

```bash
python src/main.py --input data.csv --method min_max
```

Apply z-score normalization:

```bash
python src/main.py --input data.csv --method z_score
```

Apply robust scaling:

```bash
python src/main.py --input data.csv --method robust
```

Apply quantile transformation:

```bash
python src/main.py --input data.csv --method quantile
```

Apply power transformation:

```bash
python src/main.py --input data.csv --method power
```

Apply both methods:

```bash
python src/main.py --input data.csv --method both
```

Scale specific columns:

```bash
python src/main.py --input data.csv --method min_max --columns age score
```

Custom feature range:

```bash
python src/main.py --input data.csv --method min_max --range -1 1
```

Save scaled data:

```bash
python src/main.py --input data.csv --method z_score --output scaled_data.csv
```

### Complete Example

```python
from src.main import FeatureScaler
import pandas as pd

# Initialize scaler
scaler = FeatureScaler()

# Load data
scaler.load_data(file_path="sales_data.csv")

# Get numerical columns
numeric_cols = scaler.get_numeric_columns()
print(f"Numerical columns: {numeric_cols}")

# Apply min-max scaling to [0, 1]
scaled_data = scaler.min_max_scale(feature_range=(0, 1))

# Apply z-score normalization
normalized_data = scaler.z_score_normalize()

# Get scaling summary
summary = scaler.get_scaling_summary()
print("Scaling parameters:")
for col, params in summary.items():
    print(f"  {col}: {params}")

# Inverse transform back to original scale
original_data = scaler.inverse_transform(scaled_data)

# Save scaled data
scaler.save_scaled_data("scaled_sales_data.csv")
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import FeatureScaler

df = pd.read_csv("data.csv")
scaler = FeatureScaler()
scaler.load_data(dataframe=df)

scaled_df = scaler.min_max_scale()
normalized_df = scaler.z_score_normalize()
```

## Project Structure

```
feature-scaling/
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

- `src/main.py`: Core implementation with `FeatureScaler` class
- `config.yaml`: Configuration file for scaling parameters
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
- Min-max scaling with different ranges
- Z-score normalization
- Column-specific scaling
- Inverse transformation
- Edge cases (zero variance, zero range)
- Error handling

## Troubleshooting

### Common Issues

**Issue**: All values become NaN after scaling

**Solution**: Check that columns are numeric and have valid values (not all NaN).

**Issue**: Zero variance error

**Solution**: Columns with zero variance (all same values) are handled automatically.

**Issue**: Inverse transform doesn't match original

**Solution**: Ensure you use the same scaler instance that performed the scaling.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: feature_range min must be less than max`: Fix range values

## Scaling Methods

### Min-Max Scaling (Normalization)

- **Formula**: (x - min) / (max - min) * (max_range - min_range) + min_range
- **Range**: Typically [0, 1] or [-1, 1]
- **Use for**: When you know the bounds of your data
- **Pros**: Preserves relationships, bounded output
- **Cons**: Sensitive to outliers

### Z-Score Normalization (Standardization)

- **Formula**: (x - mean) / std
- **Range**: Unbounded (typically [-3, 3] for most data)
- **Use for**: When data distribution is unknown or has outliers
- **Pros**: Less sensitive to outliers, mean=0, std=1
- **Cons**: Unbounded output

### Robust Scaling

- **Approach**: Center using median and scale using IQR (Q3 - Q1)
- **Use for**: Data with outliers or heavy-tailed distributions
- **Pros**: Outlier resistant
- **Cons**: Output is unbounded, interpretation depends on IQR

### Quantile Transformation

- **Approach**: Map data to a target distribution using the empirical CDF
- **Output**: "uniform" or "normal"
- **Use for**: Reducing outlier impact, making distributions more Gaussian-like
- **Pros**: Strong outlier handling, distribution shaping
- **Cons**: Can distort linear relationships

### Power Transformation

- **Methods**: Yeo-Johnson (supports non-positive values), Box-Cox (positive only)
- **Use for**: Stabilizing variance and making data more Gaussian-like
- **Pros**: Useful for skewed features
- **Cons**: Box-Cox requires strictly positive values

## When to Use Each Method

### Use Min-Max Scaling When:
- You know the bounds of your data
- Data is uniformly distributed
- You need values in a specific range
- Using algorithms sensitive to feature ranges (e.g., neural networks)

### Use Z-Score Normalization When:
- Data has outliers
- Data distribution is unknown
- Using algorithms that assume normal distribution
- Features have different units

## Best Practices

1. **Scale before splitting**: Scale on training data, then apply same parameters to test data
2. **Choose method wisely**: Consider your data distribution and ML algorithm
3. **Track parameters**: Save scaling parameters for inverse transformation
4. **Handle edge cases**: Zero variance and zero range are handled automatically
5. **Validate**: Check scaled data statistics match expectations

## Real-World Applications

- Preprocessing for ML models (SVM, neural networks, k-means)
- Feature engineering pipelines
- Data normalization for analysis
- Preparing data for distance-based algorithms
- Standardizing features from different sources

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
