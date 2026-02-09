# Outlier Detection and Handling Tool

A Python tool for detecting and handling outliers using IQR method, Z-score, and isolation forest techniques. This is the eighth project in the ML learning series, focusing on outlier detection and data quality improvement.

## Project Title and Description

The Outlier Detection and Handling Tool provides automated detection and handling of outliers in datasets using multiple proven methods. It supports IQR (Interquartile Range), Z-score, and Isolation Forest techniques, with options to remove or cap outliers based on detection results.

This tool solves the problem of outliers that can significantly impact ML model performance. Outliers can skew model training, affect predictions, and reduce model accuracy. This tool provides systematic approaches to identify and handle outliers appropriately.

**Target Audience**: Beginners learning machine learning, data scientists cleaning datasets, and anyone who needs to detect and handle outliers in their data.

## Features

- Load data from CSV files or pandas DataFrames
- IQR method for outlier detection
- Z-score method for outlier detection
- Isolation Forest for multivariate outlier detection
- Outlier removal functionality
- Outlier capping (winsorization) functionality
- Outlier summary statistics
- Column-specific detection
- Automatic detection of numerical columns
- Configurable thresholds and parameters
- Save cleaned data to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/outlier-detection
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
python src/main.py --input sample.csv --method iqr
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
outlier_detection:
  iqr_multiplier: 1.5
  z_score_threshold: 3.0
  isolation_contamination: 0.1
  isolation_random_state: 42

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `iqr_multiplier`: IQR multiplier (1.5 for normal, 3.0 for extreme outliers)
- `z_score_threshold`: Z-score threshold (3.0 for normal, 2.0 for more sensitive)
- `isolation_contamination`: Expected proportion of outliers (0.0 to 0.5)
- `isolation_random_state`: Random seed for Isolation Forest
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import OutlierDetector

detector = OutlierDetector()
detector.load_data(file_path="data.csv")

# Detect outliers using IQR
outlier_mask = detector.detect_iqr()

# Get summary
summary = detector.get_outlier_summary()
```

### Command-Line Usage

Detect outliers using IQR:

```bash
python src/main.py --input data.csv --method iqr
```

Detect outliers using Z-score:

```bash
python src/main.py --input data.csv --method zscore
```

Detect outliers using Isolation Forest:

```bash
python src/main.py --input data.csv --method isolation_forest
```

Detect using all methods:

```bash
python src.main.py --input data.csv --method all
```

Remove outliers:

```bash
python src/main.py --input data.csv --method iqr --action remove --output cleaned.csv
```

Cap outliers:

```bash
python src/main.py --input data.csv --method iqr --action cap --output capped.csv
```

Analyze specific columns:

```bash
python src/main.py --input data.csv --method all --columns age score
```

### Complete Example

```python
from src.main import OutlierDetector
import pandas as pd

# Initialize detector
detector = OutlierDetector()

# Load data
detector.load_data(file_path="sales_data.csv")

# Detect outliers using multiple methods
iqr_mask = detector.detect_iqr()
zscore_mask = detector.detect_zscore()
iso_mask = detector.detect_isolation_forest()

# Get summary
summary = detector.get_outlier_summary()
print(f"Outliers detected: {summary['outlier_count']} ({summary['outlier_percentage']:.2f}%)")

# Remove outliers
cleaned_data = detector.remove_outliers()
print(f"Cleaned dataset shape: {cleaned_data.shape}")

# Or cap outliers
capped_data = detector.cap_outliers(method="iqr")
print(f"Capped dataset shape: {capped_data.shape}")

# Save cleaned data
cleaned_data.to_csv("cleaned_sales_data.csv", index=False)
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import OutlierDetector

df = pd.read_csv("data.csv")
detector = OutlierDetector()
detector.load_data(dataframe=df)

outlier_mask = detector.detect_iqr()
cleaned_df = detector.remove_outliers(outlier_mask)
```

## Project Structure

```
outlier-detection/
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

- `src/main.py`: Core implementation with `OutlierDetector` class
- `config.yaml`: Configuration file for detection parameters
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
- IQR outlier detection
- Z-score outlier detection
- Isolation Forest detection
- Outlier removal
- Outlier capping
- Summary generation
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Too many outliers detected

**Solution**: Adjust thresholds (increase IQR multiplier or Z-score threshold).

**Issue**: Isolation Forest detects no outliers

**Solution**: Increase contamination parameter or check data quality.

**Issue**: Outlier removal removes too much data

**Solution**: Consider capping instead of removing, or use less sensitive thresholds.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: No outliers detected`: Run detection methods first

## Detection Methods

### IQR Method

- **Method**: Uses interquartile range (Q3 - Q1)
- **Formula**: Outliers are values < Q1 - k*IQR or > Q3 + k*IQR
- **Use for**: Univariate detection, robust to outliers
- **Pros**: Simple, interpretable, robust
- **Cons**: Only works for univariate detection

### Z-Score Method

- **Method**: Uses standard deviations from mean
- **Formula**: Outliers are values with |z-score| > threshold
- **Use for**: Normally distributed data
- **Pros**: Simple, widely understood
- **Cons**: Sensitive to outliers (mean/std affected), assumes normal distribution

### Isolation Forest

- **Method**: Machine learning-based anomaly detection
- **Use for**: Multivariate detection, complex patterns
- **Pros**: Handles multivariate data, detects complex patterns
- **Cons**: Requires tuning, less interpretable

## Handling Strategies

### Removal

- **Use when**: Outliers are clearly errors or noise
- **Pros**: Clean dataset, no distortion
- **Cons**: Loss of data, may remove valid extreme values

### Capping (Winsorization)

- **Use when**: Outliers may be valid but extreme
- **Pros**: Preserves data, reduces impact
- **Cons**: May still affect distributions

## Best Practices

1. **Visualize first**: Always visualize data before removing outliers
2. **Understand context**: Consider domain knowledge before removing
3. **Try multiple methods**: Compare results from different methods
4. **Start conservative**: Use less sensitive thresholds initially
5. **Document decisions**: Keep track of what was removed and why

## Real-World Applications

- Data quality improvement
- Preprocessing for ML models
- Cleaning sensor data
- Removing measurement errors
- Improving model performance
- Statistical analysis preparation

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
