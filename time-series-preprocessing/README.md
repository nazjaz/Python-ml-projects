# Time Series Data Preprocessing Tool

A Python tool for performing time series data preprocessing including resampling, interpolation, and trend removal. This is the thirteenth project in the ML learning series, focusing on preparing time series data for analysis and modeling.

## Project Title and Description

The Time Series Preprocessing Tool provides comprehensive preprocessing capabilities for time series data. It supports resampling to different frequencies, multiple interpolation methods for handling missing values, and various trend removal techniques to prepare data for time series analysis and modeling.

This tool solves the problem of preparing raw time series data for analysis by providing automated preprocessing operations. It handles common time series challenges like irregular sampling, missing values, and trend components that can affect analysis and model performance.

**Target Audience**: Beginners learning time series analysis, data scientists working with temporal data, and anyone who needs to preprocess time series data for ML models.

## Features

- Load time series data from CSV files or pandas DataFrames
- Automatic datetime column detection
- Resampling to different frequencies (daily, hourly, weekly, monthly)
- Multiple interpolation methods (linear, polynomial, spline, time-based)
- Trend removal techniques (linear, polynomial, moving average)
- Time series information and statistics
- Save preprocessed data to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/time-series-preprocessing
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
preprocessing:
  default_resample_freq: "D"
  default_interpolation_method: "linear"
  default_trend_removal_method: "linear"

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `default_resample_freq`: Default resampling frequency (D, H, W, M)
- `default_interpolation_method`: Default interpolation method
- `default_trend_removal_method`: Default trend removal method
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()
preprocessor.load_data(file_path="data.csv")

# Resample data
resampled = preprocessor.resample(frequency="D", method="mean")

# Interpolate missing values
interpolated = preprocessor.interpolate_missing(method="linear")

# Remove trend
detrended = preprocessor.remove_trend(method="linear")
```

### Command-Line Usage

Resample time series:

```bash
python src/main.py --input data.csv --resample D --resample-method mean
```

Interpolate missing values:

```bash
python src/main.py --input data.csv --interpolate linear
```

Remove trend:

```bash
python src/main.py --input data.csv --remove-trend linear
```

Complete preprocessing pipeline:

```bash
python src/main.py --input data.csv --resample H --interpolate time --remove-trend polynomial --output processed_data.csv
```

### Complete Example

```python
import pandas as pd
from src.main import TimeSeriesPreprocessor

# Initialize preprocessor
preprocessor = TimeSeriesPreprocessor()

# Load data
preprocessor.load_data(
    file_path="sales_data.csv",
    datetime_column="date"
)

# Get time series information
info = preprocessor.get_time_series_info()
print(f"Date range: {info['start_date']} to {info['end_date']}")
print(f"Frequency: {info['frequency']}")

# Resample to daily frequency
daily_data = preprocessor.resample(
    frequency="D",
    method="mean"
)

# Interpolate missing values
interpolated_data = preprocessor.interpolate_missing(
    method="time",
    columns=["sales", "revenue"]
)

# Remove linear trend
detrended_data = preprocessor.remove_trend(
    method="linear",
    columns=["sales"]
)

# Save preprocessed data
preprocessor.save_preprocessed_data("processed_sales_data.csv")
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import TimeSeriesPreprocessor

df = pd.read_csv("data.csv")
df["date"] = pd.to_datetime(df["date"])

preprocessor = TimeSeriesPreprocessor()
preprocessor.load_data(dataframe=df, datetime_column="date")

resampled = preprocessor.resample(frequency="W")
```

## Project Structure

```
time-series-preprocessing/
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

- `src/main.py`: Core implementation with `TimeSeriesPreprocessor` class
- `config.yaml`: Configuration file for preprocessing parameters
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
- Resampling with different frequencies
- Interpolation with different methods
- Trend removal with different methods
- Time series information retrieval
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Datetime column not detected

**Solution**: Specify datetime column explicitly using `datetime_column` parameter.

**Issue**: Resampling fails

**Solution**: Ensure data has datetime index before resampling.

**Issue**: Interpolation doesn't work

**Solution**: Check that columns are numeric and have missing values.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Data must have datetime index`: Set datetime index before resampling
- `ValueError: Invalid method`: Check method name spelling

## Preprocessing Techniques

### Resampling

- **Purpose**: Change data frequency
- **Methods**: mean, sum, min, max, median, first, last
- **Frequencies**: D (daily), H (hourly), W (weekly), M (monthly)
- **Use for**: Aligning data to regular intervals

### Interpolation

- **Linear**: Simple linear interpolation between points
- **Polynomial**: Polynomial interpolation (order 2)
- **Spline**: Spline interpolation
- **Time**: Time-aware interpolation for datetime index
- **Use for**: Filling missing values in time series

### Trend Removal

- **Linear**: Remove linear trend using least squares
- **Polynomial**: Remove polynomial trend
- **Moving Average**: Remove trend using moving average
- **Use for**: Detrending data for stationary analysis

## Best Practices

1. **Set datetime index**: Ensure data has proper datetime index
2. **Handle missing values**: Interpolate before other operations
3. **Choose appropriate frequency**: Match resampling to analysis needs
4. **Validate preprocessing**: Check data after each preprocessing step
5. **Preserve original**: Keep original data for comparison

## Real-World Applications

- Financial time series preprocessing
- Sensor data preprocessing
- Stock market data preparation
- Weather data preprocessing
- Sales forecasting data preparation
- Time series forecasting pipelines

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
