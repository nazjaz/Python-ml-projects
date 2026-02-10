# Time Series Feature Engineering

A Python implementation of comprehensive time series feature engineering including lag features, rolling statistics, seasonal decomposition, time-based features, difference features, and ratio features. This is the forty-first project in the ML learning series, focusing on understanding time series feature engineering techniques for machine learning.

## Project Title and Description

The Time Series Feature Engineering tool provides a complete set of feature engineering capabilities for time series data. It includes lag features (past values), rolling statistics (moving averages, standard deviations), seasonal decomposition (trend, seasonal, residual components), time-based features (hour, day, month, etc.), difference features, and ratio features. It helps users understand how to engineer features from time series data for machine learning models.

This tool solves the problem of preparing time series data for machine learning by providing comprehensive feature engineering capabilities without relying on external ML libraries. It demonstrates lag features, rolling statistics, seasonal decomposition, and other time series feature engineering techniques from scratch.

**Target Audience**: Beginners learning machine learning, students studying time series analysis, and anyone who needs to engineer features from time series data for ML models.

## Features

- Lag features (past values at different time steps)
- Rolling statistics (mean, std, min, max, median, sum)
- Expanding statistics (cumulative mean, std, min, max)
- Seasonal decomposition (trend, seasonal, residual components)
- Time-based features (year, month, day, dayofweek, hour, etc.)
- Difference features (first difference, seasonal difference)
- Ratio features (current value / rolling mean)
- Visualization of seasonal decomposition
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (CSV files, pandas DataFrames)

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/time-series-feature-engineering
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
python src/main.py --input sample.csv --date-column date --target-column value --all-features --output features.csv
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

features:
  date_column: null
  target_column: null
  freq: null
  lags: [1, 2, 3, 7, 14, 30]
  rolling_windows: [3, 7, 14, 30]
  rolling_statistics: ["mean", "std", "min", "max"]
  expanding_statistics: ["mean", "std", "min", "max"]
  difference_periods: [1, 7, 30]
  ratio_windows: [7, 30]
  time_features: ["year", "month", "day", "dayofweek", "dayofyear", "week", "quarter", "hour", "is_weekend"]
  decomposition_model: "additive"
  decomposition_period: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `features.date_column`: Name of date/time column
- `features.target_column`: Name of target column for feature engineering
- `features.freq`: Frequency of time series (D=daily, H=hourly, M=monthly, etc.)
- `features.lags`: Lag periods for lag features
- `features.rolling_windows`: Rolling window sizes
- `features.rolling_statistics`: Statistics to compute (mean, std, min, max, median, sum)
- `features.expanding_statistics`: Expanding statistics to compute
- `features.difference_periods`: Difference periods
- `features.ratio_windows`: Ratio feature windows
- `features.time_features`: Time features to create
- `features.decomposition_model`: "additive" or "multiplicative"
- `features.decomposition_period`: Seasonal period (null: auto-detect)

## Usage

### Basic Usage

```python
from src.main import TimeSeriesFeatureEngineering
import pandas as pd

# Create sample time series
dates = pd.date_range("2023-01-01", periods=100, freq="D")
df = pd.DataFrame({
    "value": np.random.randn(100),
}, index=dates)

# Initialize feature engineering
ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")

# Create lag features
df = ts_fe.create_lag_features(df, lags=[1, 2, 7, 14])

# Create rolling statistics
df = ts_fe.create_rolling_statistics(df, windows=[7, 30], statistics=["mean", "std"])

# Create time features
df = ts_fe.create_time_features(df)
```

### Lag Features

```python
from src.main import TimeSeriesFeatureEngineering
import pandas as pd
import numpy as np

dates = pd.date_range("2023-01-01", periods=100, freq="D")
df = pd.DataFrame({"value": np.random.randn(100)}, index=dates)

ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_lag_features(df, lags=[1, 2, 3, 7, 14, 30])

# Features created: value_lag_1, value_lag_2, value_lag_3, value_lag_7, value_lag_14, value_lag_30
```

### Rolling Statistics

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_rolling_statistics(
    df,
    windows=[3, 7, 14, 30],
    statistics=["mean", "std", "min", "max"]
)

# Features created: value_rolling_mean_3, value_rolling_std_3, value_rolling_min_3, etc.
```

### Expanding Statistics

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_expanding_statistics(
    df,
    statistics=["mean", "std", "min", "max"]
)

# Features created: value_expanding_mean, value_expanding_std, etc.
```

### Seasonal Decomposition

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")
df, components = ts_fe.seasonal_decomposition(
    df,
    column="value",
    model="additive",
    period=7
)

# Features created: value_trend, value_seasonal, value_residual

# Plot decomposition
ts_fe.plot_decomposition(components, "value")
```

### Time Features

```python
ts_fe = TimeSeriesFeatureEngineering()
df = ts_fe.create_time_features(
    df,
    features=["year", "month", "day", "dayofweek", "hour", "is_weekend"]
)

# Features created: year, month, day, dayofweek, hour, is_weekend
```

### Difference Features

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_difference_features(df, periods=[1, 7, 30])

# Features created: value_diff_1, value_diff_7, value_diff_30
```

### Ratio Features

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_ratio_features(df, windows=[7, 30])

# Features created: value_ratio_7, value_ratio_30
```

### Complete Example

```python
from src.main import TimeSeriesFeatureEngineering
import pandas as pd
import numpy as np

# Generate sample time series with trend and seasonality
dates = pd.date_range("2023-01-01", periods=365, freq="D")
trend = np.linspace(0, 10, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365) / 7)
noise = np.random.randn(365) * 0.5
values = trend + seasonal + noise

df = pd.DataFrame({"value": values}, index=dates)

# Initialize feature engineering
ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")

# Create all feature types
df = ts_fe.create_lag_features(df, lags=[1, 7, 14, 30])
df = ts_fe.create_rolling_statistics(df, windows=[7, 30], statistics=["mean", "std"])
df = ts_fe.create_expanding_statistics(df, statistics=["mean", "std"])
df = ts_fe.create_time_features(df, features=["month", "dayofweek", "is_weekend"])
df = ts_fe.create_difference_features(df, periods=[1, 7])
df = ts_fe.create_ratio_features(df, windows=[7, 30])

# Seasonal decomposition
df, components = ts_fe.seasonal_decomposition(df, "value", period=7)

# Plot decomposition
ts_fe.plot_decomposition(components, "value")

print(f"Total features created: {len(ts_fe.feature_names_)}")
print(f"Final shape: {df.shape}")
```

### Command-Line Usage

Create all features:

```bash
python src/main.py --input data.csv --date-column date --target-column value --all-features --output features.csv
```

Create specific feature types:

```bash
python src/main.py --input data.csv --date-column date --target-column value --lag-features --rolling-stats --output features.csv
```

With seasonal decomposition and plot:

```bash
python src/main.py --input data.csv --date-column date --target-column value --seasonal-decomposition --plot-decomposition --output features.csv
```

Save decomposition plot:

```bash
python src/main.py --input data.csv --date-column date --target-column value --seasonal-decomposition --save-decomposition-plot decomposition.png --output features.csv
```

## Project Structure

```
time-series-feature-engineering/
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

- `src/main.py`: Core implementation with `TimeSeriesFeatureEngineering` class
- `config.yaml`: Configuration file for feature engineering settings
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
- Lag feature creation
- Rolling statistics creation
- Expanding statistics creation
- Time feature creation
- Difference feature creation
- Ratio feature creation
- Seasonal decomposition
- Dataframe validation
- Error handling

## Understanding Time Series Feature Engineering

### Lag Features

**Purpose:**
- Capture temporal dependencies
- Include past values as features
- Common lags: 1, 7, 14, 30 (for daily data)

**Use Cases:**
- Forecasting models
- Anomaly detection
- Pattern recognition

### Rolling Statistics

**Purpose:**
- Capture local patterns
- Smooth out noise
- Capture trends over windows

**Common Statistics:**
- Mean: Average over window
- Std: Variability over window
- Min/Max: Extremes over window
- Median: Robust average

**Window Sizes:**
- Short (3-7): Recent patterns
- Medium (14-30): Medium-term trends
- Long (60-90): Long-term trends

### Seasonal Decomposition

**Components:**
- **Trend**: Long-term direction
- **Seasonal**: Repeating patterns
- **Residual**: Random noise

**Models:**
- **Additive**: `value = trend + seasonal + residual`
- **Multiplicative**: `value = trend * seasonal * residual`

**Use Cases:**
- Understanding time series structure
- Removing seasonality
- Trend analysis

### Time Features

**Purpose:**
- Capture calendar effects
- Day of week patterns
- Month/season patterns
- Holiday effects

**Common Features:**
- Year, month, day
- Day of week, day of year
- Hour, minute (for hourly data)
- Is weekend, is holiday

### Difference Features

**Purpose:**
- Remove trend
- Stationarize data
- Capture changes

**Types:**
- First difference: `diff(t) = value(t) - value(t-1)`
- Seasonal difference: `diff(t) = value(t) - value(t-period)`

### Ratio Features

**Purpose:**
- Normalize by rolling average
- Capture relative changes
- Remove scale effects

**Formula:**
- `ratio = current_value / rolling_mean`

## Troubleshooting

### Common Issues

**Issue**: DatetimeIndex error

**Solution**: 
- Specify date_column in config or command line
- Ensure date column is parseable
- Check date format

**Issue**: Missing values in lag features

**Solution**: 
- Expected for first N rows (where N is lag)
- Fill with forward fill or backward fill if needed
- Consider using min_periods in rolling stats

**Issue**: Seasonal decomposition fails

**Solution**: 
- Ensure enough data (at least 2 * period)
- Specify period manually
- Check for sufficient seasonality

### Error Messages

- `ValueError: Dataframe must have a DatetimeIndex`: Specify date_column or use DatetimeIndex
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: Series too short for decomposition`: Use more data or smaller period

## Best Practices

1. **Start with lag features**: Most important for time series
2. **Use rolling statistics**: Capture local patterns
3. **Add time features**: Capture calendar effects
4. **Decompose seasonality**: Understand structure
5. **Use difference features**: For non-stationary data
6. **Feature selection**: Remove highly correlated features
7. **Handle missing values**: Fill appropriately
8. **Scale features**: Normalize for ML models

## Real-World Applications

- Sales forecasting
- Stock price prediction
- Energy demand forecasting
- Weather prediction
- Anomaly detection
- Educational purposes for learning time series feature engineering

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
