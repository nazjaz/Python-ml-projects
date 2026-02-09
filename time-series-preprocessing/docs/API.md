# Time Series Preprocessing API Documentation

## Classes

### TimeSeriesPreprocessor

Main class for time series preprocessing operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize TimeSeriesPreprocessor with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
preprocessor = TimeSeriesPreprocessor()
```

##### `load_data(file_path: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None, datetime_column: Optional[str] = None) -> pd.DataFrame`

Load time series data from file or use provided DataFrame.

**Parameters:**
- `file_path` (Optional[str]): Path to CSV file (optional)
- `dataframe` (Optional[pd.DataFrame]): Pandas DataFrame (optional)
- `datetime_column` (Optional[str]): Name of datetime column (optional, auto-detect if None)

**Returns:**
- `pd.DataFrame`: Loaded DataFrame with datetime index

**Raises:**
- `ValueError`: If neither file_path nor dataframe provided
- `FileNotFoundError`: If file doesn't exist

**Example:**
```python
preprocessor.load_data(file_path="data.csv")
# or
preprocessor.load_data(dataframe=df, datetime_column="date")
```

##### `resample(frequency: str, method: str = "mean", columns: Optional[List[str]] = None) -> pd.DataFrame`

Resample time series data to specified frequency.

**Parameters:**
- `frequency` (str): Resampling frequency (e.g., 'D', 'H', 'W', 'M')
- `method` (str): Aggregation method (mean, sum, min, max, median)
- `columns` (Optional[List[str]]): List of columns to resample (None for all)

**Returns:**
- `pd.DataFrame`: Resampled DataFrame

**Raises:**
- `ValueError`: If no data loaded or invalid method

**Example:**
```python
resampled = preprocessor.resample(frequency="D", method="mean")
```

##### `interpolate_missing(method: Optional[str] = None, columns: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame`

Interpolate missing values in time series data.

**Parameters:**
- `method` (Optional[str]): Interpolation method (linear, polynomial, spline, time)
- `columns` (Optional[List[str]]): List of columns to interpolate (None for all numeric)
- `limit` (Optional[int]): Maximum number of consecutive NaNs to fill

**Returns:**
- `pd.DataFrame`: DataFrame with interpolated values

**Raises:**
- `ValueError`: If no data loaded or invalid method

**Example:**
```python
interpolated = preprocessor.interpolate_missing(method="linear")
```

##### `remove_trend(method: Optional[str] = None, columns: Optional[List[str]] = None, order: int = 1) -> pd.DataFrame`

Remove trend from time series data.

**Parameters:**
- `method` (Optional[str]): Trend removal method (linear, polynomial, moving_average)
- `columns` (Optional[List[str]]): List of columns to detrend (None for all numeric)
- `order` (int): Polynomial order (for polynomial method)

**Returns:**
- `pd.DataFrame`: DataFrame with trend removed

**Raises:**
- `ValueError`: If no data loaded or invalid method

**Example:**
```python
detrended = preprocessor.remove_trend(method="linear")
```

##### `get_time_series_info() -> Dict[str, any]`

Get information about the time series data.

**Returns:**
- `Dict[str, any]`: Dictionary with time series information

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
info = preprocessor.get_time_series_info()
print(f"Date range: {info['start_date']} to {info['end_date']}")
```

##### `save_preprocessed_data(output_path: str) -> None`

Save preprocessed data to CSV file.

**Parameters:**
- `output_path` (str): Path to output CSV file

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
preprocessor.save_preprocessed_data("processed_data.csv")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

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

- `default_resample_freq` (str): Default resampling frequency
- `default_interpolation_method` (str): Default interpolation method
- `default_trend_removal_method` (str): Default trend removal method
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()
preprocessor.load_data(file_path="data.csv")

resampled = preprocessor.resample(frequency="D")
interpolated = preprocessor.interpolate_missing()
detrended = preprocessor.remove_trend()
```

### Complete Workflow

```python
import pandas as pd
from src.main import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()

# Load data
preprocessor.load_data(
    file_path="sales_data.csv",
    datetime_column="date"
)

# Get information
info = preprocessor.get_time_series_info()

# Resample
daily_data = preprocessor.resample(frequency="D", method="mean")

# Interpolate
interpolated = preprocessor.interpolate_missing(method="time")

# Remove trend
detrended = preprocessor.remove_trend(method="linear")

# Save
preprocessor.save_preprocessed_data("processed_data.csv")
```

### Resampling Frequencies

Common resampling frequencies:
- `'D'`: Daily
- `'H'`: Hourly
- `'W'`: Weekly
- `'M'`: Monthly
- `'Q'`: Quarterly
- `'Y'`: Yearly

### Interpolation Methods

- `'linear'`: Linear interpolation
- `'polynomial'`: Polynomial interpolation (order 2)
- `'spline'`: Spline interpolation
- `'time'`: Time-aware interpolation

### Trend Removal Methods

- `'linear'`: Remove linear trend
- `'polynomial'`: Remove polynomial trend
- `'moving_average'`: Remove trend using moving average
