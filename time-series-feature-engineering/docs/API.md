# Time Series Feature Engineering API Documentation

## TimeSeriesFeatureEngineering Class

### `TimeSeriesFeatureEngineering(date_column=None, target_column=None, freq=None)`

Time Series Feature Engineering with lag features, rolling stats, and decomposition.

#### Parameters

- `date_column` (str, optional): Name of date/time column.
- `target_column` (str, optional): Name of target column for feature engineering.
- `freq` (str, optional): Frequency of time series (e.g., 'D', 'H', 'M').

#### Attributes

- `feature_names_` (list): List of created feature names.

### Methods

#### `create_lag_features(df, columns=None, lags=[1, 2, 3, 7, 14, 30])`

Create lag features for specified columns.

**Parameters:**
- `df` (DataFrame): Input dataframe.
- `columns` (list, optional): Columns to create lags for (default: all numeric or target).
- `lags` (int or list): Lag periods (default: [1, 2, 3, 7, 14, 30]).

**Returns:**
- `DataFrame`: Dataframe with lag features added.

**Example:**
```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_lag_features(df, lags=[1, 7, 14])
```

#### `create_rolling_statistics(df, columns=None, windows=[3, 7, 14, 30], statistics=None)`

Create rolling statistics features.

**Parameters:**
- `df` (DataFrame): Input dataframe.
- `columns` (list, optional): Columns to create rolling stats for.
- `windows` (int or list): Rolling window sizes (default: [3, 7, 14, 30]).
- `statistics` (list, optional): Statistics to compute (default: ['mean', 'std', 'min', 'max']).

**Returns:**
- `DataFrame`: Dataframe with rolling statistics features added.

**Example:**
```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_rolling_statistics(df, windows=[7, 30], statistics=["mean", "std"])
```

#### `create_expanding_statistics(df, columns=None, statistics=None)`

Create expanding window statistics features.

**Parameters:**
- `df` (DataFrame): Input dataframe.
- `columns` (list, optional): Columns to create expanding stats for.
- `statistics` (list, optional): Statistics to compute (default: ['mean', 'std', 'min', 'max']).

**Returns:**
- `DataFrame`: Dataframe with expanding statistics features added.

**Example:**
```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_expanding_statistics(df, statistics=["mean", "std"])
```

#### `seasonal_decomposition(df, column, model='additive', period=None)`

Perform seasonal decomposition.

**Parameters:**
- `df` (DataFrame): Input dataframe.
- `column` (str): Column to decompose.
- `model` (str): Decomposition model ('additive' or 'multiplicative') (default: 'additive').
- `period` (int, optional): Seasonal period (default: auto-detect).

**Returns:**
- `tuple`: (Dataframe with decomposition features, decomposition components dict).

**Example:**
```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")
df, components = ts_fe.seasonal_decomposition(df, "value", period=7)
```

#### `create_time_features(df, features=None)`

Create time-based features from datetime index.

**Parameters:**
- `df` (DataFrame): Input dataframe.
- `features` (list, optional): Time features to create (default: all).

**Returns:**
- `DataFrame`: Dataframe with time features added.

**Example:**
```python
ts_fe = TimeSeriesFeatureEngineering()
df = ts_fe.create_time_features(df, features=["year", "month", "dayofweek"])
```

#### `create_difference_features(df, columns=None, periods=[1, 7, 30])`

Create difference features.

**Parameters:**
- `df` (DataFrame): Input dataframe.
- `columns` (list, optional): Columns to create differences for.
- `periods` (int or list): Difference periods (default: [1, 7, 30]).

**Returns:**
- `DataFrame`: Dataframe with difference features added.

**Example:**
```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_difference_features(df, periods=[1, 7])
```

#### `create_ratio_features(df, columns=None, windows=[7, 30])`

Create ratio features (current value / rolling mean).

**Parameters:**
- `df` (DataFrame): Input dataframe.
- `columns` (list, optional): Columns to create ratios for.
- `windows` (int or list): Rolling window sizes (default: [7, 30]).

**Returns:**
- `DataFrame`: Dataframe with ratio features added.

**Example:**
```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_ratio_features(df, windows=[7, 30])
```

#### `plot_decomposition(components, column, save_path=None, show=True)`

Plot seasonal decomposition components.

**Parameters:**
- `components` (dict): Decomposition components dictionary.
- `column` (str): Column name for title.
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).

**Example:**
```python
ts_fe.plot_decomposition(components, "value", save_path="decomposition.png")
```

## Usage Examples

### Complete Feature Engineering Pipeline

```python
from src.main import TimeSeriesFeatureEngineering
import pandas as pd
import numpy as np

dates = pd.date_range("2023-01-01", periods=365, freq="D")
df = pd.DataFrame({"value": np.random.randn(365)}, index=dates)

ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")

# Create all feature types
df = ts_fe.create_lag_features(df, lags=[1, 7, 14, 30])
df = ts_fe.create_rolling_statistics(df, windows=[7, 30])
df = ts_fe.create_time_features(df)
df = ts_fe.create_difference_features(df, periods=[1, 7])
df = ts_fe.create_ratio_features(df, windows=[7, 30])
df, components = ts_fe.seasonal_decomposition(df, "value")

print(f"Total features: {len(ts_fe.feature_names_)}")
```

### Lag Features Only

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_lag_features(df, lags=[1, 2, 3, 7, 14, 30])
```

### Rolling Statistics Only

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value")
df = ts_fe.create_rolling_statistics(
    df,
    windows=[3, 7, 14, 30],
    statistics=["mean", "std", "min", "max"]
)
```

### Seasonal Decomposition with Plot

```python
ts_fe = TimeSeriesFeatureEngineering(target_column="value", freq="D")
df, components = ts_fe.seasonal_decomposition(df, "value", period=7)
ts_fe.plot_decomposition(components, "value")
```
