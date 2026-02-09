# Feature Scaling API Documentation

## Classes

### FeatureScaler

Main class for feature scaling operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize FeatureScaler with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
scaler = FeatureScaler()
```

##### `load_data(file_path: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame`

Load data from file or use provided DataFrame.

**Parameters:**
- `file_path` (Optional[str]): Path to CSV file (optional)
- `dataframe` (Optional[pd.DataFrame]): Pandas DataFrame (optional)

**Returns:**
- `pd.DataFrame`: Loaded DataFrame

**Raises:**
- `ValueError`: If neither file_path nor dataframe provided
- `FileNotFoundError`: If file doesn't exist

**Example:**
```python
scaler.load_data(file_path="data.csv")
# or
scaler.load_data(dataframe=df)
```

##### `get_numeric_columns() -> List[str]`

Get list of numerical columns in the dataset.

**Returns:**
- `List[str]`: List of numerical column names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
numeric_cols = scaler.get_numeric_columns()
```

##### `min_max_scale(columns: Optional[List[str]] = None, feature_range: Optional[Tuple[float, float]] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Apply min-max scaling to numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to scale (None for all numeric)
- `feature_range` (Optional[Tuple[float, float]]): Tuple of (min, max) for output range
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with scaled values

**Raises:**
- `ValueError`: If no data loaded, columns invalid, or range invalid

**Example:**
```python
result = scaler.min_max_scale()
# or with custom range
result = scaler.min_max_scale(feature_range=(-1, 1))
# or for specific columns
result = scaler.min_max_scale(columns=["age", "score"])
```

##### `z_score_normalize(columns: Optional[List[str]] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Apply z-score normalization (standardization) to numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to normalize (None for all numeric)
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with normalized values

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
result = scaler.z_score_normalize()
# or for specific columns
result = scaler.z_score_normalize(columns=["age", "score"])
```

##### `inverse_transform(scaled_data: Optional[pd.DataFrame] = None, columns: Optional[List[str]] = None) -> pd.DataFrame`

Inverse transform scaled data back to original scale.

**Parameters:**
- `scaled_data` (Optional[pd.DataFrame]): Scaled DataFrame (None uses internal data)
- `columns` (Optional[List[str]]): List of columns to inverse transform (None for all scaled)

**Returns:**
- `pd.DataFrame`: DataFrame with original scale values

**Raises:**
- `ValueError`: If no scaling parameters or invalid columns

**Example:**
```python
original_data = scaler.inverse_transform(scaled_data)
```

##### `get_scaling_summary() -> Dict[str, Dict[str, float]]`

Get summary of scaling parameters used.

**Returns:**
- `Dict[str, Dict[str, float]]`: Dictionary mapping column names to scaling parameters

**Example:**
```python
summary = scaler.get_scaling_summary()
for col, params in summary.items():
    print(f"{col}: {params}")
```

##### `save_scaled_data(output_path: str) -> None`

Save scaled data to CSV file.

**Parameters:**
- `output_path` (str): Path to output CSV file

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
scaler.save_scaled_data("scaled_data.csv")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
scaling:
  min_max_range: [0, 1]
  inplace: false

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `min_max_range` (List[float]): Feature range for min-max scaling [min, max] (default: [0, 1])
- `inplace` (bool): Whether to modify data in place (False creates copy)
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import FeatureScaler

scaler = FeatureScaler()
scaler.load_data(file_path="data.csv")

scaled_data = scaler.min_max_scale()
normalized_data = scaler.z_score_normalize()
```

### Complete Workflow

```python
from src.main import FeatureScaler

scaler = FeatureScaler(config_path="config.yaml")
scaler.load_data(file_path="sales_data.csv")

# Get numerical columns
numeric_cols = scaler.get_numeric_columns()

# Apply min-max scaling
scaled = scaler.min_max_scale(feature_range=(0, 1))

# Apply z-score normalization
normalized = scaler.z_score_normalize()

# Get summary
summary = scaler.get_scaling_summary()

# Inverse transform
original = scaler.inverse_transform(scaled)

# Save
scaler.save_scaled_data("scaled_sales_data.csv")
```

### Column-Specific Scaling

```python
scaler = FeatureScaler()
scaler.load_data(file_path="data.csv")

# Scale specific columns
scaler.min_max_scale(columns=["age", "score"])
scaler.z_score_normalize(columns=["height", "weight"])
```
