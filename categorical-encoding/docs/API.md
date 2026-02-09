# Categorical Encoding API Documentation

## Classes

### CategoricalEncoder

Main class for categorical variable encoding operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize CategoricalEncoder with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
encoder = CategoricalEncoder()
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
encoder.load_data(file_path="data.csv")
# or
encoder.load_data(dataframe=df)
```

##### `get_categorical_columns() -> List[str]`

Get list of categorical columns in the dataset.

**Returns:**
- `List[str]`: List of categorical column names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
categorical_cols = encoder.get_categorical_columns()
```

##### `one_hot_encode(columns: Optional[List[str]] = None, drop_first: Optional[bool] = None, prefix: Optional[str] = None, prefix_sep: Optional[str] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Apply one-hot encoding to categorical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to encode (None for all categorical)
- `drop_first` (Optional[bool]): Whether to drop first category (default from config)
- `prefix` (Optional[str]): Prefix for new column names (default from config)
- `prefix_sep` (Optional[str]): Separator for prefix (default from config)
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with one-hot encoded columns

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
result = encoder.one_hot_encode()
# or with options
result = encoder.one_hot_encode(
    columns=["category"],
    drop_first=True,
    prefix="cat"
)
```

##### `label_encode(columns: Optional[List[str]] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Apply label encoding to categorical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to encode (None for all categorical)
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with label encoded columns

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
result = encoder.label_encode()
# or for specific columns
result = encoder.label_encode(columns=["priority", "status"])
```

##### `inverse_label_encode(encoded_data: Optional[pd.DataFrame] = None, columns: Optional[List[str]] = None) -> pd.DataFrame`

Inverse transform label encoded columns back to original categories.

**Parameters:**
- `encoded_data` (Optional[pd.DataFrame]): Encoded DataFrame (None uses internal data)
- `columns` (Optional[List[str]]): List of columns to inverse transform (None for all encoded)

**Returns:**
- `pd.DataFrame`: DataFrame with original categorical values

**Raises:**
- `ValueError`: If no label encodings or invalid columns

**Example:**
```python
original_data = encoder.inverse_label_encode(encoded_data)
```

##### `compare_encodings(columns: Optional[List[str]] = None) -> Dict[str, any]`

Compare one-hot and label encoding for specified columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of columns to compare (None for all categorical)

**Returns:**
- `Dict[str, any]`: Dictionary with comparison analysis

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
comparison = encoder.compare_encodings()
for col, info in comparison.items():
    if isinstance(info, dict) and "column" in info:
        print(f"{info['column']}: {info['recommendation']}")
```

##### `get_encoding_summary() -> Dict[str, Dict]`

Get summary of encoding operations performed.

**Returns:**
- `Dict[str, Dict]`: Dictionary mapping column names to encoding information

**Example:**
```python
summary = encoder.get_encoding_summary()
for col, info in summary.items():
    print(f"{col}: {info['method']}")
```

##### `save_encoded_data(output_path: str) -> None`

Save encoded data to CSV file.

**Parameters:**
- `output_path` (str): Path to output CSV file

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
encoder.save_encoded_data("encoded_data.csv")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

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

- `drop_first` (bool): Whether to drop first category in one-hot encoding
- `prefix` (str or None): Prefix for one-hot encoded column names
- `prefix_sep` (str): Separator for prefix in one-hot encoding
- `inplace` (bool): Whether to modify data in place (False creates copy)
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import CategoricalEncoder

encoder = CategoricalEncoder()
encoder.load_data(file_path="data.csv")

encoded_data = encoder.one_hot_encode()
# or
encoded_data = encoder.label_encode()
```

### Complete Workflow

```python
from src.main import CategoricalEncoder

encoder = CategoricalEncoder(config_path="config.yaml")
encoder.load_data(file_path="sales_data.csv")

# Compare methods
comparison = encoder.compare_encodings()

# Apply encoding
one_hot_data = encoder.one_hot_encode(columns=["category"])
label_data = encoder.label_encode(columns=["priority"])

# Get summary
summary = encoder.get_encoding_summary()

# Inverse transform
original = encoder.inverse_label_encode(label_data, columns=["priority"])

# Save
encoder.save_encoded_data("encoded_sales_data.csv")
```

### Column-Specific Encoding

```python
encoder = CategoricalEncoder()
encoder.load_data(file_path="data.csv")

# One-hot for nominal data
encoder.one_hot_encode(columns=["color", "type"])

# Label for ordinal data
encoder.label_encode(columns=["priority", "rating"])
```
