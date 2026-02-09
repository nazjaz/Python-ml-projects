# CSV Data Explorer API Documentation

## Classes

### CSVDataExplorer

Main class for exploring CSV datasets.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize CSVDataExplorer with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
explorer = CSVDataExplorer()
```

##### `load_csv(file_path: str, separator: Optional[str] = None, encoding: Optional[str] = None) -> pd.DataFrame`

Load CSV file into pandas DataFrame.

**Parameters:**
- `file_path` (str): Path to CSV file
- `separator` (Optional[str]): Column separator (default from config)
- `encoding` (Optional[str]): File encoding (default from config)

**Returns:**
- `pd.DataFrame`: Loaded DataFrame

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `pd.errors.EmptyDataError`: If file is empty
- `pd.errors.ParserError`: If file cannot be parsed

**Example:**
```python
df = explorer.load_csv("data.csv")
```

##### `get_basic_info() -> Dict[str, any]`

Get basic information about the dataset.

**Returns:**
- `Dict[str, any]`: Dictionary with basic dataset information
  - `shape`: Tuple of (rows, columns)
  - `rows`: Number of rows
  - `columns`: Number of columns
  - `column_names`: List of column names
  - `memory_usage_bytes`: Memory usage in bytes

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
info = explorer.get_basic_info()
print(f"Shape: {info['shape']}")
```

##### `get_data_types() -> Dict[str, str]`

Get data types for each column.

**Returns:**
- `Dict[str, str]`: Dictionary mapping column names to data types

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
dtypes = explorer.get_data_types()
for col, dtype in dtypes.items():
    print(f"{col}: {dtype}")
```

##### `get_basic_statistics() -> pd.DataFrame`

Get basic statistical summary for numerical columns.

**Returns:**
- `pd.DataFrame`: DataFrame with statistical summary (count, mean, std, min, max, etc.)

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
stats = explorer.get_basic_statistics()
print(stats)
```

##### `get_missing_value_analysis() -> Dict[str, any]`

Analyze missing values in the dataset.

**Returns:**
- `Dict[str, any]`: Dictionary with missing value analysis
  - `total_missing`: Total number of missing values
  - `total_rows`: Number of rows
  - `missing_percentage`: Percentage of missing values
  - `columns_with_missing`: Dictionary of columns with missing values
  - `columns_without_missing`: List of columns without missing values

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
missing = explorer.get_missing_value_analysis()
print(f"Total missing: {missing['total_missing']}")
```

##### `get_categorical_summary() -> Dict[str, any]`

Get summary for categorical columns.

**Returns:**
- `Dict[str, any]`: Dictionary with categorical column summaries
  - For each column: `unique_count`, `most_frequent`, `most_frequent_count`

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
categorical = explorer.get_categorical_summary()
for col, info in categorical.items():
    print(f"{col}: {info['unique_count']} unique values")
```

##### `generate_report() -> str`

Generate comprehensive exploration report.

**Returns:**
- `str`: Formatted report string

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
report = explorer.generate_report()
print(report)
```

##### `preview_data(n_rows: Optional[int] = None) -> pd.DataFrame`

Preview first n rows of the dataset.

**Parameters:**
- `n_rows` (Optional[int]): Number of rows to preview (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with preview rows

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
preview = explorer.preview_data(n_rows=10)
print(preview)
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
csv_explorer:
  separator: ","
  encoding: "utf-8"
  max_rows_preview: 10

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `separator` (str): Column separator (comma, semicolon, tab, etc.)
- `encoding` (str): File encoding (utf-8, latin-1, etc.)
- `max_rows_preview` (int): Default number of rows to preview
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import CSVDataExplorer

explorer = CSVDataExplorer()
explorer.load_csv("data.csv")

info = explorer.get_basic_info()
dtypes = explorer.get_data_types()
missing = explorer.get_missing_value_analysis()
```

### Complete Workflow

```python
from src.main import CSVDataExplorer

explorer = CSVDataExplorer(config_path="config.yaml")
explorer.load_csv("sales_data.csv")

report = explorer.generate_report()
print(report)

stats = explorer.get_basic_statistics()
categorical = explorer.get_categorical_summary()
preview = explorer.preview_data(n_rows=5)
```

### Custom Configuration

```python
explorer = CSVDataExplorer(config_path="custom_config.yaml")
df = explorer.load_csv("data.csv", separator=";", encoding="latin-1")
```
