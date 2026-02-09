# Missing Value Imputation API Documentation

## Classes

### MissingValueImputer

Main class for handling missing value imputation.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize MissingValueImputer with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
imputer = MissingValueImputer()
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
imputer.load_data(file_path="data.csv")
# or
imputer.load_data(dataframe=df)
```

##### `analyze_missing_values() -> Dict[str, any]`

Analyze missing values in the dataset.

**Returns:**
- `Dict[str, any]`: Dictionary with missing value analysis
  - `total_missing`: Total number of missing values
  - `total_cells`: Total number of cells
  - `missing_percentage`: Percentage of missing values
  - `columns_with_missing`: Dictionary of columns with missing values
  - `columns_without_missing`: List of columns without missing values
  - `numeric_columns_with_missing`: List of numeric columns with missing values
  - `categorical_columns_with_missing`: List of categorical columns with missing values

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
analysis = imputer.analyze_missing_values()
print(f"Total missing: {analysis['total_missing']}")
```

##### `impute_mean(columns: Optional[List[str]] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Impute missing values using mean for numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to impute (None for all numeric)
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with imputed values

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
result = imputer.impute_mean()
# or for specific columns
result = imputer.impute_mean(columns=["age", "score"])
```

##### `impute_median(columns: Optional[List[str]] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Impute missing values using median for numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to impute (None for all numeric)
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with imputed values

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
result = imputer.impute_median()
```

##### `impute_mode(columns: Optional[List[str]] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Impute missing values using mode for categorical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to impute (None for all categorical)
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with imputed values

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
result = imputer.impute_mode()
```

##### `impute_all(numeric_strategy: Optional[str] = None, categorical_strategy: Optional[str] = None, inplace: Optional[bool] = None) -> pd.DataFrame`

Impute all missing values using specified strategies.

**Parameters:**
- `numeric_strategy` (Optional[str]): Strategy for numeric columns (mean, median)
- `categorical_strategy` (Optional[str]): Strategy for categorical columns (mode)
- `inplace` (Optional[bool]): Whether to modify in place (default from config)

**Returns:**
- `pd.DataFrame`: DataFrame with all missing values imputed

**Raises:**
- `ValueError`: If no data loaded or invalid strategy

**Example:**
```python
result = imputer.impute_all()
# or with custom strategies
result = imputer.impute_all(
    numeric_strategy="median",
    categorical_strategy="mode"
)
```

##### `get_imputation_summary() -> Dict[str, str]`

Get summary of imputation strategies used.

**Returns:**
- `Dict[str, str]`: Dictionary mapping column names to imputation strategies

**Example:**
```python
summary = imputer.get_imputation_summary()
for col, strategy in summary.items():
    print(f"{col}: {strategy}")
```

##### `save_cleaned_data(output_path: str) -> None`

Save cleaned data to CSV file.

**Parameters:**
- `output_path` (str): Path to output CSV file

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
imputer.save_cleaned_data("cleaned_data.csv")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
imputation:
  default_numeric_strategy: "mean"
  default_categorical_strategy: "mode"
  inplace: false

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `default_numeric_strategy` (str): Default strategy for numerical columns (mean or median)
- `default_categorical_strategy` (str): Default strategy for categorical columns (mode)
- `inplace` (bool): Whether to modify data in place (False creates copy)
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import MissingValueImputer

imputer = MissingValueImputer()
imputer.load_data(file_path="data.csv")

analysis = imputer.analyze_missing_values()
cleaned_data = imputer.impute_all()
```

### Complete Workflow

```python
from src.main import MissingValueImputer

imputer = MissingValueImputer(config_path="config.yaml")
imputer.load_data(file_path="sales_data.csv")

# Analyze
analysis = imputer.analyze_missing_values()
print(f"Missing: {analysis['total_missing']}")

# Impute
cleaned = imputer.impute_all(
    numeric_strategy="median",
    categorical_strategy="mode"
)

# Get summary
summary = imputer.get_imputation_summary()

# Save
imputer.save_cleaned_data("cleaned_sales_data.csv")
```

### Column-Specific Imputation

```python
imputer = MissingValueImputer()
imputer.load_data(file_path="data.csv")

# Impute specific columns
imputer.impute_mean(columns=["age", "score"])
imputer.impute_mode(columns=["category", "type"])
```
