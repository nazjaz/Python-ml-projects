# Feature Selection API Documentation

## Classes

### FeatureSelector

Main class for feature selection operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize FeatureSelector with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
selector = FeatureSelector()
```

##### `load_data(file_path: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]`

Load data from file or use provided DataFrame.

**Parameters:**
- `file_path` (Optional[str]): Path to CSV file (optional)
- `dataframe` (Optional[pd.DataFrame]): Pandas DataFrame (optional)
- `target_column` (Optional[str]): Name of target column (optional)

**Returns:**
- `Tuple[pd.DataFrame, Optional[pd.Series]]`: Tuple of (features DataFrame, target Series)

**Raises:**
- `ValueError`: If neither file_path nor dataframe provided
- `FileNotFoundError`: If file doesn't exist

**Example:**
```python
X, y = selector.load_data(file_path="data.csv", target_column="label")
```

##### `get_numeric_columns() -> List[str]`

Get list of numerical columns in the dataset.

**Returns:**
- `List[str]`: List of numerical column names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
numeric_cols = selector.get_numeric_columns()
```

##### `select_variance_threshold(threshold: Optional[float] = None) -> List[str]`

Select features using variance threshold.

**Parameters:**
- `threshold` (Optional[float]): Variance threshold (default from config)

**Returns:**
- `List[str]`: List of selected feature names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
selected = selector.select_variance_threshold(threshold=0.01)
```

##### `select_correlation(threshold: Optional[float] = None) -> List[str]`

Select features using correlation analysis.

**Parameters:**
- `threshold` (Optional[float]): Correlation threshold (default from config)

**Returns:**
- `List[str]`: List of selected feature names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
selected = selector.select_correlation(threshold=0.9)
```

##### `select_univariate(k: Optional[int] = None, score_func: Optional[str] = None) -> List[str]`

Select features using univariate statistical tests.

**Parameters:**
- `k` (Optional[int]): Number of top features to select (default from config)
- `score_func` (Optional[str]): Score function (f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression)

**Returns:**
- `List[str]`: List of selected feature names

**Raises:**
- `ValueError`: If no data loaded, no target, or invalid score function

**Example:**
```python
selected = selector.select_univariate(k=10, score_func="f_classif")
```

##### `select_all(variance_threshold: Optional[float] = None, correlation_threshold: Optional[float] = None, univariate_k: Optional[int] = None, univariate_score_func: Optional[str] = None) -> List[str]`

Apply all feature selection methods sequentially.

**Parameters:**
- `variance_threshold` (Optional[float]): Variance threshold (default from config)
- `correlation_threshold` (Optional[float]): Correlation threshold (default from config)
- `univariate_k` (Optional[int]): Number of top features (default from config)
- `univariate_score_func` (Optional[str]): Score function (default from config)

**Returns:**
- `List[str]`: List of selected feature names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
selected = selector.select_all()
```

##### `get_selection_summary() -> Dict[str, any]`

Get summary of feature selection results.

**Returns:**
- `Dict[str, any]`: Dictionary with selection summary

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
summary = selector.get_selection_summary()
print(f"Selected: {summary['selected_features']} features")
```

##### `apply_selection(features: Optional[List[str]] = None, inplace: bool = False) -> pd.DataFrame`

Apply feature selection to data.

**Parameters:**
- `features` (Optional[List[str]]): List of features to select (None uses selected_features)
- `inplace` (bool): Whether to modify in place

**Returns:**
- `pd.DataFrame`: DataFrame with selected features

**Raises:**
- `ValueError`: If no data loaded or features invalid

**Example:**
```python
selected_data = selector.apply_selection()
```

##### `save_selected_data(output_path: str) -> None`

Save selected features to CSV file.

**Parameters:**
- `output_path` (str): Path to output CSV file

**Raises:**
- `ValueError`: If no data loaded or no features selected

**Example:**
```python
selector.save_selected_data("selected_features.csv")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
feature_selection:
  variance_threshold: 0.0
  correlation_threshold: 0.95
  univariate_k: 10
  univariate_score_func: "f_classif"

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `variance_threshold` (float): Variance threshold (features below this are removed)
- `correlation_threshold` (float): Correlation threshold (features above this are removed)
- `univariate_k` (int): Number of top features for univariate selection
- `univariate_score_func` (str): Score function for univariate selection
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import FeatureSelector

selector = FeatureSelector()
selector.load_data(file_path="data.csv", target_column="label")

selected = selector.select_variance_threshold()
selected_data = selector.apply_selection()
```

### Complete Workflow

```python
from src.main import FeatureSelector

selector = FeatureSelector(config_path="config.yaml")
selector.load_data(file_path="sales_data.csv", target_column="category")

# Apply all methods
selected = selector.select_all(
    variance_threshold=0.01,
    correlation_threshold=0.9,
    univariate_k=15
)

# Get summary
summary = selector.get_selection_summary()

# Apply and save
selector.save_selected_data("selected_sales_data.csv")
```

### Individual Methods

```python
selector = FeatureSelector()
selector.load_data(file_path="data.csv", target_column="label")

# Variance threshold
variance_features = selector.select_variance_threshold(threshold=0.01)

# Correlation
correlation_features = selector.select_correlation(threshold=0.9)

# Univariate
univariate_features = selector.select_univariate(k=10, score_func="f_classif")
```
