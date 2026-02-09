# Outlier Detection API Documentation

## Classes

### OutlierDetector

Main class for outlier detection and handling operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize OutlierDetector with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
detector = OutlierDetector()
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
detector.load_data(file_path="data.csv")
# or
detector.load_data(dataframe=df)
```

##### `get_numeric_columns() -> List[str]`

Get list of numerical columns in the dataset.

**Returns:**
- `List[str]`: List of numerical column names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
numeric_cols = detector.get_numeric_columns()
```

##### `detect_iqr(columns: Optional[List[str]] = None, multiplier: Optional[float] = None) -> pd.Series`

Detect outliers using IQR (Interquartile Range) method.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to analyze (None for all numeric)
- `multiplier` (Optional[float]): IQR multiplier (default from config)

**Returns:**
- `pd.Series`: Boolean Series indicating outliers (True = outlier)

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
outlier_mask = detector.detect_iqr()
# or with options
outlier_mask = detector.detect_iqr(columns=["age"], multiplier=3.0)
```

##### `detect_zscore(columns: Optional[List[str]] = None, threshold: Optional[float] = None) -> pd.Series`

Detect outliers using Z-score method.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to analyze (None for all numeric)
- `threshold` (Optional[float]): Z-score threshold (default from config)

**Returns:**
- `pd.Series`: Boolean Series indicating outliers (True = outlier)

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
outlier_mask = detector.detect_zscore()
# or with options
outlier_mask = detector.detect_zscore(columns=["age"], threshold=2.0)
```

##### `detect_isolation_forest(columns: Optional[List[str]] = None, contamination: Optional[float] = None, random_state: Optional[int] = None) -> pd.Series`

Detect outliers using Isolation Forest method.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to analyze (None for all numeric)
- `contamination` (Optional[float]): Expected proportion of outliers (default from config)
- `random_state` (Optional[int]): Random seed (default from config)

**Returns:**
- `pd.Series`: Boolean Series indicating outliers (True = outlier)

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
outlier_mask = detector.detect_isolation_forest()
# or with options
outlier_mask = detector.detect_isolation_forest(contamination=0.2)
```

##### `get_outlier_summary(outlier_mask: Optional[pd.Series] = None) -> Dict[str, any]`

Get summary of detected outliers.

**Parameters:**
- `outlier_mask` (Optional[pd.Series]): Boolean Series indicating outliers (None uses all methods)

**Returns:**
- `Dict[str, any]`: Dictionary with outlier summary statistics

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
summary = detector.get_outlier_summary()
print(f"Outliers: {summary['outlier_count']} ({summary['outlier_percentage']:.2f}%)")
```

##### `remove_outliers(outlier_mask: Optional[pd.Series] = None, inplace: bool = False) -> pd.DataFrame`

Remove outliers from dataset.

**Parameters:**
- `outlier_mask` (Optional[pd.Series]): Boolean Series indicating outliers (None uses all methods)
- `inplace` (bool): Whether to modify in place

**Returns:**
- `pd.DataFrame`: DataFrame with outliers removed

**Raises:**
- `ValueError`: If no data loaded or no outliers detected

**Example:**
```python
cleaned_data = detector.remove_outliers()
# or with specific mask
cleaned_data = detector.remove_outliers(outlier_mask)
```

##### `cap_outliers(columns: Optional[List[str]] = None, method: str = "iqr", inplace: bool = False) -> pd.DataFrame`

Cap outliers to boundary values.

**Parameters:**
- `columns` (Optional[List[str]]): List of columns to cap (None for all numeric)
- `method` (str): Detection method to use for boundaries (iqr or zscore)
- `inplace` (bool): Whether to modify in place

**Returns:**
- `pd.DataFrame`: DataFrame with outliers capped

**Raises:**
- `ValueError`: If no data loaded or invalid method

**Example:**
```python
capped_data = detector.cap_outliers(method="iqr")
# or with specific columns
capped_data = detector.cap_outliers(columns=["age"], method="zscore")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

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

- `iqr_multiplier` (float): IQR multiplier (typically 1.5 for normal, 3.0 for extreme)
- `z_score_threshold` (float): Z-score threshold (typically 3.0 for normal, 2.0 for sensitive)
- `isolation_contamination` (float): Expected proportion of outliers (0.0 to 0.5)
- `isolation_random_state` (int): Random seed for Isolation Forest
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import OutlierDetector

detector = OutlierDetector()
detector.load_data(file_path="data.csv")

outlier_mask = detector.detect_iqr()
summary = detector.get_outlier_summary()
```

### Complete Workflow

```python
from src.main import OutlierDetector

detector = OutlierDetector(config_path="config.yaml")
detector.load_data(file_path="sales_data.csv")

# Detect using multiple methods
iqr_mask = detector.detect_iqr()
zscore_mask = detector.detect_zscore()
iso_mask = detector.detect_isolation_forest()

# Get summary
summary = detector.get_outlier_summary()

# Handle outliers
cleaned_data = detector.remove_outliers()
# or
capped_data = detector.cap_outliers(method="iqr")
```

### Column-Specific Detection

```python
detector = OutlierDetector()
detector.load_data(file_path="data.csv")

# Detect outliers in specific columns
outlier_mask = detector.detect_iqr(columns=["age", "score"])
cleaned_data = detector.remove_outliers(outlier_mask)
```
