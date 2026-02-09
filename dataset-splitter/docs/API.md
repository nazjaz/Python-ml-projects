# Dataset Splitter API Documentation

## Classes

### DatasetSplitter

Main class for splitting datasets into train/validation/test sets.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize DatasetSplitter with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
splitter = DatasetSplitter()
```

##### `load_data(file_path: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]`

Load data from file or use provided DataFrame.

**Parameters:**
- `file_path` (Optional[str]): Path to CSV file (optional)
- `dataframe` (Optional[pd.DataFrame]): Pandas DataFrame (optional)
- `target_column` (Optional[str]): Name of target column for stratification (optional)

**Returns:**
- `Tuple[pd.DataFrame, Optional[pd.Series]]`: Tuple of (features DataFrame, target Series)

**Raises:**
- `ValueError`: If neither file_path nor dataframe provided
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If target column not found

**Example:**
```python
X, y = splitter.load_data(file_path="data.csv", target_column="label")
# or
X, y = splitter.load_data(dataframe=df, target_column="label")
```

##### `split(train_ratio: Optional[float] = None, val_ratio: Optional[float] = None, test_ratio: Optional[float] = None, stratify: Optional[bool] = None, random_state: Optional[int] = None, shuffle: Optional[bool] = None) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]`

Split dataset into training, validation, and test sets.

**Parameters:**
- `train_ratio` (Optional[float]): Training set ratio (default from config)
- `val_ratio` (Optional[float]): Validation set ratio (default from config)
- `test_ratio` (Optional[float]): Test set ratio (default from config)
- `stratify` (Optional[bool]): Whether to stratify by target (default from config)
- `random_state` (Optional[int]): Random seed (default from config)
- `shuffle` (Optional[bool]): Whether to shuffle data (default from config)

**Returns:**
- `Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]`: Dictionary with keys 'train', 'val', 'test', each containing tuple of (features DataFrame, target Series)

**Raises:**
- `ValueError`: If no data loaded or ratios invalid

**Example:**
```python
splits = splitter.split()
# or with custom ratios
splits = splitter.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# Access splits
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]
```

##### `get_split_summary(splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]) -> Dict[str, any]`

Get summary of dataset splits.

**Parameters:**
- `splits` (Dict): Dictionary returned from split() method

**Returns:**
- `Dict[str, any]`: Dictionary with split summary statistics

**Example:**
```python
summary = splitter.get_split_summary(splits)
for split_name, info in summary.items():
    print(f"{split_name}: {info['samples']} samples")
```

##### `save_splits(splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]], output_dir: str = "splits") -> None`

Save splits to CSV files.

**Parameters:**
- `splits` (Dict): Dictionary returned from split() method
- `output_dir` (str): Directory to save split files

**Raises:**
- `ValueError`: If splits invalid

**Example:**
```python
splitter.save_splits(splits, output_dir="my_splits")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_state: 42
  shuffle: true
  stratify: true

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `train_ratio` (float): Training set ratio (must sum to 1.0 with val_ratio and test_ratio)
- `val_ratio` (float): Validation set ratio
- `test_ratio` (float): Test set ratio
- `random_state` (int): Random seed for reproducibility
- `shuffle` (bool): Whether to shuffle data before splitting
- `stratify` (bool): Whether to stratify by target column (for classification)
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import DatasetSplitter

splitter = DatasetSplitter()
splitter.load_data(file_path="data.csv", target_column="label")

splits = splitter.split()
X_train, y_train = splits["train"]
X_test, y_test = splits["test"]
```

### Complete Workflow

```python
from src.main import DatasetSplitter

splitter = DatasetSplitter(config_path="config.yaml")
splitter.load_data(file_path="sales_data.csv", target_column="category")

# Split with custom ratios
splits = splitter.split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,
    random_state=42
)

# Get summary
summary = splitter.get_split_summary(splits)

# Save splits
splitter.save_splits(splits, output_dir="splits")
```

### Without Target Column

```python
splitter = DatasetSplitter()
splitter.load_data(file_path="data.csv")

splits = splitter.split(stratify=False)
X_train, _ = splits["train"]
X_test, _ = splits["test"]
```
