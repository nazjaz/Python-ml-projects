# Data Augmentation API Documentation

## Classes

### DataAugmenter

Main class for data augmentation operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize DataAugmenter with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
augmenter = DataAugmenter()
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
augmenter.load_data(file_path="data.csv")
# or
augmenter.load_data(dataframe=df)
```

##### `get_numeric_columns() -> List[str]`

Get list of numerical columns in the dataset.

**Returns:**
- `List[str]`: List of numerical column names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
numeric_cols = augmenter.get_numeric_columns()
```

##### `inject_noise(columns: Optional[List[str]] = None, noise_type: Optional[str] = None, noise_std: Optional[float] = None, noise_mean: Optional[float] = None, random_state: Optional[int] = None, inplace: bool = False) -> pd.DataFrame`

Inject noise into numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to augment (None for all numeric)
- `noise_type` (Optional[str]): Type of noise (gaussian, uniform, laplace)
- `noise_std` (Optional[float]): Standard deviation for noise (default from config)
- `noise_mean` (Optional[float]): Mean for noise (default from config)
- `random_state` (Optional[int]): Random seed (default from config)
- `inplace` (bool): Whether to modify in place

**Returns:**
- `pd.DataFrame`: DataFrame with noise injected

**Raises:**
- `ValueError`: If no data loaded, columns invalid, or noise type invalid

**Example:**
```python
augmented = augmenter.inject_noise(noise_type="gaussian", noise_std=0.1)
# or for specific columns
augmented = augmenter.inject_noise(columns=["age"], noise_type="uniform")
```

##### `apply_scaling_variations(columns: Optional[List[str]] = None, scaling_type: Optional[str] = None, scaling_factor_min: Optional[float] = None, scaling_factor_max: Optional[float] = None, random_state: Optional[int] = None, inplace: bool = False) -> pd.DataFrame`

Apply scaling variations to numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to augment (None for all numeric)
- `scaling_type` (Optional[str]): Type of scaling (multiplicative, additive, percentage)
- `scaling_factor_min` (Optional[float]): Minimum scaling factor (default from config)
- `scaling_factor_max` (Optional[float]): Maximum scaling factor (default from config)
- `random_state` (Optional[int]): Random seed (default from config)
- `inplace` (bool): Whether to modify in place

**Returns:**
- `pd.DataFrame`: DataFrame with scaling variations applied

**Raises:**
- `ValueError`: If no data loaded, columns invalid, or scaling type invalid

**Example:**
```python
augmented = augmenter.apply_scaling_variations(
    scaling_type="multiplicative",
    scaling_factor_min=0.9,
    scaling_factor_max=1.1
)
```

##### `augment_all(noise_type: Optional[str] = None, noise_std: Optional[float] = None, scaling_type: Optional[str] = None, scaling_factor_min: Optional[float] = None, scaling_factor_max: Optional[float] = None, columns: Optional[List[str]] = None, inplace: bool = False) -> pd.DataFrame`

Apply all augmentation techniques.

**Parameters:**
- `noise_type` (Optional[str]): Type of noise (default from config)
- `noise_std` (Optional[float]): Standard deviation for noise (default from config)
- `scaling_type` (Optional[str]): Type of scaling (default from config)
- `scaling_factor_min` (Optional[float]): Minimum scaling factor (default from config)
- `scaling_factor_max` (Optional[float]): Maximum scaling factor (default from config)
- `columns` (Optional[List[str]]): List of columns to augment (None for all numeric)
- `inplace` (bool): Whether to modify in place

**Returns:**
- `pd.DataFrame`: DataFrame with all augmentations applied

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
augmented = augmenter.augment_all()
```

##### `get_augmentation_summary() -> Dict[str, any]`

Get summary of augmentation operations performed.

**Returns:**
- `Dict[str, any]`: Dictionary with augmentation summary

**Example:**
```python
summary = augmenter.get_augmentation_summary()
print(f"Total operations: {summary['total_operations']}")
```

##### `save_augmented_data(output_path: str) -> None`

Save augmented data to CSV file.

**Parameters:**
- `output_path` (str): Path to output CSV file

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
augmenter.save_augmented_data("augmented_data.csv")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
augmentation:
  noise_type: "gaussian"
  noise_std: 0.1
  noise_mean: 0.0
  scaling_type: "multiplicative"
  scaling_factor_min: 0.9
  scaling_factor_max: 1.1
  random_state: 42

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `noise_type` (str): Type of noise (gaussian, uniform, laplace)
- `noise_std` (float): Standard deviation for noise
- `noise_mean` (float): Mean for noise
- `scaling_type` (str): Type of scaling (multiplicative, additive, percentage)
- `scaling_factor_min` (float): Minimum scaling factor
- `scaling_factor_max` (float): Maximum scaling factor
- `random_state` (int): Random seed for reproducibility
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import DataAugmenter

augmenter = DataAugmenter()
augmenter.load_data(file_path="data.csv")

augmented = augmenter.inject_noise()
augmented = augmenter.apply_scaling_variations()
```

### Complete Workflow

```python
from src.main import DataAugmenter

augmenter = DataAugmenter(config_path="config.yaml")
augmenter.load_data(file_path="sales_data.csv")

# Apply all augmentations
augmented = augmenter.augment_all(
    noise_type="gaussian",
    noise_std=0.05,
    scaling_type="multiplicative",
    scaling_factor_min=0.9,
    scaling_factor_max=1.1
)

# Get summary
summary = augmenter.get_augmentation_summary()

# Save
augmenter.save_augmented_data("augmented_sales_data.csv")
```

### Column-Specific Augmentation

```python
augmenter = DataAugmenter()
augmenter.load_data(file_path="data.csv")

# Augment specific columns
augmented = augmenter.inject_noise(columns=["age", "score"])
augmented = augmenter.apply_scaling_variations(columns=["height", "weight"])
```
