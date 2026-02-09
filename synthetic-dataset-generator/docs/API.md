# Synthetic Dataset Generator API Documentation

## Classes

### SyntheticDatasetGenerator

Main class for generating synthetic datasets.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize SyntheticDatasetGenerator with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
generator = SyntheticDatasetGenerator()
```

##### `generate_classification(n_samples: Optional[int] = None, n_features: Optional[int] = None, n_informative: Optional[int] = None, n_redundant: Optional[int] = None, n_repeated: Optional[int] = None, n_classes: Optional[int] = None, n_clusters_per_class: Optional[int] = None, weights: Optional[List[float]] = None, flip_y: Optional[float] = None, class_sep: Optional[float] = None, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]`

Generate synthetic classification dataset.

**Parameters:**
- `n_samples` (Optional[int]): Number of samples (default from config)
- `n_features` (Optional[int]): Number of features (default from config)
- `n_informative` (Optional[int]): Number of informative features
- `n_redundant` (Optional[int]): Number of redundant features
- `n_repeated` (Optional[int]): Number of repeated features
- `n_classes` (Optional[int]): Number of classes
- `n_clusters_per_class` (Optional[int]): Number of clusters per class
- `weights` (Optional[List[float]]): Class weights (proportions)
- `flip_y` (Optional[float]): Fraction of samples with flipped labels
- `class_sep` (Optional[float]): Class separation factor
- `random_state` (Optional[int]): Random seed (default from config)

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Tuple of (features DataFrame, target Series)

**Example:**
```python
data, target = generator.generate_classification(
    n_samples=1000,
    n_features=10,
    n_classes=3
)
```

##### `generate_regression(n_samples: Optional[int] = None, n_features: Optional[int] = None, n_informative: Optional[int] = None, noise: Optional[float] = None, bias: Optional[float] = None, effective_rank: Optional[int] = None, tail_strength: Optional[float] = None, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]`

Generate synthetic regression dataset.

**Parameters:**
- `n_samples` (Optional[int]): Number of samples (default from config)
- `n_features` (Optional[int]): Number of features (default from config)
- `n_informative` (Optional[int]): Number of informative features
- `noise` (Optional[float]): Standard deviation of gaussian noise
- `bias` (Optional[float]): Bias term in linear model
- `effective_rank` (Optional[int]): Approximate number of singular vectors
- `tail_strength` (Optional[float]): Strength of tail in singular values
- `random_state` (Optional[int]): Random seed (default from config)

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Tuple of (features DataFrame, target Series)

**Example:**
```python
data, target = generator.generate_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1
)
```

##### `generate_custom_classification(n_samples: int, n_features: int, n_classes: int = 2, class_distribution: Optional[List[float]] = None, feature_ranges: Optional[List[Tuple[float, float]]] = None, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]`

Generate custom classification dataset with specified distributions.

**Parameters:**
- `n_samples` (int): Number of samples
- `n_features` (int): Number of features
- `n_classes` (int): Number of classes
- `class_distribution` (Optional[List[float]]): Distribution of classes (must sum to 1.0)
- `feature_ranges` (Optional[List[Tuple[float, float]]]): List of (min, max) tuples for each feature
- `random_state` (Optional[int]): Random seed (default from config)

**Returns:**
- `Tuple[pd.DataFrame, pd.Series]`: Tuple of (features DataFrame, target Series)

**Raises:**
- `ValueError`: If parameters invalid

**Example:**
```python
data, target = generator.generate_custom_classification(
    n_samples=500,
    n_features=5,
    n_classes=2,
    class_distribution=[0.7, 0.3]
)
```

##### `get_dataset_info() -> Dict[str, any]`

Get information about generated dataset.

**Returns:**
- `Dict[str, any]`: Dictionary with dataset information

**Raises:**
- `ValueError`: If no data generated

**Example:**
```python
info = generator.get_dataset_info()
print(f"Shape: {info['shape']}")
print(f"Task type: {info['task_type']}")
```

##### `save_dataset(output_path: str, include_target: bool = True) -> None`

Save generated dataset to CSV file.

**Parameters:**
- `output_path` (str): Path to output CSV file
- `include_target` (bool): Whether to include target column

**Raises:**
- `ValueError`: If no data generated

**Example:**
```python
generator.save_dataset("dataset.csv")
# or without target
generator.save_dataset("features_only.csv", include_target=False)
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
generation:
  n_samples: 1000
  n_features: 10
  random_state: 42
  noise: 0.1

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `n_samples` (int): Number of samples to generate
- `n_features` (int): Number of features to generate
- `random_state` (int): Random seed for reproducibility
- `noise` (float): Noise level for regression datasets
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator()

# Classification
data, target = generator.generate_classification()

# Regression
data, target = generator.generate_regression()
```

### Complete Workflow

```python
from src.main import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator(config_path="config.yaml")

# Generate classification dataset
data, target = generator.generate_classification(
    n_samples=1000,
    n_features=15,
    n_classes=3,
    class_sep=1.5
)

# Get information
info = generator.get_dataset_info()

# Save dataset
generator.save_dataset("classification_data.csv")
```

### Custom Classification

```python
generator = SyntheticDatasetGenerator()

# Generate with custom distribution
data, target = generator.generate_custom_classification(
    n_samples=500,
    n_features=5,
    n_classes=2,
    class_distribution=[0.7, 0.3],
    feature_ranges=[(-5, 5), (0, 10), (-10, 0), (1, 2), (100, 200)]
)
```
