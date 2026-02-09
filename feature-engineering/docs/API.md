# Feature Engineering API Documentation

## Classes

### FeatureEngineer

Main class for feature engineering operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize FeatureEngineer with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
engineer = FeatureEngineer()
```

##### `create_polynomial_features(X: Union[pd.DataFrame, np.ndarray], degree: Optional[int] = None, include_bias: Optional[bool] = None, columns: Optional[List[str]] = None) -> pd.DataFrame`

Create polynomial features from input data.

**Parameters:**
- `X` (Union[pd.DataFrame, np.ndarray]): Input DataFrame or numpy array
- `degree` (Optional[int]): Degree of polynomial features (default from config)
- `include_bias` (Optional[bool]): Whether to include bias term (default from config)
- `columns` (Optional[List[str]]): List of column names for DataFrame (optional)

**Returns:**
- `pd.DataFrame`: DataFrame with polynomial features

**Raises:**
- `ValueError`: If degree is invalid or columns don't match

**Example:**
```python
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
polynomial_df = engineer.create_polynomial_features(df, degree=2)
```

##### `create_interaction_terms(X: Union[pd.DataFrame, np.ndarray], columns: Optional[List[str]] = None, max_interactions: Optional[int] = None, feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame`

Create interaction terms between features.

**Parameters:**
- `X` (Union[pd.DataFrame, np.ndarray]): Input DataFrame or numpy array
- `columns` (Optional[List[str]]): List of column names for DataFrame (optional)
- `max_interactions` (Optional[int]): Maximum number of interactions to create (None for all)
- `feature_pairs` (Optional[List[Tuple[str, str]]]): Specific pairs of features to create interactions for (optional)

**Returns:**
- `pd.DataFrame`: DataFrame with original features and interaction terms

**Raises:**
- `ValueError`: If columns don't match or feature pairs invalid

**Example:**
```python
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
interaction_df = engineer.create_interaction_terms(df)
```

##### `create_polynomial_and_interactions(X: Union[pd.DataFrame, np.ndarray], polynomial_degree: Optional[int] = None, include_interactions: bool = True, max_interactions: Optional[int] = None, columns: Optional[List[str]] = None) -> pd.DataFrame`

Create both polynomial features and interaction terms.

**Parameters:**
- `X` (Union[pd.DataFrame, np.ndarray]): Input DataFrame or numpy array
- `polynomial_degree` (Optional[int]): Degree of polynomial features (default from config)
- `include_interactions` (bool): Whether to include interaction terms
- `max_interactions` (Optional[int]): Maximum number of interactions to create
- `columns` (Optional[List[str]]): List of column names for DataFrame (optional)

**Returns:**
- `pd.DataFrame`: DataFrame with polynomial features and interaction terms

**Example:**
```python
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
engineered_df = engineer.create_polynomial_and_interactions(
    df, polynomial_degree=2, include_interactions=True
)
```

##### `get_feature_info() -> Dict[str, any]`

Get information about created features.

**Returns:**
- `Dict[str, any]`: Dictionary with feature information

**Example:**
```python
info = engineer.get_feature_info()
print(f"Total features: {info['n_features']}")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
feature_engineering:
  default_polynomial_degree: 2
  include_bias: false

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `default_polynomial_degree` (int): Default degree for polynomial features
- `include_bias` (bool): Whether to include bias term in polynomial features
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import FeatureEngineer
import pandas as pd

engineer = FeatureEngineer()

# Load data
df = pd.read_csv("data.csv")

# Create polynomial features
polynomial_df = engineer.create_polynomial_features(df, degree=2)

# Create interaction terms
interaction_df = engineer.create_interaction_terms(df)
```

### Complete Workflow

```python
import pandas as pd
from src.main import FeatureEngineer

engineer = FeatureEngineer()

# Load data
df = pd.read_csv("sales_data.csv")

# Create polynomial and interaction features
engineered_df = engineer.create_polynomial_and_interactions(
    df[["price", "quantity", "discount"]],
    polynomial_degree=2,
    include_interactions=True
)

# Get feature information
info = engineer.get_feature_info()

# Save
engineered_df.to_csv("engineered_features.csv", index=False)
```

### Custom Feature Pairs

```python
engineer = FeatureEngineer()

# Create interactions for specific pairs
interaction_df = engineer.create_interaction_terms(
    df,
    feature_pairs=[("price", "quantity"), ("discount", "quantity")]
)
```

### NumPy Array Support

```python
import numpy as np
from src.main import FeatureEngineer

engineer = FeatureEngineer()

X = np.array([[1, 2], [3, 4], [5, 6]])
polynomial_df = engineer.create_polynomial_features(
    X,
    degree=2,
    columns=["feature_1", "feature_2"]
)
```
