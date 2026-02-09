# Data Pipeline API Documentation

## Classes

### BaseTransformer

Abstract base class for all custom transformers.

#### Methods

##### `fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> BaseTransformer`

Fit the transformer on the data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional)

**Returns:**
- `BaseTransformer`: Self for method chaining

##### `transform(X: pd.DataFrame) -> pd.DataFrame`

Transform the data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

**Raises:**
- `ValueError`: If transformer not fitted

##### `fit_transform(X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame`

Fit transformer and transform data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional)

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

### StandardScalerTransformer

Standard scaler transformer (z-score normalization).

#### Methods

##### `__init__(columns: Optional[List[str]] = None)`

Initialize standard scaler transformer.

**Parameters:**
- `columns` (Optional[List[str]]): List of columns to scale (None for all numeric)

**Example:**
```python
scaler = StandardScalerTransformer(columns=["age", "score"])
```

##### `fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> StandardScalerTransformer`

Fit the scaler on the data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional, ignored)

**Returns:**
- `StandardScalerTransformer`: Self for method chaining

**Raises:**
- `ValueError`: If columns not found or not numeric

##### `transform(X: pd.DataFrame) -> pd.DataFrame`

Transform the data using fitted parameters.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

**Raises:**
- `ValueError`: If transformer not fitted

### MinMaxScalerTransformer

Min-max scaler transformer (normalization to [0, 1]).

#### Methods

##### `__init__(columns: Optional[List[str]] = None, feature_range: Tuple[float, float] = (0, 1))`

Initialize min-max scaler transformer.

**Parameters:**
- `columns` (Optional[List[str]]): List of columns to scale (None for all numeric)
- `feature_range` (Tuple[float, float]): Desired range of transformed data

**Example:**
```python
scaler = MinMaxScalerTransformer(feature_range=(-1, 1))
```

##### `fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> MinMaxScalerTransformer`

Fit the scaler on the data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional, ignored)

**Returns:**
- `MinMaxScalerTransformer`: Self for method chaining

##### `transform(X: pd.DataFrame) -> pd.DataFrame`

Transform the data using fitted parameters.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

**Raises:**
- `ValueError`: If transformer not fitted

### ImputerTransformer

Imputer transformer for handling missing values.

#### Methods

##### `__init__(strategy: str = "mean", columns: Optional[List[str]] = None)`

Initialize imputer transformer.

**Parameters:**
- `strategy` (str): Imputation strategy (mean, median, mode, constant)
- `columns` (Optional[List[str]]): List of columns to impute (None for all numeric)

**Example:**
```python
imputer = ImputerTransformer(strategy="median", columns=["age"])
```

##### `fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> ImputerTransformer`

Fit the imputer on the data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional, ignored)

**Returns:**
- `ImputerTransformer`: Self for method chaining

**Raises:**
- `ValueError`: If strategy invalid

##### `transform(X: pd.DataFrame) -> pd.DataFrame`

Transform the data using fitted parameters.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

**Raises:**
- `ValueError`: If transformer not fitted

### OneHotEncoderTransformer

One-hot encoder transformer for categorical variables.

#### Methods

##### `__init__(columns: Optional[List[str]] = None)`

Initialize one-hot encoder transformer.

**Parameters:**
- `columns` (Optional[List[str]]): List of columns to encode (None for all categorical)

**Example:**
```python
encoder = OneHotEncoderTransformer(columns=["category"])
```

##### `fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> OneHotEncoderTransformer`

Fit the encoder on the data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional, ignored)

**Returns:**
- `OneHotEncoderTransformer`: Self for method chaining

##### `transform(X: pd.DataFrame) -> pd.DataFrame`

Transform the data using fitted parameters.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

**Raises:**
- `ValueError`: If transformer not fitted

### DataPipeline

Pipeline for chaining multiple transformers together.

#### Methods

##### `__init__(transformers: List[BaseTransformer])`

Initialize data pipeline.

**Parameters:**
- `transformers` (List[BaseTransformer]): List of transformers to chain together

**Raises:**
- `ValueError`: If transformers list is empty

**Example:**
```python
pipeline = DataPipeline(transformers=[imputer, scaler])
```

##### `fit(X: pd.DataFrame, y: Optional[pd.Series] = None) -> DataPipeline`

Fit all transformers in the pipeline.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional)

**Returns:**
- `DataPipeline`: Self for method chaining

##### `transform(X: pd.DataFrame) -> pd.DataFrame`

Transform data through all transformers in the pipeline.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

**Raises:**
- `ValueError`: If pipeline not fitted

##### `fit_transform(X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame`

Fit pipeline and transform data.

**Parameters:**
- `X` (pd.DataFrame): Input DataFrame
- `y` (Optional[pd.Series]): Target Series (optional)

**Returns:**
- `pd.DataFrame`: Transformed DataFrame

##### `get_pipeline_info() -> Dict[str, Any]`

Get information about the pipeline.

**Returns:**
- `Dict[str, Any]`: Dictionary with pipeline information

**Example:**
```python
info = pipeline.get_pipeline_info()
print(f"Transformers: {info['transformer_types']}")
```

## Examples

### Basic Usage

```python
from src.main import (
    DataPipeline,
    ImputerTransformer,
    StandardScalerTransformer,
)

# Create transformers
imputer = ImputerTransformer(strategy="mean")
scaler = StandardScalerTransformer()

# Create pipeline
pipeline = DataPipeline(transformers=[imputer, scaler])

# Fit and transform
transformed_df = pipeline.fit_transform(df)
```

### Complete Workflow

```python
import pandas as pd
from src.main import (
    DataPipeline,
    ImputerTransformer,
    StandardScalerTransformer,
    OneHotEncoderTransformer,
)

# Load data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Create pipeline
pipeline = DataPipeline(
    transformers=[
        ImputerTransformer(strategy="mean"),
        StandardScalerTransformer(),
        OneHotEncoderTransformer(columns=["category"]),
    ]
)

# Fit on training data
pipeline.fit(df_train)

# Transform both datasets
df_train_transformed = pipeline.transform(df_train)
df_test_transformed = pipeline.transform(df_test)
```

### Individual Transformers

```python
from src.main import StandardScalerTransformer

# Create and use transformer
scaler = StandardScalerTransformer(columns=["age", "score"])
scaled_df = scaler.fit_transform(df)
```
