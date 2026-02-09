# Correlation Analysis API Documentation

## Classes

### CorrelationAnalyzer

Main class for correlation analysis and visualization.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize CorrelationAnalyzer with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
analyzer = CorrelationAnalyzer()
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
analyzer.load_data(file_path="data.csv")
# or
analyzer.load_data(dataframe=df)
```

##### `get_numeric_columns() -> List[str]`

Get list of numerical columns in the dataset.

**Returns:**
- `List[str]`: List of numerical column names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
numeric_cols = analyzer.get_numeric_columns()
```

##### `calculate_correlation(method: Optional[str] = None, columns: Optional[List[str]] = None) -> pd.DataFrame`

Calculate correlation matrix for numerical columns.

**Parameters:**
- `method` (Optional[str]): Correlation method (pearson, spearman, kendall)
- `columns` (Optional[List[str]]): List of columns to include (None for all numeric)

**Returns:**
- `pd.DataFrame`: Correlation matrix DataFrame

**Raises:**
- `ValueError`: If no data loaded or invalid method

**Example:**
```python
corr_matrix = analyzer.calculate_correlation()
# or with options
corr_matrix = analyzer.calculate_correlation(method="spearman", columns=["age", "score"])
```

##### `get_strong_correlations(threshold: float = 0.7, method: Optional[str] = None) -> List[Tuple[str, str, float]]`

Get pairs of features with strong correlations.

**Parameters:**
- `threshold` (float): Minimum absolute correlation value
- `method` (Optional[str]): Correlation method (None uses calculated matrix)

**Returns:**
- `List[Tuple[str, str, float]]`: List of tuples (col1, col2, correlation)

**Raises:**
- `ValueError`: If no correlation matrix calculated

**Example:**
```python
strong_corr = analyzer.get_strong_correlations(threshold=0.8)
for col1, col2, corr in strong_corr:
    print(f"{col1} - {col2}: {corr:.3f}")
```

##### `plot_heatmap(method: Optional[str] = None, columns: Optional[List[str]] = None, annot: bool = True, fmt: str = ".2f", save_path: Optional[str] = None) -> None`

Create correlation heatmap visualization.

**Parameters:**
- `method` (Optional[str]): Correlation method (None uses calculated matrix)
- `columns` (Optional[List[str]]): List of columns to include (None for all numeric)
- `annot` (bool): Whether to annotate cells with correlation values
- `fmt` (str): Format string for annotations
- `save_path` (Optional[str]): Path to save plot (None displays plot)

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
analyzer.plot_heatmap()
# or with options
analyzer.plot_heatmap(method="spearman", annot=True, save_path="heatmap.png")
```

##### `plot_scatter(x_col: str, y_col: str, save_path: Optional[str] = None) -> None`

Create scatter plot for two features.

**Parameters:**
- `x_col` (str): Name of x-axis column
- `y_col` (str): Name of y-axis column
- `save_path` (Optional[str]): Path to save plot (None displays plot)

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
analyzer.plot_scatter("age", "score")
# or save
analyzer.plot_scatter("age", "score", save_path="scatter.png")
```

##### `plot_scatter_matrix(columns: Optional[List[str]] = None, save_path: Optional[str] = None) -> None`

Create scatter plot matrix for multiple features.

**Parameters:**
- `columns` (Optional[List[str]]): List of columns to plot (None for all numeric, max 5)
- `save_path` (Optional[str]): Path to save plot (None displays plot)

**Raises:**
- `ValueError`: If no data loaded or too many columns

**Example:**
```python
analyzer.plot_scatter_matrix()
# or with specific columns
analyzer.plot_scatter_matrix(columns=["age", "score", "height"])
```

##### `analyze_correlations(method: Optional[str] = None, threshold: float = 0.7) -> Dict[str, any]`

Analyze correlations and generate summary.

**Parameters:**
- `method` (Optional[str]): Correlation method (None uses config default)
- `threshold` (float): Threshold for strong correlations

**Returns:**
- `Dict[str, any]`: Dictionary with correlation analysis

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
analysis = analyzer.analyze_correlations(threshold=0.8)
print(f"Mean correlation: {analysis['mean_correlation']:.3f}")
print(f"Strong pairs: {analysis['strong_correlation_count']}")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
correlation:
  method: "pearson"
  figsize: [10, 8]
  dpi: 100
  colormap: "coolwarm"
  output_dir: "plots"

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `method` (str): Correlation method (pearson, spearman, kendall)
- `figsize` (List[float]): Figure size [width, height] in inches
- `dpi` (int): DPI for saved plots
- `colormap` (str): Colormap for heatmap (coolwarm, RdYlBu, viridis, etc.)
- `output_dir` (str): Directory for saved plots
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
analyzer.load_data(file_path="data.csv")

corr_matrix = analyzer.calculate_correlation()
analyzer.plot_heatmap()
analyzer.plot_scatter("age", "score")
```

### Complete Workflow

```python
from src.main import CorrelationAnalyzer

analyzer = CorrelationAnalyzer(config_path="config.yaml")
analyzer.load_data(file_path="sales_data.csv")

# Calculate and analyze
corr_matrix = analyzer.calculate_correlation(method="pearson")
strong_corr = analyzer.get_strong_correlations(threshold=0.7)

# Analyze
analysis = analyzer.analyze_correlations(threshold=0.7)

# Visualize
analyzer.plot_heatmap()
analyzer.plot_scatter_matrix(columns=["age", "score", "height"])
```

### Column-Specific Analysis

```python
analyzer = CorrelationAnalyzer()
analyzer.load_data(file_path="data.csv")

# Analyze specific columns
analyzer.calculate_correlation(columns=["age", "score", "height"])
analyzer.plot_heatmap(columns=["age", "score", "height"])
```
