# Data Visualization API Documentation

## Classes

### DataVisualizer

Main class for creating data visualizations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize DataVisualizer with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
visualizer = DataVisualizer()
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
visualizer.load_data(file_path="data.csv")
# or
visualizer.load_data(dataframe=df)
```

##### `get_numeric_columns() -> List[str]`

Get list of numerical columns in the dataset.

**Returns:**
- `List[str]`: List of numerical column names

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
numeric_cols = visualizer.get_numeric_columns()
```

##### `plot_histogram(columns: Optional[List[str]] = None, bins: Optional[int] = None, kde: bool = False, save_path: Optional[str] = None) -> None`

Create histogram plots for numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to plot (None for all numeric)
- `bins` (Optional[int]): Number of bins for histogram (None for auto)
- `kde` (bool): Whether to overlay kernel density estimate
- `save_path` (Optional[str]): Path to save plot (None displays plot)

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
visualizer.plot_histogram()
# or with options
visualizer.plot_histogram(columns=["age"], bins=20, kde=True, save_path="hist.png")
```

##### `plot_boxplot(columns: Optional[List[str]] = None, save_path: Optional[str] = None) -> None`

Create box plots for numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to plot (None for all numeric)
- `save_path` (Optional[str]): Path to save plot (None displays plot)

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
visualizer.plot_boxplot()
# or with options
visualizer.plot_boxplot(columns=["age", "score"], save_path="boxplot.png")
```

##### `plot_density(columns: Optional[List[str]] = None, save_path: Optional[str] = None) -> None`

Create density plots (KDE) for numerical columns.

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to plot (None for all numeric)
- `save_path` (Optional[str]): Path to save plot (None displays plot)

**Raises:**
- `ValueError`: If no data loaded or columns invalid

**Example:**
```python
visualizer.plot_density()
# or with options
visualizer.plot_density(columns=["age"], save_path="density.png")
```

##### `plot_all_distributions(columns: Optional[List[str]] = None, bins: Optional[int] = None, save_dir: Optional[str] = None) -> None`

Create all distribution plots (histogram, box plot, density).

**Parameters:**
- `columns` (Optional[List[str]]): List of column names to plot (None for all numeric)
- `bins` (Optional[int]): Number of bins for histogram (None for auto)
- `save_dir` (Optional[str]): Directory to save plots (None uses config default)

**Raises:**
- `ValueError`: If no data loaded

**Example:**
```python
visualizer.plot_all_distributions()
# or with options
visualizer.plot_all_distributions(columns=["age", "score"], bins=20)
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

```yaml
visualization:
  figsize: [10, 6]
  dpi: 100
  style: "whitegrid"
  color_palette: "husl"
  output_dir: "plots"

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `figsize` (List[float]): Figure size [width, height] in inches
- `dpi` (int): DPI for saved plots
- `style` (str): Seaborn style (whitegrid, darkgrid, white, dark, ticks)
- `color_palette` (str): Color palette (husl, Set2, pastel, etc.)
- `output_dir` (str): Directory for saved plots
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import DataVisualizer

visualizer = DataVisualizer()
visualizer.load_data(file_path="data.csv")

visualizer.plot_histogram()
visualizer.plot_boxplot()
visualizer.plot_density()
```

### Complete Workflow

```python
from src.main import DataVisualizer

visualizer = DataVisualizer(config_path="config.yaml")
visualizer.load_data(file_path="sales_data.csv")

# Get numerical columns
numeric_cols = visualizer.get_numeric_columns()

# Create all plots
visualizer.plot_all_distributions(columns=numeric_cols, bins=20)
```

### Column-Specific Visualization

```python
visualizer = DataVisualizer()
visualizer.load_data(file_path="data.csv")

# Plot specific columns
visualizer.plot_histogram(columns=["age", "score"], bins=20, kde=True)
visualizer.plot_boxplot(columns=["age", "score"])
```
