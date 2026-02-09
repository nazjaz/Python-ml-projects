# Data Visualization Tool for Exploratory Data Analysis

A Python tool for visualizing data distributions using histograms, box plots, and density plots for exploratory data analysis. This is the fifth project in the ML learning series, focusing on data visualization skills.

## Project Title and Description

The Data Visualization Tool provides automated creation of distribution plots essential for exploratory data analysis (EDA). It generates histograms, box plots, and density plots to help understand data distributions, identify outliers, and detect patterns in numerical features.

This tool solves the problem of manually creating visualization code for EDA by providing automated plotting with configurable options. It helps data scientists quickly understand their data distributions before building ML models.

**Target Audience**: Beginners learning machine learning, data scientists performing EDA, and anyone who needs to visualize data distributions.

## Features

- Load data from CSV files or pandas DataFrames
- Histogram plots with optional KDE overlay
- Box plots for outlier detection
- Density plots (KDE) for distribution shape
- Automatic detection of numerical columns
- Column-specific visualization support
- Multiple plots in grid layout
- Save plots to files
- Configurable styling and appearance
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/data-visualization
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --input sample.csv --plot histogram
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

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

- `figsize`: Figure size [width, height] in inches
- `dpi`: DPI for saved plots
- `style`: Seaborn style (whitegrid, darkgrid, white, dark, ticks)
- `color_palette`: Color palette (husl, Set2, pastel, etc.)
- `output_dir`: Directory for saved plots
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import DataVisualizer

visualizer = DataVisualizer()
visualizer.load_data(file_path="data.csv")

# Create histogram
visualizer.plot_histogram()

# Create box plot
visualizer.plot_boxplot()

# Create density plot
visualizer.plot_density()
```

### Command-Line Usage

Create histogram:

```bash
python src/main.py --input data.csv --plot histogram
```

Create box plot:

```bash
python src/main.py --input data.csv --plot boxplot
```

Create density plot:

```bash
python src/main.py --input data.csv --plot density
```

Create all plots:

```bash
python src/main.py --input data.csv --plot all
```

Plot specific columns:

```bash
python src/main.py --input data.csv --plot histogram --columns age score
```

Save plot to file:

```bash
python src/main.py --input data.csv --plot histogram --output histogram.png
```

Custom bins for histogram:

```bash
python src/main.py --input data.csv --plot histogram --bins 20
```

### Complete Example

```python
from src.main import DataVisualizer
import pandas as pd

# Initialize visualizer
visualizer = DataVisualizer()

# Load data
visualizer.load_data(file_path="sales_data.csv")

# Get numerical columns
numeric_cols = visualizer.get_numeric_columns()
print(f"Numerical columns: {numeric_cols}")

# Create histogram with KDE
visualizer.plot_histogram(columns=["age", "score"], bins=20, kde=True)

# Create box plot
visualizer.plot_boxplot(columns=["age", "score"])

# Create density plot
visualizer.plot_density(columns=["age", "score"])

# Create all plots and save
visualizer.plot_all_distributions(columns=numeric_cols, bins=15)
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import DataVisualizer

df = pd.read_csv("data.csv")
visualizer = DataVisualizer()
visualizer.load_data(dataframe=df)

visualizer.plot_histogram()
```

## Project Structure

```
data-visualization/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py              # Main implementation
├── tests/
│   └── test_main.py         # Unit tests
├── docs/
│   └── API.md               # API documentation
└── logs/
    └── .gitkeep             # Log directory
```

### File Descriptions

- `src/main.py`: Core implementation with `DataVisualizer` class
- `config.yaml`: Configuration file for visualization parameters
- `tests/test_main.py`: Comprehensive unit tests
- `docs/API.md`: Detailed API documentation
- `logs/`: Directory for application logs
- `plots/`: Directory for saved plots (created automatically)

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

Tests cover:
- Data loading from files and DataFrames
- Histogram creation
- Box plot creation
- Density plot creation
- Column-specific visualization
- Plot saving functionality
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Plots not displaying

**Solution**: Ensure you have a display backend. Use `--output` to save plots instead.

**Issue**: Memory errors with large datasets

**Solution**: Filter data or plot specific columns, reduce number of bins.

**Issue**: Plots look crowded

**Solution**: Adjust `figsize` in config.yaml or plot fewer columns at once.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Column 'X' not found`: Check column names
- `ImportError`: Install matplotlib and seaborn

## Visualization Types

### Histogram

- **Purpose**: Shows frequency distribution of numerical data
- **Use for**: Understanding data distribution, detecting skewness
- **Features**: Configurable bins, optional KDE overlay
- **Insights**: Distribution shape, central tendency, spread

### Box Plot

- **Purpose**: Shows quartiles, median, and outliers
- **Use for**: Detecting outliers, comparing distributions
- **Features**: Automatic outlier detection
- **Insights**: Median, quartiles, outliers, range

### Density Plot (KDE)

- **Purpose**: Shows smooth probability density function
- **Use for**: Understanding distribution shape
- **Features**: Smooth curve representation
- **Insights**: Distribution shape, peaks, tails

## Best Practices

1. **Start with histograms**: Get overview of distributions
2. **Check for outliers**: Use box plots to identify outliers
3. **Understand shape**: Use density plots for smooth distributions
4. **Compare columns**: Plot multiple columns to compare distributions
5. **Save important plots**: Document your EDA findings

## Real-World Applications

- Exploratory data analysis in ML projects
- Data quality assessment
- Feature distribution analysis
- Outlier detection
- Data preprocessing decisions
- Model feature selection

## Contributing

### Development Setup

1. Clone the repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Run tests: `pytest tests/`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Include docstrings for all public functions and classes
- Write tests for all new functionality

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. See LICENSE file in parent directory for details.
