# Correlation Analysis Tool

A Python tool for calculating correlation matrices and visualizing feature relationships using heatmaps and scatter plots. This is the sixth project in the ML learning series, focusing on correlation analysis for feature selection and understanding.

## Project Title and Description

The Correlation Analysis Tool provides automated calculation and visualization of correlations between numerical features. It helps identify relationships, detect multicollinearity, and guide feature selection decisions in machine learning projects.

This tool solves the problem of understanding relationships between features in datasets. High correlations can indicate redundant features, while strong correlations can reveal important relationships. This tool makes it easy to analyze and visualize these relationships.

**Target Audience**: Beginners learning machine learning, data scientists performing feature analysis, and anyone who needs to understand feature relationships.

## Features

- Load data from CSV files or pandas DataFrames
- Calculate correlation matrices (Pearson, Spearman, Kendall)
- Heatmap visualization of correlation matrix
- Scatter plots for feature pairs
- Scatter plot matrix for multiple features
- Identify strong correlations
- Correlation analysis summary
- Automatic detection of numerical columns
- Column-specific analysis support
- Save plots to files
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/correlation-analysis
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
python src/main.py --input sample.csv --plot heatmap
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

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

- `method`: Correlation method (pearson, spearman, kendall)
- `figsize`: Figure size [width, height] in inches
- `dpi`: DPI for saved plots
- `colormap`: Colormap for heatmap (coolwarm, RdYlBu, viridis, etc.)
- `output_dir`: Directory for saved plots
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
analyzer.load_data(file_path="data.csv")

# Calculate correlation matrix
corr_matrix = analyzer.calculate_correlation()

# Create heatmap
analyzer.plot_heatmap()

# Create scatter plot
analyzer.plot_scatter("age", "score")
```

### Command-Line Usage

Create correlation heatmap:

```bash
python src/main.py --input data.csv --plot heatmap
```

Create scatter plot:

```bash
python src/main.py --input data.csv --plot scatter --x age --y score
```

Create scatter plot matrix:

```bash
python src/main.py --input data.csv --plot scatter_matrix
```

Create all visualizations:

```bash
python src/main.py --input data.csv --plot all
```

Use different correlation method:

```bash
python src/main.py --input data.csv --plot heatmap --method spearman
```

Analyze specific columns:

```bash
python src/main.py --input data.csv --plot heatmap --columns age score height
```

Find strong correlations:

```bash
python src/main.py --input data.csv --threshold 0.8
```

### Complete Example

```python
from src.main import CorrelationAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = CorrelationAnalyzer()

# Load data
analyzer.load_data(file_path="sales_data.csv")

# Get numerical columns
numeric_cols = analyzer.get_numeric_columns()
print(f"Numerical columns: {numeric_cols}")

# Calculate correlation matrix
corr_matrix = analyzer.calculate_correlation(method="pearson")
print(f"\nCorrelation matrix shape: {corr_matrix.shape}")

# Find strong correlations
strong_corr = analyzer.get_strong_correlations(threshold=0.7)
print(f"\nStrong correlations (|r| >= 0.7): {len(strong_corr)}")
for col1, col2, corr in strong_corr[:5]:
    print(f"  {col1} - {col2}: {corr:.3f}")

# Analyze correlations
analysis = analyzer.analyze_correlations(threshold=0.7)
print(f"\nMean correlation: {analysis['mean_correlation']:.3f}")
print(f"Max correlation: {analysis['max_correlation']:.3f}")

# Create visualizations
analyzer.plot_heatmap()
analyzer.plot_scatter("age", "score")
analyzer.plot_scatter_matrix(columns=numeric_cols[:5])
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import CorrelationAnalyzer

df = pd.read_csv("data.csv")
analyzer = CorrelationAnalyzer()
analyzer.load_data(dataframe=df)

analyzer.plot_heatmap()
```

## Project Structure

```
correlation-analysis/
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

- `src/main.py`: Core implementation with `CorrelationAnalyzer` class
- `config.yaml`: Configuration file for correlation parameters
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
- Correlation matrix calculation
- Heatmap visualization
- Scatter plot creation
- Scatter plot matrix
- Strong correlation identification
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Correlation matrix is all NaN

**Solution**: Check that columns have sufficient variance (not all same values).

**Issue**: Heatmap is too crowded

**Solution**: Filter to specific columns or increase figure size in config.

**Issue**: Scatter plot shows no relationship

**Solution**: Check that both columns are numerical and have valid data.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: Invalid method`: Use 'pearson', 'spearman', or 'kendall'

## Correlation Methods

### Pearson Correlation

- **Use for**: Linear relationships, normally distributed data
- **Range**: -1 to 1
- **Interpretation**: Measures linear correlation
- **Sensitive to**: Outliers

### Spearman Correlation

- **Use for**: Monotonic relationships, non-normal data
- **Range**: -1 to 1
- **Interpretation**: Measures rank correlation
- **Robust to**: Outliers

### Kendall Correlation

- **Use for**: Ordinal relationships, small datasets
- **Range**: -1 to 1
- **Interpretation**: Measures concordance
- **Robust to**: Outliers, small sample sizes

## Interpreting Correlations

- **|r| > 0.7**: Strong correlation (may indicate multicollinearity)
- **0.5 < |r| < 0.7**: Moderate correlation
- **|r| < 0.5**: Weak correlation
- **r > 0**: Positive relationship (increase together)
- **r < 0**: Negative relationship (one increases, other decreases)

## Best Practices

1. **Check assumptions**: Use appropriate correlation method
2. **Visualize**: Always visualize correlations, don't just rely on numbers
3. **Consider context**: High correlation doesn't always mean causation
4. **Feature selection**: Remove highly correlated features to reduce multicollinearity
5. **Document findings**: Save important visualizations for reports

## Real-World Applications

- Feature selection for ML models
- Multicollinearity detection
- Understanding feature relationships
- Data quality assessment
- Exploratory data analysis
- Model interpretability

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
