# CSV Dataset Explorer

A Python tool for loading and exploring CSV datasets with comprehensive statistics, data type analysis, and missing value detection. This is the first project in the ML learning series, focusing on fundamental data exploration skills.

## Project Title and Description

The CSV Dataset Explorer provides a comprehensive solution for quickly understanding the structure and quality of CSV datasets. It automates the initial data exploration phase that is crucial for any machine learning project, providing detailed insights into data types, missing values, and basic statistical summaries.

This tool solves the problem of manually inspecting large CSV files by providing automated analysis and reporting. It's designed for data scientists, ML engineers, and students who need to quickly understand their datasets before proceeding with data preprocessing and modeling.

**Target Audience**: Beginners learning machine learning, data scientists starting new projects, and anyone who needs to quickly explore CSV datasets.

## Features

- Load CSV files with configurable separators and encoding
- Basic dataset information (shape, memory usage, column names)
- Data type analysis for all columns
- Missing value analysis with counts and percentages
- Basic statistical summary for numerical columns
- Categorical column summaries
- Comprehensive exploration reports
- Data preview functionality
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/csv-data-explorer
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
python src/main.py --file sample.csv --report
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
csv_explorer:
  separator: ","
  encoding: "utf-8"
  max_rows_preview: 10

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `separator`: Column separator (comma, semicolon, tab, etc.)
- `encoding`: File encoding (utf-8, latin-1, etc.)
- `max_rows_preview`: Default number of rows to preview
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import CSVDataExplorer

explorer = CSVDataExplorer()
explorer.load_csv("data.csv")

# Get basic information
info = explorer.get_basic_info()
print(f"Dataset shape: {info['shape']}")

# Get data types
dtypes = explorer.get_data_types()

# Analyze missing values
missing = explorer.get_missing_value_analysis()
```

### Command-Line Usage

Explore a CSV file:

```bash
python src/main.py --file data.csv
```

Generate comprehensive report:

```bash
python src/main.py --file data.csv --report
```

Preview first 20 rows:

```bash
python src/main.py --file data.csv --preview 20
```

Use custom configuration:

```bash
python src/main.py --config custom_config.yaml --file data.csv
```

### Complete Example

```python
from src.main import CSVDataExplorer

# Initialize explorer
explorer = CSVDataExplorer()

# Load CSV file
explorer.load_csv("sales_data.csv")

# Generate comprehensive report
report = explorer.generate_report()
print(report)

# Get specific analyses
info = explorer.get_basic_info()
dtypes = explorer.get_data_types()
missing = explorer.get_missing_value_analysis()
stats = explorer.get_basic_statistics()
categorical = explorer.get_categorical_summary()

# Preview data
preview = explorer.preview_data(n_rows=5)
print(preview)
```

## Project Structure

```
csv-data-explorer/
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

- `src/main.py`: Core implementation with `CSVDataExplorer` class
- `config.yaml`: Configuration file for CSV loading and logging
- `tests/test_main.py`: Comprehensive unit tests
- `docs/API.md`: Detailed API documentation
- `logs/`: Directory for application logs

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
- CSV file loading with different separators and encodings
- Basic information extraction
- Data type analysis
- Missing value detection
- Statistical summaries
- Categorical summaries
- Report generation
- Error handling

## Troubleshooting

### Common Issues

**Issue**: File not found error

**Solution**: Check that the CSV file path is correct and the file exists.

**Issue**: Encoding errors

**Solution**: Try different encodings (utf-8, latin-1, cp1252) in config.yaml.

**Issue**: Parsing errors

**Solution**: Check the separator in config.yaml matches your CSV file format.

**Issue**: Memory errors with large files

**Solution**: Consider using chunking or processing files in parts.

### Error Messages

- `FileNotFoundError`: CSV file doesn't exist - check file path
- `pd.errors.EmptyDataError`: CSV file is empty
- `pd.errors.ParserError`: CSV format issue - check separator
- `ValueError`: No data loaded - call load_csv() first

## Data Exploration Workflow

1. **Load Data**: Use `load_csv()` to load your CSV file
2. **Basic Info**: Get dataset shape and memory usage
3. **Data Types**: Understand column data types
4. **Missing Values**: Identify data quality issues
5. **Statistics**: Analyze numerical columns
6. **Categorical**: Understand categorical distributions
7. **Report**: Generate comprehensive report

## Performance Considerations

- Memory usage depends on dataset size
- Large files may require chunking
- Statistical calculations are optimized for pandas
- Consider sampling for very large datasets

## Real-World Applications

- Initial data exploration in ML projects
- Data quality assessment
- Feature engineering preparation
- Dataset documentation
- Data cleaning workflow

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
