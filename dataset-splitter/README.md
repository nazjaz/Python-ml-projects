# Dataset Splitter Tool

A Python tool for splitting datasets into training, validation, and test sets with configurable ratios and stratification. This is the seventh project in the ML learning series, focusing on proper dataset splitting for machine learning.

## Project Title and Description

The Dataset Splitter Tool provides automated splitting of datasets into training, validation, and test sets with proper stratification for classification tasks. It ensures reproducible splits and maintains class distributions across splits, which is essential for building reliable ML models.

This tool solves the problem of manually splitting datasets, which can lead to data leakage, improper class distributions, and non-reproducible results. It provides a standardized way to split data that follows ML best practices.

**Target Audience**: Beginners learning machine learning, data scientists preparing datasets, and anyone who needs to properly split data for ML model training.

## Features

- Load data from CSV files or pandas DataFrames
- Split into training, validation, and test sets
- Configurable split ratios
- Stratification support for classification tasks
- Reproducible splits with random seed
- Shuffle option for data randomization
- Split summary statistics
- Save splits to CSV files
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/dataset-splitter
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
python src/main.py --input sample.csv --target label
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

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

- `train_ratio`: Training set ratio (default: 0.7)
- `val_ratio`: Validation set ratio (default: 0.15)
- `test_ratio`: Test set ratio (default: 0.15)
- `random_state`: Random seed for reproducibility (default: 42)
- `shuffle`: Whether to shuffle data before splitting (default: true)
- `stratify`: Whether to stratify by target column (default: true)
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

**Note**: Ratios must sum to 1.0. If they don't, they will be automatically normalized.

## Usage

### Basic Usage

```python
from src.main import DatasetSplitter

splitter = DatasetSplitter()
splitter.load_data(file_path="data.csv", target_column="label")

# Split dataset
splits = splitter.split()

# Access splits
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]
```

### Command-Line Usage

Basic split:

```bash
python src/main.py --input data.csv --target label
```

Custom ratios:

```bash
python src/main.py --input data.csv --target label --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Without stratification:

```bash
python src/main.py --input data.csv --target label --no-stratify
```

Custom output directory:

```bash
python src/main.py --input data.csv --target label --output-dir my_splits
```

Custom random seed:

```bash
python src/main.py --input data.csv --target label --random-state 123
```

### Complete Example

```python
from src.main import DatasetSplitter
import pandas as pd

# Initialize splitter
splitter = DatasetSplitter()

# Load data with target column
splitter.load_data(
    file_path="sales_data.csv",
    target_column="sales_category"
)

# Split with custom ratios
splits = splitter.split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify=True,
    random_state=42
)

# Get split summary
summary = splitter.get_split_summary(splits)
print("Split Summary:")
for split_name, info in summary.items():
    print(f"{split_name}: {info['samples']} samples")

# Access individual splits
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]

# Save splits to files
splitter.save_splits(splits, output_dir="splits")
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import DatasetSplitter

df = pd.read_csv("data.csv")
splitter = DatasetSplitter()
splitter.load_data(dataframe=df, target_column="target")

splits = splitter.split()
```

## Project Structure

```
dataset-splitter/
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

- `src/main.py`: Core implementation with `DatasetSplitter` class
- `config.yaml`: Configuration file for split parameters
- `tests/test_main.py`: Comprehensive unit tests
- `docs/API.md`: Detailed API documentation
- `logs/`: Directory for application logs
- `splits/`: Directory for saved split files (created automatically)

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
- Train/validation/test splitting
- Custom ratios
- Stratification
- Split summary generation
- Saving splits
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Ratios don't sum to 1.0

**Solution**: Ratios are automatically normalized. Ensure they're reasonable (e.g., 0.7, 0.15, 0.15).

**Issue**: Stratification fails

**Solution**: Ensure target column has sufficient samples per class. Stratification requires at least 2 samples per class.

**Issue**: Split sizes don't match expected ratios

**Solution**: Due to rounding, exact ratios may not be possible. The tool uses the closest possible split.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Target column 'X' not found`: Check target column name
- `ValueError: Ratios invalid`: Ensure ratios are between 0 and 1

## Split Ratios

### Common Split Ratios

- **70/15/15**: Standard split (train/val/test)
- **80/10/10**: More training data
- **60/20/20**: More validation/test data
- **90/5/5**: Very large training set

### When to Use Each

- **70/15/15**: General purpose, balanced
- **80/10/10**: Large datasets, need more training data
- **60/20/20**: Small datasets, need more validation data
- **90/5/5**: Very large datasets, minimal validation needed

## Stratification

### What is Stratification?

Stratification ensures that each split maintains the same class distribution as the original dataset. This is crucial for classification tasks.

### When to Use Stratification

- **Use**: Classification tasks with imbalanced classes
- **Use**: When you need representative splits
- **Don't use**: Regression tasks (no classes)
- **Don't use**: When classes are too small (< 2 samples per class)

## Best Practices

1. **Always stratify**: Use stratification for classification tasks
2. **Set random seed**: Use fixed random_state for reproducibility
3. **Validate ratios**: Ensure ratios sum to 1.0
4. **Check distributions**: Verify class distributions in splits
5. **Save splits**: Save splits to avoid re-splitting

## Real-World Applications

- Preparing datasets for ML model training
- Creating train/val/test splits for cross-validation
- Ensuring reproducible experiments
- Maintaining class balance in classification
- Data preprocessing pipelines

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
