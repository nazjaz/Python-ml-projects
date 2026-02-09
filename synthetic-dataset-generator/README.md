# Synthetic Dataset Generator

A Python tool for generating synthetic datasets for classification and regression tasks with configurable parameters. This is the tenth project in the ML learning series, focusing on creating test datasets for ML model development and testing.

## Project Title and Description

The Synthetic Dataset Generator provides automated creation of synthetic datasets for machine learning tasks. It supports classification and regression datasets with extensive configurability, allowing users to create datasets with specific characteristics for testing, prototyping, and learning ML algorithms.

This tool solves the problem of needing test datasets for ML development. Real datasets may be unavailable, too large, or have privacy concerns. This tool generates realistic synthetic data with known properties, making it ideal for algorithm testing, benchmarking, and educational purposes.

**Target Audience**: Beginners learning machine learning, data scientists testing algorithms, and anyone who needs synthetic datasets for ML development.

## Features

- Generate classification datasets with configurable classes
- Generate regression datasets with configurable noise
- Custom classification dataset generation with specified distributions
- Configurable number of samples and features
- Control over informative, redundant, and repeated features
- Class separation and distribution control
- Noise level control for regression
- Dataset information and statistics
- Save datasets to CSV files
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/synthetic-dataset-generator
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
python src/main.py --task classification --output test.csv
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

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

- `n_samples`: Number of samples to generate (default: 1000)
- `n_features`: Number of features to generate (default: 10)
- `random_state`: Random seed for reproducibility (default: 42)
- `noise`: Noise level for regression datasets (default: 0.1)
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import SyntheticDatasetGenerator

generator = SyntheticDatasetGenerator()

# Generate classification dataset
data, target = generator.generate_classification(
    n_samples=1000,
    n_features=10,
    n_classes=3
)

# Generate regression dataset
data, target = generator.generate_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1
)
```

### Command-Line Usage

Generate classification dataset:

```bash
python src/main.py --task classification --output classification_data.csv
```

Generate regression dataset:

```bash
python src/main.py --task regression --output regression_data.csv
```

Generate custom classification:

```bash
python src/main.py --task custom_classification --output custom_data.csv
```

Custom parameters:

```bash
python src/main.py --task classification --n-samples 5000 --n-features 20 --n-classes 3 --output data.csv
```

Custom random seed:

```bash
python src/main.py --task regression --random-state 123 --output data.csv
```

### Complete Example

```python
from src.main import SyntheticDatasetGenerator
import pandas as pd

# Initialize generator
generator = SyntheticDatasetGenerator()

# Generate classification dataset
data, target = generator.generate_classification(
    n_samples=1000,
    n_features=15,
    n_classes=3,
    n_informative=10,
    n_redundant=3,
    class_sep=1.5,
    random_state=42
)

# Get dataset information
info = generator.get_dataset_info()
print(f"Dataset shape: {info['shape']}")
print(f"Task type: {info['task_type']}")
print(f"Classes: {info['n_classes']}")

# Generate regression dataset
data, target = generator.generate_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=0.2,
    random_state=42
)

# Generate custom classification
data, target = generator.generate_custom_classification(
    n_samples=500,
    n_features=5,
    n_classes=2,
    class_distribution=[0.7, 0.3],
    random_state=42
)

# Save datasets
generator.save_dataset("classification_data.csv")
```

### Using Generated Datasets

```python
from src.main import SyntheticDatasetGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate dataset
generator = SyntheticDatasetGenerator()
X, y = generator.generate_classification(n_samples=1000, n_classes=3)

# Use for ML training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.3f}")
```

## Project Structure

```
synthetic-dataset-generator/
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

- `src/main.py`: Core implementation with `SyntheticDatasetGenerator` class
- `config.yaml`: Configuration file for generation parameters
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
- Classification dataset generation
- Regression dataset generation
- Custom classification generation
- Parameter validation
- Dataset information retrieval
- Saving datasets
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Generated dataset has too many features

**Solution**: Reduce `n_features` or adjust `n_informative`, `n_redundant` parameters.

**Issue**: Classes are not well separated

**Solution**: Increase `class_sep` parameter for classification datasets.

**Issue**: Regression target has too much noise

**Solution**: Reduce `noise` parameter for regression datasets.

### Error Messages

- `ValueError: No dataset generated`: Call generation method first
- `ValueError: class_distribution must sum to 1.0`: Fix class distribution
- `ValueError: feature_ranges length mismatch`: Ensure feature_ranges matches n_features

## Dataset Types

### Classification Datasets

- **Use for**: Testing classification algorithms
- **Parameters**: n_classes, n_informative, n_redundant, class_sep, flip_y
- **Output**: Features and categorical target
- **Applications**: Algorithm testing, benchmarking, education

### Regression Datasets

- **Use for**: Testing regression algorithms
- **Parameters**: n_informative, noise, bias, effective_rank
- **Output**: Features and continuous target
- **Applications**: Algorithm testing, benchmarking, education

### Custom Classification

- **Use for**: Specific class distributions or feature ranges
- **Parameters**: class_distribution, feature_ranges
- **Output**: Customized classification dataset
- **Applications**: Imbalanced learning, specific test scenarios

## Best Practices

1. **Set random seed**: Use fixed random_state for reproducibility
2. **Choose appropriate size**: Balance between too small and too large
3. **Control complexity**: Adjust informative/redundant features
4. **Validate datasets**: Check generated dataset properties
5. **Document parameters**: Keep track of generation parameters

## Real-World Applications

- Testing ML algorithms
- Prototyping models
- Educational purposes
- Benchmarking algorithms
- Creating test datasets for pipelines
- Algorithm development

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
