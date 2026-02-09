# Feature Engineering Utilities

A Python tool for creating polynomial features and interaction terms for machine learning models. This is the fifteenth project in the ML learning series, focusing on feature engineering to improve model performance.

## Project Title and Description

The Feature Engineering Tool provides utilities for creating polynomial features and interaction terms from existing features. These engineered features can capture non-linear relationships and feature interactions that may improve machine learning model performance.

This tool solves the problem of limited feature expressiveness by automatically generating polynomial combinations and interaction terms. It helps discover complex relationships between features that linear models might miss, potentially improving model accuracy and predictive power.

**Target Audience**: Beginners learning machine learning, data scientists building ML models, and anyone who needs to create advanced features for model training.

## Features

- Create polynomial features up to specified degree
- Generate interaction terms between feature pairs
- Combine polynomial features and interactions
- Support for pandas DataFrames and numpy arrays
- Configurable polynomial degree
- Optional bias term inclusion
- Limit number of interactions
- Specify custom feature pairs for interactions
- Feature information and statistics
- Save engineered features to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/feature-engineering
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
python src/main.py --input sample.csv --polynomial-degree 2
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
feature_engineering:
  default_polynomial_degree: 2
  include_bias: false

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `default_polynomial_degree`: Default degree for polynomial features
- `include_bias`: Whether to include bias term in polynomial features
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

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

### Command-Line Usage

Create polynomial features:

```bash
python src/main.py --input data.csv --polynomial-degree 2 --output polynomial_features.csv
```

Create interaction terms:

```bash
python src/main.py --input data.csv --interactions --output interaction_features.csv
```

Create both:

```bash
python src/main.py --input data.csv --polynomial-degree 2 --interactions --output engineered_features.csv
```

### Complete Example

```python
import pandas as pd
from src.main import FeatureEngineer

# Initialize engineer
engineer = FeatureEngineer()

# Load data
df = pd.read_csv("sales_data.csv")

# Create polynomial features (degree 2)
polynomial_df = engineer.create_polynomial_features(
    df[["price", "quantity"]],
    degree=2
)

# Create interaction terms
interaction_df = engineer.create_interaction_terms(
    df[["price", "quantity", "discount"]]
)

# Create both polynomial and interactions
engineered_df = engineer.create_polynomial_and_interactions(
    df[["price", "quantity", "discount"]],
    polynomial_degree=2,
    include_interactions=True
)

# Get feature information
info = engineer.get_feature_info()
print(f"Created {info['n_features']} features")

# Save engineered features
engineered_df.to_csv("engineered_sales_data.csv", index=False)
```

### Custom Feature Pairs

```python
from src.main import FeatureEngineer

engineer = FeatureEngineer()

# Create interactions for specific feature pairs
interaction_df = engineer.create_interaction_terms(
    df,
    feature_pairs=[("price", "quantity"), ("discount", "quantity")]
)
```

### Using with NumPy Arrays

```python
import numpy as np
from src.main import FeatureEngineer

engineer = FeatureEngineer()

# Create from numpy array
X = np.array([[1, 2], [3, 4], [5, 6]])
polynomial_df = engineer.create_polynomial_features(
    X,
    degree=2,
    columns=["feature_1", "feature_2"]
)
```

## Project Structure

```
feature-engineering/
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

- `src/main.py`: Core implementation with `FeatureEngineer` class
- `config.yaml`: Configuration file for feature engineering parameters
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
- Polynomial feature creation
- Interaction term creation
- Combined feature engineering
- Custom feature pairs
- NumPy array support
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Too many features created

**Solution**: Reduce polynomial degree or limit number of interactions using `max_interactions`.

**Issue**: Memory issues with large datasets

**Solution**: Process features in batches or reduce polynomial degree.

**Issue**: Feature names not matching

**Solution**: Ensure column names are provided when using numpy arrays.

### Error Messages

- `ValueError: Degree must be >= 1`: Use degree >= 1
- `ValueError: Feature pair contains invalid column names`: Check feature pair column names

## Feature Engineering Techniques

### Polynomial Features

- **Purpose**: Capture non-linear relationships
- **Degree**: Controls complexity (degree 2 = quadratic, degree 3 = cubic)
- **Use for**: Non-linear models, capturing curvature

### Interaction Terms

- **Purpose**: Capture feature interactions
- **Method**: Multiply pairs of features
- **Use for**: Discovering feature relationships

## Best Practices

1. **Start small**: Begin with degree 2 and increase if needed
2. **Monitor feature count**: Polynomial features grow exponentially
3. **Use domain knowledge**: Create interactions for features that likely interact
4. **Validate improvements**: Test if engineered features improve model performance
5. **Consider regularization**: Use with models that handle many features well

## Real-World Applications

- Improving linear model performance
- Capturing non-linear relationships
- Feature interaction discovery
- Model performance enhancement
- Feature engineering pipelines

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
