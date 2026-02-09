# Data Augmentation Tool for Numerical Data

A Python tool for performing data augmentation techniques on numerical data including noise injection and scaling variations. This is the eleventh project in the ML learning series, focusing on data augmentation to increase dataset size and improve model robustness.

## Project Title and Description

The Data Augmentation Tool provides automated augmentation of numerical datasets using noise injection and scaling variations. It helps increase dataset size, improve model generalization, and make models more robust to variations in the data.

This tool solves the problem of limited training data by generating augmented versions of existing datasets. Data augmentation is particularly useful when collecting more real data is expensive or time-consuming, and helps improve model performance through increased training diversity.

**Target Audience**: Beginners learning machine learning, data scientists working with small datasets, and anyone who needs to augment numerical data for ML model training.

## Features

- Load data from CSV files or pandas DataFrames
- Noise injection with multiple types (Gaussian, uniform, Laplace)
- Scaling variations (multiplicative, additive, percentage)
- Configurable noise parameters (mean, standard deviation)
- Configurable scaling factors (min, max)
- Column-specific augmentation support
- Automatic detection of numerical columns
- Augmentation history tracking
- Save augmented data to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/data-augmentation
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
python src/main.py --input sample.csv --method noise
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
augmentation:
  noise_type: "gaussian"
  noise_std: 0.1
  noise_mean: 0.0
  scaling_type: "multiplicative"
  scaling_factor_min: 0.9
  scaling_factor_max: 1.1
  random_state: 42

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `noise_type`: Type of noise (gaussian, uniform, laplace)
- `noise_std`: Standard deviation for noise
- `noise_mean`: Mean for noise
- `scaling_type`: Type of scaling (multiplicative, additive, percentage)
- `scaling_factor_min`: Minimum scaling factor
- `scaling_factor_max`: Maximum scaling factor
- `random_state`: Random seed for reproducibility
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import DataAugmenter

augmenter = DataAugmenter()
augmenter.load_data(file_path="data.csv")

# Inject noise
augmented_data = augmenter.inject_noise()

# Apply scaling variations
augmented_data = augmenter.apply_scaling_variations()
```

### Command-Line Usage

Inject noise:

```bash
python src/main.py --input data.csv --method noise
```

Apply scaling variations:

```bash
python src/main.py --input data.csv --method scaling
```

Apply all augmentations:

```bash
python src/main.py --input data.csv --method all
```

Custom noise parameters:

```bash
python src/main.py --input data.csv --method noise --noise-type gaussian --noise-std 0.2
```

Custom scaling parameters:

```bash
python src/main.py --input data.csv --method scaling --scaling-type multiplicative --scaling-min 0.8 --scaling-max 1.2
```

Augment specific columns:

```bash
python src/main.py --input data.csv --method all --columns age score
```

Save augmented data:

```bash
python src/main.py --input data.csv --method all --output augmented_data.csv
```

### Complete Example

```python
from src.main import DataAugmenter
import pandas as pd

# Initialize augmenter
augmenter = DataAugmenter()

# Load data
augmenter.load_data(file_path="sales_data.csv")

# Inject Gaussian noise
noisy_data = augmenter.inject_noise(
    noise_type="gaussian",
    noise_std=0.1,
    noise_mean=0.0
)

# Apply multiplicative scaling
scaled_data = augmenter.apply_scaling_variations(
    scaling_type="multiplicative",
    scaling_factor_min=0.9,
    scaling_factor_max=1.1
)

# Apply all augmentations
augmented_data = augmenter.augment_all(
    noise_type="gaussian",
    noise_std=0.05,
    scaling_type="percentage",
    scaling_factor_min=-0.1,
    scaling_factor_max=0.1
)

# Get augmentation summary
summary = augmenter.get_augmentation_summary()
print(f"Total operations: {summary['total_operations']}")

# Save augmented data
augmenter.save_augmented_data("augmented_sales_data.csv")
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.main import DataAugmenter

df = pd.read_csv("data.csv")
augmenter = DataAugmenter()
augmenter.load_data(dataframe=df)

augmented_df = augmenter.inject_noise()
```

## Project Structure

```
data-augmentation/
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

- `src/main.py`: Core implementation with `DataAugmenter` class
- `config.yaml`: Configuration file for augmentation parameters
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
- Data loading from files and DataFrames
- Noise injection with different types
- Scaling variations with different types
- Column-specific augmentation
- Augmentation history tracking
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Augmented values are out of expected range

**Solution**: Adjust noise_std or scaling factors to smaller values.

**Issue**: Too much variation in augmented data

**Solution**: Reduce noise_std and narrow scaling factor range.

**Issue**: Augmentation doesn't preserve data characteristics

**Solution**: Use smaller augmentation parameters to preserve original distribution.

### Error Messages

- `ValueError: No data loaded`: Call `load_data()` first
- `ValueError: Column 'X' not found`: Check column names
- `ValueError: Invalid noise type`: Use 'gaussian', 'uniform', or 'laplace'
- `ValueError: Invalid scaling type`: Use 'multiplicative', 'additive', or 'percentage'

## Augmentation Techniques

### Noise Injection

- **Gaussian**: Normal distribution noise (most common)
- **Uniform**: Uniform distribution noise
- **Laplace**: Laplace distribution noise (heavier tails)
- **Use for**: Adding realistic measurement noise, improving robustness

### Scaling Variations

- **Multiplicative**: Multiply by random factor
- **Additive**: Add random value
- **Percentage**: Multiply by (1 + percentage)
- **Use for**: Simulating measurement variations, improving generalization

## Best Practices

1. **Start small**: Use small noise_std and narrow scaling ranges initially
2. **Preserve distribution**: Monitor augmented data to ensure it maintains original characteristics
3. **Validate augmentation**: Check that augmented data makes sense for your domain
4. **Track operations**: Use augmentation history to document what was done
5. **Test model performance**: Compare model performance on original vs augmented data

## Real-World Applications

- Increasing training dataset size
- Improving model robustness
- Simulating measurement variations
- Handling limited data scenarios
- Data augmentation pipelines
- Model generalization improvement

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
