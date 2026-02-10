# Time Series Cross-Validation

A Python implementation of time series cross-validation techniques including walk-forward validation and expanding window strategies for proper evaluation of time series models while respecting temporal order.

## Description

This project provides robust time series cross-validation methods that are essential for evaluating time series models. Unlike standard cross-validation, time series cross-validation respects the temporal order of data, preventing data leakage and providing realistic performance estimates.

The implementation includes two main strategies:
- **Walk-Forward Validation**: Creates multiple train-test splits by moving forward in time
- **Expanding Window**: Uses an expanding training window with fixed-size test sets

## Features

- **Walk-Forward Validation**: Multiple train-test splits with configurable test size and gap
- **Expanding Window Strategy**: Growing training window with step-by-step forward movement
- **Flexible Configuration**: YAML-based configuration for easy parameter tuning
- **Multiple Scoring Metrics**: Support for MSE, RMSE, MAE, and R²
- **Comprehensive Evaluation**: Detailed split information and statistics
- **Scikit-learn Compatible**: Works with any scikit-learn estimator
- **Pandas Integration**: Supports both NumPy arrays and pandas DataFrames
- **Command-Line Interface**: Easy-to-use CLI for batch processing
- **Detailed Logging**: Comprehensive logging for debugging and monitoring
- **Input Validation**: Robust error handling and validation

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone or navigate to the project directory:
```bash
cd time-series-cross-validation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a YAML configuration file (`config.yaml`) for default settings:

```yaml
logging:
  level: INFO
  file: logs/app.log

cross_validation:
  strategy: walk_forward
  n_splits: 5
  test_size: null
  gap: 0
  initial_train_size: null
  step_size: 1
  max_train_size: null
  scoring: mse
  return_train_score: false
```

### Configuration Parameters

- **strategy**: Validation strategy - `walk_forward` or `expanding_window`
- **n_splits**: Number of splits for walk-forward validation
- **test_size**: Size of test set for each split (auto-calculated if None)
- **gap**: Number of samples to skip between train and test sets
- **initial_train_size**: Initial training size for expanding window
- **step_size**: Step size for expanding window (number of samples to move forward)
- **max_train_size**: Maximum size of training window
- **scoring**: Scoring metric - `mse`, `rmse`, `mae`, or `r2`
- **return_train_score**: Whether to return training scores

## Usage

### Command-Line Interface

Basic usage with walk-forward validation:
```bash
python src/main.py \
  --input data.csv \
  --target-col target \
  --model model.pkl \
  --strategy walk_forward \
  --n-splits 5 \
  --output results.json
```

Using expanding window strategy:
```bash
python src/main.py \
  --input data.csv \
  --target-col target \
  --model model.pkl \
  --strategy expanding_window \
  --initial-train-size 50 \
  --step-size 10 \
  --output results.json
```

With custom scoring metric:
```bash
python src/main.py \
  --input data.csv \
  --target-col target \
  --model model.pkl \
  --strategy walk_forward \
  --scoring r2 \
  --output results.json
```

### Programmatic Usage

#### Walk-Forward Validation

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from src.main import TimeSeriesCrossValidator

# Prepare data
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Initialize model
model = LinearRegression()

# Create validator
validator = TimeSeriesCrossValidator(
    strategy="walk_forward",
    n_splits=5,
    test_size=10,
    gap=0
)

# Perform cross-validation
results = validator.cross_validate(
    model, X, y, scoring="mse", return_train_score=True
)

print(f"Mean Test Score: {results['test_mean']:.4f}")
print(f"Std Test Score: {results['test_std']:.4f}")
```

#### Expanding Window Validation

```python
from src.main import TimeSeriesCrossValidator

validator = TimeSeriesCrossValidator(
    strategy="expanding_window",
    initial_train_size=30,
    step_size=10,
    n_splits=5
)

results = validator.cross_validate(
    model, X, y, scoring="r2", return_train_score=True
)
```

#### Using TimeSeriesValidator (with config)

```python
from pathlib import Path
from src.main import TimeSeriesValidator

validator = TimeSeriesValidator(config_path=Path("config.yaml"))
results = validator.validate(
    model, X, y, strategy="walk_forward", n_splits=5
)
```

#### With Pandas DataFrames

```python
import pandas as pd
from src.main import TimeSeriesValidator

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

validator = TimeSeriesValidator()
results = validator.validate(model, X, y, strategy="walk_forward")
```

## Project Structure

```
time-series-cross-validation/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── src/
│   └── main.py
├── tests/
│   └── test_main.py
├── docs/
│   └── API.md (if applicable)
└── logs/
    └── .gitkeep
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Walk-Forward Validation

Walk-forward validation creates multiple train-test splits by moving forward in time:

1. **Split 1**: Train on [0:60], Test on [60:70]
2. **Split 2**: Train on [0:70], Test on [70:80]
3. **Split 3**: Train on [0:80], Test on [80:90]
4. **Split 4**: Train on [0:90], Test on [90:100]
5. **Split 5**: Train on [0:100], Test on [100:110]

This approach ensures that:
- Training data always comes before test data
- Each split uses more training data than the previous
- Test sets are non-overlapping

## Expanding Window Strategy

Expanding window validation uses a growing training window:

1. **Split 1**: Train on [0:30], Test on [30:40]
2. **Split 2**: Train on [0:40], Test on [40:50]
3. **Split 3**: Train on [0:50], Test on [50:60]
4. **Split 4**: Train on [0:60], Test on [60:70]
5. **Split 5**: Train on [0:70], Test on [70:80]

This approach:
- Gradually increases training data size
- Maintains fixed test set size
- Provides insights into model performance as more data becomes available

## Scoring Metrics

The implementation supports multiple scoring metrics:

- **MSE (Mean Squared Error)**: Average squared differences
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **MAE (Mean Absolute Error)**: Average absolute differences
- **R² (Coefficient of Determination)**: Proportion of variance explained

## Output Format

The cross-validation results are returned as a dictionary:

```python
{
    "strategy": "walk_forward",
    "n_splits": 5,
    "test_scores": [0.123, 0.145, 0.132, 0.138, 0.141],
    "test_mean": 0.1358,
    "test_std": 0.0082,
    "split_info": [
        {
            "fold": 1,
            "train_size": 60,
            "test_size": 10,
            "train_start": 0,
            "train_end": 60,
            "test_start": 60,
            "test_end": 70
        },
        ...
    ]
}
```

## Troubleshooting

### Common Issues

1. **"Cannot have n_splits > n_samples"**
   - Reduce `n_splits` or increase dataset size
   - For expanding window, reduce `initial_train_size`

2. **"test_size and n_splits too large"**
   - Reduce `test_size` or `n_splits`
   - Increase dataset size

3. **"initial_train_size >= n_samples"**
   - Reduce `initial_train_size` for expanding window
   - Ensure dataset is large enough

4. **"X and y must have same number of samples"**
   - Check that X and y have matching lengths
   - Ensure no missing values or filtering issues

### Performance Tips

- Use `max_train_size` to limit training window size for faster computation
- Set `gap` to simulate realistic prediction delays
- Use `step_size` > 1 for expanding window to reduce computation time
- Consider using simpler models for initial validation

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write docstrings for all public functions and classes
4. Add tests for new features
5. Update README.md if adding new features

## License

This project is part of the Python ML Projects collection.
