# Cross-Validation Tool

A Python tool for performing cross-validation with k-fold, stratified k-fold, and leave-one-out strategies. This is the twentieth project in the ML learning series, focusing on understanding and implementing cross-validation techniques for model evaluation.

## Project Title and Description

The Cross-Validation Tool provides automated cross-validation splitting with multiple strategies. It helps users understand different cross-validation approaches and when to use each one, which is essential for reliable model evaluation and avoiding overfitting.

This tool solves the problem of properly evaluating machine learning models by providing standardized cross-validation implementations. It supports k-fold, stratified k-fold, and leave-one-out cross-validation strategies.

**Target Audience**: Beginners learning machine learning, data scientists evaluating models, and anyone who needs to understand or implement cross-validation from scratch.

## Features

- K-fold cross-validation
- Stratified k-fold cross-validation (maintains class distribution)
- Leave-one-out cross-validation
- Configurable number of folds
- Shuffle option for data randomization
- Reproducible splits with random seed
- Split summary statistics
- Class distribution analysis (for stratified splits)
- Save splits to JSON
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas DataFrames)

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/cross-validation
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
python src/main.py --input sample.csv --strategy kfold --n-splits 5
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

cross_validation:
  n_splits: 5
  random_state: 42
  shuffle: true
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `cross_validation.n_splits`: Default number of folds (default: 5)
- `cross_validation.random_state`: Default random seed (default: 42)
- `cross_validation.shuffle`: Whether to shuffle data by default (default: true)

## Usage

### Basic Usage

```python
from src.main import CrossValidator

cv = CrossValidator()

# K-fold cross-validation
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
splits = cv.k_fold_split(X, n_splits=5)

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X[train_idx], X[test_idx]
    print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
```

### Stratified K-Fold

```python
from src.main import CrossValidator

cv = CrossValidator()

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# Stratified k-fold maintains class distribution
splits = cv.stratified_k_fold_split(X, y, n_splits=5)

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
```

### Leave-One-Out

```python
from src.main import CrossValidator

cv = CrossValidator()

X = [[1], [2], [3], [4], [5]]

# Leave-one-out creates n splits (one for each sample)
splits = cv.leave_one_out_split(X)

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X[train_idx], X[test_idx]
    print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
```

### Get Summary Statistics

```python
from src.main import CrossValidator

cv = CrossValidator()

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

splits = cv.stratified_k_fold_split(X, y, n_splits=5)

# Get summary
summary = cv.get_split_summary(splits, y)
print(f"Number of folds: {summary['n_folds']}")

# Print formatted summary
cv.print_summary(splits, y, strategy="Stratified K-Fold")
```

### Save Splits

```python
from src.main import CrossValidator

cv = CrossValidator()

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

splits = cv.k_fold_split(X, n_splits=5)

# Save splits to JSON
cv.save_splits(splits, y, output_path="cv_splits.json")
```

### Command-Line Usage

K-fold cross-validation:

```bash
python src/main.py --input data.csv --strategy kfold --n-splits 5
```

Stratified k-fold cross-validation:

```bash
python src/main.py --input data.csv --target label --strategy stratified --n-splits 5
```

Leave-one-out cross-validation:

```bash
python src/main.py --input data.csv --strategy loo
```

With shuffle:

```bash
python src/main.py --input data.csv --strategy kfold --n-splits 5 --shuffle
```

With custom random state:

```bash
python src/main.py --input data.csv --strategy kfold --n-splits 5 --random-state 123
```

Save splits to JSON:

```bash
python src/main.py --input data.csv --strategy kfold --n-splits 5 --save splits.json
```

### Complete Example

```python
from src.main import CrossValidator
import numpy as np
import pandas as pd

# Initialize cross-validator
cv = CrossValidator()

# Load or create data
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

X = df[["feature1", "feature2"]]
y = df["target"]

# Perform stratified k-fold cross-validation
splits = cv.stratified_k_fold_split(
    X, y, n_splits=5, shuffle=True, random_state=42
)

# Print summary
cv.print_summary(splits, y, strategy="Stratified K-Fold")

# Use splits for model evaluation
for fold_idx, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"\nFold {fold_idx + 1}:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train and evaluate model here
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)

# Save splits
cv.save_splits(splits, y, output_path="cv_splits.json")
```

## Project Structure

```
cross-validation/
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

- `src/main.py`: Core implementation with `CrossValidator` class
- `config.yaml`: Configuration file for logging and cross-validation settings
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
- K-fold cross-validation
- Stratified k-fold cross-validation
- Leave-one-out cross-validation
- Input validation
- Error handling
- Different input types (lists, numpy arrays, pandas DataFrames)
- Reproducibility
- Class distribution maintenance

## Understanding Cross-Validation Strategies

### K-Fold Cross-Validation

Divides data into k folds. Each fold is used once as test set while the remaining k-1 folds are used for training.

**When to use:**
- General purpose cross-validation
- Balanced datasets
- Sufficient data (typically k=5 or k=10)

**Advantages:**
- Simple and widely used
- Good balance between bias and variance
- Computationally efficient

**Disadvantages:**
- May not work well with imbalanced datasets
- Class distribution may vary across folds

### Stratified K-Fold Cross-Validation

Similar to k-fold but maintains class distribution in each fold.

**When to use:**
- Classification tasks
- Imbalanced datasets
- When class distribution is important

**Advantages:**
- Maintains class distribution
- Better for imbalanced datasets
- More reliable estimates for classification

**Disadvantages:**
- Requires target labels
- Slightly more complex implementation

### Leave-One-Out Cross-Validation

Creates n splits where each split uses one sample as test and the rest as training.

**When to use:**
- Small datasets
- When maximum data usage is needed
- When computational cost is not a concern

**Advantages:**
- Uses maximum data for training
- Unbiased estimate
- No randomness

**Disadvantages:**
- Computationally expensive (n models)
- High variance in estimates
- May be slow for large datasets

## Troubleshooting

### Common Issues

**Issue**: Stratified k-fold requires target column

**Solution**: Provide target column using `--target` flag or pass `y` parameter in code.

**Issue**: n_splits too large

**Solution**: Ensure n_splits is less than number of samples (and less than samples per class for stratified).

**Issue**: Class distribution not maintained

**Solution**: Use stratified k-fold instead of regular k-fold for classification tasks.

### Error Messages

- `ValueError: n_splits must be at least 2`: n_splits must be 2 or greater
- `ValueError: n_splits cannot be greater than number of samples`: Too many folds requested
- `ValueError: y is required for stratified k-fold`: Target labels required for stratified splits
- `ValueError: Length mismatch`: X and y have different lengths

## Best Practices

1. **Use stratified k-fold for classification**: Maintains class distribution
2. **Use k-fold for regression**: Stratification not applicable
3. **Use leave-one-out for small datasets**: Maximum data usage
4. **Set random_state for reproducibility**: Ensures consistent splits
5. **Shuffle data when appropriate**: Reduces order bias
6. **Choose appropriate k**: k=5 or k=10 are common choices

## Real-World Applications

- Model evaluation and selection
- Hyperparameter tuning
- Feature selection
- Model comparison
- Performance estimation
- Avoiding overfitting
- Educational purposes for understanding cross-validation

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
