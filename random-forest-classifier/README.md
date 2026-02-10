# Random Forest Classifier with Bootstrap Sampling and Feature Importance

A Python implementation of Random Forest Classifier from scratch with bootstrap sampling and feature importance calculation. This is the thirty-sixth project in the ML learning series, focusing on understanding ensemble methods, bootstrap sampling, and feature importance in random forests.

## Project Title and Description

The Random Forest Classifier tool provides a complete implementation of random forests from scratch, including bootstrap sampling, random feature selection, majority voting, and feature importance aggregation. It helps users understand how random forests work, how bootstrap sampling creates diversity, and how feature importance is calculated across multiple trees.

This tool solves the problem of learning ensemble methods fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates bootstrap sampling, random feature selection, tree aggregation, and feature importance calculation from scratch.

**Target Audience**: Beginners learning machine learning, students studying ensemble methods, and anyone who needs to understand random forests, bootstrap sampling, and feature importance from scratch.

## Features

- Random Forest implementation from scratch
- Bootstrap sampling (sampling with replacement)
- Random feature selection for each split
- Multiple decision trees (configurable n_estimators)
- Majority voting for classification
- Class probability prediction
- Feature importance calculation and visualization
- Configurable tree parameters (max_depth, min_samples_split, min_samples_leaf)
- Flexible max_features options (sqrt, log2, int, float)
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas DataFrames)
- Multiclass classification support

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/random-forest-classifier
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
python src/main.py --input sample.csv --target class_label --n-estimators 10
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  n_estimators: 100
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: "sqrt"
  bootstrap: true
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_estimators`: Number of trees in forest (default: 100)
- `model.max_depth`: Maximum depth of trees. If null, nodes expanded until pure (default: null)
- `model.min_samples_split`: Minimum samples required to split node (default: 2)
- `model.min_samples_leaf`: Minimum samples required at leaf node (default: 1)
- `model.max_features`: Maximum features to consider for split. Options: int, "sqrt", "log2", float 0-1 (default: "sqrt")
- `model.bootstrap`: Whether to use bootstrap sampling (default: true)
- `model.random_state`: Random seed (default: null)

## Usage

### Basic Usage

```python
from src.main import RandomForestClassifier
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit model
forest = RandomForestClassifier(n_estimators=10)
forest.fit(X, y)

# Predict
predictions = forest.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
accuracy = forest.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Different Parameters

```python
from src.main import RandomForestClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Custom parameters
forest = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    max_features="sqrt",
    bootstrap=True
)
forest.fit(X, y)
```

### Feature Importance

```python
from src.main import RandomForestClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

forest = RandomForestClassifier(n_estimators=10)
forest.feature_names_ = ["feature1", "feature2"]
forest.fit(X, y)

# Get feature importance
importances = forest.get_feature_importances()
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.6f}")

# Plot feature importance
forest.plot_feature_importance()
```

### Class Probabilities

```python
from src.main import RandomForestClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

forest = RandomForestClassifier(n_estimators=10)
forest.fit(X, y)

# Predict probabilities
probabilities = forest.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")
```

### With Different max_features

```python
from src.main import RandomForestClassifier
import numpy as np

X = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
y = np.array([0, 0, 1, 1])

# sqrt of features
forest_sqrt = RandomForestClassifier(n_estimators=10, max_features="sqrt")
forest_sqrt.fit(X, y)

# log2 of features
forest_log2 = RandomForestClassifier(n_estimators=10, max_features="log2")
forest_log2.fit(X, y)

# Fixed number
forest_int = RandomForestClassifier(n_estimators=10, max_features=2)
forest_int.fit(X, y)

# Fraction
forest_float = RandomForestClassifier(n_estimators=10, max_features=0.5)
forest_float.fit(X, y)
```

### With Pandas DataFrame

```python
from src.main import RandomForestClassifier
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values
y = df["target"].values

forest = RandomForestClassifier(n_estimators=10)
forest.feature_names_ = ["feature1", "feature2", "feature3"]
forest.fit(X, y)

predictions = forest.predict(X)
```

### Command-Line Usage

Basic random forest:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100
```

With custom parameters:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 50 --max-depth 5 --max-features sqrt
```

Without bootstrap sampling:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100 --no-bootstrap
```

Plot feature importance:

```bash
python src/main.py --input data.csv --target class_label --plot-importance
```

Save predictions:

```bash
python src/main.py --input data.csv --target class_label --output predictions.csv
```

Make predictions on new data:

```bash
python src/main.py --input train.csv --target class_label --predict test.csv --output-predictions predictions.csv
```

### Complete Example

```python
from src.main import RandomForestClassifier
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)

# Fit random forest
forest = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    max_features="sqrt",
    bootstrap=True
)
forest.feature_names_ = [f"feature_{i}" for i in range(5)]
forest.fit(X, y)

print(f"Number of trees: {len(forest.estimators_)}")
print(f"Classes: {forest.classes_}")

# Evaluate
accuracy = forest.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
importances = forest.get_feature_importances()
print("\nFeature Importance:")
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.6f}")

# Visualize
forest.plot_feature_importance()
```

## Project Structure

```
random-forest-classifier/
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

- `src/main.py`: Core implementation with `RandomForestClassifier` class
- `config.yaml`: Configuration file for model settings
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
- Model initialization
- Fitting with different parameters
- Bootstrap sampling
- Prediction
- Class probabilities
- Feature importance
- Different max_features options
- Error handling
- Different input types
- Multiclass classification
- Feature importance visualization

## Understanding Random Forest

### Random Forest Algorithm

Random Forest is an ensemble method that combines multiple decision trees:

1. **Bootstrap Sampling**: Create multiple bootstrap samples from training data
2. **Build Trees**: Train decision tree on each bootstrap sample
3. **Random Features**: At each split, consider only random subset of features
4. **Aggregate**: Combine predictions using majority voting

### Bootstrap Sampling

**Process:**
- Sample n samples with replacement from n training samples
- Each tree trained on different bootstrap sample
- Creates diversity among trees
- About 63% of samples appear in each bootstrap sample

**Benefits:**
- Reduces overfitting
- Creates diverse trees
- Improves generalization

### Random Feature Selection

**max_features Options:**
- `"sqrt"`: sqrt(n_features) features per split
- `"log2"`: log2(n_features) features per split
- `int`: Exact number of features
- `float`: Fraction of features (0-1)

**Benefits:**
- Reduces correlation between trees
- Improves diversity
- Prevents overfitting

### Majority Voting

**Classification:**
- Each tree votes for a class
- Final prediction is majority vote
- Probabilities = fraction of trees voting for each class

### Feature Importance

Feature importance calculated by:
1. For each tree, calculate feature importance based on information gain
2. Average importance across all trees
3. Normalize to sum to 1.0

**Formula:**
```
Importance(feature) = (1/n_trees) * Σ(tree_importance(feature))
```

### Advantages of Random Forest

1. **Reduces Overfitting**: Multiple trees average out errors
2. **Handles Non-linearity**: Can capture complex patterns
3. **Feature Importance**: Provides interpretability
4. **Robust**: Less sensitive to outliers
5. **No Feature Scaling**: Works well without scaling
6. **Handles Missing Values**: Can work with missing data (with modifications)

## Troubleshooting

### Common Issues

**Issue**: Low accuracy

**Solution**: 
- Increase n_estimators
- Tune max_depth
- Adjust max_features
- Check data quality
- Ensure balanced classes

**Issue**: Slow training

**Solution**: 
- Reduce n_estimators
- Set max_depth
- Use fewer features
- Reduce dataset size for testing

**Issue**: Overfitting

**Solution**: 
- Reduce max_depth
- Increase min_samples_split
- Increase min_samples_leaf
- Reduce n_estimators
- Use more regularization

**Issue**: Feature importance doesn't make sense

**Solution**: 
- Increase n_estimators for stability
- Check for correlated features
- Verify data quality
- Use domain knowledge to validate

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Choose n_estimators**: Start with 100, increase if needed
2. **Use bootstrap**: Generally improves performance
3. **Tune max_features**: sqrt or log2 are good defaults
4. **Control tree depth**: Set max_depth to prevent overfitting
5. **Analyze feature importance**: Helps understand which features matter
6. **Use cross-validation**: Tune hyperparameters with CV
7. **Handle imbalanced classes**: Consider class weights or resampling
8. **Monitor training time**: Balance accuracy and speed

## Real-World Applications

- Medical diagnosis
- Credit risk assessment
- Customer segmentation
- Fraud detection
- Image classification
- Text classification
- Educational purposes for learning ensemble methods

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
