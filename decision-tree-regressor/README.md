# Decision Tree Regressor with Pruning and Feature Importance

A Python implementation of Decision Tree Regressor from scratch with pruning techniques (pre-pruning and post-pruning) and feature importance analysis. This is the thirty-fifth project in the ML learning series, focusing on understanding decision trees for regression, pruning techniques, and feature importance calculation.

## Project Title and Description

The Decision Tree Regressor tool provides a complete implementation of decision trees for regression from scratch, including MSE and MAE splitting criteria, cost-complexity pruning, feature importance calculation, and comprehensive visualization. It helps users understand how decision trees work for regression, how pruning prevents overfitting, and how to interpret feature importance.

This tool solves the problem of learning decision tree regression fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates variance reduction, pruning techniques, feature importance calculation, and tree visualization from scratch.

**Target Audience**: Beginners learning machine learning, students studying regression algorithms, and anyone who needs to understand decision trees for regression, pruning, and feature importance from scratch.

## Features

- Decision tree regressor implementation from scratch
- MSE (Mean Squared Error) splitting criterion
- MAE (Mean Absolute Error) splitting criterion
- Pre-pruning techniques (max_depth, min_samples_split, min_samples_leaf)
- Post-pruning using cost-complexity pruning (CCP)
- Feature importance calculation and visualization
- Recursive tree building algorithm
- Tree depth and node count statistics
- Text-based tree visualization
- Graphical tree visualization
- R-squared, MSE, and MAE evaluation metrics
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
cd /path/to/Python-ml-projects/decision-tree-regressor
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
python src/main.py --input sample.csv --target target_value --criterion mse
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  criterion: "mse"
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  min_impurity_decrease: 0.0
  ccp_alpha: 0.0
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.criterion`: Splitting criterion. Options: "mse", "mae" (default: "mse")
- `model.max_depth`: Maximum depth of tree. If null, nodes expanded until pure (default: null)
- `model.min_samples_split`: Minimum samples required to split node (default: 2)
- `model.min_samples_leaf`: Minimum samples required at leaf node (default: 1)
- `model.min_impurity_decrease`: Minimum impurity decrease for split (default: 0.0)
- `model.ccp_alpha`: Complexity parameter for cost-complexity pruning. 0.0 = no pruning (default: 0.0)
- `model.random_state`: Random seed (default: null)

## Usage

### Basic Usage

```python
from src.main import DecisionTreeRegressor
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

# Initialize and fit model
tree = DecisionTreeRegressor(criterion="mse")
tree.fit(X, y)

# Predict
predictions = tree.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
r2 = tree.score(X, y)
mse = tree.mse(X, y)
mae = tree.mae(X, y)
print(f"R-squared: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
```

### With Different Criteria

```python
from src.main import DecisionTreeRegressor
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

# MSE criterion
tree_mse = DecisionTreeRegressor(criterion="mse")
tree_mse.fit(X, y)

# MAE criterion
tree_mae = DecisionTreeRegressor(criterion="mae")
tree_mae.fit(X, y)
```

### With Pre-Pruning

```python
from src.main import DecisionTreeRegressor
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

# Limit tree depth
tree = DecisionTreeRegressor(max_depth=3, min_samples_split=5, min_samples_leaf=2)
tree.fit(X, y)

print(f"Tree depth: {tree.get_depth()}")
print(f"Number of nodes: {tree.get_n_nodes()}")
```

### With Post-Pruning (Cost-Complexity Pruning)

```python
from src.main import DecisionTreeRegressor
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])

# Apply cost-complexity pruning
tree = DecisionTreeRegressor(ccp_alpha=0.1)
tree.fit(X, y)

print(f"Tree depth after pruning: {tree.get_depth()}")
print(f"Number of nodes after pruning: {tree.get_n_nodes()}")
```

### Feature Importance

```python
from src.main import DecisionTreeRegressor
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

tree = DecisionTreeRegressor()
tree.feature_names_ = ["feature1", "feature2"]
tree.fit(X, y)

# Get feature importance
importances = tree.get_feature_importances()
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.6f}")

# Plot feature importance
tree.plot_feature_importance()
```

### Tree Visualization

```python
from src.main import DecisionTreeRegressor
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

tree = DecisionTreeRegressor()
tree.fit(X, y)

# Print tree structure
tree.print_tree()

# Plot tree visualization
tree.plot_tree()
```

### With Pandas DataFrame

```python
from src.main import DecisionTreeRegressor
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2"]].values
y = df["target"].values

tree = DecisionTreeRegressor(criterion="mse")
tree.feature_names_ = ["feature1", "feature2"]
tree.fit(X, y)

predictions = tree.predict(X)
```

### Command-Line Usage

Basic decision tree regressor:

```bash
python src/main.py --input data.csv --target target_value --criterion mse
```

With MAE criterion:

```bash
python src/main.py --input data.csv --target target_value --criterion mae
```

With max depth (pre-pruning):

```bash
python src/main.py --input data.csv --target target_value --max-depth 5
```

With cost-complexity pruning:

```bash
python src/main.py --input data.csv --target target_value --ccp-alpha 0.1
```

Print tree structure:

```bash
python src/main.py --input data.csv --target target_value --print-tree
```

Plot tree visualization:

```bash
python src/main.py --input data.csv --target target_value --plot-tree
```

Plot feature importance:

```bash
python src/main.py --input data.csv --target target_value --plot-importance
```

Save predictions:

```bash
python src/main.py --input data.csv --target target_value --output predictions.csv
```

Make predictions on new data:

```bash
python src/main.py --input train.csv --target target_value --predict test.csv --output-predictions predictions.csv
```

### Complete Example

```python
from src.main import DecisionTreeRegressor
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.5, 100)

# Fit with pruning
tree = DecisionTreeRegressor(criterion="mse", max_depth=5, ccp_alpha=0.01)
tree.feature_names_ = ["feature1", "feature2", "feature3"]
tree.fit(X, y)

print(f"Tree depth: {tree.get_depth()}")
print(f"Number of nodes: {tree.get_n_nodes()}")

# Evaluate
r2 = tree.score(X, y)
mse = tree.mse(X, y)
mae = tree.mae(X, y)
print(f"R-squared: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

# Feature importance
importances = tree.get_feature_importances()
print("\nFeature Importance:")
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.6f}")

# Visualize
tree.plot_feature_importance()
tree.plot_tree()
```

## Project Structure

```
decision-tree-regressor/
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

- `src/main.py`: Core implementation with `DecisionTreeRegressor` class
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
- Fitting with different criteria
- Prediction
- Evaluation metrics (R-squared, MSE, MAE)
- Feature importance
- Pre-pruning parameters
- Post-pruning (CCP)
- Tree depth and node count
- Error handling
- Different input types
- Tree visualization

## Understanding Decision Tree Regression

### Decision Tree Algorithm for Regression

Decision trees for regression recursively partition data based on feature values:

1. **Start with root node** containing all data
2. **Find best split** that maximizes variance reduction
3. **Create child nodes** based on split
4. **Repeat recursively** for each child node
5. **Stop when** stopping criteria met (pure node, max depth, etc.)
6. **Apply pruning** if specified (post-pruning)

### Splitting Criteria

**MSE (Mean Squared Error):**
```
MSE = (1/n) * Σ(yᵢ - ȳ)²
```

**MAE (Mean Absolute Error):**
```
MAE = (1/n) * Σ|yᵢ - median(y)|
```

**Variance Reduction:**
```
Reduction = Impurity(parent) - Weighted_Impurity(children)
```

### Pruning Techniques

**Pre-Pruning (Early Stopping):**
- `max_depth`: Limit tree depth
- `min_samples_split`: Minimum samples to split
- `min_samples_leaf`: Minimum samples at leaf
- Applied during tree building

**Post-Pruning (Cost-Complexity Pruning):**
- `ccp_alpha`: Complexity parameter
- Prunes nodes that don't improve cost-complexity
- Cost = MSE + α * complexity
- Applied after tree building

### Feature Importance

Feature importance is calculated based on:
- Total variance reduction contributed by each feature
- Weighted by number of samples at each split
- Normalized to sum to 1.0

**Formula:**
```
Importance(feature) = Σ(samples_at_split * variance_reduction) / total_variance_reduction
```

### Choosing Pruning Parameters

**Pre-Pruning:**
- Start with `max_depth=5-10`
- Use `min_samples_split=10-20` for larger datasets
- Use `min_samples_leaf=5-10` to prevent overfitting

**Post-Pruning:**
- Start with `ccp_alpha=0.01`
- Increase for more aggressive pruning
- Use cross-validation to find optimal α

## Troubleshooting

### Common Issues

**Issue**: Tree too deep (overfitting)

**Solution**: 
- Set max_depth parameter
- Increase min_samples_split
- Increase min_samples_leaf
- Use cost-complexity pruning (ccp_alpha)

**Issue**: Tree too shallow (underfitting)

**Solution**: 
- Increase max_depth
- Decrease min_samples_split
- Decrease min_samples_leaf
- Reduce ccp_alpha

**Issue**: Poor R-squared score

**Solution**: 
- Try different criterion (mse vs mae)
- Tune pruning parameters
- Check feature quality
- Consider feature engineering
- Check for outliers

**Issue**: Feature importance doesn't make sense

**Solution**: 
- Ensure features are properly scaled
- Check for correlated features
- Verify data quality
- Use domain knowledge to validate

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: Unknown criterion`: Use "mse" or "mae"
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Choose appropriate criterion**: MSE is more sensitive to outliers, MAE is more robust
2. **Use pre-pruning**: Set max_depth to prevent overfitting
3. **Use post-pruning**: Apply cost-complexity pruning for optimal tree size
4. **Analyze feature importance**: Helps understand which features matter most
5. **Visualize tree**: Helps understand model decisions
6. **Check tree statistics**: Depth and node count indicate complexity
7. **Handle outliers**: Consider robust criteria (MAE) or outlier removal
8. **Cross-validation**: Use CV to tune pruning parameters

## Real-World Applications

- House price prediction
- Sales forecasting
- Temperature prediction
- Stock price prediction
- Quality control
- Educational purposes for learning regression algorithms

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
