# XGBoost Classifier with Tree Construction, Regularization, and Early Stopping

A Python implementation of XGBoost (Extreme Gradient Boosting) Classifier from scratch with tree construction, regularization (L1/L2), and early stopping. This is the thirty-ninth project in the ML learning series, focusing on understanding XGBoost, second-order gradients, regularization, and early stopping.

## Project Title and Description

The XGBoost Classifier tool provides a complete implementation of XGBoost from scratch, including second-order gradient optimization, gain-based tree construction, L1/L2 regularization, and early stopping with validation sets. It helps users understand how XGBoost works, how it differs from regular gradient boosting, and how regularization and early stopping prevent overfitting.

This tool solves the problem of learning advanced gradient boosting fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates second-order gradients (hessians), gain calculation with regularization, tree construction, and early stopping from scratch.

**Target Audience**: Beginners learning machine learning, students studying advanced ensemble methods, and anyone who needs to understand XGBoost, regularization, and early stopping from scratch.

## Features

- XGBoost implementation from scratch
- Second-order gradient optimization (gradients and hessians)
- Gain-based tree construction with regularization
- L1 regularization (reg_alpha)
- L2 regularization (reg_lambda)
- Early stopping with validation set
- Subsampling support (stochastic XGBoost)
- Feature importance calculation and visualization
- Class probability prediction
- Configurable tree parameters (max_depth, min_child_weight, gamma)
- Binary classification support
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
cd /path/to/Python-ml-projects/xgboost-classifier
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
  learning_rate: 0.1
  max_depth: 6
  min_child_weight: 1.0
  gamma: 0.0
  reg_lambda: 1.0
  reg_alpha: 0.0
  subsample: 1.0
  early_stopping_rounds: null
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_estimators`: Number of boosting rounds (trees) (default: 100)
- `model.learning_rate`: Learning rate (shrinkage) (default: 0.1)
- `model.max_depth`: Maximum depth of trees (default: 6)
- `model.min_child_weight`: Minimum sum of instance weight in child (default: 1.0)
- `model.gamma`: Minimum loss reduction for split (default: 0.0)
- `model.reg_lambda`: L2 regularization (default: 1.0)
- `model.reg_alpha`: L1 regularization (default: 0.0)
- `model.subsample`: Fraction of samples to use for each tree, 0-1 (default: 1.0)
- `model.early_stopping_rounds`: Early stopping rounds. If null, no early stopping (default: null)
- `model.random_state`: Random seed (default: null)

## Usage

### Basic Usage

```python
from src.main import XGBoostClassifier
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit model
xgb = XGBoostClassifier(n_estimators=10, learning_rate=0.1)
xgb.fit(X, y)

# Predict
predictions = xgb.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
accuracy = xgb.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Regularization

```python
from src.main import XGBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# L2 regularization
xgb_l2 = XGBoostClassifier(n_estimators=50, reg_lambda=2.0)
xgb_l2.fit(X, y)

# L1 regularization
xgb_l1 = XGBoostClassifier(n_estimators=50, reg_alpha=0.5)
xgb_l1.fit(X, y)

# Both L1 and L2
xgb_both = XGBoostClassifier(n_estimators=50, reg_lambda=1.0, reg_alpha=0.1)
xgb_both.fit(X, y)
```

### With Early Stopping

```python
from src.main import XGBoostClassifier
import numpy as np

X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_val = np.array([[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]])
y_val = np.array([0, 0, 1])

xgb = XGBoostClassifier(
    n_estimators=100,
    early_stopping_rounds=5
)
xgb.fit(X_train, y_train, eval_set=(X_val, y_val))

print(f"Best iteration: {xgb.best_iteration_ + 1}")
print(f"Trees fitted: {len(xgb.estimators_)}")
```

### With Subsampling

```python
from src.main import XGBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Use 80% of samples for each tree
xgb = XGBoostClassifier(n_estimators=50, subsample=0.8)
xgb.fit(X, y)
```

### Feature Importance

```python
from src.main import XGBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

xgb = XGBoostClassifier(n_estimators=50)
xgb.feature_names_ = ["feature1", "feature2"]
xgb.fit(X, y)

# Get feature importance
importances = xgb.get_feature_importances()
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.6f}")

# Plot feature importance
xgb.plot_feature_importance()
```

### Class Probabilities

```python
from src.main import XGBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

xgb = XGBoostClassifier(n_estimators=50)
xgb.fit(X, y)

# Predict probabilities
probabilities = xgb.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")
```

### With Pandas DataFrame

```python
from src.main import XGBoostClassifier
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values
y = df["target"].values

xgb = XGBoostClassifier(n_estimators=50, learning_rate=0.1)
xgb.feature_names_ = ["feature1", "feature2", "feature3"]
xgb.fit(X, y)

predictions = xgb.predict(X)
```

### Command-Line Usage

Basic XGBoost:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100
```

With regularization:

```bash
python src/main.py --input data.csv --target class_label --reg-lambda 2.0 --reg-alpha 0.5
```

With early stopping:

```bash
python src/main.py --input train.csv --target class_label --eval-set val.csv --early-stopping-rounds 10
```

With custom parameters:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100 --max-depth 5 --learning-rate 0.05 --subsample 0.8
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
from src.main import XGBoostClassifier
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)

# Split for early stopping
X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]

# Fit with regularization and early stopping
xgb = XGBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    reg_lambda=1.0,
    reg_alpha=0.1,
    subsample=0.8,
    early_stopping_rounds=10
)
xgb.feature_names_ = [f"feature_{i}" for i in range(5)]
xgb.fit(X_train, y_train, eval_set=(X_val, y_val))

print(f"Number of trees fitted: {len(xgb.estimators_)}")
if xgb.best_iteration_ is not None:
    print(f"Best iteration: {xgb.best_iteration_ + 1}")
print(f"Classes: {xgb.classes_}")

# Evaluate
accuracy = xgb.score(X_train, y_train)
val_accuracy = xgb.score(X_val, y_val)
print(f"Training accuracy: {accuracy:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")

# Feature importance
importances = xgb.get_feature_importances()
print("\nFeature Importance:")
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.6f}")

# Visualize
xgb.plot_feature_importance()
```

## Project Structure

```
xgboost-classifier/
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

- `src/main.py`: Core implementation with `XGBoostClassifier` and `XGBoostTree` classes
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
- Tree construction
- Regularization (L1/L2)
- Early stopping
- Gradient and hessian calculation
- Prediction
- Class probabilities
- Feature importance
- Error handling
- Different input types
- Feature importance visualization

## Understanding XGBoost

### XGBoost Algorithm

XGBoost is an advanced gradient boosting algorithm that uses second-order gradients:

1. **Initialize**: Start with initial prediction (log odds)
2. **For each tree**:
   - Calculate gradients and hessians
   - Build tree using gain-based splitting with regularization
   - Update predictions: `predictions += learning_rate * tree_predictions`
3. **Convert**: Transform final predictions to probabilities using sigmoid

### Mathematical Foundation

**Gradients (First Derivatives):**
```
gᵢ = ∂L/∂F(xᵢ) = pᵢ - yᵢ
```

**Hessians (Second Derivatives):**
```
hᵢ = ∂²L/∂F(xᵢ)² = pᵢ(1 - pᵢ)
```

**Gain Calculation:**
```
Gain = (GL²/(HL + λ)) + (GR²/(HR + λ)) - ((GL + GR)²/(HL + HR + λ)) - γ
```

Where:
- GL, GR: Sum of gradients in left/right child
- HL, HR: Sum of hessians in left/right child
- λ: L2 regularization (reg_lambda)
- γ: Minimum loss reduction (gamma)

**Leaf Value:**
```
value = -G / (H + λ)
```

Where:
- G: Sum of gradients
- H: Sum of hessians
- λ: L2 regularization

**L1 Regularization:**
```
value = sign(-G/(H+λ)) * max(0, |G/(H+λ)| - α)
```

Where α is L1 regularization (reg_alpha).

### Key Differences from Gradient Boosting

1. **Second-Order Gradients**: Uses hessians in addition to gradients
2. **Regularization**: Built-in L1 and L2 regularization
3. **Gain-Based Splitting**: Uses gain calculation with regularization
4. **Early Stopping**: Built-in early stopping with validation set
5. **Subsampling**: Supports stochastic boosting

### Regularization

**L2 Regularization (reg_lambda):**
- Penalizes large leaf values
- Prevents overfitting
- Typical range: 0.1-10.0

**L1 Regularization (reg_alpha):**
- Can set leaf values to exactly zero
- Feature selection effect
- Typical range: 0.0-1.0

### Early Stopping

**Process:**
1. Monitor validation loss after each iteration
2. Track best iteration
3. Stop if no improvement for N rounds
4. Use best iteration for prediction

**Benefits:**
- Prevents overfitting
- Saves computation
- Finds optimal number of trees

### Hyperparameter Tuning

**Learning Rate:**
- Lower (0.01-0.1): More conservative, needs more trees
- Higher (0.1-0.3): Faster convergence, risk of overfitting

**Max Depth:**
- Shallow (3-5): Less overfitting, faster
- Deep (6-10): More complex, risk of overfitting

**Regularization:**
- Higher reg_lambda: More regularization
- Higher reg_alpha: More L1 regularization

## Troubleshooting

### Common Issues

**Issue**: Overfitting

**Solution**: 
- Increase reg_lambda
- Increase reg_alpha
- Reduce max_depth
- Reduce learning_rate
- Use early stopping
- Use subsampling

**Issue**: Underfitting

**Solution**: 
- Decrease reg_lambda
- Increase max_depth
- Increase learning_rate
- Increase n_estimators

**Issue**: Slow training

**Solution**: 
- Reduce n_estimators
- Reduce max_depth
- Use early stopping
- Use subsampling
- Reduce dataset size

**Issue**: Early stopping too early

**Solution**: 
- Increase early_stopping_rounds
- Check validation set quality
- Reduce learning_rate

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: XGBoost currently supports binary classification only`: Use binary classification data
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Start with defaults**: n_estimators=100, learning_rate=0.1, max_depth=6
2. **Use early stopping**: Prevents overfitting and finds optimal trees
3. **Tune regularization**: Start with reg_lambda=1.0, adjust as needed
4. **Control depth**: Shallow trees (3-6) work well
5. **Use subsampling**: 0.8-0.9 for regularization
6. **Monitor validation**: Use separate validation set
7. **Analyze feature importance**: Understand which features matter
8. **Cross-validation**: Use CV to tune hyperparameters

## Real-World Applications

- Click-through rate prediction
- Fraud detection
- Medical diagnosis
- Credit risk assessment
- Recommendation systems
- Educational purposes for learning advanced ensemble methods

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
