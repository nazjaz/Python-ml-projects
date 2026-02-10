# Gradient Boosting Classifier with Learning Rate and Optimization

A Python implementation of Gradient Boosting Classifier from scratch with learning rate, depth, and tree count optimization. This is the thirty-seventh project in the ML learning series, focusing on understanding gradient boosting, sequential tree building, and hyperparameter optimization.

## Project Title and Description

The Gradient Boosting Classifier tool provides a complete implementation of gradient boosting from scratch, including sequential tree building, learning rate control, depth optimization, and tree count tuning. It helps users understand how gradient boosting works, how trees are built sequentially to correct errors, and how hyperparameters affect model performance.

This tool solves the problem of learning ensemble boosting methods fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates gradient descent, log loss minimization, sequential tree building, and hyperparameter optimization from scratch.

**Target Audience**: Beginners learning machine learning, students studying ensemble methods, and anyone who needs to understand gradient boosting, sequential learning, and hyperparameter tuning from scratch.

## Features

- Gradient boosting implementation from scratch
- Sequential tree building (each tree corrects previous errors)
- Learning rate (shrinkage) control
- Configurable tree depth (max_depth)
- Configurable number of trees (n_estimators)
- Log loss minimization
- Gradient calculation (negative gradient of log loss)
- Subsampling support (stochastic gradient boosting)
- Feature importance calculation and visualization
- Class probability prediction
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
cd /path/to/Python-ml-projects/gradient-boosting-classifier
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
  max_depth: 3
  min_samples_split: 2
  min_samples_leaf: 1
  subsample: 1.0
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_estimators`: Number of boosting stages (trees) (default: 100)
- `model.learning_rate`: Learning rate (shrinkage) (default: 0.1)
- `model.max_depth`: Maximum depth of trees (default: 3)
- `model.min_samples_split`: Minimum samples required to split node (default: 2)
- `model.min_samples_leaf`: Minimum samples required at leaf node (default: 1)
- `model.subsample`: Fraction of samples to use for each tree, 0-1 (default: 1.0)
- `model.random_state`: Random seed (default: null)

## Usage

### Basic Usage

```python
from src.main import GradientBoostingClassifier
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit model
gb = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)
gb.fit(X, y)

# Predict
predictions = gb.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
accuracy = gb.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Different Learning Rates

```python
from src.main import GradientBoostingClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Low learning rate (more trees needed)
gb_low = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01)
gb_low.fit(X, y)

# High learning rate (fewer trees needed, risk of overfitting)
gb_high = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5)
gb_high.fit(X, y)
```

### With Depth Optimization

```python
from src.main import GradientBoostingClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Shallow trees (less overfitting)
gb_shallow = GradientBoostingClassifier(n_estimators=50, max_depth=2)
gb_shallow.fit(X, y)

# Deeper trees (more complex, risk of overfitting)
gb_deep = GradientBoostingClassifier(n_estimators=50, max_depth=5)
gb_deep.fit(X, y)
```

### With Tree Count Optimization

```python
from src.main import GradientBoostingClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Few trees (faster, may underfit)
gb_few = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)
gb_few.fit(X, y)

# Many trees (slower, better accuracy, risk of overfitting)
gb_many = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
gb_many.fit(X, y)
```

### With Subsampling (Stochastic Gradient Boosting)

```python
from src.main import GradientBoostingClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Use 80% of samples for each tree
gb = GradientBoostingClassifier(n_estimators=50, subsample=0.8)
gb.fit(X, y)
```

### Feature Importance

```python
from src.main import GradientBoostingClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

gb = GradientBoostingClassifier(n_estimators=50)
gb.feature_names_ = ["feature1", "feature2"]
gb.fit(X, y)

# Get feature importance
importances = gb.get_feature_importances()
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.6f}")

# Plot feature importance
gb.plot_feature_importance()
```

### Class Probabilities

```python
from src.main import GradientBoostingClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

gb = GradientBoostingClassifier(n_estimators=50)
gb.fit(X, y)

# Predict probabilities
probabilities = gb.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")
```

### With Pandas DataFrame

```python
from src.main import GradientBoostingClassifier
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values
y = df["target"].values

gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
gb.feature_names_ = ["feature1", "feature2", "feature3"]
gb.fit(X, y)

predictions = gb.predict(X)
```

### Command-Line Usage

Basic gradient boosting:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100
```

With custom learning rate:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100 --learning-rate 0.05
```

With depth optimization:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100 --max-depth 5
```

With subsampling:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 100 --subsample 0.8
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
from src.main import GradientBoostingClassifier
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)

# Fit with optimized parameters
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8
)
gb.feature_names_ = [f"feature_{i}" for i in range(5)]
gb.fit(X, y)

print(f"Number of trees: {len(gb.estimators_)}")
print(f"Classes: {gb.classes_}")

# Evaluate
accuracy = gb.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
importances = gb.get_feature_importances()
print("\nFeature Importance:")
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.6f}")

# Visualize
gb.plot_feature_importance()
```

## Project Structure

```
gradient-boosting-classifier/
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

- `src/main.py`: Core implementation with `GradientBoostingClassifier` class
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
- Learning rate variations
- Depth variations
- Tree count variations
- Subsampling
- Prediction
- Class probabilities
- Feature importance
- Error handling
- Different input types

## Understanding Gradient Boosting

### Gradient Boosting Algorithm

Gradient boosting builds trees sequentially, where each tree corrects errors of previous trees:

1. **Initialize**: Start with initial prediction (log odds)
2. **For each tree**:
   - Calculate negative gradient (residuals) of loss function
   - Fit tree to residuals
   - Update predictions: `predictions += learning_rate * tree_predictions`
3. **Convert**: Transform final predictions to probabilities using sigmoid

### Mathematical Foundation

**Initial Prediction:**
```
F₀(x) = log(p / (1 - p))
```

Where `p` is proportion of positive class.

**Loss Function (Log Loss):**
```
L(y, F(x)) = -y*log(σ(F(x))) - (1-y)*log(1 - σ(F(x)))
```

Where `σ` is sigmoid function.

**Negative Gradient (Residuals):**
```
rᵢ = yᵢ - σ(F(xᵢ))
```

**Update Rule:**
```
Fₘ(x) = Fₘ₋₁(x) + α * hₘ(x)
```

Where:
- `α` is learning rate
- `hₘ(x)` is prediction from m-th tree

### Learning Rate

**Low Learning Rate (e.g., 0.01-0.1):**
- More conservative updates
- Requires more trees
- Less prone to overfitting
- Better generalization

**High Learning Rate (e.g., 0.5-1.0):**
- More aggressive updates
- Requires fewer trees
- Higher risk of overfitting
- Faster convergence

**Trade-off:**
- Lower learning rate + more trees = better generalization
- Higher learning rate + fewer trees = faster but riskier

### Depth Optimization

**Shallow Trees (depth 1-3):**
- Less complex
- Less prone to overfitting
- Good for high-dimensional data
- Default: depth 3

**Deep Trees (depth 4-8):**
- More complex
- Can capture interactions
- Higher risk of overfitting
- Use with regularization

### Tree Count Optimization

**Few Trees (10-50):**
- Faster training
- May underfit
- Good for simple problems

**Many Trees (100-500):**
- Better accuracy
- Slower training
- Risk of overfitting
- Use with early stopping

**Optimal Strategy:**
- Start with moderate count (100)
- Use validation set to find optimal
- Consider early stopping

### Subsampling (Stochastic Gradient Boosting)

**Benefits:**
- Reduces overfitting
- Increases diversity
- Improves generalization
- Faster training

**Typical Values:**
- 0.8-1.0: Common range
- 0.5-0.8: More regularization
- 1.0: No subsampling (default)

## Troubleshooting

### Common Issues

**Issue**: Overfitting

**Solution**: 
- Reduce learning rate
- Reduce max_depth
- Increase min_samples_split
- Increase min_samples_leaf
- Use subsampling
- Reduce n_estimators

**Issue**: Underfitting

**Solution**: 
- Increase learning rate
- Increase max_depth
- Increase n_estimators
- Reduce regularization

**Issue**: Slow training

**Solution**: 
- Reduce n_estimators
- Reduce max_depth
- Use subsampling
- Reduce dataset size

**Issue**: Poor accuracy

**Solution**: 
- Tune learning rate
- Optimize tree count
- Adjust depth
- Check data quality
- Feature engineering

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: Gradient boosting currently supports binary classification only`: Use binary classification data
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Start with defaults**: n_estimators=100, learning_rate=0.1, max_depth=3
2. **Tune learning rate first**: Lower is usually better (0.01-0.1)
3. **Optimize tree count**: Use validation set to find optimal
4. **Control depth**: Shallow trees (2-4) work well
5. **Use subsampling**: 0.8-0.9 for regularization
6. **Monitor overfitting**: Use validation set
7. **Early stopping**: Stop when validation error stops improving
8. **Feature importance**: Analyze to understand model

## Real-World Applications

- Click-through rate prediction
- Fraud detection
- Medical diagnosis
- Credit risk assessment
- Customer churn prediction
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
