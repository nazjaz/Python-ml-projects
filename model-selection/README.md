# Model Selection with Nested Cross-Validation and Learning Curves

A Python implementation of model selection using nested cross-validation and learning curves for bias-variance analysis. This is the forty-fifth project in the ML learning series, focusing on understanding model selection, nested cross-validation, and bias-variance trade-off analysis.

## Project Title and Description

The Model Selection tool provides complete implementations of nested cross-validation and learning curves from scratch. It includes outer CV for model selection, inner CV for hyperparameter tuning, learning curve generation, and bias-variance analysis. It helps users understand how to properly select models, avoid overfitting in model selection, and diagnose bias-variance issues.

This tool solves the problem of proper model selection and bias-variance diagnosis by providing clear, educational implementations without relying on external ML libraries. It demonstrates nested cross-validation, learning curves, and bias-variance analysis from scratch.

**Target Audience**: Beginners learning machine learning, students studying model selection, and anyone who needs to understand nested cross-validation and bias-variance analysis from scratch.

## Features

- Nested cross-validation (outer CV for model selection, inner CV for hyperparameter tuning)
- Learning curves generation
- Bias-variance analysis and diagnosis
- Model comparison across multiple estimators
- Cross-validation support
- Configurable scoring functions
- Learning curve visualization
- Result analysis and reporting
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for CSV input files

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/model-selection
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
python src/main.py --input sample.csv --target label --nested-cv --estimators-config estimators.json
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model_selection:
  outer_cv: 5
  inner_cv: 5
  cv: 5
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model_selection.outer_cv`: Number of outer CV folds for nested CV (default: 5)
- `model_selection.inner_cv`: Number of inner CV folds for nested CV (default: 5)
- `model_selection.cv`: Number of CV folds for learning curves (default: 5)
- `model_selection.random_state`: Random seed (default: null)

## Usage

### Nested Cross-Validation

```python
from src.main import NestedCrossValidation
from src.example_estimator import SimpleClassifier
import numpy as np

X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

estimators = {
    "classifier1": SimpleClassifier(),
    "classifier2": SimpleClassifier(),
}

param_grids = {
    "classifier1": {"max_depth": [3, 5], "min_samples_split": [2, 5]},
    "classifier2": {"max_depth": [5, 7], "min_samples_split": [2, 10]},
}

nested_cv = NestedCrossValidation(
    estimators=estimators,
    param_grids=param_grids,
    outer_cv=5,
    inner_cv=5,
    verbose=1,
)
nested_cv.fit(X, y)

print(f"Best estimator: {nested_cv.best_estimator_name_}")
print(f"Best score: {nested_cv.best_score_:.6f}")
```

### Learning Curves

```python
from src.main import LearningCurves
from src.example_estimator import SimpleClassifier
import numpy as np

X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

estimator = SimpleClassifier()
lc = LearningCurves(estimator=estimator, cv=5, verbose=1)
lc.fit(X, y)

analysis = lc.get_bias_variance_analysis()
print(f"Final training score: {analysis['final_train_score']:.6f}")
print(f"Final validation score: {analysis['final_val_score']:.6f}")
print(f"Gap: {analysis['gap']:.6f}")
print(f"Diagnosis: {analysis['diagnosis']}")

lc.plot_learning_curves()
```

### Bias-Variance Analysis

```python
from src.main import LearningCurves
from src.example_estimator import SimpleClassifier
import numpy as np

X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

estimator = SimpleClassifier()
lc = LearningCurves(estimator=estimator, cv=5)
lc.fit(X, y)

analysis = lc.get_bias_variance_analysis()

if analysis["diagnosis"] == "High Variance (Overfitting)":
    print("Model is overfitting - consider regularization")
elif analysis["diagnosis"] == "High Bias (Underfitting)":
    print("Model is underfitting - consider more complex model")
else:
    print("Model is well-balanced")
```

### Complete Example

```python
from src.main import NestedCrossValidation, LearningCurves
from src.example_estimator import SimpleClassifier, SimpleRegressor
import numpy as np

# Generate sample data
X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

# Nested Cross-Validation
estimators = {
    "classifier1": SimpleClassifier(),
    "classifier2": SimpleClassifier(),
}

param_grids = {
    "classifier1": {"max_depth": [3, 5], "min_samples_split": [2, 5]},
    "classifier2": {"max_depth": [5, 7], "min_samples_split": [2, 10]},
}

nested_cv = NestedCrossValidation(
    estimators=estimators,
    param_grids=param_grids,
    outer_cv=5,
    inner_cv=5,
    verbose=1,
)
nested_cv.fit(X, y)

print(f"\n=== Nested CV Results ===")
print(f"Best estimator: {nested_cv.best_estimator_name_}")
print(f"Best score: {nested_cv.best_score_:.6f}")

# Learning Curves
best_estimator = SimpleClassifier()
lc = LearningCurves(estimator=best_estimator, cv=5, verbose=1)
lc.fit(X, y)

analysis = lc.get_bias_variance_analysis()
print(f"\n=== Bias-Variance Analysis ===")
print(f"Final training score: {analysis['final_train_score']:.6f}")
print(f"Final validation score: {analysis['final_val_score']:.6f}")
print(f"Gap: {analysis['gap']:.6f}")
print(f"Diagnosis: {analysis['diagnosis']}")

lc.plot_learning_curves()
```

### Command-Line Usage

Nested cross-validation:

```bash
python src/main.py --input data.csv --target label --nested-cv --estimators-config estimators.json --output results.csv
```

Learning curves:

```bash
python src/main.py --input data.csv --target label --learning-curves --plot-learning-curves
```

Both:

```bash
python src/main.py --input data.csv --target label --nested-cv --learning-curves --estimators-config estimators.json --plot-learning-curves --output results.csv
```

With custom CV:

```bash
python src/main.py --input data.csv --target label --nested-cv --outer-cv 10 --inner-cv 5 --estimators-config estimators.json
```

### Estimators Configuration JSON Format

```json
{
  "estimators": [
    {
      "name": "classifier1",
      "type": "SimpleClassifier",
      "param_grid": {
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "learning_rate": [0.01, 0.1, 0.5]
      }
    },
    {
      "name": "classifier2",
      "type": "SimpleClassifier",
      "param_grid": {
        "max_depth": [5, 7, 10],
        "min_samples_split": [2, 10],
        "learning_rate": [0.1, 0.5]
      }
    }
  ]
}
```

## Project Structure

```
model-selection/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   ├── main.py              # Main implementation
│   └── example_estimator.py  # Example estimators
├── tests/
│   └── test_main.py         # Unit tests
├── docs/
│   └── API.md               # API documentation
└── logs/
    └── .gitkeep             # Log directory
```

### File Descriptions

- `src/main.py`: Core implementation with `NestedCrossValidation` and `LearningCurves` classes
- `src/example_estimator.py`: Example estimators for demonstration
- `config.yaml`: Configuration file for model selection settings
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
- Nested cross-validation functionality
- Learning curves calculation
- Bias-variance analysis
- Error handling

## Understanding Model Selection

### Nested Cross-Validation

**Concept:**
- Outer CV: Model selection
- Inner CV: Hyperparameter tuning
- Prevents overfitting in model selection

**Process:**
1. **Outer loop**: Split data into train/test
2. **Inner loop**: On training data, perform CV for hyperparameter tuning
3. **Evaluation**: Test best model on test data
4. **Selection**: Choose model with best average test score

**Why Nested:**
- Prevents data leakage
- Unbiased model selection
- Proper hyperparameter tuning

**Structure:**
```
Outer Fold 1:
  Train: [Inner CV for hyperparameter tuning]
  Test: [Evaluate best model]
Outer Fold 2:
  ...
```

### Learning Curves

**Concept:**
- Plot training and validation scores vs training set size
- Shows how model performance changes with more data
- Diagnoses bias-variance issues

**Process:**
1. Start with small training set
2. Train model and evaluate on train and validation
3. Increase training set size
4. Repeat and plot scores

**Interpretation:**
- **High gap**: Overfitting (high variance)
- **Low scores**: Underfitting (high bias)
- **Converging**: Good fit

### Bias-Variance Analysis

**High Bias (Underfitting):**
- Both train and validation scores are low
- Gap is small
- Model too simple
- Solution: Increase model complexity

**High Variance (Overfitting):**
- Train score high, validation score low
- Large gap
- Model too complex
- Solution: Regularization, simpler model

**Balanced:**
- Both scores are high
- Gap is small
- Good model fit

## Troubleshooting

### Common Issues

**Issue**: Nested CV taking too long

**Solution**: 
- Reduce outer_cv and inner_cv
- Reduce parameter grid size
- Use smaller dataset for testing

**Issue**: Learning curves not showing convergence

**Solution**: 
- Increase training set sizes
- Check if enough data
- Verify model is learning

**Issue**: High variance diagnosis

**Solution**: 
- Add regularization
- Reduce model complexity
- Get more training data
- Use simpler model

**Issue**: High bias diagnosis

**Solution**: 
- Increase model complexity
- Add more features
- Use more powerful model
- Reduce regularization

### Error Messages

- `ValueError: --estimators-config required for nested CV`: Provide estimators config file
- `ValueError: Target column not found`: Check target column name
- `ValueError: Learning curves must be fitted before analysis`: Call `fit()` first

## Best Practices

1. **Use nested CV**: For proper model selection
2. **Learning curves**: Always check for bias-variance issues
3. **Outer CV**: 5-10 folds for model selection
4. **Inner CV**: 3-5 folds for hyperparameter tuning
5. **Training sizes**: Use 10-20 points for learning curves
6. **Diagnosis**: Act on bias-variance diagnosis
7. **Validation set**: Keep separate final validation set

## Real-World Applications

- Model comparison
- Algorithm selection
- Hyperparameter tuning
- Bias-variance diagnosis
- Educational purposes for learning model selection

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
