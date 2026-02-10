# AdaBoost Classifier with Weak Learners and Adaptive Boosting

A Python implementation of AdaBoost (Adaptive Boosting) Classifier from scratch with weak learners (decision stumps) and adaptive boosting iterations. This is the thirty-eighth project in the ML learning series, focusing on understanding adaptive boosting, weak learners, and sample weight adaptation.

## Project Title and Description

The AdaBoost Classifier tool provides a complete implementation of AdaBoost from scratch, including decision stumps as weak learners, adaptive sample weight updates, and weighted voting. It helps users understand how AdaBoost works, how weak learners are combined, and how the algorithm adaptively focuses on hard-to-classify samples.

This tool solves the problem of learning adaptive boosting fundamentals by providing a clear, educational implementation without relying on external ML libraries. It demonstrates weak learners, sample weight adaptation, error-based learner weighting, and weighted voting from scratch.

**Target Audience**: Beginners learning machine learning, students studying ensemble methods, and anyone who needs to understand AdaBoost, adaptive boosting, and weak learners from scratch.

## Features

- AdaBoost implementation from scratch
- Decision stump weak learners (single-level decision trees)
- Adaptive sample weight updates
- Error-based learner weighting (alpha calculation)
- Weighted voting for final predictions
- Class probability prediction
- Feature importance calculation and visualization
- Learning rate control
- Configurable number of iterations
- Early stopping on perfect classification
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for various input types (lists, numpy arrays, pandas DataFrames)
- Binary classification support

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/adaboost-classifier
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
  n_estimators: 50
  learning_rate: 1.0
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_estimators`: Number of weak learners (decision stumps) (default: 50)
- `model.learning_rate`: Learning rate (shrinkage) (default: 1.0)
- `model.random_state`: Random seed (default: null)

## Usage

### Basic Usage

```python
from src.main import AdaBoostClassifier
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Initialize and fit model
adaboost = AdaBoostClassifier(n_estimators=10)
adaboost.fit(X, y)

# Predict
predictions = adaboost.predict(X)
print(f"Predictions: {predictions}")

# Evaluate
accuracy = adaboost.score(X, y)
print(f"Accuracy: {accuracy:.4f}")
```

### With Different Learning Rates

```python
from src.main import AdaBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Lower learning rate (more conservative)
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.5)
adaboost.fit(X, y)

# Higher learning rate (more aggressive)
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1.5)
adaboost.fit(X, y)
```

### With Different Number of Iterations

```python
from src.main import AdaBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Few iterations
adaboost_few = AdaBoostClassifier(n_estimators=10)
adaboost_few.fit(X, y)

# Many iterations
adaboost_many = AdaBoostClassifier(n_estimators=100)
adaboost_many.fit(X, y)
```

### Feature Importance

```python
from src.main import AdaBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.feature_names_ = ["feature1", "feature2"]
adaboost.fit(X, y)

# Get feature importance
importances = adaboost.get_feature_importances()
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.6f}")

# Plot feature importance
adaboost.plot_feature_importance()
```

### Class Probabilities

```python
from src.main import AdaBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X, y)

# Predict probabilities
probabilities = adaboost.predict_proba(X)
print(f"Probabilities shape: {probabilities.shape}")
```

### Get Estimator Errors

```python
from src.main import AdaBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X, y)

# Get error rates for each iteration
errors = adaboost.get_estimator_errors()
for i, error in enumerate(errors):
    print(f"Iteration {i+1}: error={error:.6f}")
```

### With Pandas DataFrame

```python
from src.main import AdaBoostClassifier
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["feature1", "feature2", "feature3"]].values
y = df["target"].values

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.feature_names_ = ["feature1", "feature2", "feature3"]
adaboost.fit(X, y)

predictions = adaboost.predict(X)
```

### Command-Line Usage

Basic AdaBoost:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 50
```

With custom learning rate:

```bash
python src/main.py --input data.csv --target class_label --n-estimators 50 --learning-rate 0.5
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
from src.main import AdaBoostClassifier
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)

# Fit AdaBoost
adaboost = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0
)
adaboost.feature_names_ = [f"feature_{i}" for i in range(5)]
adaboost.fit(X, y)

print(f"Number of weak learners: {len(adaboost.estimators_)}")
print(f"Classes: {adaboost.classes_}")

# Evaluate
accuracy = adaboost.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
importances = adaboost.get_feature_importances()
print("\nFeature Importance:")
for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.6f}")

# Estimator errors
errors = adaboost.get_estimator_errors()
print(f"\nAverage error: {np.mean(errors):.6f}")

# Visualize
adaboost.plot_feature_importance()
```

## Project Structure

```
adaboost-classifier/
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

- `src/main.py`: Core implementation with `AdaBoostClassifier` and `DecisionStump` classes
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
- Decision stump functionality
- Fitting with different parameters
- Adaptive weight updates
- Prediction
- Class probabilities
- Feature importance
- Learning rate variations
- Different number of iterations
- Error handling
- Different input types
- Feature importance visualization

## Understanding AdaBoost

### AdaBoost Algorithm

AdaBoost (Adaptive Boosting) is an ensemble method that combines weak learners:

1. **Initialize**: Equal weights for all samples
2. **For each iteration**:
   - Train weak learner on weighted data
   - Calculate weighted error
   - Calculate learner weight (alpha)
   - Update sample weights (increase for misclassified)
3. **Combine**: Weighted voting using learner weights

### Mathematical Foundation

**Initial Weights:**
```
w₁ = 1/n for all samples
```

**Weighted Error:**
```
εₘ = Σ(wᵢ * I(hₘ(xᵢ) ≠ yᵢ))
```

**Learner Weight (Alpha):**
```
αₘ = (1/2) * log((1 - εₘ) / εₘ)
```

**Weight Update:**
```
wᵢ = wᵢ * exp(αₘ * I(hₘ(xᵢ) ≠ yᵢ))
wᵢ = wᵢ / Σwⱼ  (normalize)
```

**Final Prediction:**
```
H(x) = sign(Σ(αₘ * hₘ(x)))
```

### Decision Stump (Weak Learner)

**Definition:**
- Single-level decision tree
- One feature, one threshold
- Binary split

**Why Weak:**
- Simple model
- Slightly better than random
- Error < 0.5 required

### Adaptive Weight Updates

**Process:**
1. Misclassified samples get higher weights
2. Correctly classified samples get lower weights
3. Next iteration focuses on hard samples
4. Weights normalized to sum to 1

**Benefits:**
- Focuses on difficult samples
- Improves overall accuracy
- Creates diverse weak learners

### Learning Rate

**Default (1.0):**
- Full contribution of each learner
- Standard AdaBoost

**Lower (< 1.0):**
- More conservative updates
- Smoother learning
- Better generalization

**Higher (> 1.0):**
- More aggressive updates
- Faster convergence
- Risk of overfitting

### Early Stopping

**Conditions:**
- Error >= 0.5: Weak learner not better than random
- Error == 0: Perfect classification achieved

**Benefits:**
- Prevents poor learners
- Stops when perfect
- Saves computation

## Troubleshooting

### Common Issues

**Issue**: Low accuracy

**Solution**: 
- Increase n_estimators
- Adjust learning rate
- Check data quality
- Ensure classes are balanced
- Verify feature quality

**Issue**: Slow training

**Solution**: 
- Reduce n_estimators
- Use smaller dataset for testing
- Check for early stopping

**Issue**: Overfitting

**Solution**: 
- Reduce learning rate
- Reduce n_estimators
- Use validation set
- Check for early stopping

**Issue**: Error >= 0.5 warning

**Solution**: 
- Check data quality
- Verify features are informative
- Ensure classes are separable
- May need more features

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: AdaBoost currently supports binary classification only`: Use binary classification data
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Start with defaults**: n_estimators=50, learning_rate=1.0
2. **Tune learning rate**: Lower for smoother learning
3. **Monitor errors**: Check estimator errors to understand learning
4. **Use early stopping**: Let algorithm stop when error >= 0.5
5. **Analyze feature importance**: Understand which features matter
6. **Handle imbalanced classes**: Consider class weights or resampling
7. **Check for perfect fit**: Early stopping when error == 0
8. **Validate on test set**: Use separate validation set

## Real-World Applications

- Face detection
- Text classification
- Medical diagnosis
- Fraud detection
- Quality control
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
