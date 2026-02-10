# Ensemble Methods: Voting, Bagging, and Stacking

A Python implementation of ensemble methods from scratch including voting (hard and soft), bagging (bootstrap aggregating), and stacking (stacked generalization) with multiple base models. This is the fortieth project in the ML learning series, focusing on understanding ensemble learning, combining multiple models, and improving prediction accuracy.

## Project Title and Description

The Ensemble Methods tool provides complete implementations of three major ensemble techniques from scratch: voting classifiers, bagging classifiers, and stacking classifiers. It includes multiple base models (Decision Trees, KNN) and demonstrates how combining models can improve performance. It helps users understand how ensemble methods work, how different models can be combined, and how each technique addresses different aspects of model combination.

This tool solves the problem of learning ensemble learning fundamentals by providing clear, educational implementations without relying on external ML libraries. It demonstrates voting (hard and soft), bootstrap aggregating, and stacked generalization from scratch.

**Target Audience**: Beginners learning machine learning, students studying ensemble methods, and anyone who needs to understand voting, bagging, and stacking from scratch.

## Features

- Voting Classifier with hard and soft voting
- Bagging Classifier with bootstrap sampling
- Stacking Classifier with meta-learner
- Multiple base models (Decision Tree, KNN)
- Bootstrap sampling with replacement
- Cross-validation for stacking
- Feature subsampling for bagging
- Class probability prediction
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
cd /path/to/Python-ml-projects/ensemble-methods
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
python src/main.py --input sample.csv --target class_label --method voting
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  n_estimators: 10
  max_samples: 0.8
  max_features: 0.8
  cv: 5
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.n_estimators`: Number of estimators for bagging (default: 10)
- `model.max_samples`: Fraction of samples for bagging, 0-1 (default: 0.8)
- `model.max_features`: Fraction of features for bagging, 0-1 (default: 0.8)
- `model.cv`: Cross-validation folds for stacking (default: 5)
- `model.random_state`: Random seed (default: null)

## Usage

### Voting Classifier

#### Hard Voting

```python
from src.main import VotingClassifier, SimpleDecisionTree, SimpleKNN
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

dt1 = SimpleDecisionTree(max_depth=3)
dt2 = SimpleDecisionTree(max_depth=5)
knn = SimpleKNN(n_neighbors=5)

voting = VotingClassifier(
    estimators=[("dt1", dt1), ("dt2", dt2), ("knn", knn)],
    voting="hard"
)
voting.fit(X, y)

predictions = voting.predict(X)
accuracy = voting.score(X, y)
```

#### Soft Voting

```python
voting = VotingClassifier(
    estimators=[("dt1", dt1), ("dt2", dt2), ("knn", knn)],
    voting="soft"
)
voting.fit(X, y)

predictions = voting.predict(X)
probabilities = voting.predict_proba(X)
```

### Bagging Classifier

```python
from src.main import BaggingClassifier, SimpleDecisionTree
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

base_estimator = SimpleDecisionTree(max_depth=3)
bagging = BaggingClassifier(
    base_estimator=base_estimator,
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8
)
bagging.fit(X, y)

predictions = bagging.predict(X)
accuracy = bagging.score(X, y)
```

### Stacking Classifier

```python
from src.main import StackingClassifier, SimpleDecisionTree, SimpleKNN
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

dt = SimpleDecisionTree(max_depth=3)
knn = SimpleKNN(n_neighbors=5)
meta_estimator = SimpleDecisionTree(max_depth=2)

stacking = StackingClassifier(
    base_estimators=[("dt", dt), ("knn", knn)],
    meta_estimator=meta_estimator,
    cv=5
)
stacking.fit(X, y)

predictions = stacking.predict(X)
accuracy = stacking.score(X, y)
```

### Command-Line Usage

Voting classifier:

```bash
python src/main.py --input data.csv --target class_label --method voting --voting hard
```

Soft voting:

```bash
python src/main.py --input data.csv --target class_label --method voting --voting soft
```

Bagging classifier:

```bash
python src/main.py --input data.csv --target class_label --method bagging --n-estimators 20
```

Stacking classifier:

```bash
python src/main.py --input data.csv --target class_label --method stacking
```

Save predictions:

```bash
python src/main.py --input data.csv --target class_label --method voting --output predictions.csv
```

Make predictions on new data:

```bash
python src/main.py --input train.csv --target class_label --method voting --predict test.csv --output-predictions predictions.csv
```

### Complete Example

```python
from src.main import (
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier,
    SimpleDecisionTree,
    SimpleKNN,
)
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int)

# Voting Classifier
dt1 = SimpleDecisionTree(max_depth=3)
dt2 = SimpleDecisionTree(max_depth=5)
knn = SimpleKNN(n_neighbors=5)

voting = VotingClassifier(
    estimators=[("dt1", dt1), ("dt2", dt2), ("knn", knn)],
    voting="soft"
)
voting.fit(X, y)
voting_accuracy = voting.score(X, y)
print(f"Voting accuracy: {voting_accuracy:.4f}")

# Bagging Classifier
base_estimator = SimpleDecisionTree(max_depth=3)
bagging = BaggingClassifier(
    base_estimator=base_estimator,
    n_estimators=20,
    max_samples=0.8,
    max_features=0.8
)
bagging.fit(X, y)
bagging_accuracy = bagging.score(X, y)
print(f"Bagging accuracy: {bagging_accuracy:.4f}")

# Stacking Classifier
dt = SimpleDecisionTree(max_depth=3)
knn = SimpleKNN(n_neighbors=5)
meta = SimpleDecisionTree(max_depth=2)

stacking = StackingClassifier(
    base_estimators=[("dt", dt), ("knn", knn)],
    meta_estimator=meta,
    cv=5
)
stacking.fit(X, y)
stacking_accuracy = stacking.score(X, y)
print(f"Stacking accuracy: {stacking_accuracy:.4f}")
```

## Project Structure

```
ensemble-methods/
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

- `src/main.py`: Core implementation with ensemble method classes
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
- Base model functionality (Decision Tree, KNN)
- Voting classifier (hard and soft)
- Bagging classifier
- Stacking classifier
- Error handling
- Different input types

## Understanding Ensemble Methods

### Voting Classifier

**Hard Voting:**
- Each model votes for a class
- Final prediction is majority vote
- Simple and effective

**Soft Voting:**
- Each model provides probabilities
- Final prediction is average of probabilities
- Can be more accurate than hard voting

**When to Use:**
- When models are diverse
- When models have similar performance
- Quick ensemble method

### Bagging (Bootstrap Aggregating)

**Process:**
1. Create multiple bootstrap samples (with replacement)
2. Train model on each sample
3. Aggregate predictions (majority vote or average)

**Benefits:**
- Reduces variance
- Reduces overfitting
- Works well with high-variance models

**When to Use:**
- When base model is unstable (e.g., decision trees)
- When you have enough data
- To reduce overfitting

### Stacking (Stacked Generalization)

**Process:**
1. Train base models on data
2. Use cross-validation to get base model predictions
3. Train meta-learner on base model predictions
4. Final prediction uses meta-learner

**Benefits:**
- Can learn optimal combination
- Often best performance
- Handles model diversity well

**When to Use:**
- When you have diverse base models
- When you want best possible performance
- When you have enough data for CV

### Comparison

| Method | Complexity | Performance | Use Case |
|--------|-----------|-------------|----------|
| Voting | Low | Good | Quick ensemble |
| Bagging | Medium | Very Good | Unstable models |
| Stacking | High | Best | Maximum accuracy |

## Troubleshooting

### Common Issues

**Issue**: Low accuracy with voting

**Solution**: 
- Use more diverse models
- Try soft voting instead of hard
- Check base model performance
- Ensure models are trained properly

**Issue**: Bagging not improving performance

**Solution**: 
- Increase n_estimators
- Adjust max_samples and max_features
- Check base model quality
- Ensure enough data

**Issue**: Stacking overfitting

**Solution**: 
- Increase CV folds
- Use simpler meta-learner
- Reduce base model complexity
- Use more data

### Error Messages

- `ValueError: Model must be fitted before prediction`: Call `fit()` before `predict()`
- `ValueError: X and y must have the same length`: Ensure matching lengths

## Best Practices

1. **Diversity**: Use diverse base models for better ensembles
2. **Voting**: Start with voting for quick results
3. **Bagging**: Use bagging for unstable models (trees)
4. **Stacking**: Use stacking when you need best performance
5. **Cross-validation**: Use proper CV for stacking
6. **Base models**: Ensure base models are reasonably good
7. **Data size**: Ensure enough data for ensemble methods

## Real-World Applications

- Classification competitions
- Medical diagnosis
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
