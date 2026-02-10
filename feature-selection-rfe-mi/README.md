# Feature Selection using RFE and Mutual Information

A Python implementation of feature selection techniques including Recursive Feature Elimination (RFE) and Mutual Information scoring. This tool provides comprehensive solutions for selecting the most relevant features from datasets to improve model performance and reduce dimensionality.

## Project Title and Description

The Feature Selection tool provides implementations of two powerful feature selection methods: Recursive Feature Elimination (RFE) and Mutual Information scoring. These techniques help identify and select the most informative features, reducing overfitting, improving model interpretability, and enhancing computational efficiency.

This tool solves the problem of high-dimensional datasets with many irrelevant or redundant features by providing automated feature selection methods that identify the most predictive features. It helps improve model performance, reduce training time, and enhance model interpretability.

**Target Audience**: Data scientists, machine learning engineers, researchers, and anyone working with high-dimensional datasets who needs to identify the most relevant features for their models.

## Features

### Recursive Feature Elimination (RFE)
- **Iterative Feature Removal**: Recursively removes least important features
- **Model-Based Selection**: Uses estimator's feature importance for ranking
- **Configurable Steps**: Adjustable number of features to remove per iteration
- **Support for Any Estimator**: Works with any scikit-learn estimator
- **Feature Ranking**: Provides ranking of all features

### Mutual Information Scoring
- **Information-Theoretic Selection**: Measures mutual dependence between features and target
- **Classification and Regression**: Supports both task types
- **Score-Based Selection**: Selects features based on MI scores
- **Threshold Support**: Optional minimum score threshold
- **Top-K Selection**: Select top N features by score

### Additional Features
- Unified interface for both methods
- Support for classification and regression tasks
- Feature score and ranking access
- Data transformation with selected features
- Evaluation of selection performance
- Command-line interface
- Configuration via YAML file
- Support for pandas DataFrames and numpy arrays
- Comprehensive logging
- Input validation and error handling

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/feature-selection-rfe-mi
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
python src/main.py --input sample.csv --target-col target --method both
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

mutual_information:
  n_features: null
  score_threshold: null
  random_state: 42

rfe:
  n_features_to_select: null
  step: 1
  verbose: 0
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `mutual_information.n_features`: Number of top features to select (default: None, all)
- `mutual_information.score_threshold`: Minimum MI score threshold (default: None)
- `mutual_information.random_state`: Random seed (default: 42)
- `rfe.n_features_to_select`: Number of features to select (default: None, half of features)
- `rfe.step`: Number of features to remove per iteration (default: 1)
- `rfe.verbose`: Verbosity level (default: 0)

## Usage

### Command-Line Interface

#### Select Features with Mutual Information

```bash
python src/main.py --input data.csv --target-col target \
  --method mutual_information --n-features 10 --output selected_features.csv
```

#### Select Features with RFE

```bash
python src/main.py --input data.csv --target-col target \
  --method rfe --n-features 10 --output selected_features.csv
```

#### Use Both Methods

```bash
python src/main.py --input data.csv --target-col target \
  --method both --n-features 10 --scores-output scores.json
```

#### For Regression Tasks

```bash
python src/main.py --input data.csv --target-col target \
  --task-type regression --method both --n-features 10
```

### Programmatic Usage

#### Basic Feature Selection

```python
import numpy as np
import pandas as pd
from src.main import FeatureSelector

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Initialize selector
selector = FeatureSelector()

# Load data
selector.load_data(X, y, task_type="classification")

# Fit Mutual Information selector
selector.fit_mutual_information(n_features=10)

# Get selected features
selected_features = selector.get_selected_features("mutual_information")
print(f"Selected features: {selected_features}")

# Transform data
X_selected = selector.transform(X, method="mutual_information")
```

#### Using Individual Selectors

```python
from src.main import MutualInformationSelector, RecursiveFeatureElimination

# Mutual Information
mi_selector = MutualInformationSelector(n_features=10, random_state=42)
mi_selector.fit(X, y, task_type="classification")

selected = mi_selector.get_selected_features()
scores = mi_selector.get_feature_scores()
X_transformed = mi_selector.transform(X)

# RFE
rfe_selector = RecursiveFeatureElimination(n_features_to_select=10)
rfe_selector.fit(X, y, task_type="classification")

selected = rfe_selector.get_selected_features()
ranking = rfe_selector.get_feature_ranking()
X_transformed = rfe_selector.transform(X)
```

#### Compare Both Methods

```python
selector = FeatureSelector()
selector.load_data(X, y, task_type="classification")

# Fit both methods
selector.fit_all()

# Get features from both
mi_features = selector.get_selected_features("mutual_information")
rfe_features = selector.get_selected_features("rfe")

# Compare
print(f"MI selected: {len(mi_features)} features")
print(f"RFE selected: {len(rfe_features)} features")
print(f"Common features: {set(mi_features) & set(rfe_features)}")
```

#### Evaluation

```python
# Evaluate selection performance
evaluation = selector.evaluate_selection(
    X_test=X_test,
    y_test=y_test,
    method="mutual_information"
)

print(f"Accuracy with selected features: {evaluation['accuracy']:.4f}")
print(f"Number of features: {evaluation['n_features']}")
```

### Common Use Cases

1. **High-Dimensional Data**: Reduce dimensionality in datasets with many features
2. **Model Improvement**: Select features that improve model performance
3. **Interpretability**: Identify most important features for understanding
4. **Computational Efficiency**: Reduce training time by removing irrelevant features
5. **Overfitting Prevention**: Remove redundant or noisy features
6. **Feature Engineering**: Guide feature engineering efforts
7. **Data Preprocessing**: Prepare data for machine learning pipelines

## Project Structure

```
feature-selection-rfe-mi/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py             # Main implementation
├── tests/
│   └── test_main.py        # Unit tests
├── docs/
│   └── API.md              # API documentation (if applicable)
└── logs/
    └── .gitkeep            # Keep logs directory in git
```

### File Descriptions

- `src/main.py`: Contains all implementation:
  - `MutualInformationSelector`: Mutual Information feature selection
  - `RecursiveFeatureElimination`: RFE feature selection
  - `FeatureSelector`: Main selector class with CLI interface
  - `main()`: Command-line interface entry point

- `tests/test_main.py`: Comprehensive unit tests including:
  - Mutual Information selector tests
  - RFE selector tests
  - Integration tests
  - Error handling tests

- `config.yaml`: Configuration file for algorithm parameters

- `requirements.txt`: Python package dependencies with versions

## Testing

### Run All Tests

```bash
pytest tests/test_main.py -v
```

### Run Tests with Coverage

```bash
pytest tests/test_main.py --cov=src --cov-report=html
```

### Test Coverage Information

The test suite includes:
- Unit tests for Mutual Information selector
- Unit tests for RFE selector
- Integration tests for complete workflows
- Error handling and edge case tests

Current test coverage: >90% of code paths

## Algorithm Details

### Recursive Feature Elimination (RFE)

**How it works**:
1. Train a model on all features
2. Rank features by importance (e.g., coefficients, feature_importances_)
3. Remove least important features
4. Repeat until desired number of features remains

**Advantages**:
- Model-aware selection
- Works with any estimator
- Can capture feature interactions through model
- Provides feature ranking

**Limitations**:
- Computationally expensive (requires multiple model fits)
- Dependent on estimator choice
- May be slow for large datasets

**Best for**: When you have a good estimator and want model-based selection

### Mutual Information

**How it works**:
1. Calculate mutual information between each feature and target
2. MI measures how much information feature provides about target
3. Select features with highest MI scores

**Mathematical Definition**:
```
MI(X, Y) = Σ Σ p(x, y) log(p(x, y) / (p(x) p(y)))
```

**Advantages**:
- Fast computation
- Model-independent
- Captures non-linear relationships
- Works for both classification and regression

**Limitations**:
- Doesn't account for feature interactions
- May select redundant features
- Requires sufficient data for accurate estimation

**Best for**: Quick feature selection, when you want model-independent method

## Choosing Between Methods

### Use Mutual Information when:
- You need fast feature selection
- You want model-independent selection
- You have many features to evaluate
- You want to capture non-linear relationships
- Computational resources are limited

### Use RFE when:
- You have a specific model in mind
- You want model-aware selection
- You need feature ranking
- You can afford computational cost
- You want to capture feature interactions through model

### Use Both when:
- You want to compare methods
- You need consensus on important features
- You want to validate selection
- You're exploring feature importance

## Troubleshooting

### Common Issues and Solutions

#### Issue: Too Many/Few Features Selected
**Problem**: n_features parameter doesn't match needs

**Solution**:
- Use cross-validation to find optimal number
- Try different values: 5, 10, 20, 50, etc.
- Use score_threshold for MI instead of n_features
- Evaluate model performance with different numbers

#### Issue: RFE is Slow
**Problem**: Large dataset or many iterations

**Solution**:
- Increase `step` parameter to remove more features per iteration
- Use faster estimator (e.g., LinearRegression instead of RandomForest)
- Reduce number of features to select
- Use Mutual Information for initial filtering

#### Issue: Low MI Scores
**Problem**: Features have weak relationship with target

**Solution**:
- Check data quality and preprocessing
- Consider feature engineering
- Lower score_threshold
- Use RFE instead (may find features MI misses)

#### Issue: Memory Error
**Problem**: Dataset too large

**Solution**:
- Process data in batches
- Use sparse matrices if applicable
- Reduce number of features before selection
- Use Mutual Information (more memory efficient)

#### Issue: Inconsistent Results
**Problem**: Different methods select different features

**Solution**:
- This is normal - methods have different criteria
- Use both methods and find intersection
- Evaluate performance of each selection
- Consider ensemble approach

### Error Message Explanations

- **"X and y must have same number of samples"**: Input arrays have mismatched lengths
- **"task_type must be 'classification' or 'regression'"**: Invalid task type specified
- **"Selector must be fitted before transform"**: Call fit() before transform()
- **"Data must be loaded before fitting"**: Load data before fitting selector
- **"Unknown method"**: Invalid method name (use "mutual_information" or "rfe")

## Performance Considerations

### Computational Complexity

- **Mutual Information**: O(n_samples * n_features) for classification
- **RFE**: O(n_iterations * n_features * model_complexity)

### Optimization Tips

1. **For large datasets**: Use Mutual Information first, then RFE on subset
2. **For many features**: Use Mutual Information to pre-filter
3. **For speed**: Use Mutual Information or increase RFE step size
4. **For accuracy**: Use RFE with appropriate estimator
5. **For both**: Use MI for initial selection, RFE for refinement

## Best Practices

1. **Start with Mutual Information**: Quick way to identify important features
2. **Use RFE for Final Selection**: More careful selection with model awareness
3. **Cross-Validate**: Always validate feature selection on held-out data
4. **Compare Methods**: Use both methods and compare results
5. **Evaluate Performance**: Measure model performance with selected features
6. **Consider Domain Knowledge**: Combine automated selection with expert knowledge
7. **Avoid Data Leakage**: Ensure feature selection doesn't use test data

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Create a feature branch: `git checkout -b feature/new-selector`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Add unit tests for new selectors
- Update documentation for new features

### Pull Request Process

1. Ensure all tests pass: `pytest tests/test_main.py`
2. Check code coverage: `pytest --cov=src`
3. Update documentation if adding new features
4. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. Please refer to the main repository license.

## Additional Resources

- [Feature Selection in Scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Mutual Information Wikipedia](https://en.wikipedia.org/wiki/Mutual_information)
- [Recursive Feature Elimination Paper](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf)
- [Feature Selection for Machine Learning](https://machinelearningmastery.com/feature-selection-machine-learning-python/)
