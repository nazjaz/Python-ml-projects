# Model Selection API Documentation

## NestedCrossValidation Class

### `NestedCrossValidation(estimators, param_grids, outer_cv=5, inner_cv=5, scoring=None, verbose=0, random_state=None)`

Nested Cross-Validation for model selection.

#### Parameters

- `estimators` (dict): Dictionary of estimator names and instances.
- `param_grids` (dict): Dictionary of estimator names and parameter grids.
- `outer_cv` (int): Number of outer CV folds (default: 5).
- `inner_cv` (int): Number of inner CV folds (default: 5).
- `scoring` (callable, optional): Scoring function (default: estimator.score).
- `verbose` (int): Verbosity level (default: 0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `best_estimator_name_` (str): Name of best estimator.
- `best_params_` (dict): Best parameters for best estimator.
- `best_score_` (float): Best cross-validation score.
- `cv_results_` (dict): Cross-validation results.

### Methods

#### `fit(X, y)`

Fit nested cross-validation.

**Parameters:**
- `X` (ndarray): Feature matrix.
- `y` (ndarray): Target values.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
nested_cv = NestedCrossValidation(estimators=estimators, param_grids=param_grids)
nested_cv.fit(X, y)
```

## LearningCurves Class

### `LearningCurves(estimator, train_sizes=None, cv=5, scoring=None, verbose=0, random_state=None)`

Learning Curves for bias-variance analysis.

#### Parameters

- `estimator`: Base estimator.
- `train_sizes` (ndarray, optional): Training set sizes (default: auto).
- `cv` (int): Number of cross-validation folds (default: 5).
- `scoring` (callable, optional): Scoring function (default: estimator.score).
- `verbose` (int): Verbosity level (default: 0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `train_scores_` (ndarray): Training scores for each training size.
- `val_scores_` (ndarray): Validation scores for each training size.
- `train_sizes_` (ndarray): Training set sizes used.

### Methods

#### `fit(X, y)`

Fit learning curves.

**Parameters:**
- `X` (ndarray): Feature matrix.
- `y` (ndarray): Target values.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
lc = LearningCurves(estimator=estimator, cv=5)
lc.fit(X, y)
```

#### `plot_learning_curves(save_path=None, show=True, title=None)`

Plot learning curves.

**Parameters:**
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).
- `title` (str, optional): Plot title.

**Example:**
```python
lc.plot_learning_curves(save_path="learning_curves.png")
```

#### `get_bias_variance_analysis()`

Get bias-variance analysis.

**Returns:**
- `dict`: Dictionary with bias and variance estimates.

**Raises:**
- `ValueError`: If learning curves not fitted.

**Example:**
```python
analysis = lc.get_bias_variance_analysis()
print(analysis["diagnosis"])
```

## Usage Examples

### Nested Cross-Validation

```python
from src.main import NestedCrossValidation
from src.example_estimator import SimpleClassifier

estimators = {
    "classifier1": SimpleClassifier(),
    "classifier2": SimpleClassifier(),
}

param_grids = {
    "classifier1": {"max_depth": [3, 5]},
    "classifier2": {"max_depth": [5, 7]},
}

nested_cv = NestedCrossValidation(
    estimators=estimators,
    param_grids=param_grids,
    outer_cv=5,
    inner_cv=5,
)
nested_cv.fit(X, y)

print(f"Best estimator: {nested_cv.best_estimator_name_}")
print(f"Best score: {nested_cv.best_score_}")
```

### Learning Curves

```python
from src.main import LearningCurves
from src.example_estimator import SimpleClassifier

estimator = SimpleClassifier()
lc = LearningCurves(estimator=estimator, cv=5)
lc.fit(X, y)

analysis = lc.get_bias_variance_analysis()
lc.plot_learning_curves()
```

### Bias-Variance Analysis

```python
lc = LearningCurves(estimator=estimator, cv=5)
lc.fit(X, y)

analysis = lc.get_bias_variance_analysis()
print(f"Diagnosis: {analysis['diagnosis']}")
print(f"Gap: {analysis['gap']:.6f}")
```
