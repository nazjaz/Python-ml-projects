# AdaBoost Classifier API Documentation

## AdaBoostClassifier Class

### `AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=None)`

AdaBoost Classifier with weak learners and adaptive boosting.

#### Parameters

- `n_estimators` (int): Number of weak learners (decision stumps) (default: 50).
- `learning_rate` (float): Learning rate (shrinkage) (default: 1.0).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `estimators_` (list): List of fitted decision stumps.
- `estimator_weights_` (list): List of learner weights (alpha values).
- `n_features_` (int): Number of features.
- `classes_` (ndarray): Unique class labels.
- `feature_importances_` (ndarray): Feature importance values.
- `feature_names_` (list, optional): Names of features.

### Methods

#### `fit(X, y)`

Fit the AdaBoost classifier.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,) (binary: 0, 1 or -1, 1).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X, y)
```

#### `predict(X)`

Predict class labels using weighted voting.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Predicted class labels of shape (n_samples,).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
predictions = adaboost.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Class probabilities of shape (n_samples, 2).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
probabilities = adaboost.predict_proba(X)
```

#### `score(X, y)`

Calculate classification accuracy.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): True labels.

**Returns:**
- `float`: Classification accuracy (0-1).

**Example:**
```python
accuracy = adaboost.score(X, y)
```

#### `get_feature_importances()`

Get feature importance as dictionary.

**Returns:**
- `dict`: Dictionary mapping feature names to importance values.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
importances = adaboost.get_feature_importances()
```

#### `get_estimator_errors()`

Get error rates for each estimator.

**Returns:**
- `list`: List of error rates for each iteration.

**Example:**
```python
errors = adaboost.get_estimator_errors()
```

#### `plot_feature_importance(save_path=None, show=True, top_n=None)`

Plot feature importance.

**Parameters:**
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).
- `top_n` (int, optional): Number of top features to show (default: all).

**Example:**
```python
adaboost.plot_feature_importance(save_path="importance.png")
```

## DecisionStump Class

### `DecisionStump()`

Decision Stump (weak learner) - single-level decision tree.

#### Attributes

- `feature_index` (int, optional): Index of feature used for split.
- `threshold` (float, optional): Threshold value for split.
- `polarity` (int): Polarity of split (1 or -1).
- `alpha` (float): Learner weight.

### Methods

#### `fit(X, y, sample_weights)`

Fit decision stump to weighted data.

**Parameters:**
- `X` (array-like): Feature matrix.
- `y` (array-like): Target labels (-1, 1).
- `sample_weights` (array-like): Sample weights.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
stump = DecisionStump()
stump.fit(X, y, sample_weights)
```

#### `predict(X)`

Predict class labels.

**Parameters:**
- `X` (array-like): Feature matrix.

**Returns:**
- `ndarray`: Predicted labels (-1, 1).

**Example:**
```python
predictions = stump.predict(X)
```

## Usage Examples

### Basic AdaBoost

```python
from src.main import AdaBoostClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X, y)
predictions = adaboost.predict(X)
```

### With Learning Rate

```python
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.5)
adaboost.fit(X, y)
```

### Feature Importance

```python
adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.feature_names_ = ["feature1", "feature2"]
adaboost.fit(X, y)

importances = adaboost.get_feature_importances()
adaboost.plot_feature_importance()
```

### Estimator Errors

```python
adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X, y)

errors = adaboost.get_estimator_errors()
for i, error in enumerate(errors):
    print(f"Iteration {i+1}: {error:.6f}")
```

### Class Probabilities

```python
adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X, y)

probabilities = adaboost.predict_proba(X)
```
