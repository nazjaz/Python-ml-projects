# Random Forest Classifier API Documentation

## RandomForestClassifier Class

### `RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None)`

Random Forest Classifier with bootstrap sampling and feature importance.

#### Parameters

- `n_estimators` (int): Number of trees in forest (default: 100).
- `max_depth` (int, optional): Maximum depth of trees. If None, nodes expanded until pure (default: None).
- `min_samples_split` (int): Minimum samples required to split node (default: 2).
- `min_samples_leaf` (int): Minimum samples required at leaf node (default: 1).
- `max_features` (int, str, float): Maximum features to consider for split. Options: int, "sqrt", "log2", float 0-1 (default: "sqrt").
- `bootstrap` (bool): Whether to use bootstrap sampling (default: True).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `estimators_` (list): List of fitted decision trees.
- `n_features_` (int): Number of features.
- `classes_` (ndarray): Unique class labels.
- `feature_importances_` (ndarray): Feature importance values.
- `feature_names_` (list, optional): Names of features.

### Methods

#### `fit(X, y)`

Fit the random forest classifier.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).
- `y` (array-like): Target labels of shape (n_samples,).

**Returns:**
- `self`: Returns self for method chaining.

**Raises:**
- `ValueError`: If inputs are invalid.

**Example:**
```python
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X, y)
```

#### `predict(X)`

Predict class labels using majority voting.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Predicted class labels of shape (n_samples,).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
predictions = forest.predict(X)
```

#### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (array-like): Feature matrix of shape (n_samples, n_features).

**Returns:**
- `ndarray`: Class probabilities of shape (n_samples, n_classes).

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
probabilities = forest.predict_proba(X)
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
accuracy = forest.score(X, y)
```

#### `get_feature_importances()`

Get feature importance as dictionary.

**Returns:**
- `dict`: Dictionary mapping feature names to importance values.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
importances = forest.get_feature_importances()
```

#### `plot_feature_importance(save_path=None, show=True, top_n=None)`

Plot feature importance.

**Parameters:**
- `save_path` (str, optional): Path to save figure.
- `show` (bool): Whether to display plot (default: True).
- `top_n` (int, optional): Number of top features to show (default: all).

**Example:**
```python
forest.plot_feature_importance(save_path="importance.png")
```

## Usage Examples

### Basic Random Forest

```python
from src.main import RandomForestClassifier
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

forest = RandomForestClassifier(n_estimators=10)
forest.fit(X, y)
predictions = forest.predict(X)
```

### With Bootstrap Sampling

```python
# With bootstrap (default)
forest = RandomForestClassifier(n_estimators=50, bootstrap=True)
forest.fit(X, y)

# Without bootstrap
forest_no_bootstrap = RandomForestClassifier(n_estimators=50, bootstrap=False)
forest_no_bootstrap.fit(X, y)
```

### Feature Importance

```python
forest = RandomForestClassifier(n_estimators=100)
forest.feature_names_ = ["feature1", "feature2", "feature3"]
forest.fit(X, y)

importances = forest.get_feature_importances()
forest.plot_feature_importance()
```

### Class Probabilities

```python
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X, y)

probabilities = forest.predict_proba(X)
```

### Different max_features

```python
# sqrt of features
forest = RandomForestClassifier(n_estimators=50, max_features="sqrt")
forest.fit(X, y)

# log2 of features
forest = RandomForestClassifier(n_estimators=50, max_features="log2")
forest.fit(X, y)

# Fixed number
forest = RandomForestClassifier(n_estimators=50, max_features=3)
forest.fit(X, y)

# Fraction
forest = RandomForestClassifier(n_estimators=50, max_features=0.5)
forest.fit(X, y)
```
